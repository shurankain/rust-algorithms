// Paged Attention
//
// Memory-efficient attention mechanism that manages KV cache using paging,
// inspired by virtual memory systems. Key innovation from vLLM (SOSP 2023).
//
// Key concepts:
// - Block-based storage: KV cache divided into fixed-size blocks (pages)
// - Block table: Maps logical blocks to physical blocks (like page table)
// - Dynamic allocation: Blocks allocated on-demand, freed when done
// - Copy-on-Write (CoW): Enables memory sharing for beam search/sampling
// - Memory pooling: Reuse freed blocks to avoid fragmentation
//
// Benefits:
// - Near-zero memory waste (vs pre-allocation for max sequence length)
// - Memory sharing between sequences with common prefixes
// - Efficient batched inference with varying sequence lengths
// - Support for beam search without memory explosion
//
// Memory layout:
// - Physical blocks: Contiguous memory storing KV pairs
// - Block size: Number of tokens per block (typically 16-64)
// - Each block stores: [num_heads, block_size, head_dim] for K and V

use std::collections::{HashMap, HashSet};

/// KV cache data: (keys, values) where each is [num_layers][data...]
pub type KvData = (Vec<Vec<f32>>, Vec<Vec<f32>>);

/// Configuration for paged attention
#[derive(Debug, Clone)]
pub struct PagedAttentionConfig {
    /// Number of tokens per block
    pub block_size: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Dimension per head
    pub head_dim: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Maximum number of blocks in the pool
    pub max_blocks: usize,
}

impl PagedAttentionConfig {
    /// Create a new configuration
    pub fn new(
        block_size: usize,
        num_heads: usize,
        head_dim: usize,
        num_layers: usize,
        max_blocks: usize,
    ) -> Self {
        Self {
            block_size,
            num_heads,
            head_dim,
            num_layers,
            max_blocks,
        }
    }

    /// Memory per block in bytes (f32 storage)
    pub fn block_memory_bytes(&self) -> usize {
        // K and V each: num_heads * block_size * head_dim * 4 bytes
        // Per layer: K + V
        // Total: num_layers * 2 * num_heads * block_size * head_dim * 4
        self.num_layers * 2 * self.num_heads * self.block_size * self.head_dim * 4
    }

    /// Total memory pool size in bytes
    pub fn total_memory_bytes(&self) -> usize {
        self.max_blocks * self.block_memory_bytes()
    }

    /// Elements per block per layer (K or V separately)
    pub fn elements_per_block(&self) -> usize {
        self.num_heads * self.block_size * self.head_dim
    }
}

impl Default for PagedAttentionConfig {
    fn default() -> Self {
        Self {
            block_size: 16,
            num_heads: 32,
            head_dim: 128,
            num_layers: 32,
            max_blocks: 1024,
        }
    }
}

/// Physical block ID (index into the block pool)
pub type PhysicalBlockId = usize;

/// Logical block ID (index into sequence's block table)
pub type LogicalBlockId = usize;

/// Sequence ID for tracking multiple sequences
pub type SequenceId = usize;

/// Reference count for copy-on-write
#[derive(Debug, Clone, Copy, Default)]
pub struct RefCount(usize);

impl RefCount {
    pub fn new() -> Self {
        RefCount(1)
    }

    pub fn increment(&mut self) {
        self.0 += 1;
    }

    pub fn decrement(&mut self) -> usize {
        self.0 = self.0.saturating_sub(1);
        self.0
    }

    pub fn count(&self) -> usize {
        self.0
    }
}

/// A physical block storing KV cache data for multiple layers
#[derive(Debug, Clone)]
pub struct PhysicalBlock {
    /// Key data: [num_layers, num_heads, block_size, head_dim]
    pub keys: Vec<f32>,
    /// Value data: [num_layers, num_heads, block_size, head_dim]
    pub values: Vec<f32>,
    /// Number of tokens currently stored
    pub num_tokens: usize,
    /// Reference count for CoW
    pub ref_count: RefCount,
    /// Block ID
    pub id: PhysicalBlockId,
}

impl PhysicalBlock {
    /// Create a new empty block
    pub fn new(id: PhysicalBlockId, config: &PagedAttentionConfig) -> Self {
        let size = config.num_layers * config.elements_per_block();
        Self {
            keys: vec![0.0; size],
            values: vec![0.0; size],
            num_tokens: 0,
            ref_count: RefCount::new(),
            id,
        }
    }

    /// Check if block is full
    pub fn is_full(&self, block_size: usize) -> bool {
        self.num_tokens >= block_size
    }

    /// Check if block is shared (ref_count > 1)
    pub fn is_shared(&self) -> bool {
        self.ref_count.count() > 1
    }

    /// Get remaining capacity
    pub fn remaining_capacity(&self, block_size: usize) -> usize {
        block_size.saturating_sub(self.num_tokens)
    }
}

/// Block table mapping logical to physical blocks for a sequence
#[derive(Debug, Clone)]
pub struct BlockTable {
    /// Mapping from logical block index to physical block ID
    pub mappings: Vec<PhysicalBlockId>,
    /// Sequence ID this table belongs to
    pub sequence_id: SequenceId,
}

impl BlockTable {
    /// Create a new empty block table
    pub fn new(sequence_id: SequenceId) -> Self {
        Self {
            mappings: Vec::new(),
            sequence_id,
        }
    }

    /// Get number of logical blocks
    pub fn num_blocks(&self) -> usize {
        self.mappings.len()
    }

    /// Get physical block for logical index
    pub fn get_physical(&self, logical: LogicalBlockId) -> Option<PhysicalBlockId> {
        self.mappings.get(logical).copied()
    }

    /// Append a new physical block mapping
    pub fn append(&mut self, physical: PhysicalBlockId) {
        self.mappings.push(physical);
    }

    /// Update mapping at logical index
    pub fn update(&mut self, logical: LogicalBlockId, physical: PhysicalBlockId) {
        if logical < self.mappings.len() {
            self.mappings[logical] = physical;
        }
    }
}

/// Block allocator managing the physical block pool
#[derive(Debug)]
pub struct BlockAllocator {
    /// Free block IDs
    free_blocks: Vec<PhysicalBlockId>,
    /// Set of allocated block IDs
    allocated_blocks: HashSet<PhysicalBlockId>,
    /// Maximum number of blocks
    max_blocks: usize,
    /// Next block ID for new allocations
    next_id: PhysicalBlockId,
}

impl BlockAllocator {
    /// Create a new allocator
    pub fn new(max_blocks: usize) -> Self {
        Self {
            free_blocks: Vec::new(),
            allocated_blocks: HashSet::new(),
            max_blocks,
            next_id: 0,
        }
    }

    /// Allocate a block, returning its ID
    pub fn allocate(&mut self) -> Option<PhysicalBlockId> {
        // Try to reuse a freed block first
        if let Some(id) = self.free_blocks.pop() {
            self.allocated_blocks.insert(id);
            return Some(id);
        }

        // Allocate a new block if we haven't reached max
        if self.next_id < self.max_blocks {
            let id = self.next_id;
            self.next_id += 1;
            self.allocated_blocks.insert(id);
            return Some(id);
        }

        None // Out of memory
    }

    /// Free a block, making it available for reuse
    pub fn free(&mut self, id: PhysicalBlockId) {
        if self.allocated_blocks.remove(&id) {
            self.free_blocks.push(id);
        }
    }

    /// Get number of free blocks
    pub fn num_free(&self) -> usize {
        self.free_blocks.len() + (self.max_blocks - self.next_id)
    }

    /// Get number of allocated blocks
    pub fn num_allocated(&self) -> usize {
        self.allocated_blocks.len()
    }

    /// Check if a block ID is allocated
    pub fn is_allocated(&self, id: PhysicalBlockId) -> bool {
        self.allocated_blocks.contains(&id)
    }
}

/// Statistics for paged attention
#[derive(Debug, Clone, Default)]
pub struct PagedAttentionStats {
    /// Total blocks allocated
    pub blocks_allocated: usize,
    /// Blocks currently in use
    pub blocks_in_use: usize,
    /// Total sequences
    pub num_sequences: usize,
    /// Copy-on-write operations performed
    pub cow_operations: usize,
    /// Total tokens stored
    pub total_tokens: usize,
    /// Memory efficiency (used / allocated)
    pub memory_efficiency: f32,
}

/// Paged Attention Manager
/// Handles block allocation, sequence management, and attention computation
#[derive(Debug)]
pub struct PagedAttention {
    /// Configuration
    config: PagedAttentionConfig,
    /// Physical block storage
    blocks: HashMap<PhysicalBlockId, PhysicalBlock>,
    /// Block tables per sequence
    block_tables: HashMap<SequenceId, BlockTable>,
    /// Block allocator
    allocator: BlockAllocator,
    /// Next sequence ID
    next_sequence_id: SequenceId,
    /// Statistics
    stats: PagedAttentionStats,
}

impl PagedAttention {
    /// Create a new paged attention manager
    pub fn new(config: PagedAttentionConfig) -> Self {
        let allocator = BlockAllocator::new(config.max_blocks);
        Self {
            config,
            blocks: HashMap::new(),
            block_tables: HashMap::new(),
            allocator,
            next_sequence_id: 0,
            stats: PagedAttentionStats::default(),
        }
    }

    /// Get configuration
    pub fn config(&self) -> &PagedAttentionConfig {
        &self.config
    }

    /// Create a new sequence, returns its ID
    pub fn create_sequence(&mut self) -> SequenceId {
        let id = self.next_sequence_id;
        self.next_sequence_id += 1;
        self.block_tables.insert(id, BlockTable::new(id));
        self.stats.num_sequences += 1;
        id
    }

    /// Remove a sequence and free its blocks
    pub fn remove_sequence(&mut self, seq_id: SequenceId) {
        if let Some(table) = self.block_tables.remove(&seq_id) {
            for &physical_id in &table.mappings {
                self.decrement_ref_count(physical_id);
            }
            self.stats.num_sequences = self.stats.num_sequences.saturating_sub(1);
        }
    }

    /// Decrement reference count and potentially free block
    fn decrement_ref_count(&mut self, block_id: PhysicalBlockId) {
        let should_free = if let Some(block) = self.blocks.get_mut(&block_id) {
            let count = block.ref_count.decrement();
            count == 0
        } else {
            false
        };

        if should_free {
            self.blocks.remove(&block_id);
            self.allocator.free(block_id);
            self.stats.blocks_in_use = self.stats.blocks_in_use.saturating_sub(1);
        }
    }

    /// Allocate a new block for a sequence
    fn allocate_block(&mut self, seq_id: SequenceId) -> Option<PhysicalBlockId> {
        let block_id = self.allocator.allocate()?;

        let block = PhysicalBlock::new(block_id, &self.config);
        self.blocks.insert(block_id, block);

        if let Some(table) = self.block_tables.get_mut(&seq_id) {
            table.append(block_id);
        }

        self.stats.blocks_allocated += 1;
        self.stats.blocks_in_use += 1;

        Some(block_id)
    }

    /// Get or allocate a block for appending tokens
    fn get_or_allocate_block(&mut self, seq_id: SequenceId) -> Option<PhysicalBlockId> {
        let table = self.block_tables.get(&seq_id)?;

        // Check if last block has space
        if let Some(&last_id) = table.mappings.last()
            && let Some(block) = self.blocks.get(&last_id)
            && !block.is_full(self.config.block_size)
        {
            // Need to check if shared (CoW)
            if block.is_shared() {
                // Perform copy-on-write
                return self.copy_on_write(seq_id, table.num_blocks() - 1);
            }
            return Some(last_id);
        }

        // Need a new block
        self.allocate_block(seq_id)
    }

    /// Perform copy-on-write for a logical block
    fn copy_on_write(
        &mut self,
        seq_id: SequenceId,
        logical_id: LogicalBlockId,
    ) -> Option<PhysicalBlockId> {
        let old_physical_id = self.block_tables.get(&seq_id)?.get_physical(logical_id)?;

        // Allocate new block
        let new_physical_id = self.allocator.allocate()?;

        // Copy data
        let old_block = self.blocks.get(&old_physical_id)?;
        let mut new_block = PhysicalBlock::new(new_physical_id, &self.config);
        new_block.keys.copy_from_slice(&old_block.keys);
        new_block.values.copy_from_slice(&old_block.values);
        new_block.num_tokens = old_block.num_tokens;

        self.blocks.insert(new_physical_id, new_block);

        // Update block table
        if let Some(table) = self.block_tables.get_mut(&seq_id) {
            table.update(logical_id, new_physical_id);
        }

        // Decrement old block ref count
        self.decrement_ref_count(old_physical_id);

        self.stats.cow_operations += 1;
        self.stats.blocks_allocated += 1;
        self.stats.blocks_in_use += 1;

        Some(new_physical_id)
    }

    /// Append KV cache for new tokens
    /// keys/values shape per layer: [num_heads, num_new_tokens, head_dim]
    pub fn append_kv(
        &mut self,
        seq_id: SequenceId,
        keys: &[Vec<f32>],
        values: &[Vec<f32>],
        num_tokens: usize,
    ) -> Result<(), PagedAttentionError> {
        if keys.len() != self.config.num_layers || values.len() != self.config.num_layers {
            return Err(PagedAttentionError::InvalidShape);
        }

        let expected_size = self.config.num_heads * num_tokens * self.config.head_dim;
        for layer in 0..self.config.num_layers {
            if keys[layer].len() != expected_size || values[layer].len() != expected_size {
                return Err(PagedAttentionError::InvalidShape);
            }
        }

        // Process tokens one at a time (could be optimized for bulk)
        for token_idx in 0..num_tokens {
            let block_id = self
                .get_or_allocate_block(seq_id)
                .ok_or(PagedAttentionError::OutOfMemory)?;

            let block = self
                .blocks
                .get_mut(&block_id)
                .ok_or(PagedAttentionError::BlockNotFound)?;

            let slot_idx = block.num_tokens;

            // Copy KV for each layer
            for layer in 0..self.config.num_layers {
                let layer_offset = layer * self.config.elements_per_block();

                for head in 0..self.config.num_heads {
                    let head_offset_in_block = head * self.config.block_size * self.config.head_dim
                        + slot_idx * self.config.head_dim;

                    let head_offset_in_input =
                        head * num_tokens * self.config.head_dim + token_idx * self.config.head_dim;

                    let block_start = layer_offset + head_offset_in_block;
                    let input_start = head_offset_in_input;

                    for d in 0..self.config.head_dim {
                        block.keys[block_start + d] = keys[layer][input_start + d];
                        block.values[block_start + d] = values[layer][input_start + d];
                    }
                }
            }

            block.num_tokens += 1;
            self.stats.total_tokens += 1;
        }

        self.update_efficiency();
        Ok(())
    }

    /// Fork a sequence (for beam search), sharing blocks via CoW
    pub fn fork_sequence(&mut self, source_id: SequenceId) -> Option<SequenceId> {
        // Clone the mappings to avoid borrow issues
        let source_mappings = self.block_tables.get(&source_id)?.mappings.clone();

        let new_id = self.create_sequence();

        // Increment ref counts for shared blocks
        for &physical_id in &source_mappings {
            if let Some(block) = self.blocks.get_mut(&physical_id) {
                block.ref_count.increment();
            }
        }

        // Update the new table with mappings
        if let Some(new_table) = self.block_tables.get_mut(&new_id) {
            new_table.mappings = source_mappings;
        }

        Some(new_id)
    }

    /// Get KV cache for a sequence (for attention computation)
    /// Returns (keys, values) where each is [num_layers][seq_len * num_heads * head_dim]
    pub fn get_kv(&self, seq_id: SequenceId) -> Result<KvData, PagedAttentionError> {
        let table = self
            .block_tables
            .get(&seq_id)
            .ok_or(PagedAttentionError::SequenceNotFound)?;

        let total_tokens = self.get_sequence_length(seq_id).unwrap_or(0);
        if total_tokens == 0 {
            let empty: Vec<Vec<f32>> = (0..self.config.num_layers).map(|_| vec![]).collect();
            return Ok((empty.clone(), empty));
        }

        let mut all_keys: Vec<Vec<f32>> = (0..self.config.num_layers)
            .map(|_| {
                Vec::with_capacity(total_tokens * self.config.num_heads * self.config.head_dim)
            })
            .collect();
        let mut all_values: Vec<Vec<f32>> = (0..self.config.num_layers)
            .map(|_| {
                Vec::with_capacity(total_tokens * self.config.num_heads * self.config.head_dim)
            })
            .collect();

        // Iterate through blocks
        for (block_idx, &physical_id) in table.mappings.iter().enumerate() {
            let block = self
                .blocks
                .get(&physical_id)
                .ok_or(PagedAttentionError::BlockNotFound)?;

            let tokens_in_block = if block_idx == table.num_blocks() - 1 {
                block.num_tokens
            } else {
                self.config.block_size
            };

            // Extract data for each layer
            for layer in 0..self.config.num_layers {
                let layer_offset = layer * self.config.elements_per_block();

                for head in 0..self.config.num_heads {
                    for slot in 0..tokens_in_block {
                        let offset = layer_offset
                            + head * self.config.block_size * self.config.head_dim
                            + slot * self.config.head_dim;

                        all_keys[layer]
                            .extend_from_slice(&block.keys[offset..offset + self.config.head_dim]);
                        all_values[layer].extend_from_slice(
                            &block.values[offset..offset + self.config.head_dim],
                        );
                    }
                }
            }
        }

        Ok((all_keys, all_values))
    }

    /// Get sequence length (number of tokens)
    pub fn get_sequence_length(&self, seq_id: SequenceId) -> Option<usize> {
        let table = self.block_tables.get(&seq_id)?;
        let mut total = 0;

        for (i, &physical_id) in table.mappings.iter().enumerate() {
            if let Some(block) = self.blocks.get(&physical_id) {
                if i == table.num_blocks() - 1 {
                    total += block.num_tokens;
                } else {
                    total += self.config.block_size;
                }
            }
        }

        Some(total)
    }

    /// Get statistics
    pub fn stats(&self) -> &PagedAttentionStats {
        &self.stats
    }

    /// Update memory efficiency stat
    fn update_efficiency(&mut self) {
        let allocated = self.stats.blocks_in_use * self.config.block_size;
        if allocated > 0 {
            self.stats.memory_efficiency = self.stats.total_tokens as f32 / allocated as f32;
        }
    }

    /// Get number of free blocks
    pub fn num_free_blocks(&self) -> usize {
        self.allocator.num_free()
    }

    /// Get number of allocated blocks
    pub fn num_allocated_blocks(&self) -> usize {
        self.allocator.num_allocated()
    }

    /// Get memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        self.stats.blocks_in_use * self.config.block_memory_bytes()
    }

    /// Check if can allocate n more blocks
    pub fn can_allocate(&self, n: usize) -> bool {
        self.allocator.num_free() >= n
    }

    /// Compute attention for a query against cached KV
    /// query shape: [num_heads, query_len, head_dim]
    /// Returns attention output: [num_heads, query_len, head_dim]
    pub fn compute_attention(
        &self,
        seq_id: SequenceId,
        layer: usize,
        query: &[f32],
        query_len: usize,
    ) -> Result<Vec<f32>, PagedAttentionError> {
        if layer >= self.config.num_layers {
            return Err(PagedAttentionError::InvalidLayer);
        }

        let expected_query_size = self.config.num_heads * query_len * self.config.head_dim;
        if query.len() != expected_query_size {
            return Err(PagedAttentionError::InvalidShape);
        }

        let seq_len = self
            .get_sequence_length(seq_id)
            .ok_or(PagedAttentionError::SequenceNotFound)?;

        if seq_len == 0 {
            return Ok(vec![0.0; expected_query_size]);
        }

        let table = self
            .block_tables
            .get(&seq_id)
            .ok_or(PagedAttentionError::SequenceNotFound)?;

        let mut output = vec![0.0; expected_query_size];
        let scale = 1.0 / (self.config.head_dim as f32).sqrt();

        // Process each head independently
        for head in 0..self.config.num_heads {
            // Process each query position
            for q_pos in 0..query_len {
                let q_offset =
                    head * query_len * self.config.head_dim + q_pos * self.config.head_dim;
                let query_vec = &query[q_offset..q_offset + self.config.head_dim];

                // Compute attention scores for all KV positions
                let mut scores = Vec::with_capacity(seq_len);
                let mut max_score = f32::NEG_INFINITY;

                // Iterate through blocks to get keys
                for (block_idx, &physical_id) in table.mappings.iter().enumerate() {
                    let block = self
                        .blocks
                        .get(&physical_id)
                        .ok_or(PagedAttentionError::BlockNotFound)?;

                    let tokens_in_block = if block_idx == table.num_blocks() - 1 {
                        block.num_tokens
                    } else {
                        self.config.block_size
                    };

                    let layer_offset = layer * self.config.elements_per_block();

                    for slot in 0..tokens_in_block {
                        let k_offset = layer_offset
                            + head * self.config.block_size * self.config.head_dim
                            + slot * self.config.head_dim;
                        let key_vec = &block.keys[k_offset..k_offset + self.config.head_dim];

                        // Dot product
                        let score: f32 = query_vec
                            .iter()
                            .zip(key_vec.iter())
                            .map(|(&q, &k)| q * k)
                            .sum::<f32>()
                            * scale;

                        max_score = max_score.max(score);
                        scores.push(score);
                    }
                }

                // Softmax
                let mut sum_exp = 0.0;
                for score in &mut scores {
                    *score = (*score - max_score).exp();
                    sum_exp += *score;
                }
                for score in &mut scores {
                    *score /= sum_exp;
                }

                // Weighted sum of values
                let mut weighted_value = vec![0.0; self.config.head_dim];
                let mut kv_pos = 0;

                for (block_idx, &physical_id) in table.mappings.iter().enumerate() {
                    let block = self
                        .blocks
                        .get(&physical_id)
                        .ok_or(PagedAttentionError::BlockNotFound)?;

                    let tokens_in_block = if block_idx == table.num_blocks() - 1 {
                        block.num_tokens
                    } else {
                        self.config.block_size
                    };

                    let layer_offset = layer * self.config.elements_per_block();

                    for slot in 0..tokens_in_block {
                        let v_offset = layer_offset
                            + head * self.config.block_size * self.config.head_dim
                            + slot * self.config.head_dim;
                        let value_vec = &block.values[v_offset..v_offset + self.config.head_dim];

                        let weight = scores[kv_pos];
                        for (i, &v) in value_vec.iter().enumerate() {
                            weighted_value[i] += weight * v;
                        }
                        kv_pos += 1;
                    }
                }

                // Store output
                let out_offset =
                    head * query_len * self.config.head_dim + q_pos * self.config.head_dim;
                output[out_offset..out_offset + self.config.head_dim]
                    .copy_from_slice(&weighted_value);
            }
        }

        Ok(output)
    }
}

/// Error types for paged attention
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PagedAttentionError {
    /// Out of memory (no free blocks)
    OutOfMemory,
    /// Sequence not found
    SequenceNotFound,
    /// Block not found
    BlockNotFound,
    /// Invalid tensor shape
    InvalidShape,
    /// Invalid layer index
    InvalidLayer,
}

impl std::fmt::Display for PagedAttentionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PagedAttentionError::OutOfMemory => write!(f, "Out of memory: no free blocks"),
            PagedAttentionError::SequenceNotFound => write!(f, "Sequence not found"),
            PagedAttentionError::BlockNotFound => write!(f, "Block not found"),
            PagedAttentionError::InvalidShape => write!(f, "Invalid tensor shape"),
            PagedAttentionError::InvalidLayer => write!(f, "Invalid layer index"),
        }
    }
}

impl std::error::Error for PagedAttentionError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> PagedAttentionConfig {
        PagedAttentionConfig {
            block_size: 4,
            num_heads: 2,
            head_dim: 8,
            num_layers: 2,
            max_blocks: 16,
        }
    }

    fn random_kv(
        config: &PagedAttentionConfig,
        num_tokens: usize,
    ) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let size = config.num_heads * num_tokens * config.head_dim;
        let mut keys = Vec::with_capacity(config.num_layers);
        let mut values = Vec::with_capacity(config.num_layers);

        for layer in 0..config.num_layers {
            let k: Vec<f32> = (0..size)
                .map(|i| {
                    let mut hasher = DefaultHasher::new();
                    (layer * 1000 + i).hash(&mut hasher);
                    ((hasher.finish() % 10000) as f32 / 5000.0) - 1.0
                })
                .collect();
            let v: Vec<f32> = (0..size)
                .map(|i| {
                    let mut hasher = DefaultHasher::new();
                    (layer * 2000 + i).hash(&mut hasher);
                    ((hasher.finish() % 10000) as f32 / 5000.0) - 1.0
                })
                .collect();
            keys.push(k);
            values.push(v);
        }

        (keys, values)
    }

    #[test]
    fn test_config_memory() {
        let config = test_config();
        // Per block: 2 layers * 2 * (K+V) * 2 heads * 4 block_size * 8 head_dim * 4 bytes
        // But we store K and V separately, each is: num_layers * num_heads * block_size * head_dim
        let expected = 2 * 2 * 2 * 4 * 8 * 4; // layers * 2(K+V) * heads * block_size * head_dim * 4
        assert_eq!(config.block_memory_bytes(), expected);
    }

    #[test]
    fn test_create_sequence() {
        let config = test_config();
        let mut pa = PagedAttention::new(config);

        let seq1 = pa.create_sequence();
        let seq2 = pa.create_sequence();

        assert_eq!(seq1, 0);
        assert_eq!(seq2, 1);
        assert_eq!(pa.stats().num_sequences, 2);
    }

    #[test]
    fn test_append_single_token() {
        let config = test_config();
        let mut pa = PagedAttention::new(config.clone());

        let seq_id = pa.create_sequence();
        let (keys, values) = random_kv(&config, 1);

        pa.append_kv(seq_id, &keys, &values, 1).unwrap();

        assert_eq!(pa.get_sequence_length(seq_id), Some(1));
        assert_eq!(pa.num_allocated_blocks(), 1);
    }

    #[test]
    fn test_append_multiple_tokens() {
        let config = test_config();
        let mut pa = PagedAttention::new(config.clone());

        let seq_id = pa.create_sequence();

        // Append 3 tokens (fits in one block of size 4)
        let (keys, values) = random_kv(&config, 3);
        pa.append_kv(seq_id, &keys, &values, 3).unwrap();

        assert_eq!(pa.get_sequence_length(seq_id), Some(3));
        assert_eq!(pa.num_allocated_blocks(), 1);

        // Append 2 more (needs second block)
        let (keys2, values2) = random_kv(&config, 2);
        pa.append_kv(seq_id, &keys2, &values2, 2).unwrap();

        assert_eq!(pa.get_sequence_length(seq_id), Some(5));
        assert_eq!(pa.num_allocated_blocks(), 2);
    }

    #[test]
    fn test_remove_sequence() {
        let config = test_config();
        let mut pa = PagedAttention::new(config.clone());

        let seq_id = pa.create_sequence();
        let (keys, values) = random_kv(&config, 4);
        pa.append_kv(seq_id, &keys, &values, 4).unwrap();

        assert_eq!(pa.num_allocated_blocks(), 1);

        pa.remove_sequence(seq_id);

        assert_eq!(pa.stats().num_sequences, 0);
        assert_eq!(pa.num_allocated_blocks(), 0);
        assert_eq!(pa.num_free_blocks(), config.max_blocks);
    }

    #[test]
    fn test_fork_sequence() {
        let config = test_config();
        let mut pa = PagedAttention::new(config.clone());

        let seq1 = pa.create_sequence();
        let (keys, values) = random_kv(&config, 2);
        pa.append_kv(seq1, &keys, &values, 2).unwrap();

        // Fork the sequence
        let seq2 = pa.fork_sequence(seq1).unwrap();

        assert_eq!(pa.get_sequence_length(seq1), Some(2));
        assert_eq!(pa.get_sequence_length(seq2), Some(2));
        // Should share the same block
        assert_eq!(pa.num_allocated_blocks(), 1);

        // The block should have ref_count = 2
        let table1 = pa.block_tables.get(&seq1).unwrap();
        let block_id = table1.get_physical(0).unwrap();
        assert_eq!(pa.blocks.get(&block_id).unwrap().ref_count.count(), 2);
    }

    #[test]
    fn test_copy_on_write() {
        let config = test_config();
        let mut pa = PagedAttention::new(config.clone());

        let seq1 = pa.create_sequence();
        let (keys, values) = random_kv(&config, 2);
        pa.append_kv(seq1, &keys, &values, 2).unwrap();

        // Fork
        let seq2 = pa.fork_sequence(seq1).unwrap();
        assert_eq!(pa.num_allocated_blocks(), 1);

        // Append to seq2 - should trigger CoW
        let (keys2, values2) = random_kv(&config, 1);
        pa.append_kv(seq2, &keys2, &values2, 1).unwrap();

        // Now should have 2 blocks (original for seq1, copy for seq2)
        assert_eq!(pa.num_allocated_blocks(), 2);
        assert_eq!(pa.stats().cow_operations, 1);
    }

    #[test]
    fn test_get_kv() {
        let config = test_config();
        let mut pa = PagedAttention::new(config.clone());

        let seq_id = pa.create_sequence();
        let (keys, values) = random_kv(&config, 3);
        pa.append_kv(seq_id, &keys, &values, 3).unwrap();

        let (retrieved_keys, retrieved_values) = pa.get_kv(seq_id).unwrap();

        assert_eq!(retrieved_keys.len(), config.num_layers);
        assert_eq!(retrieved_values.len(), config.num_layers);

        for layer in 0..config.num_layers {
            assert_eq!(
                retrieved_keys[layer].len(),
                config.num_heads * 3 * config.head_dim
            );
            assert_eq!(
                retrieved_values[layer].len(),
                config.num_heads * 3 * config.head_dim
            );
        }
    }

    #[test]
    fn test_out_of_memory() {
        let config = PagedAttentionConfig {
            block_size: 2,
            num_heads: 1,
            head_dim: 4,
            num_layers: 1,
            max_blocks: 2,
        };
        let mut pa = PagedAttention::new(config.clone());

        let seq_id = pa.create_sequence();

        // Fill up 4 tokens (2 blocks)
        let (keys, values) = random_kv(&config, 4);
        pa.append_kv(seq_id, &keys, &values, 4).unwrap();

        // Try to add more - should fail
        let (keys2, values2) = random_kv(&config, 1);
        let result = pa.append_kv(seq_id, &keys2, &values2, 1);

        assert_eq!(result, Err(PagedAttentionError::OutOfMemory));
    }

    #[test]
    fn test_compute_attention() {
        let config = PagedAttentionConfig {
            block_size: 4,
            num_heads: 1,
            head_dim: 4,
            num_layers: 1,
            max_blocks: 8,
        };
        let mut pa = PagedAttention::new(config.clone());

        let seq_id = pa.create_sequence();

        // Create simple KV cache
        let keys = vec![vec![
            // 2 tokens, 1 head, 4 dim each
            1.0, 0.0, 0.0, 0.0, // token 0
            0.0, 1.0, 0.0, 0.0, // token 1
        ]];
        let values = vec![vec![
            1.0, 2.0, 3.0, 4.0, // token 0
            5.0, 6.0, 7.0, 8.0, // token 1
        ]];

        pa.append_kv(seq_id, &keys, &values, 2).unwrap();

        // Query that should attend equally to both tokens
        let query = vec![
            0.5, 0.5, 0.0, 0.0, // will have equal dot products
        ];

        let output = pa.compute_attention(seq_id, 0, &query, 1).unwrap();

        assert_eq!(output.len(), config.head_dim);
        // Output should be roughly average of two values
        // (softmax of equal scores = 0.5, 0.5)
        for i in 0..config.head_dim {
            let expected = (values[0][i] + values[0][i + config.head_dim]) / 2.0;
            assert!(
                (output[i] - expected).abs() < 0.1,
                "dim {}: expected {}, got {}",
                i,
                expected,
                output[i]
            );
        }
    }

    #[test]
    fn test_block_allocator() {
        let mut allocator = BlockAllocator::new(4);

        assert_eq!(allocator.num_free(), 4);
        assert_eq!(allocator.num_allocated(), 0);

        let b1 = allocator.allocate().unwrap();
        let _b2 = allocator.allocate().unwrap();

        assert_eq!(allocator.num_free(), 2);
        assert_eq!(allocator.num_allocated(), 2);

        allocator.free(b1);

        assert_eq!(allocator.num_free(), 3);
        assert_eq!(allocator.num_allocated(), 1);

        // Reuse freed block
        let b3 = allocator.allocate().unwrap();
        assert_eq!(b3, b1);
    }

    #[test]
    fn test_block_table() {
        let mut table = BlockTable::new(0);

        table.append(5);
        table.append(10);

        assert_eq!(table.num_blocks(), 2);
        assert_eq!(table.get_physical(0), Some(5));
        assert_eq!(table.get_physical(1), Some(10));
        assert_eq!(table.get_physical(2), None);

        table.update(0, 15);
        assert_eq!(table.get_physical(0), Some(15));
    }

    #[test]
    fn test_ref_count() {
        let mut rc = RefCount::new();
        assert_eq!(rc.count(), 1);

        rc.increment();
        assert_eq!(rc.count(), 2);

        rc.decrement();
        assert_eq!(rc.count(), 1);

        rc.decrement();
        assert_eq!(rc.count(), 0);

        // Should not go negative
        rc.decrement();
        assert_eq!(rc.count(), 0);
    }

    #[test]
    fn test_memory_efficiency() {
        let config = PagedAttentionConfig {
            block_size: 4,
            num_heads: 2,
            head_dim: 8,
            num_layers: 2,
            max_blocks: 16,
        };
        let mut pa = PagedAttention::new(config.clone());

        let seq_id = pa.create_sequence();

        // Add 3 tokens (block capacity 4)
        let (keys, values) = random_kv(&config, 3);
        pa.append_kv(seq_id, &keys, &values, 3).unwrap();

        // Efficiency should be 3/4 = 0.75
        assert!((pa.stats().memory_efficiency - 0.75).abs() < 0.01);
    }

    #[test]
    fn test_multiple_sequences() {
        let config = test_config();
        let mut pa = PagedAttention::new(config.clone());

        let seq1 = pa.create_sequence();
        let seq2 = pa.create_sequence();

        let (keys1, values1) = random_kv(&config, 2);
        let (keys2, values2) = random_kv(&config, 3);

        pa.append_kv(seq1, &keys1, &values1, 2).unwrap();
        pa.append_kv(seq2, &keys2, &values2, 3).unwrap();

        assert_eq!(pa.get_sequence_length(seq1), Some(2));
        assert_eq!(pa.get_sequence_length(seq2), Some(3));
        assert_eq!(pa.num_allocated_blocks(), 2);
    }

    #[test]
    fn test_config_builder() {
        let config = PagedAttentionConfig::new(32, 8, 64, 12, 512);

        assert_eq!(config.block_size, 32);
        assert_eq!(config.num_heads, 8);
        assert_eq!(config.head_dim, 64);
        assert_eq!(config.num_layers, 12);
        assert_eq!(config.max_blocks, 512);
    }

    #[test]
    fn test_empty_sequence() {
        let config = test_config();
        let pa = PagedAttention::new(config.clone());

        let result = pa.get_kv(999);
        assert_eq!(result, Err(PagedAttentionError::SequenceNotFound));
    }

    #[test]
    fn test_invalid_shape() {
        let config = test_config();
        let mut pa = PagedAttention::new(config.clone());

        let seq_id = pa.create_sequence();

        // Wrong number of layers
        let keys = vec![vec![1.0; 10]];
        let values = vec![vec![1.0; 10]];

        let result = pa.append_kv(seq_id, &keys, &values, 1);
        assert_eq!(result, Err(PagedAttentionError::InvalidShape));
    }

    #[test]
    fn test_cross_block_kv_retrieval() {
        let config = PagedAttentionConfig {
            block_size: 2,
            num_heads: 1,
            head_dim: 2,
            num_layers: 1,
            max_blocks: 8,
        };
        let mut pa = PagedAttention::new(config.clone());

        let seq_id = pa.create_sequence();

        // Add 5 tokens (spans 3 blocks)
        let keys = vec![vec![
            1.0, 2.0, // token 0
            3.0, 4.0, // token 1
            5.0, 6.0, // token 2
            7.0, 8.0, // token 3
            9.0, 10.0, // token 4
        ]];
        let values = vec![vec![
            10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0,
        ]];

        pa.append_kv(seq_id, &keys, &values, 5).unwrap();

        let (retrieved_keys, retrieved_values) = pa.get_kv(seq_id).unwrap();

        assert_eq!(retrieved_keys[0], keys[0]);
        assert_eq!(retrieved_values[0], values[0]);
    }
}

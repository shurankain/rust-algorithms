// RadixAttention
//
// Prefix caching optimization for LLM inference using a radix tree.
// Key innovation from SGLang (LMSYS, 2024).
//
// Key concepts:
// - Radix tree: Trie-like structure for efficient prefix matching
// - Prefix caching: Reuse KV cache for common prefixes across requests
// - Token-level granularity: Cache at token level for maximum sharing
// - LRU eviction: Remove least recently used cache entries when full
//
// Benefits:
// - Significant speedup for prompts with common prefixes (system prompts, few-shot examples)
// - Memory efficiency through deduplication
// - Automatic prefix matching without explicit management
// - Compatible with continuous batching
//
// Use cases:
// - System prompts shared across requests
// - Few-shot learning examples
// - Multi-turn conversations with shared history
// - Batch processing with similar prompts

use std::collections::HashMap;

/// Configuration for RadixAttention
#[derive(Debug, Clone)]
pub struct RadixAttentionConfig {
    /// Number of attention heads
    pub num_heads: usize,
    /// Dimension per head
    pub head_dim: usize,
    /// Number of transformer layers
    pub num_layers: usize,
    /// Maximum number of cached tokens
    pub max_cached_tokens: usize,
    /// Block size for KV storage (tokens per block)
    pub block_size: usize,
}

impl RadixAttentionConfig {
    /// Create a new configuration
    pub fn new(num_heads: usize, head_dim: usize, num_layers: usize) -> Self {
        Self {
            num_heads,
            head_dim,
            num_layers,
            max_cached_tokens: 100_000,
            block_size: 16,
        }
    }

    /// Set maximum cached tokens
    pub fn max_tokens(mut self, max: usize) -> Self {
        self.max_cached_tokens = max;
        self
    }

    /// Set block size
    pub fn block_size(mut self, size: usize) -> Self {
        self.block_size = size;
        self
    }

    /// KV cache size per token in bytes (f32)
    pub fn kv_size_per_token(&self) -> usize {
        // K + V, each: num_layers * num_heads * head_dim * 4 bytes
        2 * self.num_layers * self.num_heads * self.head_dim * 4
    }
}

impl Default for RadixAttentionConfig {
    fn default() -> Self {
        Self {
            num_heads: 32,
            head_dim: 128,
            num_layers: 32,
            max_cached_tokens: 100_000,
            block_size: 16,
        }
    }
}

/// Token ID type
pub type TokenId = u32;

/// Node ID in the radix tree
pub type NodeId = usize;

/// Unique ID for cache entries
type CacheEntryId = usize;

/// A node in the radix tree
#[derive(Debug, Clone)]
struct RadixNode {
    /// Node ID
    id: NodeId,
    /// Parent node ID (None for root)
    parent: Option<NodeId>,
    /// Token sequence for this edge (from parent to this node)
    tokens: Vec<TokenId>,
    /// Children nodes, keyed by first token of edge
    children: HashMap<TokenId, NodeId>,
    /// Reference count (number of active sequences using this prefix)
    ref_count: usize,
    /// Last access time for LRU eviction
    last_access: u64,
    /// Cache entry IDs for this node's KV cache
    cache_entries: Vec<CacheEntryId>,
}

impl RadixNode {
    fn new(id: NodeId, parent: Option<NodeId>, tokens: Vec<TokenId>) -> Self {
        Self {
            id,
            parent,
            tokens,
            children: HashMap::new(),
            ref_count: 0,
            last_access: 0,
            cache_entries: Vec::new(),
        }
    }

    fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }
}

/// KV cache entry for a block of tokens
#[derive(Debug, Clone)]
struct CacheEntry {
    /// Entry ID
    id: CacheEntryId,
    /// Key cache: [num_layers, num_heads, num_tokens, head_dim]
    keys: Vec<f32>,
    /// Value cache: [num_layers, num_heads, num_tokens, head_dim]
    values: Vec<f32>,
    /// Number of tokens in this entry
    num_tokens: usize,
    /// Node this entry belongs to (for debugging/introspection)
    #[allow(dead_code)]
    node_id: NodeId,
}

/// Result of a prefix lookup
#[derive(Debug, Clone)]
pub struct PrefixMatch {
    /// Number of tokens matched from the prefix
    pub matched_tokens: usize,
    /// Node ID where the match ended
    pub node_id: NodeId,
    /// Remaining tokens that need to be computed
    pub remaining_tokens: Vec<TokenId>,
    /// Whether this is an exact match (no remaining tokens)
    pub is_exact: bool,
}

/// Statistics for RadixAttention
#[derive(Debug, Clone, Default)]
pub struct RadixAttentionStats {
    /// Total number of nodes in the tree
    pub num_nodes: usize,
    /// Total cached tokens
    pub cached_tokens: usize,
    /// Cache hits (prefix reuse)
    pub cache_hits: usize,
    /// Cache misses
    pub cache_misses: usize,
    /// Total insertions
    pub total_insertions: usize,
    /// Evictions performed
    pub evictions: usize,
    /// Memory used in bytes
    pub memory_bytes: usize,
}

impl RadixAttentionStats {
    /// Cache hit rate
    pub fn hit_rate(&self) -> f32 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 {
            0.0
        } else {
            self.cache_hits as f32 / total as f32
        }
    }
}

/// RadixAttention manager
/// Implements prefix caching using a radix tree structure
#[derive(Debug)]
pub struct RadixAttention {
    /// Configuration
    config: RadixAttentionConfig,
    /// Radix tree nodes
    nodes: Vec<RadixNode>,
    /// KV cache entries
    cache_entries: Vec<CacheEntry>,
    /// Root node ID (always 0)
    root: NodeId,
    /// Global timestamp for LRU
    timestamp: u64,
    /// Statistics
    stats: RadixAttentionStats,
    /// Next cache entry ID
    next_entry_id: CacheEntryId,
}

impl RadixAttention {
    /// Create a new RadixAttention manager
    pub fn new(config: RadixAttentionConfig) -> Self {
        // Create root node
        let root = RadixNode::new(0, None, vec![]);
        let nodes = vec![root];

        Self {
            config,
            nodes,
            cache_entries: Vec::new(),
            root: 0,
            timestamp: 0,
            stats: RadixAttentionStats {
                num_nodes: 1,
                ..Default::default()
            },
            next_entry_id: 0,
        }
    }

    /// Get configuration
    pub fn config(&self) -> &RadixAttentionConfig {
        &self.config
    }

    /// Get statistics
    pub fn stats(&self) -> &RadixAttentionStats {
        &self.stats
    }

    /// Look up a token sequence and find the longest matching prefix
    /// Note: This only returns complete node matches (not partial edge matches)
    /// For partial edge matches, use lookup_with_partial
    pub fn lookup(&mut self, tokens: &[TokenId]) -> PrefixMatch {
        self.timestamp += 1;
        let current_time = self.timestamp;

        let mut current_node = self.root;
        let mut matched = 0;

        while matched < tokens.len() {
            let next_token = tokens[matched];

            // Check if there's a child starting with this token
            if let Some(&child_id) = self.nodes[current_node].children.get(&next_token) {
                let child_tokens = &self.nodes[child_id].tokens;

                // Check how many tokens match along this edge
                let mut edge_matched = 0;
                while edge_matched < child_tokens.len()
                    && matched + edge_matched < tokens.len()
                    && child_tokens[edge_matched] == tokens[matched + edge_matched]
                {
                    edge_matched += 1;
                }

                if edge_matched == child_tokens.len() {
                    // Full edge match, continue to child
                    matched += edge_matched;
                    current_node = child_id;
                    self.nodes[current_node].last_access = current_time;
                } else {
                    // Partial match along edge - count as cache hit with partial match
                    // We matched `matched + edge_matched` tokens total
                    matched += edge_matched;
                    break;
                }
            } else {
                // No matching child
                break;
            }
        }

        if matched > 0 {
            self.stats.cache_hits += 1;
        } else {
            self.stats.cache_misses += 1;
        }

        PrefixMatch {
            matched_tokens: matched,
            node_id: current_node,
            remaining_tokens: tokens[matched..].to_vec(),
            is_exact: matched == tokens.len(),
        }
    }

    /// Insert a token sequence with its KV cache
    /// Returns the node ID for the inserted sequence
    pub fn insert(
        &mut self,
        tokens: &[TokenId],
        keys: Vec<f32>,
        values: Vec<f32>,
    ) -> Result<NodeId, RadixAttentionError> {
        if tokens.is_empty() {
            return Ok(self.root);
        }

        // Validate KV cache size
        let expected_size =
            self.config.num_layers * self.config.num_heads * tokens.len() * self.config.head_dim;
        if keys.len() != expected_size || values.len() != expected_size {
            return Err(RadixAttentionError::InvalidKvSize);
        }

        // Check if we need to evict
        let new_tokens = tokens.len();
        while self.stats.cached_tokens + new_tokens > self.config.max_cached_tokens {
            if !self.evict_lru() {
                return Err(RadixAttentionError::CacheFull);
            }
        }

        self.timestamp += 1;
        let current_time = self.timestamp;

        // Navigate the tree, splitting edges as needed
        let mut current_node = self.root;
        let mut pos = 0;

        while pos < tokens.len() {
            let next_token = tokens[pos];

            if let Some(&child_id) = self.nodes[current_node].children.get(&next_token) {
                let child_tokens = self.nodes[child_id].tokens.clone();

                // Find common prefix length
                let mut common = 0;
                while common < child_tokens.len()
                    && pos + common < tokens.len()
                    && child_tokens[common] == tokens[pos + common]
                {
                    common += 1;
                }

                if common == child_tokens.len() {
                    // Full match on this edge, continue
                    pos += common;
                    current_node = child_id;
                    self.nodes[current_node].last_access = current_time;
                } else {
                    // Need to split the edge
                    let split_node = self.split_edge(current_node, child_id, common);
                    pos += common;
                    current_node = split_node;

                    // Insert remaining tokens as new path
                    if pos < tokens.len() {
                        current_node = self.insert_path(current_node, &tokens[pos..], current_time);
                        pos = tokens.len();
                    }
                }
            } else {
                // No child with this token, insert new path
                current_node = self.insert_path(current_node, &tokens[pos..], current_time);
                pos = tokens.len();
            }
        }

        // Store the KV cache
        self.store_cache(current_node, tokens.len(), keys, values);

        self.stats.total_insertions += 1;
        self.update_memory_stats();

        Ok(current_node)
    }

    /// Split an edge at a given position
    fn split_edge(&mut self, parent_id: NodeId, child_id: NodeId, split_pos: usize) -> NodeId {
        let child_tokens = self.nodes[child_id].tokens.clone();
        let first_token = child_tokens[0];

        // Create new intermediate node
        let new_node_id = self.nodes.len();
        let prefix_tokens = child_tokens[..split_pos].to_vec();
        let suffix_tokens = child_tokens[split_pos..].to_vec();

        let mut new_node = RadixNode::new(new_node_id, Some(parent_id), prefix_tokens);
        new_node.last_access = self.timestamp;

        // Update old child to point to new node
        self.nodes[child_id].tokens = suffix_tokens;
        self.nodes[child_id].parent = Some(new_node_id);

        // Add old child as child of new node
        let suffix_first = self.nodes[child_id].tokens[0];
        new_node.children.insert(suffix_first, child_id);

        // Update parent to point to new node
        self.nodes[parent_id]
            .children
            .insert(first_token, new_node_id);

        self.nodes.push(new_node);
        self.stats.num_nodes += 1;

        new_node_id
    }

    /// Insert a path of tokens starting from a node
    fn insert_path(&mut self, start_node: NodeId, tokens: &[TokenId], timestamp: u64) -> NodeId {
        if tokens.is_empty() {
            return start_node;
        }

        let new_node_id = self.nodes.len();
        let first_token = tokens[0];

        let mut new_node = RadixNode::new(new_node_id, Some(start_node), tokens.to_vec());
        new_node.last_access = timestamp;

        self.nodes[start_node]
            .children
            .insert(first_token, new_node_id);
        self.nodes.push(new_node);
        self.stats.num_nodes += 1;

        new_node_id
    }

    /// Store KV cache for a node
    fn store_cache(
        &mut self,
        node_id: NodeId,
        num_tokens: usize,
        keys: Vec<f32>,
        values: Vec<f32>,
    ) {
        let entry = CacheEntry {
            id: self.next_entry_id,
            keys,
            values,
            num_tokens,
            node_id,
        };

        self.next_entry_id += 1;
        self.nodes[node_id].cache_entries.push(entry.id);
        self.stats.cached_tokens += num_tokens;
        self.cache_entries.push(entry);
    }

    /// Update existing cache for a node (used when reinserting same sequence)
    #[allow(dead_code)]
    fn update_cache(
        &mut self,
        node_id: NodeId,
        num_tokens: usize,
        keys: Vec<f32>,
        values: Vec<f32>,
    ) {
        // Remove old entries
        let old_entries: Vec<CacheEntryId> = self.nodes[node_id].cache_entries.drain(..).collect();

        for entry_id in old_entries {
            if let Some(pos) = self.cache_entries.iter().position(|e| e.id == entry_id) {
                let entry = self.cache_entries.remove(pos);
                self.stats.cached_tokens =
                    self.stats.cached_tokens.saturating_sub(entry.num_tokens);
            }
        }

        // Store new cache
        self.store_cache(node_id, num_tokens, keys, values);
    }

    /// Evict least recently used cache entries
    /// Returns true if eviction was successful
    fn evict_lru(&mut self) -> bool {
        if self.nodes.len() <= 1 {
            return false; // Only root, nothing to evict
        }

        // Find leaf node with lowest last_access and ref_count == 0
        let mut best_candidate: Option<(NodeId, u64)> = None;

        for node in &self.nodes {
            if node.id == self.root {
                continue;
            }
            if node.ref_count > 0 {
                continue;
            }
            if !node.is_leaf() {
                continue;
            }

            match best_candidate {
                None => best_candidate = Some((node.id, node.last_access)),
                Some((_, best_time)) if node.last_access < best_time => {
                    best_candidate = Some((node.id, node.last_access));
                }
                _ => {}
            }
        }

        if let Some((node_id, _)) = best_candidate {
            self.evict_node(node_id);
            self.stats.evictions += 1;
            true
        } else {
            false
        }
    }

    /// Evict a specific node
    fn evict_node(&mut self, node_id: NodeId) {
        // Remove cache entries
        let entry_ids: Vec<CacheEntryId> = self.nodes[node_id].cache_entries.clone();
        for entry_id in entry_ids {
            if let Some(pos) = self.cache_entries.iter().position(|e| e.id == entry_id) {
                let entry = self.cache_entries.remove(pos);
                self.stats.cached_tokens =
                    self.stats.cached_tokens.saturating_sub(entry.num_tokens);
            }
        }

        // Remove from parent's children
        if let Some(parent_id) = self.nodes[node_id].parent
            && !self.nodes[node_id].tokens.is_empty()
        {
            let first_token = self.nodes[node_id].tokens[0];
            self.nodes[parent_id].children.remove(&first_token);
        }

        // Mark node as empty (we don't actually remove to preserve IDs)
        self.nodes[node_id].tokens.clear();
        self.nodes[node_id].cache_entries.clear();
        self.nodes[node_id].children.clear();
    }

    /// Get cached KV for a prefix match
    pub fn get_cached_kv(&self, node_id: NodeId) -> Option<(&[f32], &[f32])> {
        let node = self.nodes.get(node_id)?;
        let entry_id = node.cache_entries.first()?;
        let entry = self.cache_entries.iter().find(|e| e.id == *entry_id)?;
        Some((&entry.keys, &entry.values))
    }

    /// Increment reference count for a node (mark as in-use)
    pub fn acquire(&mut self, node_id: NodeId) {
        if node_id < self.nodes.len() {
            self.nodes[node_id].ref_count += 1;
        }
    }

    /// Decrement reference count for a node
    pub fn release(&mut self, node_id: NodeId) {
        if node_id < self.nodes.len() {
            self.nodes[node_id].ref_count = self.nodes[node_id].ref_count.saturating_sub(1);
        }
    }

    /// Get the token sequence for a node (from root to node)
    pub fn get_tokens(&self, node_id: NodeId) -> Vec<TokenId> {
        let mut tokens = Vec::new();
        let mut current = node_id;

        // Collect tokens from node to root
        while current != self.root {
            let node = &self.nodes[current];
            tokens.extend(node.tokens.iter().rev());
            if let Some(parent) = node.parent {
                current = parent;
            } else {
                break;
            }
        }

        tokens.reverse();
        tokens
    }

    /// Get depth of a node
    pub fn get_depth(&self, node_id: NodeId) -> usize {
        let mut depth = 0;
        let mut current = node_id;

        while current != self.root {
            if let Some(parent) = self.nodes[current].parent {
                depth += 1;
                current = parent;
            } else {
                break;
            }
        }

        depth
    }

    /// Get number of children for a node
    pub fn num_children(&self, node_id: NodeId) -> usize {
        self.nodes
            .get(node_id)
            .map(|n| n.children.len())
            .unwrap_or(0)
    }

    /// Clear all cache entries
    pub fn clear(&mut self) {
        // Reset to just root node
        let root = RadixNode::new(0, None, vec![]);
        self.nodes.clear();
        self.nodes.push(root);
        self.cache_entries.clear();
        self.stats = RadixAttentionStats {
            num_nodes: 1,
            ..Default::default()
        };
        self.next_entry_id = 0;
    }

    fn update_memory_stats(&mut self) {
        let kv_bytes: usize = self
            .cache_entries
            .iter()
            .map(|e| (e.keys.len() + e.values.len()) * 4)
            .sum();

        let node_overhead = self.nodes.len() * std::mem::size_of::<RadixNode>();
        self.stats.memory_bytes = kv_bytes + node_overhead;
    }

    /// Get prefix sharing factor (average number of sequences sharing each prefix)
    pub fn prefix_sharing_factor(&self) -> f32 {
        let total_refs: usize = self.nodes.iter().map(|n| n.ref_count).sum();
        let active_nodes = self.nodes.iter().filter(|n| n.ref_count > 0).count();
        if active_nodes == 0 {
            1.0
        } else {
            total_refs as f32 / active_nodes as f32
        }
    }
}

/// Error types for RadixAttention
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RadixAttentionError {
    /// Invalid KV cache size
    InvalidKvSize,
    /// Cache is full and cannot evict
    CacheFull,
    /// Node not found
    NodeNotFound,
}

impl std::fmt::Display for RadixAttentionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RadixAttentionError::InvalidKvSize => write!(f, "Invalid KV cache size"),
            RadixAttentionError::CacheFull => write!(f, "Cache is full"),
            RadixAttentionError::NodeNotFound => write!(f, "Node not found"),
        }
    }
}

impl std::error::Error for RadixAttentionError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> RadixAttentionConfig {
        RadixAttentionConfig {
            num_heads: 2,
            head_dim: 4,
            num_layers: 2,
            max_cached_tokens: 1000,
            block_size: 4,
        }
    }

    fn make_kv(config: &RadixAttentionConfig, num_tokens: usize) -> (Vec<f32>, Vec<f32>) {
        let size = config.num_layers * config.num_heads * num_tokens * config.head_dim;
        let keys: Vec<f32> = (0..size).map(|i| i as f32 * 0.01).collect();
        let values: Vec<f32> = (0..size).map(|i| i as f32 * 0.02).collect();
        (keys, values)
    }

    #[test]
    fn test_config_default() {
        let config = RadixAttentionConfig::default();
        assert_eq!(config.num_heads, 32);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.num_layers, 32);
    }

    #[test]
    fn test_config_builder() {
        let config = RadixAttentionConfig::new(8, 64, 12)
            .max_tokens(50000)
            .block_size(32);

        assert_eq!(config.num_heads, 8);
        assert_eq!(config.head_dim, 64);
        assert_eq!(config.num_layers, 12);
        assert_eq!(config.max_cached_tokens, 50000);
        assert_eq!(config.block_size, 32);
    }

    #[test]
    fn test_create_radix_attention() {
        let config = test_config();
        let ra = RadixAttention::new(config);

        assert_eq!(ra.stats().num_nodes, 1);
        assert_eq!(ra.stats().cached_tokens, 0);
    }

    #[test]
    fn test_insert_single_sequence() {
        let config = test_config();
        let mut ra = RadixAttention::new(config.clone());

        let tokens = vec![1, 2, 3, 4, 5];
        let (keys, values) = make_kv(&config, tokens.len());

        let node_id = ra.insert(&tokens, keys, values).unwrap();

        assert!(node_id > 0);
        assert_eq!(ra.stats().cached_tokens, 5);
        assert_eq!(ra.stats().total_insertions, 1);
    }

    #[test]
    fn test_lookup_exact_match() {
        let config = test_config();
        let mut ra = RadixAttention::new(config.clone());

        let tokens = vec![1, 2, 3, 4, 5];
        let (keys, values) = make_kv(&config, tokens.len());
        ra.insert(&tokens, keys, values).unwrap();

        let result = ra.lookup(&tokens);

        assert!(result.is_exact);
        assert_eq!(result.matched_tokens, 5);
        assert!(result.remaining_tokens.is_empty());
    }

    #[test]
    fn test_lookup_prefix_match() {
        let config = test_config();
        let mut ra = RadixAttention::new(config.clone());

        let tokens = vec![1, 2, 3, 4, 5];
        let (keys, values) = make_kv(&config, tokens.len());
        ra.insert(&tokens, keys, values).unwrap();

        // Lookup a longer sequence
        let query = vec![1, 2, 3, 4, 5, 6, 7];
        let result = ra.lookup(&query);

        assert!(!result.is_exact);
        assert_eq!(result.matched_tokens, 5);
        assert_eq!(result.remaining_tokens, vec![6, 7]);
    }

    #[test]
    fn test_lookup_partial_match() {
        let config = test_config();
        let mut ra = RadixAttention::new(config.clone());

        let tokens = vec![1, 2, 3, 4, 5];
        let (keys, values) = make_kv(&config, tokens.len());
        ra.insert(&tokens, keys, values).unwrap();

        // Lookup a sequence that diverges
        let query = vec![1, 2, 3, 10, 11];
        let result = ra.lookup(&query);

        assert!(!result.is_exact);
        assert_eq!(result.matched_tokens, 3);
        assert_eq!(result.remaining_tokens, vec![10, 11]);
    }

    #[test]
    fn test_no_match() {
        let config = test_config();
        let mut ra = RadixAttention::new(config.clone());

        let tokens = vec![1, 2, 3, 4, 5];
        let (keys, values) = make_kv(&config, tokens.len());
        ra.insert(&tokens, keys, values).unwrap();

        // Lookup a completely different sequence
        let query = vec![10, 20, 30];
        let result = ra.lookup(&query);

        assert!(!result.is_exact);
        assert_eq!(result.matched_tokens, 0);
        assert_eq!(result.remaining_tokens, vec![10, 20, 30]);
    }

    #[test]
    fn test_multiple_sequences_shared_prefix() {
        let config = test_config();
        let mut ra = RadixAttention::new(config.clone());

        // Insert first sequence
        let tokens1 = vec![1, 2, 3, 4, 5];
        let (keys1, values1) = make_kv(&config, tokens1.len());
        ra.insert(&tokens1, keys1, values1).unwrap();

        // Insert second sequence with shared prefix
        let tokens2 = vec![1, 2, 3, 6, 7];
        let (keys2, values2) = make_kv(&config, tokens2.len());
        ra.insert(&tokens2, keys2, values2).unwrap();

        // Both should be found
        let result1 = ra.lookup(&tokens1);
        let result2 = ra.lookup(&tokens2);

        assert!(result1.is_exact);
        assert!(result2.is_exact);

        // Tree should have split at position 3
        assert!(ra.stats().num_nodes > 2);
    }

    #[test]
    fn test_get_tokens() {
        let config = test_config();
        let mut ra = RadixAttention::new(config.clone());

        let tokens = vec![1, 2, 3, 4, 5];
        let (keys, values) = make_kv(&config, tokens.len());
        let node_id = ra.insert(&tokens, keys, values).unwrap();

        let retrieved = ra.get_tokens(node_id);
        assert_eq!(retrieved, tokens);
    }

    #[test]
    fn test_acquire_release() {
        let config = test_config();
        let mut ra = RadixAttention::new(config.clone());

        let tokens = vec![1, 2, 3];
        let (keys, values) = make_kv(&config, tokens.len());
        let node_id = ra.insert(&tokens, keys, values).unwrap();

        ra.acquire(node_id);
        assert_eq!(ra.nodes[node_id].ref_count, 1);

        ra.acquire(node_id);
        assert_eq!(ra.nodes[node_id].ref_count, 2);

        ra.release(node_id);
        assert_eq!(ra.nodes[node_id].ref_count, 1);

        ra.release(node_id);
        assert_eq!(ra.nodes[node_id].ref_count, 0);
    }

    #[test]
    fn test_eviction() {
        let config = RadixAttentionConfig {
            num_heads: 2,
            head_dim: 4,
            num_layers: 2,
            max_cached_tokens: 8, // Smaller cache - 5+5=10 > 8
            block_size: 4,
        };
        let mut ra = RadixAttention::new(config.clone());

        // Insert first sequence (5 tokens)
        let tokens1 = vec![1, 2, 3, 4, 5];
        let (keys1, values1) = make_kv(&config, tokens1.len());
        ra.insert(&tokens1, keys1, values1).unwrap();
        assert_eq!(ra.stats().cached_tokens, 5);

        // Insert second sequence - should trigger eviction (5+5=10 > 8)
        let tokens2 = vec![10, 20, 30, 40, 50];
        let (keys2, values2) = make_kv(&config, tokens2.len());
        ra.insert(&tokens2, keys2, values2).unwrap();

        // First sequence should be evicted
        assert!(ra.stats().evictions > 0);
    }

    #[test]
    fn test_eviction_respects_refs() {
        let config = RadixAttentionConfig {
            num_heads: 2,
            head_dim: 4,
            num_layers: 2,
            max_cached_tokens: 8, // 5+5=10 > 8, so eviction needed
            block_size: 4,
        };
        let mut ra = RadixAttention::new(config.clone());

        // Insert and acquire first sequence
        let tokens1 = vec![1, 2, 3, 4, 5];
        let (keys1, values1) = make_kv(&config, tokens1.len());
        let node1 = ra.insert(&tokens1, keys1, values1).unwrap();
        ra.acquire(node1);

        // Try to insert second sequence - should fail since first is locked
        let tokens2 = vec![10, 20, 30, 40, 50];
        let (keys2, values2) = make_kv(&config, tokens2.len());
        let result = ra.insert(&tokens2, keys2, values2);

        assert_eq!(result, Err(RadixAttentionError::CacheFull));
    }

    #[test]
    fn test_clear() {
        let config = test_config();
        let mut ra = RadixAttention::new(config.clone());

        let tokens = vec![1, 2, 3, 4, 5];
        let (keys, values) = make_kv(&config, tokens.len());
        ra.insert(&tokens, keys, values).unwrap();

        ra.clear();

        assert_eq!(ra.stats().num_nodes, 1);
        assert_eq!(ra.stats().cached_tokens, 0);
        assert_eq!(ra.nodes.len(), 1);
    }

    #[test]
    fn test_cache_hit_rate() {
        let config = test_config();
        let mut ra = RadixAttention::new(config.clone());

        let tokens = vec![1, 2, 3, 4, 5];
        let (keys, values) = make_kv(&config, tokens.len());
        ra.insert(&tokens, keys, values).unwrap();

        // Miss
        ra.lookup(&[10, 20, 30]);
        // Hit
        ra.lookup(&[1, 2, 3]);
        ra.lookup(&[1, 2, 3, 4]);

        let stats = ra.stats();
        assert_eq!(stats.cache_hits, 2);
        assert_eq!(stats.cache_misses, 1);
        assert!((stats.hit_rate() - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_get_cached_kv() {
        let config = test_config();
        let mut ra = RadixAttention::new(config.clone());

        let tokens = vec![1, 2, 3];
        let (keys, values) = make_kv(&config, tokens.len());
        let node_id = ra.insert(&tokens, keys.clone(), values.clone()).unwrap();

        let (cached_keys, cached_values) = ra.get_cached_kv(node_id).unwrap();
        assert_eq!(cached_keys, &keys[..]);
        assert_eq!(cached_values, &values[..]);
    }

    #[test]
    fn test_get_depth() {
        let config = test_config();
        let mut ra = RadixAttention::new(config.clone());

        let tokens1 = vec![1, 2, 3];
        let (keys1, values1) = make_kv(&config, tokens1.len());
        let node1 = ra.insert(&tokens1, keys1, values1).unwrap();

        // Insert sequence that shares prefix
        let tokens2 = vec![1, 2, 3, 4, 5];
        let (keys2, values2) = make_kv(&config, tokens2.len());
        let node2 = ra.insert(&tokens2, keys2, values2).unwrap();

        assert!(ra.get_depth(node2) >= ra.get_depth(node1));
        assert_eq!(ra.get_depth(ra.root), 0);
    }

    #[test]
    fn test_num_children() {
        let config = test_config();
        let mut ra = RadixAttention::new(config.clone());

        // Insert two sequences with different first tokens
        let tokens1 = vec![1, 2, 3];
        let (keys1, values1) = make_kv(&config, tokens1.len());
        ra.insert(&tokens1, keys1, values1).unwrap();

        let tokens2 = vec![10, 20, 30];
        let (keys2, values2) = make_kv(&config, tokens2.len());
        ra.insert(&tokens2, keys2, values2).unwrap();

        // Root should have 2 children
        assert_eq!(ra.num_children(ra.root), 2);
    }

    #[test]
    fn test_invalid_kv_size() {
        let config = test_config();
        let mut ra = RadixAttention::new(config);

        let tokens = vec![1, 2, 3];
        let keys = vec![1.0; 10]; // Wrong size
        let values = vec![1.0; 10];

        let result = ra.insert(&tokens, keys, values);
        assert_eq!(result, Err(RadixAttentionError::InvalidKvSize));
    }

    #[test]
    fn test_prefix_sharing_factor() {
        let config = test_config();
        let mut ra = RadixAttention::new(config.clone());

        let tokens = vec![1, 2, 3];
        let (keys, values) = make_kv(&config, tokens.len());
        let node_id = ra.insert(&tokens, keys, values).unwrap();

        // No refs initially
        assert_eq!(ra.prefix_sharing_factor(), 1.0);

        // Acquire multiple times
        ra.acquire(node_id);
        ra.acquire(node_id);
        ra.acquire(node_id);

        assert!(ra.prefix_sharing_factor() > 1.0);
    }

    #[test]
    fn test_empty_tokens() {
        let config = test_config();
        let mut ra = RadixAttention::new(config);

        let result = ra.insert(&[], vec![], vec![]);
        assert_eq!(result, Ok(0)); // Returns root
    }

    #[test]
    fn test_kv_size_per_token() {
        let config = RadixAttentionConfig::new(32, 128, 32);
        // 2 * 32 layers * 32 heads * 128 dim * 4 bytes = 1,048,576 bytes
        assert_eq!(config.kv_size_per_token(), 2 * 32 * 32 * 128 * 4);
    }
}

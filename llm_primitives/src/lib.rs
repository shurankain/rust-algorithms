// LLM Primitives
// Building blocks for Large Language Model inference optimization
// Based on latest research from ICML 2025 and industry trends

pub mod kv_cache;
pub mod paged_attention;
pub mod radix_attention;

pub use kv_cache::{
    KvQuantConfig, QuantBits, QuantGranularity, QuantMode, QuantParams, QuantizedKvCache,
    QuantizedTensor, dequantize_asymmetric_int8, dequantize_per_token_int8,
    dequantize_symmetric_int4, dequantize_symmetric_int8, quantization_error,
    quantize_asymmetric_int8, quantize_per_token_int8, quantize_symmetric_int4,
    quantize_symmetric_int8, sqnr_db,
};

pub use paged_attention::{
    BlockAllocator, BlockTable, KvData, LogicalBlockId, PagedAttention, PagedAttentionConfig,
    PagedAttentionError, PagedAttentionStats, PhysicalBlock, PhysicalBlockId, RefCount, SequenceId,
};

pub use radix_attention::{
    NodeId, PrefixMatch, RadixAttention, RadixAttentionConfig, RadixAttentionError,
    RadixAttentionStats, TokenId,
};

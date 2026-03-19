// LLM Primitives
// Building blocks for Large Language Model inference optimization
// Based on latest research from ICML 2025 and industry trends

pub mod alibi;
pub mod flash_attention;
pub mod grouped_query_attention;
pub mod kv_cache;
pub mod multi_query_attention;
pub mod paged_attention;
pub mod radix_attention;
pub mod rope;
pub mod sliding_window_attention;
pub mod speculative_decoding;

pub use flash_attention::{
    FlashAttention, FlashAttentionConfig, FlashAttentionStats, standard_attention,
    verify_equivalence,
};

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

pub use sliding_window_attention::{
    SlidingWindowAttention, SlidingWindowConfig, SlidingWindowStats, StreamingSlidingWindow,
    standard_sliding_window, verify_equivalence as verify_sliding_window_equivalence,
};

pub use multi_query_attention::{
    KvCacheComparison, MqaConfig, MqaStats, MultiQueryAttention, compare_kv_cache_size,
    standard_mha, verify_mqa_mha_equivalence,
};

pub use grouped_query_attention::{
    AttentionType, GqaConfig, GqaKvCacheComparison, GqaStats, GroupedQueryAttention,
    compare_attention_kv_cache, verify_gqa_mha_equivalence,
};

pub use rope::{
    RopeConfig, RopeFrequencies, YarnConfig, ntk_scaled_base, rope_angle, rotate_pair,
    verify_relative_position_encoding,
};

pub use alibi::{
    AlibiAttention, AlibiConfig, AlibiStats, compare_attention_patterns, compute_alibi_slopes,
    get_alibi_bias_matrix, standard_attention as alibi_standard_attention,
    verify_alibi_extrapolation,
};

pub use speculative_decoding::{
    DraftQualityAnalysis, SpeculativeConfig, SpeculativeDecoder, SpeculativeResult,
    SpeculativeStats, SpeculativeTree, analyze_draft_quality, build_speculative_tree,
    compute_expected_acceptance_rate,
};

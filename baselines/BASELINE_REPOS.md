# Baseline Repositories

This document tracks the baseline KV cache compression methods we're comparing against.

---

## Repository Overview

| Method | Paper | GitHub | Key Idea | Status |
|--------|-------|--------|----------|--------|
| H2O | [arxiv:2306.14048](https://arxiv.org/abs/2306.14048) | [FMInference/H2O](https://github.com/FMInference/H2O) | Heavy-hitter oracle: keep tokens with highest accumulated attention scores | Cloned ✓ |
| StreamingLLM | [arxiv:2309.17453](https://arxiv.org/abs/2309.17453) | [mit-han-lab/streaming-llm](https://github.com/mit-han-lab/streaming-llm) | Sliding window + attention sinks for infinite streaming | Cloned ✓ |
| SnapKV | [arxiv:2405.04443](https://arxiv.org/abs/2405.04443) | [FasterDecoding/SnapKV](https://github.com/FasterDecoding/SnapKV) | Clustered attention scores, compress at prompt boundary | Cloned ✓ |
| Quest | [arxiv:2406.10774](https://arxiv.org/abs/2406.10774) | [mit-han-lab/quest](https://github.com/mit-han-lab/quest) | Query-aware page-level KV retrieval | Cloned ✓ |
| KIVI | [arxiv:2402.02750](https://arxiv.org/abs/2402.02750) | [jy-yuan/KIVI](https://github.com/jy-yuan/KIVI) | Tuning-free asymmetric 2-bit/4-bit quantization | Cloned ✓ |
| FastGen | [arxiv:2403.11413](https://arxiv.org/abs/2403.11413) | [machilusZ/FastGen](https://github.com/machilusZ/FastGen) | Per-head adaptive compression strategies | Cloned ✓ |

---

## Detailed Analysis

### 1. H2O (Heavy-Hitter Oracle)

**Location:** `baselines/h2o_repo/`

**Key components:**
- `h2o_hf/` - HuggingFace implementation
- `h2o_flexgen/` - FlexGen implementation for offloading

**Implementation approach:**
- Accumulates attention scores for each token
- Evicts tokens with lowest accumulated scores when cache is full
- Keeps a small window of recent tokens

**Notes:**
- Simple to implement, good starting baseline
- Works with standard HuggingFace transformers

---

### 2. StreamingLLM

**Location:** `baselines/streamingllm_repo/`

**Key components:**
- `streaming_llm/` - Main implementation
- `examples/` - Usage examples

**Implementation approach:**
- Sliding window attention (keep recent N tokens)
- Attention sinks: special tokens that receive attention regardless of position
- Enables infinite-length generation without quality degradation

**Notes:**
- Very simple heuristic (window + sinks)
- Good baseline for "simple method that works"
- May not handle retrieval tasks well (tokens evicted can be needed later)

---

### 3. SnapKV

**Location:** `baselines/snapkv_repo/`

**Key components:**
- `snapkv/` - Main implementation
- `experiments/` - Evaluation scripts
- `notebooks/` - Demo notebooks

**Implementation approach:**
- Compresses KV cache during prefill phase
- Uses clustered attention scores to identify important positions
- Focuses on prompt-level compression

**Notes:**
- More sophisticated than H2O/StreamingLLM
- Good for retrieval tasks (preserves important prompt tokens)
- Has good integration with HuggingFace

---

### 4. Quest

**Location:** `baselines/quest_repo/`

**Key components:**
- `quest/` - Main implementation
- `kernels/` - Custom CUDA kernels
- `evaluation/` - Benchmark scripts

**Implementation approach:**
- Page-based KV cache organization
- Query-aware retrieval at decode time
- Only loads relevant pages into HBM

**Notes:**
- Most sophisticated baseline
- Requires custom CUDA kernels
- Good for very long context scenarios

---

### 5. KIVI

**Location:** `baselines/kivi_repo/`

**Key components:**
- `kivi/` - Main implementation (check config directory)
- `eval_long_bench.py` - LongBench evaluation
- `example.py` - Usage example

**Implementation approach:**
- Asymmetric quantization: keys use per-channel, values use per-token
- 2-bit or 4-bit precision options
- Group-wise quantization for accuracy

**Notes:**
- Pure quantization approach (no eviction)
- Compatible with other eviction methods
- Has vLLM integration available: [LLAA178/vllm-kivi](https://github.com/LLAA178/vllm-kivi)

---

### 6. FastGen

**Location:** `baselines/fastgen_repo/`

**Key components:**
- Minimal files (LICENSE, README.md only)
- May need to search for actual implementation

**Implementation approach:**
- Per-head adaptive compression strategies
- Profile-guided strategy selection
- Multiple compression methods per head

**Notes:**
- Repo appears incomplete (only README)
- May need to find alternative implementation
- Paper describes per-head specialization

---

## Next Steps

1. Study each implementation in detail
2. Create unified interface for all baselines
3. Set up evaluation harness
4. Test on dev benchmark first

---

## Hardware Constraints

**Our GPU:** NVIDIA RTX 5070 (12GB VRAM)

**Adjustments needed:**
- Use 7B/8B models instead of 70B
- Use quantized models (4-bit) to fit in memory
- Test on shorter context lengths initially (4K-16K instead of 128K)
- May need to use vLLM's built-in memory management for comparison
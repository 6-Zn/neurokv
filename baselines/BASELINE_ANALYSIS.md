# Baseline Implementation Analysis

This document summarizes the key implementation details of each baseline method.

---

## 1. H2O (Heavy-Hitter Oracle)

**Location:** `baselines/h2o_repo/h2o_hf/utils_hh/modify_llama.py`

### Key Mechanism
- **Accumulated Attention Scores**: Tracks `previous_scores` - accumulated attention scores for each token
- **Two Budgets**:
  - `heavy_budget_ratio`: Fraction of tokens to keep based on accumulated scores
  - `recent_budget_ratio`: Fraction of recent tokens (sliding window)
- **Masking**: Uses `attention_masks_next` to select which tokens to attend to

### Code Flow
```python
# Accumulate scores
current_scores_sum = attn_weights.sum(0).sum(1)  # (heads, k-tokens)
if self.previous_scores is not None:
    current_scores_sum[:, :-1] += self.previous_scores

# Select tokens
if attn_tokens_all > self.cache_budget:
    # Keep recent tokens
    attn_mask[:, :-self.recent_budget] = 0
    # Keep heavy hitters (top-k by accumulated score)
    _, keep_topk = selected_set.topk(k=self.heavy_budget, dim=-1, largest=True)
    attn_mask = attn_mask.scatter(-1, keep_topk, 1)
```

### Parameters
- `heavy_ratio`: 0.1 (10% of tokens by accumulated score)
- `recent_ratio`: 0.1 (10% recent tokens)

---

## 2. StreamingLLM

**Location:** `baselines/streamingllm_repo/streaming_llm/kv_cache.py`

### Key Mechanism
- **Attention Sinks**: Keep first `start_size` tokens (typically 4)
- **Sliding Window**: Keep last `recent_size` tokens (typically 512)
- **Simple Concatenation**: Direct slicing and concatenation of KV cache

### Code Flow
```python
class StartRecentKVCache:
    def __call__(self, past_key_values):
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if seq_len <= self.cache_size:
            return past_key_values
        return [
            [
                torch.cat([
                    self.k_slice(k, 0, self.start_size),          # Attention sinks
                    self.k_slice(k, seq_len - self.recent_size, seq_len),  # Recent
                ], dim=self.k_seq_dim),
                torch.cat([
                    self.v_slice(v, 0, self.start_size),
                    self.v_slice(v, seq_len - self.recent_size, seq_len),
                ], dim=self.v_seq_dim),
            ]
            for k, v in past_key_values
        ]
```

### Parameters
- `start_size`: 4 (attention sinks)
- `recent_size`: 512 (sliding window)

---

## 3. KIVI (Quantization)

**Location:** `baselines/kivi_repo/models/utils_quant.py`

### Key Mechanism
- **Asymmetric Quantization**: Min-max quantization per group
- **Group-wise**: Quantize in groups of 64-256 elements
- **Asymmetric Approach**:
  - Keys: Per-channel quantization (scale computed across all tokens)
  - Values: Per-token quantization (scale computed per token)

### Code Flow
```python
class AsymGroupedQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, clip_val, num_bits, group_size):
        # Split into groups
        input_in_groups = input.view(bs, seqlen, num_groups, group_size)
        
        # Compute min/max per group
        mx, mn = input_in_groups.max(dim=-1)[0], input_in_groups.min(dim=-1)[0]
        
        # Quantize
        scale = (mx - mn) / (2 ** num_bits - 1)
        input_in_groups = (input_in_groups - mn) / scale
        rounded_input = input_in_groups.round_()
        
        # Dequantize (for simulation)
        dequantized = rounded_input * scale + mn
        return dequantized
```

### Parameters
- `num_bits`: 2 or 4
- `group_size`: 64, 128, or 256

---

## 4. SnapKV

**Location:** `baselines/snapkv_repo/snapkv/`

### Key Mechanism (Based on README and structure)
- **Clustered Attention**: Groups similar attention patterns
- **Prompt-level Compression**: Compresses during prefill phase
- **Focus Positions**: Identifies important positions in prompt

---

## 5. Quest

**Location:** `baselines/quest_repo/quest/`

### Key Mechanism (Based on README and structure)
- **Page-based Organization**: KV cache organized in pages
- **Query-aware Retrieval**: Only load relevant pages at decode time
- **Custom CUDA Kernels**: Efficient page retrieval

---

## NeuroKV Design Implications

Based on baseline analysis, NeuroKV should:

1. **Replace Fixed Heuristics**: All baselines use fixed rules:
   - H2O: fixed ratio + accumulated scores
   - StreamingLLM: fixed start_size + recent_size
   - KIVI: fixed group_size + num_bits
   
   NeuroKV learns these parameters dynamically.

2. **Combine Eviction + Quantization**: 
   - H2O/StreamingLLM: eviction only
   - KIVI: quantization only
   - NeuroKV: three-tier hierarchy (FP16 → INT4 → Evict)

3. **Per-token Decision**:
   - Instead of fixed ratios, decide per-token:
   - KEEP_FP16, COMPRESS_INT4, OFFLOAD_DRAM, EVICT

4. **Layer-aware**: Different layers have different attention patterns
   - Early layers: more dense attention
   - Later layers: more sparse, focused attention

---

## Next Implementation Steps

1. Create unified baseline interface
2. Implement attention monitoring module
3. Build policy network architecture
4. Create oracle attribution pipeline
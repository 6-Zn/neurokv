# NeuroKV Execution Plan

## Project Overview
**NeuroKV**: Adaptive Hierarchical Key–Value Cache Management via Deep Reinforcement Learning for Long-Context LLM Inference

**Timeline**: May 11 – June 22, 2026 (6 weeks)

---

## Phase 1: Foundation (Week 1-2)

### Step 1.1: Environment Setup
**What to do:**
- Set up vLLM development environment
- Set up HuggingFace Transformers
- Create evaluation harness skeleton

**Simple version:** Use conda environment with pip install vllm, transformers

**Alternatives:**
- **Alternative A:** Docker-based setup (more reproducible but slower iteration)
- **Alternative B:** Use pre-built vLLM wheels vs. build from source (source needed for modification)

---

### Step 1.2: Baseline Implementation — ⚠️ HAS ALTERNATIVES
**What to do:** Implement 6 baselines (H2O, StreamingLLM, SnapKV, Quest, KIVI, FastGen)

**Simple version:** Start with just 2-3 baselines:
1. H2O (simplest to implement)
2. StreamingLLM (sliding window, very simple)
3. KIVI (quantization-only)

**Alternatives:**
- **Alternative A (Faster):** Use existing implementations from papers' repos directly
  - H2O: `https://github.com/FMInference/H2O`
  - SnapKV: `https://github.com/FasterDecoding/SnapKV`
  - Quest: `https://github.com/mit-han-lab/quest`
  - KIVI: `https://github.com/jysohn23/KIVI`
- **Alternative B (From scratch):** Implement all baselines yourself for deeper understanding
- **Recommendation:** Use existing repos for Week 1-2, refactor if needed later

---

### Step 1.3: Evaluation Pipeline
**What to do:** Set up LongBench, RULER, InfiniteBench evaluation

**Simple version:** Start with LongBench only (covers 21 tasks)

**Alternatives:**
- **Alternative A:** Full eval from Day 1 → Slow iteration
- **Alternative B:** Build a small "dev benchmark" (10-50 prompts) for fast iteration
- **Recommendation:** Build dev benchmark first (Alternative B)

---

## Phase 2: Oracle & Imitation Learning (Week 3)

### Step 2.1: Oracle Attribution Pipeline — ⚠️ HAS ALTERNATIVES
**What to do:** Compute leave-one-out attention attribution for each token

**Simple version:** Use attention weights directly as importance proxy

**Alternatives:**
- **Alternative A (Exact):** True leave-one-out: remove each KV pair, measure log-likelihood drop
  - Pro: Accurate ground truth
  - Con: O(L × S) per prompt, very expensive
- **Alternative B (Approximate):** Use attention weights directly as proxy
  - Pro: O(1) per token, very fast
  - Con: May miss long-range dependencies
- **Alternative C (Hybrid):** Use attention weights + gradient-based attribution
- **Recommendation:** Start with Alternative B, upgrade to A if results are poor

---

### Step 2.2: Policy Network v1
**What to do:** Build 4-layer Transformer policy network (~8M params)

**Simple version:** Start with smaller network (2 layers, 128 hidden)

| Component | Simple | Full |
|-----------|--------|------|
| Layers | 2 | 4 |
| Hidden size | 128 | 256 |
| Heads | 2 | 4 |
| Parameters | ~1M | ~8M |

---

### Step 2.3: Imitation Learning Training — ⚠️ HAS ALTERNATIVES
**What to do:** Train policy to match oracle actions via cross-entropy

**Simple version:** Standard supervised learning on oracle traces

**Alternatives:**
- **Alternative A:** Offline RL (DQfD, BCQ) instead of pure imitation
- **Alternative B:** Behavior cloning with data augmentation
- **Recommendation:** Start with pure behavior cloning (simplest)

---

## Phase 3: Three-Tier Hierarchy & RL (Week 4)

### Step 3.1: Three-Tier Cache Implementation — ⚠️ HAS ALTERNATIVES
**What to do:** Implement FP16 HBM / INT4 HBM / INT4 DRAM tiers

**Simple version:** Start with 2-tier (FP16 HBM / Evict only)

**Alternatives:**
- **Alternative A (2-tier):** FP16 HBM ↔ Evict (simplest)
- **Alternative B (3-tier with INT4):** FP16 HBM ↔ INT4 HBM ↔ Evict
- **Alternative C (Full 3-tier with DRAM offload):** As proposed (most complex)

**Recommendation:**
1. Week 4.1: Implement 2-tier (Alternative A)
2. Week 4.2: Add INT4 compression (Alternative B)
3. Week 4.3: Add DRAM offload (Alternative C)

---

### Step 3.2: INT4 Quantization Kernels
**Simple version:** Use KIVI's existing kernels directly

**Alternatives:**
- **Alternative A:** Use KIVI kernels as-is (fastest path)
- **Alternative B:** Implement custom CUDA kernels (more control, more work)
- **Recommendation:** Use KIVI kernels initially

---

### Step 3.3: PPO Fine-tuning — ⚠️ HAS ALTERNATIVES
**What to do:** Fine-tune policy with quality-aware reward

**Simple version:** Skip PPO, use only imitation learning for first results

**Reward alternatives:**
- **Alternative A (Full reward):** R = Q - λ·Mem - μ·Latency
- **Alternative B (Quality-only):** R = Q only
- **Alternative C (Fixed budget):** Hard constraint on memory, maximize quality

**PPO alternatives:**
- **Alternative A (PPO):** As proposed
- **Alternative B (DPO):** Direct Preference Optimization
- **Alternative C (Skip RL):** Only imitation learning, tune threshold manually
- **Recommendation:** Start with Alternative C for first results, add PPO if time permits

---

## Phase 4: Evaluation & Iteration (Week 5)

### Step 4.1: Full Benchmark Evaluation
**Simple version:** LongBench only for first pass

**Alternatives:**
- **Alternative A:** All 3 benchmarks from start
- **Alternative B:** LongBench → RULER → InfiniteBench
- **Recommendation:** Alternative B (prioritize by importance)

### Step 4.2: Ablation Studies
**Key ablations:**
1. Imitation only vs Imitation + PPO
2. Policy size (2/4/6 layers)
3. Per-layer vs shared policy
4. 2-tier vs 3-tier
5. Feature ablation
6. Cross-model transfer

**Simple version:** Run ablations 1, 2, 4 only

---

## Phase 5: Integration & Paper (Week 6)

### Step 5.1: vLLM Integration — ⚠️ HAS ALTERNATIVES
**Simple version:** Keep HuggingFace-only implementation

**Alternatives:**
- **Alternative A (vLLM integration):** Full integration with vLLM's block manager
- **Alternative B (HuggingFace only):** Stay with HuggingFace implementation
- **Recommendation:** HuggingFace for main results (Alternative B), vLLM if time

### Step 5.2: Paper Writing
- IEEE Transactions format, 10-16 pages
- System diagram, Pareto frontiers, ablation tables

---

## Summary: Recommended "Simple First" Path

| Week | Simple Version | Full Version | Priority |
|------|---------------|--------------|----------|
| 1 | Env setup + 2 baselines (H2O, StreamingLLM) | All 6 baselines | Simple ✓ |
| 2 | Add KIVI, dev benchmark | Full baseline suite | Simple ✓ |
| 3 | Attention-proxy oracle + small policy + BC | True oracle + full policy | Simple ✓ |
| 4 | 2-tier cache only | 3-tier + PPO | Simple ✓ |
| 5 | LongBench eval + key ablations | All benchmarks + all ablations | Incremental |
| 6 | HuggingFace implementation | vLLM integration | Start simple |

---

## Key Decision Points with Alternatives

1. **Oracle attribution method** (Week 3): Attention proxy vs. true leave-one-out
2. **Cache tiers** (Week 4): 2-tier vs. 3-tier
3. **RL training** (Week 4): Skip PPO vs. include
4. **vLLM integration** (Week 6): HuggingFace only vs. full vLLM
5. **Baseline implementation** (Week 1-2): Use existing repos vs. from-scratch
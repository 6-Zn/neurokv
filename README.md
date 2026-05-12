# NeuroKV: Adaptive Hierarchical KV Cache Management

A deep learning-based KV cache management system for long-context LLM inference.

## Project Overview

**Course:** Deep Learning (Spring 2026)
**Final Project:** NeuroKV - Adaptive Hierarchical Key–Value Cache Management via Deep Reinforcement Learning

### Key Features
- **Three-tier hierarchy**: FP16 HBM → INT4 HBM → INT4 DRAM
- **Learned policy**: Transformer-based policy network (~17M params)
- **Two-stage training**: Imitation learning + PPO fine-tuning
- **Target**: 4-8× memory reduction at ≤1 pp quality loss

## Project Structure

```
neurokv/
├── policy/           # Policy network (Transformer)
│   ├── network.py    # PolicyNetwork, PolicyNetworkSmall
│   └── __init__.py
├── cache/            # Cache manager (three-tier)
│   ├── manager.py    # ThreeTierCacheManager, Quantizer
│   └── __init__.py
├── oracle/           # Oracle attribution pipeline
│   ├── attribution.py # Importance scoring, trace generation
│   └── __init__.py
├── trainer/          # Training modules
│   ├── imitation.py  # Behavior cloning trainer
│   └── __init__.py
└── __init__.py

baselines/
├── unified_interface.py  # Unified interface for 6 baselines
├── BASELINE_REPOS.md     # Repo documentation
├── BASELINE_ANALYSIS.md  # Implementation analysis
├── h2o_repo/             # H2O implementation
├── streamingllm_repo/    # StreamingLLM implementation
├── snapkv_repo/          # SnapKV implementation
├── quest_repo/           # Quest implementation
├── kivi_repo/            # KIVI implementation
└── fastgen_repo/         # FastGen implementation

scripts/
├── create_dev_benchmark.py  # Dev benchmark creation

proposal/
├── proposal.tex    # IEEE Trans format proposal
├── proposal.pdf    # Compiled PDF
├── references.bib  # Bibliography
└── README.md       # Proposal overview
```

## Baselines

| Method | Paper | Key Idea |
|--------|-------|----------|
| H2O | [arxiv:2306.14048](https://arxiv.org/abs/2306.14048) | Heavy hitters + recent tokens |
| StreamingLLM | [arxiv:2309.17453](https://arxiv.org/abs/2309.17453) | Attention sinks + sliding window |
| SnapKV | [arxiv:2405.04443](https://arxiv.org/abs/2405.04443) | Clustered attention compression |
| Quest | [arxiv:2406.10774](https://arxiv.org/abs/2406.10774) | Page-based retrieval |
| KIVI | [arxiv:2402.02750](https://arxiv.org/abs/2402.02750) | 2/4-bit quantization |

## Setup

```bash
# Create environment
conda create -n neurokv python=3.10 -y
conda activate neurokv

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install vllm transformers accelerate datasets wandb tensorboard rouge-score sacrebleu nltk
pip install ipython jupyter black isort pytest pytest-cov evaluate

# Set PYTHONPATH
PYTHONPATH=/path/to/neurokv:$PYTHONPATH
```

## Quick Test

```bash
# Test policy network
python neurokv/policy/network.py

# Test cache manager
python neurokv/cache/manager.py

# Test oracle pipeline
python neurokv/oracle/attribution.py

# Test unified baselines
python baselines/unified_interface.py

# Test imitation trainer
PYTHONPATH=.:$PYTHONPATH python neurokv/trainer/imitation.py
```

## Training Results (Day 2)

| Metric | Value |
|--------|-------|
| Training accuracy | 72.44% |
| Test accuracy | 74.50% |
| KEEP_FP16 accuracy | 75.93% |
| EVICT accuracy | 97.46% |

## Timeline

| Week | Dates | Milestone |
|------|-------|-----------|
| 1 | May 11-17 | Environment, baselines, policy network |
| 2 | May 18-24 | Baseline validation, dev benchmark |
| 3 | May 25-31 | Oracle pipeline, imitation learning |
| 4 | Jun 1-7 | Three-tier hierarchy, PPO |
| 5 | Jun 8-14 | Full evaluation, ablations |
| 6 | Jun 15-22 | vLLM integration, paper |

## Documentation

- [EXECUTION_PLAN.md](EXECUTION_PLAN.md) - Detailed plan with alternatives
- [EXECUTION_LOG.md](EXECUTION_LOG.md) - Day-by-day execution log

## License

Apache-2.0

## Author

[Author Name] - Deep Learning Final Project (Spring 2026)
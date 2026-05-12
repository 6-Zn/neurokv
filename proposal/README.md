# NeuroKV — Final Project Proposal

**Course:** Deep Learning (Spring 2026)
**Title:** *NeuroKV: Adaptive Hierarchical Key–Value Cache Management via Deep Reinforcement Learning for Long-Context Large Language Model Inference*
**Proposal due:** 2026-05-11   ·   **Final report due:** 2026-06-22

## TL;DR

Long-context LLM inference is bottlenecked by the linear growth of the KV cache. Every published compression method (H2O, StreamingLLM, SnapKV, Quest, KIVI, KVQuant) uses a **hand-crafted heuristic** to decide which tokens to keep, compress, or evict. We replace that heuristic with a small **learned Transformer policy** trained via imitation learning from a full-attention oracle and fine-tuned with PPO under a quality-aware reward. The policy operates over a **three-tier hierarchy** (FP16 HBM / INT4 HBM / INT4 DRAM) and is integrated into vLLM. Target: **4–8× memory reduction at ≤1 pp quality loss** vs. full attention, beating the strongest baseline at matched memory.

## Why this is a strong final-paper topic

| Course / proposal criterion | How NeuroKV satisfies it |
| --- | --- |
| Novelty (选题的新颖性) | First learning-based KV cache management; existing work is all heuristic. |
| Innovation (解决问题的创新性) | Three-tier hierarchical action space + imitation→RL pipeline. |
| Systematic analysis (结果分析的系统性) | 6 baselines × 3 benchmarks × 6 ablations on a Pareto frontier. |
| Reproducibility (代码的可重复性) | vLLM extension + oracle traces + checkpoints under Apache-2.0. |
| Readability (报告的易读性) | IEEE Trans format, TikZ system diagram, clear notation. |
| Course alignment (deep learning) | Transformer policy network, supervised + RL training. |
| AI infrastructure focus | Sits in the LLM serving stack (vLLM extension). |
| PhD-level difficulty | Open research problem; combines RL, systems, and LLM internals. |

## Files in this directory

```
proposal/
├── proposal.tex          # IEEE Trans-format proposal (main source)
├── references.bib        # 22 cited works (all real, no fabrications)
├── proposal.pdf          # Compiled output (4 pages)
├── README.md             # This file
└── 大作业提案-20260427.pptx  # Original course requirements
```

## Building the PDF

Tested with MiKTeX 25.12 on Windows.

```powershell
cd C:\projects\yixin-vton-54\project\proposal
pdflatex -interaction=nonstopmode proposal.tex
bibtex   proposal
pdflatex -interaction=nonstopmode proposal.tex
pdflatex -interaction=nonstopmode proposal.tex
```

## Six-week plan (May 11 – Jun 22, 2026)

| Wk | Dates       | Milestone                                                              |
|----|-------------|------------------------------------------------------------------------|
| 1  | May 11–17   | Lit review; vLLM/HF environment; baseline reproduction harness.        |
| 2  | May 18–24   | Implement and validate H2O, StreamingLLM, SnapKV, Quest, KIVI.         |
| 3  | May 25–31   | Oracle attribution pipeline; imitation policy v1 (Stage 1).            |
| 4  | Jun 1–7     | Three-tier hierarchy; PPO fine-tuning (Stage 2); first E2E results.    |
| 5  | Jun 8–14    | Full eval on LongBench / RULER / InfiniteBench; ablations.             |
| 6  | Jun 15–22   | vLLM integration; throughput; paper writing; code release.             |

## Hardware

Primary: 1 × NVIDIA A800 80GB (group cluster). Stretch: 2 × A800 for 70B
tensor-parallel evaluation.

## Planned code-release layout (`neurokv-vllm/`)

```
neurokv-vllm/
├── neurokv/
│   ├── policy/             # Transformer policy network (~8M params)
│   ├── oracle/             # Full-attention attribution traces
│   ├── trainer/
│   │   ├── imitation.py    # Stage 1: cross-entropy on oracle actions
│   │   └── ppo.py          # Stage 2: quality-aware RL fine-tune
│   ├── cache/              # Three-tier KV pool, INT4 kernels
│   └── integration/        # vLLM hook + block-manager bridge
├── baselines/              # H2O / StreamingLLM / SnapKV / Quest / KIVI
├── eval/                   # LongBench / RULER / InfiniteBench runners
├── scripts/                # Repro scripts for every paper figure
└── README.md
```

## Risks and mitigations (summary)

- **Oracle attribution is expensive** → cache traces; subsample to ≤ 5K prompts.
- **RL fine-tuning unstable** → start from imitation; PPO clip; LLM frozen.
- **vLLM integration scope creep** → HuggingFace prototype first; vLLM is week-6.
- **Negative result** → matching strongest baseline at ½ memory + clean ablations is still publishable.

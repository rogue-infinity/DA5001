# DA5001 — Privacy in AI

Course repository for **DA5001: Special Topics in Data Science — Privacy in AI** (2025–2026).

---

## Assignments

| # | Notebook | Topics |
|---|----------|--------|
| HW 0 | `HW_0/HW_0.ipynb` | Python & ML warm-up |
| HW 1 | `HW_1/HW_1.ipynb` | Differential privacy fundamentals |
| HW 2 | `HW_2/HW_2.ipynb` | Local & global DP mechanisms |
| HW 3 | `HW_3/HW_3.ipynb` | Federated learning & privacy |
| HW 4 | `HW_4/HW_4.ipynb` | Membership inference attacks |

---

## Project — Membership Inference Attacks on Diffusion Language Models

> **Can an attacker determine whether a specific text was used to train a Masked Diffusion Language Model?**

This project studies privacy leakage in **Masked Diffusion Language Models (MDLMs)** — a class of generative models that reconstruct randomly masked tokens rather than predicting the next token. We show that fine-tuning an MDLM on member texts leaves a measurable memorisation signal in the model's ELBO (Evidence Lower BOund) trajectory, and exploit this signal to build membership inference attacks that substantially outperform existing baselines.

### Key Contributions

1. **White-box trajectory attack** — extract a 46-dimensional ELBO-trajectory feature vector per text (masking level × model internals), train an XGBoost/MLP classifier. Achieves **AUC 0.878** vs 0.816 for the best prior grey-box baseline (SAMA) on the MIMIR benchmark.
2. **Shadow model transfer** — train shadow MDLMs on disjoint surrogate data, transfer the classifier with *no target membership labels*. Reaches **AUC 0.858** (~98 % of oracle performance), with a gap of only 0.020 AUC.
3. **ELBO dominance finding** — leave-one-signal-out ablation shows the ELBO trajectory alone accounts for the full improvement; attention features add no signal and fail to transfer.

### Results Summary

| Method | Access | Mean AUC | TPR @ 1 % FPR |
|--------|--------|----------|---------------|
| Loss / Zlib / Ratio | black-box | 0.66–0.69 | 4–6 % |
| SAMA (Chen et al., ICLR 2026) | grey-box | 0.816 | 6.6 % |
| **XGBoost — white-box (ours)** | white-box | **0.878** | **24.7 %** |
| **Shadow transfer (ours)** | white-box† | **0.858** | **19.5 %** |

†No target membership labels required.

### Project Structure

```
Project/Project/
├── Runs/
│   ├── Run_1_Qwen_DLLM/        # Proof-of-concept: initial pipeline
│   ├── Run_2_Qwen_DLLM/        # Proof-of-concept: validated config
│   ├── Run_3_Full_MIMIR/       # Main experiment: 6-domain benchmark
│   └── Run_4_Shadow_MIA/       # Main experiment: shadow model transfer
├── Final_Report/               # LaTeX source + compiled PDF
├── Final_Presentation/         # Slides
└── Midterm_Report/             # Mid-project report
```

### Replication

The production-ready codebase (Run 3 + Run 4) with full setup instructions lives in a dedicated repository:

[![Replication Code](https://img.shields.io/badge/Replication%20Code-GitHub-black?style=for-the-badge&logo=github)](https://github.com/rogue-infinity/Whitebox-and-Shadow-Models-dLLM-Attacks)
[![Technical Report](https://img.shields.io/badge/Technical%20Report-Read%20PDF-red?style=for-the-badge&logo=adobeacrobatreader)](https://github.com/rogue-infinity/Whitebox-and-Shadow-Models-dLLM-Attacks/blob/main/Technical_Report.pdf)

### Model & Data

- **Target model**: [`dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1`](https://huggingface.co/dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1) (600 M params, Qwen3 backbone)
- **Benchmark**: [MIMIR](https://github.com/iamgroot42/mimir) — `iamgroot42/mimir`, 6 domains (arxiv, github, hackernews, pile\_cc, pubmed\_central, wikipedia), 1 000 members + 1 000 non-members per domain
- **External dependencies**: [SAMA](https://github.com/Stry233/SAMA) · [dLLM](https://github.com/ZHZisZZ/dllm)

---

## Literature

Key papers in `Papers/` and `Project/Lit_Review/`:

- Carlini et al. (2021) — *Extracting Training Data from Large Language Models*
- Mattern et al. (2023) — *Membership Inference Attacks against Language Models via Neighbourhood Comparison*
- Duan et al. (2024) — *Do Membership Inference Attacks Work on Large Language Models?* (MIMIR)
- Chen et al. (2026) — *Membership Inference Attacks Against Fine-tuned Diffusion Language Models* (SAMA, ICLR 2026)

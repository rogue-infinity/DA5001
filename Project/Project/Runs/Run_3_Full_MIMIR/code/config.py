"""
config.py — Central configuration for Run_3_Full_MIMIR.

All scripts import from here so hyperparameters are changed in one place.
"""

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
MIMIR_DATASETS = ["arxiv", "github", "hackernews", "pubmed_central", "wikipedia", "pile_cc"]
NLP_DATASETS   = ["wikitext103", "agnews", "xsum"]
ALL_DATASETS   = MIMIR_DATASETS + NLP_DATASETS

# ---------------------------------------------------------------------------
# Sample sizes
# ---------------------------------------------------------------------------
MIMIR_N = 1_000    # members + non-members per MIMIR domain
NLP_N   = 10_000   # members + non-members per NLP benchmark (Fu et al. 2024)

# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------
MAX_LENGTH = 256   # tokens; sequences shorter than this are discarded

# ---------------------------------------------------------------------------
# MIMIR HuggingFace config
# ---------------------------------------------------------------------------
MIMIR_HF_REPO = "iamgroot42/mimir"
MIMIR_SPLIT   = "ngram_13_0.8"    # higher n-gram overlap → cleaner member/nonmember

# HuggingFace config name for each MIMIR domain (may differ from internal key)
MIMIR_HF_NAME = {
    "arxiv":          "arxiv",
    "github":         "github",
    "hackernews":     "hackernews",
    "pubmed_central":  "pubmed_central",
    "wikipedia":      "wikipedia_(en)",   # HF uses "(en)" suffix
    "pile_cc":        "pile_cc",
}

# ---------------------------------------------------------------------------
# NLP benchmark HuggingFace configs
# ---------------------------------------------------------------------------
NLP_HF_CONFIGS = {
    "wikitext103": {"path": "wikitext", "name": "wikitext-103-raw-v1",
                    "text_col": "text",
                    "member_split": "train", "nonmember_split": "test"},
    "agnews":      {"path": "ag_news",  "name": None,
                    "text_col": "text",
                    "member_split": "train", "nonmember_split": "test"},
    "xsum":        {"path": "EdinburghNLP/xsum", "name": None,
                    "text_col": "document",
                    "member_split": "train", "nonmember_split": "test"},
}

# ---------------------------------------------------------------------------
# Fine-tuning hyperparameters (AdamW, bf16, single-GPU)
# ---------------------------------------------------------------------------
FT_LR          = 1e-4
FT_WD          = 0.01
FT_BATCH_SIZE  = 8
FT_EPOCHS      = 5

# ---------------------------------------------------------------------------
# Masking schedule shared by SAMA + signal extraction
# ---------------------------------------------------------------------------
T_STEPS   = 4
ALPHA_MIN = 0.05   # 5%  — minimum masking ratio
ALPHA_MAX = 0.50   # 50% — maximum masking ratio

# ---------------------------------------------------------------------------
# SAMA attack parameters
# ---------------------------------------------------------------------------
SAMA_N_SUBSETS = 128   # N: number of random subset masks per timestep
SAMA_M_TOKENS  = 10    # m: masked tokens per subset
SAMA_MC        = 4     # Monte-Carlo repetitions for variance reduction

# ---------------------------------------------------------------------------
# Signal extraction
# ---------------------------------------------------------------------------
SIG_K         = 8      # mask configs for multi-mask consistency
SIG_GRAD_T    = 2      # timesteps for gradient norm computation

# ---------------------------------------------------------------------------
# Base model
# ---------------------------------------------------------------------------
MODEL_PATH = "dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1"

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Qwen2.5 / Gemma 4 SORRY-Bench steering-vector cosine pipeline for active/passive variants.

This is a single-file pipeline for the user's dataset shape:

[
  {
    "id": 0,
    "category": "...",
    "domain": "...",
    "C1_active": "...",
    "C2_passive": "...",
    "C3_active_ctx": "...",
    "C4_passive_ctx": "..."
  },
  ...
]

Pipeline:
  1. Download/load SORRY-Bench 202503 from Hugging Face.
  2. Filter selected refusal-required category ids.
  3. Load benign prompts from CSV or Hugging Face Alpaca.
  4. Build calibration rows:
       benign
       harmful_refused from filtered SORRY-Bench
  5. Build evaluation rows:
       C1_active
       C2_passive
       C3_active_ctx
       C4_passive_ctx
  6. Extract all-layer hidden states at:
       t_inst      = final token inside user instruction text
       t_post_inst = final token of full chat-template prompt before generation
  7. Build steering vectors:
       harmfulness_t_inst = mean(SORRY harmful at t_inst) - mean(benign at t_inst)
       refusal_t_post     = mean(SORRY harmful/refused at t_post) - mean(benign at t_post)
  8. Compute cosine similarity for each evaluation variant C1/C2/C3/C4.
  9. Save layer-level, band-level, variant-level, condition-level, and paired-gap outputs.

Install:
  pip install torch transformers accelerate datasets pandas numpy tqdm matplotlib

Example debug run:
  python gemma4_sorry_active_passive_cosine_pipeline.py \
    --active_passive_json active_passive.json \
    --benign_csv benign_prompts.csv \
    --output_dir debug_gemma4_31b \
    --model_name google/gemma-4-31B-it \
    --max_length 1024 \
    --dtype bfloat16 \
    --device_map auto \
    --max_sorry_per_category 2 \
    --save_hidden_cache

Example 72B run:
  python gemma4_sorry_active_passive_cosine_pipeline.py \
    --active_passive_json active_passive.json \
    --benign_csv benign_prompts.csv \
    --output_dir gemma4_31b_steering_results \
    --model_name google/gemma-4-31B-it \
    --max_length 1024 \
    --dtype bfloat16 \
    --device_map auto \
    --save_hidden_cache

Notes:
  - SORRY-Bench 202503 can require Hugging Face access and `huggingface-cli login`.
  - Evaluation refused labels are set to -1 by default because cosine evaluation does not require generated outputs.
  - For final attack-success analysis, separately generate outputs and annotate refused/accepted.
"""

import os
import re
import json
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from datasets import load_dataset
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM, AutoModelForImageTextToText


DEFAULT_REFUSAL_REQUIRED_CATEGORIES = [
    6, 7, 8, 9, 10,
    12, 13, 14, 15,
    17, 18, 19, 20, 21,
    23, 24, 28,
]

SORRY_DATASET_NAME = "sorry-bench/sorry-bench-202503"

CANDIDATE_CATEGORY_ID_COLS = [
    "category_id",
    "category_idx",
    "category",
    "category_num",
    "taxonomy_id",
]

CANDIDATE_CATEGORY_NAME_COLS = [
    "category_name",
    "category",
    "taxonomy",
    "safety_category",
]

CANDIDATE_PROMPT_COLS = [
    "prompt",
    "question",
    "instruction",
    "user_query",
    "query",
    "turns",  # Changed: SORRY-Bench stores prompt text inside this column
]


@dataclass
class Config:
    model_name: str
    active_passive_json: str
    benign_csv: Optional[str]
    output_dir: str

    benign_dataset_name: Optional[str]  # Changed: optional Hugging Face benign/Alpaca dataset
    benign_dataset_split: str  # Changed
    benign_prompt_col: str  # Changed
    benign_input_col: Optional[str]  # Changed
    benign_source_label: str  # Changed

    hf_cache_dir: str
    prefer_local_model_cache: bool  # Changed: use cached snapshot before attempting download

    sorry_dataset_name: str
    sorry_csv: Optional[str]  # Changed
    sorry_split: Optional[str]
    category_ids: List[int]
    max_sorry_per_category: Optional[int]
    sorry_base_only: bool  # Changed: filter SORRY-Bench to prompt_style=="base" only

    mode: str
    max_length: int
    dtype: str
    device_map: str

    calibration_split: str
    evaluation_split: str

    benign_label: str
    harmful_label: str
    active_label: str
    passive_label: str

    save_hidden_cache: bool
    load_hidden_cache: bool
    save_built_dataset: bool

    max_benign_samples: Optional[int]  # Changed: cap benign/Alpaca calibration samples
    benign_sample_seed: int  # Changed: reproducible benign/Alpaca sampling

    model_loader: str  # auto | causal_lm | image_text_to_text
    processor_loader: str  # auto | tokenizer | processor


# ============================================================
# Utilities
# ============================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_torch_dtype(dtype_name: str):
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def safe_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x)


def safe_filename(x: str) -> str:
    x = str(x)
    x = x.replace("/", "__").replace("\\", "__").replace(" ", "_").replace(":", "_")
    x = re.sub(r"[^A-Za-z0-9_.=-]+", "_", x)
    return x[:180]


def find_col(df: pd.DataFrame, candidates: List[str], required: bool = True) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(
            f"Could not find any candidate columns: {candidates}. "
            f"Available columns: {list(df.columns)}"
        )
    return None


def normalize_category_id_series(s: pd.Series) -> pd.Series:
    def parse_one(x):
        if pd.isna(x):
            return None
        if isinstance(x, int):
            return x
        if isinstance(x, float):
            return int(x)

        text = str(x).strip()
        if text.isdigit():
            return int(text)

        m = re.match(r"^\s*(\d+)", text)
        if m:
            return int(m.group(1))

        return None

    return s.apply(parse_one)


def get_input_device(model):
    """
    Safer than model.device when using device_map='auto'.
    """
    return model.get_input_embeddings().weight.device




def hf_repo_id_to_cache_dir_name(repo_id: str) -> str:
    """
    Convert a Hugging Face repo id into the hub cache directory name.

    Example:
      Qwen/Qwen2.5-72B-Instruct
      -> models--Qwen--Qwen2.5-72B-Instruct
    """
    return "models--" + repo_id.replace("/", "--")


def is_valid_model_snapshot(path: Path) -> bool:
    """
    Return True when a HF snapshot directory looks usable by transformers.
    Symlinks are fine because Hugging Face snapshots commonly point into blobs/.
    """
    if not path.exists() or not path.is_dir():
        return False

    has_config = (path / "config.json").exists()
    has_tokenizer = (
        (path / "tokenizer.json").exists()
        or (path / "tokenizer.model").exists()
        or (path / "vocab.json").exists()
        or (path / "merges.txt").exists()
    )
    has_weights = (
        (path / "model.safetensors.index.json").exists()
        or any(path.glob("*.safetensors"))
        or any(path.glob("pytorch_model*.bin"))
    )

    return has_config and has_tokenizer and has_weights


def find_latest_cached_snapshot(model_name: str, hf_cache_dir: str) -> Optional[str]:
    """
    Find a local Hugging Face snapshot for a repo id.

    If multiple snapshots exist, use the most recently modified valid one.
    """
    if "/" not in model_name:
        return None

    model_cache_dir = Path(hf_cache_dir) / hf_repo_id_to_cache_dir_name(model_name)
    snapshots_dir = model_cache_dir / "snapshots"

    if not snapshots_dir.exists():
        return None

    candidates = [
        snapshot
        for snapshot in snapshots_dir.iterdir()
        if snapshot.is_dir() and is_valid_model_snapshot(snapshot)
    ]

    if not candidates:
        return None

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(candidates[0])


def resolve_model_name_or_path(
    model_name: str,
    hf_cache_dir: str,
    prefer_local_cache: bool = True,
) -> str:
    """
    Resolve the model name/path before calling transformers.

    Priority:
      1. If model_name is already a local path, use it directly.
      2. If a valid local HF snapshot exists, use that snapshot.
      3. Otherwise return the HF repo id and let transformers download/use cache_dir.
    """
    model_path = Path(model_name)

    if model_path.exists():
        print(f"[INFO] Using explicit local model path: {model_path}")
        return str(model_path)

    if prefer_local_cache:
        cached_snapshot = find_latest_cached_snapshot(model_name, hf_cache_dir)
        if cached_snapshot is not None:
            print(f"[INFO] Found cached model snapshot: {cached_snapshot}")
            return cached_snapshot

    print(f"[INFO] No valid local snapshot found for: {model_name}")
    print(f"[INFO] Falling back to Hugging Face repo id with cache_dir={hf_cache_dir}")
    return model_name


def extract_prompt_from_sorrybench_turns(turns) -> str:
    """
    Extract prompt text from SORRY-Bench prompt-like fields.

    Supports:
      - plain string prompt
      - list/tuple/array: ["prompt text"]
      - dict: {"content": "..."}
      - stringified list/dict: '["prompt text"]'
    """
    import ast

    if turns is None:
        return ""

    # numpy / pandas object arrays often expose .tolist().
    if hasattr(turns, "tolist") and not isinstance(turns, str):
        try:
            turns = turns.tolist()
        except Exception:
            pass

    if isinstance(turns, str):
        text = turns.strip()
        if not text:
            return ""

        # Handle rows that came through CSV as a stringified list/dict.
        if text.startswith("[") or text.startswith("{"):
            try:
                parsed = ast.literal_eval(text)
                parsed_text = extract_prompt_from_sorrybench_turns(parsed)
                if parsed_text.strip():
                    return parsed_text.strip()
            except Exception:
                pass

        return text

    if isinstance(turns, (list, tuple)) and len(turns) > 0:
        first = turns[0]

        if isinstance(first, str):
            return first.strip()

        if isinstance(first, dict):
            for key in ["content", "text", "value", "prompt", "question", "instruction", "user_query", "query"]:
                value = first.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()

    if isinstance(turns, dict):
        for key in ["content", "text", "value", "prompt", "question", "instruction", "user_query", "query"]:
            value = turns.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    return ""


def has_any_column(df: pd.DataFrame, candidates: List[str]) -> bool:
    return any(c in df.columns for c in candidates)


def load_local_sorry_csv(path: str) -> pd.DataFrame:
    """
    Load a local SORRY-style CSV robustly.

    Expected preferred format:
      category,prompt
      1,"..."

    Also supports a headerless two-column CSV:
      1,"..."
    """
    print("[INFO] Loading local SORRY CSV:", path)

    df = pd.read_csv(path)
    has_category_col = has_any_column(df, CANDIDATE_CATEGORY_ID_COLS)
    has_prompt_col = has_any_column(df, CANDIDATE_PROMPT_COLS)

    if not (has_category_col and has_prompt_col):
        print("[INFO] Local SORRY CSV does not expose category/prompt headers. Re-loading as headerless category,prompt.")
        df = pd.read_csv(path, header=None, names=["category", "prompt"])

    # Normalize likely column names for the rest of the pipeline.
    if "category" not in df.columns:
        category_col = find_col(df, CANDIDATE_CATEGORY_ID_COLS, required=True)
        df = df.rename(columns={category_col: "category"})

    if "prompt" not in df.columns:
        prompt_col = find_col(df, CANDIDATE_PROMPT_COLS, required=True)
        df = df.rename(columns={prompt_col: "prompt"})

    df["prompt"] = df["prompt"].apply(extract_prompt_from_sorrybench_turns)
    before = len(df)
    df = df[df["prompt"].astype(str).str.strip().ne("")].copy()
    dropped = before - len(df)
    if dropped:
        print(f"[WARN] Dropped {dropped} local SORRY rows with empty prompt.")

    if len(df) == 0:
        raise ValueError("Local SORRY CSV loaded successfully, but all prompt values are empty.")

    return df


# ============================================================
# Dataset build
# ============================================================

def load_sorry_dataset(dataset_name: str, split_name: Optional[str]) -> pd.DataFrame:
    if split_name is not None:
        ds = load_dataset(dataset_name, split=split_name)
        return ds.to_pandas()

    dsdict = load_dataset(dataset_name)
    first_split = list(dsdict.keys())[0]
    print(f"[INFO] Loaded SORRY splits: {list(dsdict.keys())}")
    print(f"[INFO] Using SORRY split: {first_split}")
    return dsdict[first_split].to_pandas()


def build_sorry_harmful_rows(
    sorry_df: pd.DataFrame,
    category_ids: List[int],
    harmful_label: str,
    max_per_category: Optional[int],
    base_only: bool = False,  # Changed: if True, filter to prompt_style=="base"
) -> pd.DataFrame:
    category_id_col = find_col(sorry_df, CANDIDATE_CATEGORY_ID_COLS, required=True)
    prompt_col = find_col(sorry_df, CANDIDATE_PROMPT_COLS, required=True)
    category_name_col = find_col(sorry_df, CANDIDATE_CATEGORY_NAME_COLS, required=False)

    df = sorry_df.copy()

    # Changed: filter to base prompt style only if requested
    if base_only:
        if "prompt_style" in df.columns:
            before = len(df)
            df = df[df["prompt_style"] == "base"].copy()
            print(f"[INFO] sorry_base_only=True: filtered {before} -> {len(df)} rows (prompt_style=='base')")
        else:
            print("[WARN] sorry_base_only=True but 'prompt_style' column not found; skipping base filter.")

    df["_category_id_int"] = normalize_category_id_series(df[category_id_col])

    if prompt_col == "turns":
        df["_prompt_text"] = df[prompt_col].apply(extract_prompt_from_sorrybench_turns)
    else:
        df["_prompt_text"] = df[prompt_col].apply(extract_prompt_from_sorrybench_turns)

    before_nonempty = len(df)
    df = df[df["_prompt_text"].astype(str).str.strip().ne("")].copy()
    dropped_empty = before_nonempty - len(df)
    if dropped_empty:
        print(f"[WARN] Dropped {dropped_empty} SORRY rows with empty prompt before category sampling.")

    selected = df[df["_category_id_int"].isin(category_ids)].copy()
    if len(selected) == 0:
        available = sorted([int(x) for x in df["_category_id_int"].dropna().unique().tolist()])
        raise ValueError(
            "No SORRY rows matched selected category IDs after prompt extraction. "
            f"Requested={category_ids}; available={available[:80]}"
        )

    counts_before = selected["_category_id_int"].value_counts().sort_index()
    print("[INFO] SORRY selected category counts before max_per_category:")
    print(counts_before.to_string())

    if max_per_category is not None:
        low = counts_before[counts_before < max_per_category]
        if not low.empty:
            print(
                f"[WARN] Some SORRY categories have fewer than --max_sorry_per_category={max_per_category} usable prompts:"
            )
            print(low.to_string())

        selected = (
            selected
            .sort_index()
            .groupby("_category_id_int", group_keys=False)
            .head(max_per_category)
            .copy()
        )

    rows = []
    for i, row in selected.iterrows():
        cid = int(row["_category_id_int"])
        cname = safe_str(row[category_name_col]) if category_name_col is not None else str(cid)
        prompt = safe_str(row["_prompt_text"]).strip()

        if not prompt:
            print(f"[WARN] Empty SORRY prompt skipped: category_id={cid}, row_index={i}")
            continue

        rows.append({
            "id": f"calib_sorry_{cid}_{i}",
            "split": "calibration",
            "source": "sorry-bench-202503",
            "category_id": cid,
            "category_name": cname,
            "domain": "",
            "pair_id": "",
            "contextualized": "",
            "variant": "sorry_filtered",
            "condition": harmful_label,
            "prompt": prompt,
            # First-pass assumption. Replace with actual model refusal labels for final experiments.
            "refused": 1,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("SORRY harmful calibration table is empty after filtering and prompt extraction.")

    print("[INFO] SORRY harmful calibration rows built:", len(out))
    print("[INFO] SORRY harmful calibration counts:")
    print(out["category_id"].value_counts().sort_index().to_string())

    return out



def make_benign_rows_from_dataframe(
    benign: pd.DataFrame,
    benign_label: str,
    source_label: str = "benign",
) -> pd.DataFrame:
    """
    Convert a benign prompt DataFrame into the unified calibration schema.

    Required column:
      - prompt
    Optional columns:
      - id, source, category_id, category_name, domain, variant, refused
    """
    if "prompt" not in benign.columns:
        raise ValueError(
            "benign data must contain a 'prompt' column. "
            f"Available columns: {list(benign.columns)}"
        )

    rows = []
    for i, row in benign.iterrows():
        prompt = safe_str(row["prompt"]).strip()
        if not prompt:
            print(f"[WARN] Empty benign prompt skipped: row_index={i}")
            continue

        rows.append({
            "id": safe_str(row["id"]) if "id" in benign.columns else f"calib_benign_{i}",
            "split": "calibration",
            "source": safe_str(row["source"]) if "source" in benign.columns else source_label,
            "category_id": safe_str(row["category_id"]) if "category_id" in benign.columns else "",
            "category_name": safe_str(row["category_name"]) if "category_name" in benign.columns else "",
            "domain": safe_str(row["domain"]) if "domain" in benign.columns else "",
            "pair_id": "",
            "contextualized": "",
            "variant": safe_str(row["variant"]) if "variant" in benign.columns else "benign",
            "condition": benign_label,
            "prompt": prompt,
            "refused": int(row["refused"]) if "refused" in benign.columns and not pd.isna(row["refused"]) else 0,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No usable benign prompts found after prompt extraction.")

    print("[INFO] Benign calibration rows built:", len(out))
    return out


def load_benign_rows(benign_csv: str, benign_label: str) -> pd.DataFrame:
    benign = pd.read_csv(benign_csv)
    if "prompt" not in benign.columns:
        raise ValueError("benign_csv must contain a 'prompt' column.")
    return make_benign_rows_from_dataframe(benign, benign_label, source_label="benign_csv")


def build_prompt_from_instruction_input(instruction: str, input_text: str) -> str:
    instruction = safe_str(instruction).strip()
    input_text = safe_str(input_text).strip()
    if input_text:
        return instruction + "\n\n" + input_text
    return instruction


def load_benign_rows_from_hf_dataset(
    dataset_name: str,
    split_name: str,
    prompt_col: str,
    input_col: Optional[str],
    benign_label: str,
    source_label: str,
) -> pd.DataFrame:
    print(f"[INFO] Loading benign dataset from Hugging Face: {dataset_name} split={split_name}")
    ds = load_dataset(dataset_name, split=split_name)
    df = ds.to_pandas()

    if prompt_col not in df.columns:
        raise ValueError(
            f"Benign dataset prompt column '{prompt_col}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    converted = pd.DataFrame()
    if input_col is not None and input_col.strip() and input_col in df.columns:
        converted["prompt"] = [
            build_prompt_from_instruction_input(inst, inp)
            for inst, inp in zip(df[prompt_col], df[input_col])
        ]
    else:
        converted["prompt"] = df[prompt_col].apply(safe_str)

    converted["id"] = [f"calib_benign_hf_{i}" for i in range(len(converted))]
    converted["source"] = source_label
    converted["variant"] = "benign"
    converted["refused"] = 0

    return make_benign_rows_from_dataframe(converted, benign_label, source_label=source_label)


def load_benign_calibration_rows(cfg: Config) -> pd.DataFrame:
    has_csv = cfg.benign_csv is not None and str(cfg.benign_csv).strip()
    has_hf = cfg.benign_dataset_name is not None and str(cfg.benign_dataset_name).strip()

    if has_csv and has_hf:
        raise ValueError("Use either --benign_csv or --benign_dataset_name, not both.")

    if has_csv:
        print("[INFO] Loading benign prompts from CSV:", cfg.benign_csv)
        return load_benign_rows(cfg.benign_csv, cfg.benign_label)

    if has_hf:
        return load_benign_rows_from_hf_dataset(
            dataset_name=cfg.benign_dataset_name,
            split_name=cfg.benign_dataset_split,
            prompt_col=cfg.benign_prompt_col,
            input_col=cfg.benign_input_col,
            benign_label=cfg.benign_label,
            source_label=cfg.benign_source_label,
        )

    raise ValueError("Provide either --benign_csv or --benign_dataset_name.")


def load_active_passive_json(path: str) -> List[Dict]:
    text = Path(path).read_text(encoding="utf-8")
    return json.loads(text)


def build_active_passive_eval_rows(
    items: List[Dict],
    active_label: str,
    passive_label: str,
    mode: str,
) -> pd.DataFrame:
    include_plain = mode in ["plain", "both", "all"]
    include_context = mode in ["context", "both", "all"]
    include_tense = mode in ["tense", "all"]

    rows = []

    for item in items:
        base_id = str(item["id"])
        category = safe_str(item.get("category", ""))
        domain = safe_str(item.get("domain", ""))

        if include_plain:
            pair_id = f"{base_id}_plain"

            rows.append({
                "id": f"eval_{base_id}_C1_active",
                "split": "evaluation",
                "source": "active_passive_dataset",
                "category_id": base_id,
                "category_name": category,
                "domain": domain,
                "pair_id": pair_id,
                "contextualized": 0,
                "variant": "C1_active",
                "condition": active_label,
                "prompt": safe_str(item["C1_active"]),
                "refused": -1,
            })

            rows.append({
                "id": f"eval_{base_id}_C2_passive",
                "split": "evaluation",
                "source": "active_passive_dataset",
                "category_id": base_id,
                "category_name": category,
                "domain": domain,
                "pair_id": pair_id,
                "contextualized": 0,
                "variant": "C2_passive",
                "condition": passive_label,
                "prompt": safe_str(item["C2_passive"]),
                "refused": -1,
            })

        if include_context:
            pair_id = f"{base_id}_ctx"

            rows.append({
                "id": f"eval_{base_id}_C3_active_ctx",
                "split": "evaluation",
                "source": "active_passive_dataset",
                "category_id": base_id,
                "category_name": category,
                "domain": domain,
                "pair_id": pair_id,
                "contextualized": 1,
                "variant": "C3_active_ctx",
                "condition": active_label,
                "prompt": safe_str(item["C3_active_ctx"]),
                "refused": -1,
            })

            rows.append({
                "id": f"eval_{base_id}_C4_passive_ctx",
                "split": "evaluation",
                "source": "active_passive_dataset",
                "category_id": base_id,
                "category_name": category,
                "domain": domain,
                "pair_id": pair_id,
                "contextualized": 1,
                "variant": "C4_passive_ctx",
                "condition": passive_label,
                "prompt": safe_str(item["C4_passive_ctx"]),
                "refused": -1,
            })

        if include_tense:
            pair_id = f"{base_id}_tense"

            rows.append({
                "id": f"eval_{base_id}_C5_tense",
                "split": "evaluation",
                "source": "active_passive_dataset",
                "category_id": base_id,
                "category_name": category,
                "domain": domain,
                "pair_id": pair_id,
                "contextualized": 0,
                "variant": "C5_tense",
                "condition": active_label,
                "prompt": safe_str(item.get("C5_tense", "")),
                "refused": -1,
            })

    return pd.DataFrame(rows)


def build_full_dataset(cfg: Config) -> pd.DataFrame:
    print("[INFO] Loading SORRY-Bench for calibration...")

    # Prefer local SORRY CSV if provided. Supports both headered category,prompt
    # and headerless two-column CSV files.
    if cfg.sorry_csv is not None and str(cfg.sorry_csv).strip():
        sorry_df = load_local_sorry_csv(cfg.sorry_csv)
    else:
        sorry_df = load_sorry_dataset(cfg.sorry_dataset_name, cfg.sorry_split)

    print("[INFO] SORRY columns:", list(sorry_df.columns))

    harmful_df = build_sorry_harmful_rows(
        sorry_df=sorry_df,
        category_ids=cfg.category_ids,
        harmful_label=cfg.harmful_label,
        max_per_category=cfg.max_sorry_per_category,
        base_only=cfg.sorry_base_only,  # Changed
    )

    benign_df = load_benign_calibration_rows(cfg)

    # Changed: cap benign/Alpaca calibration rows explicitly when requested.
    # If not set, keep the original behavior and balance benign down to the harmful count.
    if cfg.max_benign_samples is not None:
        if len(benign_df) < cfg.max_benign_samples:
            raise ValueError(
                f"benign data has only {len(benign_df)} rows, but --max_benign_samples={cfg.max_benign_samples}."
            )
        benign_df = benign_df.sample(n=cfg.max_benign_samples, random_state=cfg.benign_sample_seed).copy()
    elif len(benign_df) > len(harmful_df):
        benign_df = benign_df.sample(n=len(harmful_df), random_state=cfg.benign_sample_seed).copy()

    items = load_active_passive_json(cfg.active_passive_json)
    eval_df = build_active_passive_eval_rows(
        items=items,
        active_label=cfg.active_label,
        passive_label=cfg.passive_label,
        mode=cfg.mode,
    )

    all_cols = [
        "id", "split", "source", "category_id", "category_name", "domain",
        "pair_id", "contextualized", "variant", "condition", "prompt", "refused",
    ]

    calib_df = pd.concat([benign_df[all_cols], harmful_df[all_cols]], ignore_index=True)
    full_df = pd.concat([calib_df, eval_df[all_cols]], ignore_index=True)

    return full_df


# ============================================================
# Chat rendering and positions
# ============================================================

def make_text_messages(prompt: str, multimodal_style: bool = False) -> List[Dict]:
    """
    Build chat-template messages.

    Gemma 4 IT is registered as an image-text-to-text model on Hugging Face.
    Its examples use multimodal-style content blocks even for text prompts:
      {"role": "user", "content": [{"type": "text", "text": prompt}]}

    Most text-only instruct models, such as Qwen, use:
      {"role": "user", "content": prompt}

    We support both and fall back automatically.
    """
    if multimodal_style:
        return [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    return [{"role": "user", "content": prompt}]


def render_chat(tokenizer, prompt: str) -> str:
    # Try text-only format first for backward compatibility. If the model chat
    # template expects OpenAI/Gemma-style content blocks, retry with text blocks.
    try:
        return tokenizer.apply_chat_template(
            make_text_messages(prompt, multimodal_style=False),
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return tokenizer.apply_chat_template(
            make_text_messages(prompt, multimodal_style=True),
            tokenize=False,
            add_generation_prompt=True,
        )


def find_t_inst_and_t_post_inst(
    tokenizer,
    prompt: str,
    max_length: int,
) -> Tuple[str, torch.Tensor, int, int]:
    rendered = render_chat(tokenizer, prompt)

    prompt_start = rendered.rfind(prompt)
    if prompt_start == -1:
        raise ValueError(
            "Prompt text not found inside rendered chat template. "
            "Check whether the tokenizer template transformed the prompt."
        )

    prompt_end = prompt_start + len(prompt)

    enc = tokenizer(
        rendered,
        return_offsets_mapping=True,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )

    input_ids = enc["input_ids"]
    offsets = enc["offset_mapping"][0].tolist()

    t_inst = None
    for i, (s, e) in enumerate(offsets):
        if e > prompt_start and s < prompt_end:
            t_inst = i

    if t_inst is None:
        raise ValueError(
            "Could not identify t_inst. Prompt may have been truncated."
        )

    t_post_inst = input_ids.shape[1] - 1
    return rendered, input_ids, t_inst, t_post_inst


# ============================================================
# Hidden extraction/cache
# ============================================================

@torch.no_grad()
def extract_hidden_for_prompt(
    model,
    tokenizer,
    prompt: str,
    max_length: int,
) -> Dict[str, torch.Tensor]:
    _, input_ids, t_inst, t_post_inst = find_t_inst_and_t_post_inst(
        tokenizer=tokenizer,
        prompt=prompt,
        max_length=max_length,
    )

    input_device = get_input_device(model)
    input_ids = input_ids.to(input_device)
    attention_mask = torch.ones_like(input_ids, device=input_device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )

    # outputs.hidden_states[0] is embedding output. Exclude it.
    hidden_states = outputs.hidden_states[1:]

    t_inst_layers = []
    t_post_layers = []

    for layer_h in hidden_states:
        t_inst_layers.append(layer_h[0, t_inst, :].detach().float().cpu())
        t_post_layers.append(layer_h[0, t_post_inst, :].detach().float().cpu())

    result = {
        "t_inst": torch.stack(t_inst_layers),
        "t_post": torch.stack(t_post_layers),
        "t_inst_index": int(t_inst),
        "t_post_index": int(t_post_inst),
        "seq_len": int(input_ids.shape[1]),
    }

    del outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def record_cache_path(cache_dir: str, row_id: str, variant: str, condition: str) -> str:
    key = f"{row_id}__{variant}__{condition}"
    return os.path.join(cache_dir, f"{safe_filename(key)}.pt")


def extract_or_load_records(
    df: pd.DataFrame,
    model,
    tokenizer,
    cfg: Config,
) -> List[Dict]:
    cache_dir = os.path.join(cfg.output_dir, "hidden_cache")
    ensure_dir(cache_dir)

    records = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extract/load hidden states"):
        row_id = safe_str(row["id"])
        variant = safe_str(row["variant"])
        condition = safe_str(row["condition"])
        cache_path = record_cache_path(cache_dir, row_id, variant, condition)

        if cfg.load_hidden_cache and os.path.exists(cache_path):
            cached = torch.load(cache_path, map_location="cpu")
            t_inst = cached["t_inst"]
            t_post = cached["t_post"]
            t_inst_index = int(cached.get("t_inst_index", -1))
            t_post_index = int(cached.get("t_post_index", -1))
            seq_len = int(cached.get("seq_len", -1))
        else:
            hs = extract_hidden_for_prompt(
                model=model,
                tokenizer=tokenizer,
                prompt=safe_str(row["prompt"]),
                max_length=cfg.max_length,
            )
            t_inst = hs["t_inst"]
            t_post = hs["t_post"]
            t_inst_index = hs["t_inst_index"]
            t_post_index = hs["t_post_index"]
            seq_len = hs["seq_len"]

            if cfg.save_hidden_cache:
                torch.save(
                    {
                        "id": row_id,
                        "variant": variant,
                        "condition": condition,
                        "t_inst": t_inst,
                        "t_post": t_post,
                        "t_inst_index": t_inst_index,
                        "t_post_index": t_post_index,
                        "seq_len": seq_len,
                    },
                    cache_path,
                )

        records.append({
            "id": row_id,
            "split": safe_str(row["split"]),
            "source": safe_str(row["source"]),
            "category_id": safe_str(row["category_id"]),
            "category_name": safe_str(row["category_name"]),
            "domain": safe_str(row["domain"]),
            "pair_id": safe_str(row["pair_id"]),
            "contextualized": safe_str(row["contextualized"]),
            "variant": variant,
            "condition": condition,
            "prompt": safe_str(row["prompt"]),
            "refused": int(row["refused"]),
            "t_inst": t_inst,
            "t_post": t_post,
            "t_inst_index": t_inst_index,
            "t_post_index": t_post_index,
            "seq_len": seq_len,
        })

    return records


# ============================================================
# Steering vectors and cosine
# ============================================================

def filter_records(
    records: List[Dict],
    split: Optional[str] = None,
    condition: Optional[str] = None,
    refused: Optional[int] = None,
) -> List[Dict]:
    out = []
    for r in records:
        if split is not None and r["split"] != split:
            continue
        if condition is not None and r["condition"] != condition:
            continue
        if refused is not None and int(r["refused"]) != int(refused):
            continue
        out.append(r)
    return out


def stack_position(records: List[Dict], position: str) -> torch.Tensor:
    if len(records) == 0:
        raise ValueError(f"No records to stack for position={position}")
    return torch.stack([r[position] for r in records])


def make_mean_direction(positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
    direction = positive.mean(dim=0) - negative.mean(dim=0)
    return normalize(direction)


def build_steering_vectors(records: List[Dict], cfg: Config) -> Dict[str, torch.Tensor]:
    calib_benign = filter_records(
        records,
        split=cfg.calibration_split,
        condition=cfg.benign_label,
        refused=0,
    )
    calib_harmful = filter_records(
        records,
        split=cfg.calibration_split,
        condition=cfg.harmful_label,
    )
    calib_harmful_refused = filter_records(
        records,
        split=cfg.calibration_split,
        condition=cfg.harmful_label,
        refused=1,
    )

    if len(calib_benign) == 0:
        raise ValueError("No calibration benign records found.")
    if len(calib_harmful) == 0:
        raise ValueError("No calibration harmful records found.")
    if len(calib_harmful_refused) == 0:
        raise ValueError("No calibration harmful refused=1 records found.")

    benign_t_inst = stack_position(calib_benign, "t_inst")
    harmful_t_inst = stack_position(calib_harmful, "t_inst")

    benign_t_post = stack_position(calib_benign, "t_post")
    harmful_refused_t_post = stack_position(calib_harmful_refused, "t_post")

    harmfulness_vector = make_mean_direction(harmful_t_inst, benign_t_inst)
    refusal_vector = make_mean_direction(harmful_refused_t_post, benign_t_post)

    return {
        "harmfulness_t_inst": harmfulness_vector,
        "refusal_t_post": refusal_vector,
    }


def cosine_by_layer(hidden: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    hidden_norm = normalize(hidden)
    direction_norm = normalize(direction)
    return (hidden_norm * direction_norm).sum(dim=-1)


def projection_by_layer(hidden: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    direction_norm = normalize(direction)
    return (hidden * direction_norm).sum(dim=-1)


def compute_eval_scores(
    records: List[Dict],
    steering: Dict[str, torch.Tensor],
    cfg: Config,
) -> pd.DataFrame:
    eval_records = filter_records(records, split=cfg.evaluation_split)
    if len(eval_records) == 0:
        raise ValueError("No evaluation records found.")

    harmfulness_vec = steering["harmfulness_t_inst"]
    refusal_vec = steering["refusal_t_post"]

    rows = []
    for r in tqdm(eval_records, desc="Compute eval cosine/projection"):
        h_cos = cosine_by_layer(r["t_inst"], harmfulness_vec)
        r_cos = cosine_by_layer(r["t_post"], refusal_vec)

        h_proj = projection_by_layer(r["t_inst"], harmfulness_vec)
        r_proj = projection_by_layer(r["t_post"], refusal_vec)

        num_layers = h_cos.shape[0]

        for layer in range(num_layers):
            rows.append({
                "id": r["id"],
                "split": r["split"],
                "source": r["source"],
                "category_id": r["category_id"],
                "category_name": r["category_name"],
                "domain": r["domain"],
                "pair_id": r["pair_id"],
                "contextualized": r["contextualized"],
                "variant": r["variant"],
                "condition": r["condition"],
                "refused": r["refused"],
                "layer": layer,
                "harmfulness_cosine_t_inst": float(h_cos[layer].item()),
                "refusal_cosine_t_post": float(r_cos[layer].item()),
                "harmfulness_projection_t_inst": float(h_proj[layer].item()),
                "refusal_projection_t_post": float(r_proj[layer].item()),
            })

    return pd.DataFrame(rows)


# ============================================================
# Summaries and plots
# ============================================================

def add_layer_bands(df: pd.DataFrame, num_layers: int) -> pd.DataFrame:
    early_end = num_layers // 3
    middle_start = num_layers // 3
    middle_end = 2 * num_layers // 3
    late_start = 3 * num_layers // 4

    def band(layer: int) -> str:
        if layer < early_end:
            return "early"
        if middle_start <= layer < middle_end:
            return "middle"
        if layer >= late_start:
            return "late"
        return "other"

    out = df.copy()
    out["layer_band"] = out["layer"].apply(band)
    return out


def summarize_scores(score_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    metric_cols = [
        "harmfulness_cosine_t_inst",
        "refusal_cosine_t_post",
        "harmfulness_projection_t_inst",
        "refusal_projection_t_post",
    ]

    def agg_spec():
        spec = {}
        for c in metric_cols:
            spec[f"{c}_mean"] = (c, "mean")
            spec[f"{c}_std"] = (c, "std")
            spec[f"{c}_count"] = (c, "count")
        return spec

    by_variant_layer = (
        score_df
        .groupby(["variant", "condition", "contextualized", "layer"], as_index=False)
        .agg(**agg_spec())
    )

    by_variant_band = (
        score_df
        .groupby(["variant", "condition", "contextualized", "layer_band"], as_index=False)
        .agg(**agg_spec())
    )

    by_condition_layer = (
        score_df
        .groupby(["condition", "layer"], as_index=False)
        .agg(**agg_spec())
    )

    by_condition_band = (
        score_df
        .groupby(["condition", "layer_band"], as_index=False)
        .agg(**agg_spec())
    )

    by_category_variant_band = (
        score_df
        .groupby(["category_name", "variant", "condition", "contextualized", "layer_band"], as_index=False)
        .agg(**agg_spec())
    )

    by_sample_band = (
        score_df
        .groupby([
            "id", "pair_id", "category_id", "category_name", "domain",
            "variant", "condition", "contextualized", "refused", "layer_band",
        ], as_index=False)
        .agg(
            harmfulness_cosine_t_inst=("harmfulness_cosine_t_inst", "mean"),
            refusal_cosine_t_post=("refusal_cosine_t_post", "mean"),
            harmfulness_projection_t_inst=("harmfulness_projection_t_inst", "mean"),
            refusal_projection_t_post=("refusal_projection_t_post", "mean"),
        )
    )

    return {
        "by_variant_layer": by_variant_layer,
        "by_variant_band": by_variant_band,
        "by_condition_layer": by_condition_layer,
        "by_condition_band": by_condition_band,
        "by_category_variant_band": by_category_variant_band,
        "by_sample_band": by_sample_band,
    }


def make_pair_gap_tables(by_sample_band: pd.DataFrame, cfg: Config) -> Dict[str, pd.DataFrame]:
    """
    Builds:
      1. active_vs_passive_pair_gap:
          C1 vs C2 for *_plain
          C3 vs C4 for *_ctx
      2. context_effect_gap:
          C1 vs C3 for active
          C2 vs C4 for passive
    """
    needed_variants = {"C1_active", "C2_passive", "C3_active_ctx", "C4_passive_ctx"}
    present_variants = set(by_sample_band["variant"].unique()) if "variant" in by_sample_band.columns else set()
    if not needed_variants.issubset(present_variants):
        print("[INFO] Skipping pair gap tables: C1-C4 variants not present (tense-only mode).")
        return {
            "active_vs_passive_pair_gap": pd.DataFrame(),
            "context_effect_gap": pd.DataFrame(),
        }

    target = by_sample_band[
        by_sample_band["condition"].isin([cfg.active_label, cfg.passive_label])
    ].copy()

    rows_pair = []

    # Active vs passive for each pair_id.
    for pair_id, g in target.groupby("pair_id"):
        h_mid = g[g["layer_band"] == "middle"]
        r_late = g[g["layer_band"] == "late"]

        if h_mid.empty or r_late.empty:
            continue

        h_pivot = h_mid.pivot_table(index="pair_id", columns="condition", values="harmfulness_cosine_t_inst", aggfunc="mean")
        r_pivot = r_late.pivot_table(index="pair_id", columns="condition", values="refusal_cosine_t_post", aggfunc="mean")

        hp_pivot = h_mid.pivot_table(index="pair_id", columns="condition", values="harmfulness_projection_t_inst", aggfunc="mean")
        rp_pivot = r_late.pivot_table(index="pair_id", columns="condition", values="refusal_projection_t_post", aggfunc="mean")

        if cfg.active_label not in h_pivot.columns or cfg.passive_label not in h_pivot.columns:
            continue
        if cfg.active_label not in r_pivot.columns or cfg.passive_label not in r_pivot.columns:
            continue

        context_type = "context" if str(pair_id).endswith("_ctx") else "plain"

        rows_pair.append({
            "pair_id": pair_id,
            "context_type": context_type,
            "harmfulness_middle_cos_active": float(h_pivot.loc[pair_id, cfg.active_label]),
            "harmfulness_middle_cos_passive": float(h_pivot.loc[pair_id, cfg.passive_label]),
            "harmfulness_middle_cos_gap_active_minus_passive": float(h_pivot.loc[pair_id, cfg.active_label] - h_pivot.loc[pair_id, cfg.passive_label]),
            "refusal_late_cos_active": float(r_pivot.loc[pair_id, cfg.active_label]),
            "refusal_late_cos_passive": float(r_pivot.loc[pair_id, cfg.passive_label]),
            "refusal_late_cos_gap_active_minus_passive": float(r_pivot.loc[pair_id, cfg.active_label] - r_pivot.loc[pair_id, cfg.passive_label]),
            "harmfulness_middle_proj_active": float(hp_pivot.loc[pair_id, cfg.active_label]),
            "harmfulness_middle_proj_passive": float(hp_pivot.loc[pair_id, cfg.passive_label]),
            "harmfulness_middle_proj_gap_active_minus_passive": float(hp_pivot.loc[pair_id, cfg.active_label] - hp_pivot.loc[pair_id, cfg.passive_label]),
            "refusal_late_proj_active": float(rp_pivot.loc[pair_id, cfg.active_label]),
            "refusal_late_proj_passive": float(rp_pivot.loc[pair_id, cfg.passive_label]),
            "refusal_late_proj_gap_active_minus_passive": float(rp_pivot.loc[pair_id, cfg.active_label] - rp_pivot.loc[pair_id, cfg.passive_label]),
        })

    active_vs_passive = pd.DataFrame(rows_pair)

    # Context effect by base id: C1 vs C3 and C2 vs C4.
    middle = by_sample_band[by_sample_band["layer_band"] == "middle"].copy()
    late = by_sample_band[by_sample_band["layer_band"] == "late"].copy()

    def base_behavior_id(row):
        return str(row["category_id"])

    middle["base_behavior_id"] = middle.apply(base_behavior_id, axis=1)
    late["base_behavior_id"] = late.apply(base_behavior_id, axis=1)

    rows_ctx = []
    for base_id in sorted(set(middle["base_behavior_id"]).intersection(set(late["base_behavior_id"]))):
        gm = middle[middle["base_behavior_id"] == base_id]
        gl = late[late["base_behavior_id"] == base_id]

        needed = ["C1_active", "C2_passive", "C3_active_ctx", "C4_passive_ctx"]
        if not all(v in set(gm["variant"]) for v in needed):
            continue
        if not all(v in set(gl["variant"]) for v in needed):
            continue

        def val(df, variant, col):
            return float(df[df["variant"] == variant][col].mean())

        rows_ctx.append({
            "base_behavior_id": base_id,
            "harmfulness_middle_cos_C1": val(gm, "C1_active", "harmfulness_cosine_t_inst"),
            "harmfulness_middle_cos_C3": val(gm, "C3_active_ctx", "harmfulness_cosine_t_inst"),
            "harmfulness_middle_cos_context_gap_active_C3_minus_C1": val(gm, "C3_active_ctx", "harmfulness_cosine_t_inst") - val(gm, "C1_active", "harmfulness_cosine_t_inst"),
            "harmfulness_middle_cos_C2": val(gm, "C2_passive", "harmfulness_cosine_t_inst"),
            "harmfulness_middle_cos_C4": val(gm, "C4_passive_ctx", "harmfulness_cosine_t_inst"),
            "harmfulness_middle_cos_context_gap_passive_C4_minus_C2": val(gm, "C4_passive_ctx", "harmfulness_cosine_t_inst") - val(gm, "C2_passive", "harmfulness_cosine_t_inst"),
            "refusal_late_cos_C1": val(gl, "C1_active", "refusal_cosine_t_post"),
            "refusal_late_cos_C3": val(gl, "C3_active_ctx", "refusal_cosine_t_post"),
            "refusal_late_cos_context_gap_active_C3_minus_C1": val(gl, "C3_active_ctx", "refusal_cosine_t_post") - val(gl, "C1_active", "refusal_cosine_t_post"),
            "refusal_late_cos_C2": val(gl, "C2_passive", "refusal_cosine_t_post"),
            "refusal_late_cos_C4": val(gl, "C4_passive_ctx", "refusal_cosine_t_post"),
            "refusal_late_cos_context_gap_passive_C4_minus_C2": val(gl, "C4_passive_ctx", "refusal_cosine_t_post") - val(gl, "C2_passive", "refusal_cosine_t_post"),
        })

    context_effect = pd.DataFrame(rows_ctx)

    return {
        "active_vs_passive_pair_gap": active_vs_passive,
        "context_effect_gap": context_effect,
    }


def save_layer_info(num_layers: int, output_dir: str):
    info = {
        "num_layers": num_layers,
        "embedding_layer_excluded": True,
        "layer_indexing": "0-based transformer block output index",
        "early_layers": list(range(0, num_layers // 3)),
        "middle_layers": list(range(num_layers // 3, 2 * num_layers // 3)),
        "late_layers": list(range(3 * num_layers // 4, num_layers)),
    }
    with open(os.path.join(output_dir, "layer_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)


def plot_layer_curves(
    df: pd.DataFrame,
    group_col: str,
    metric_mean_col: str,
    metric_std_col: str,
    metric_count_col: str,
    title: str,
    ylabel: str,
    output_path: str,
):
    plt.figure(figsize=(11, 5))

    for name in sorted(df[group_col].dropna().unique()):
        sub = df[df[group_col] == name].sort_values("layer")
        if sub.empty:
            continue

        x = sub["layer"].to_numpy()
        mean = sub[metric_mean_col].to_numpy()
        std = sub[metric_std_col].fillna(0.0).to_numpy()
        count = sub[metric_count_col].to_numpy()
        stderr = std / np.sqrt(np.maximum(count, 1))

        plt.plot(x, mean, label=str(name))
        plt.fill_between(x, mean - stderr, mean + stderr, alpha=0.18)

    plt.xlabel("Layer")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_outputs(summaries: Dict[str, pd.DataFrame], output_dir: str):
    by_variant_layer = summaries["by_variant_layer"]
    by_condition_layer = summaries["by_condition_layer"]

    plot_layer_curves(
        df=by_variant_layer,
        group_col="variant",
        metric_mean_col="harmfulness_cosine_t_inst_mean",
        metric_std_col="harmfulness_cosine_t_inst_std",
        metric_count_col="harmfulness_cosine_t_inst_count",
        title="C1/C2/C3/C4 cosine to harmfulness steering vector at t_inst",
        ylabel="Cosine similarity",
        output_path=os.path.join(output_dir, "variant_harmfulness_cosine_t_inst_by_layer.png"),
    )

    plot_layer_curves(
        df=by_variant_layer,
        group_col="variant",
        metric_mean_col="refusal_cosine_t_post_mean",
        metric_std_col="refusal_cosine_t_post_std",
        metric_count_col="refusal_cosine_t_post_count",
        title="C1/C2/C3/C4 cosine to refusal steering vector at t_post_inst",
        ylabel="Cosine similarity",
        output_path=os.path.join(output_dir, "variant_refusal_cosine_t_post_by_layer.png"),
    )

    plot_layer_curves(
        df=by_condition_layer,
        group_col="condition",
        metric_mean_col="harmfulness_cosine_t_inst_mean",
        metric_std_col="harmfulness_cosine_t_inst_std",
        metric_count_col="harmfulness_cosine_t_inst_count",
        title="Condition-level cosine to harmfulness steering vector at t_inst",
        ylabel="Cosine similarity",
        output_path=os.path.join(output_dir, "condition_harmfulness_cosine_t_inst_by_layer.png"),
    )

    plot_layer_curves(
        df=by_condition_layer,
        group_col="condition",
        metric_mean_col="refusal_cosine_t_post_mean",
        metric_std_col="refusal_cosine_t_post_std",
        metric_count_col="refusal_cosine_t_post_count",
        title="Condition-level cosine to refusal steering vector at t_post_inst",
        ylabel="Cosine similarity",
        output_path=os.path.join(output_dir, "condition_refusal_cosine_t_post_by_layer.png"),
    )


def write_interpretation(
    summaries: Dict[str, pd.DataFrame],
    gaps: Dict[str, pd.DataFrame],
    cfg: Config,
):
    by_variant_band = summaries["by_variant_band"]
    active_vs_passive = gaps["active_vs_passive_pair_gap"]
    context_effect = gaps["context_effect_gap"]

    lines = []
    lines.append("Steering-vector cosine analysis summary")
    lines.append("")
    lines.append("Primary C-variant outputs:")
    lines.append("  C1_active: plain active harmful request")
    lines.append("  C2_passive: plain passive-framed harmful request")
    lines.append("  C3_active_ctx: active request with research/domain context")
    lines.append("  C4_passive_ctx: passive-framed request with research/domain context")
    lines.append("  C5_tense: past-tense reframed harmful request")
    lines.append("")
    lines.append("Expected hypothesis pattern:")
    lines.append("  Harmfulness at t_inst, middle layers:")
    lines.append("    C1_active and C2_passive should both be aligned with harmfulness vector.")
    lines.append("    C3_active_ctx and C4_passive_ctx may shift depending on context.")
    lines.append("  Refusal at t_post_inst, late layers:")
    lines.append("    C1_active should be more aligned with refusal vector than C2_passive.")
    lines.append("    Compare C3_active_ctx vs C4_passive_ctx separately.")
    lines.append("")
    lines.append("Variant-band summary:")
    lines.append(by_variant_band.to_string(index=False))
    lines.append("")

    if not active_vs_passive.empty:
        lines.append("Active-vs-passive pair gap means:")
        for context_type, g in active_vs_passive.groupby("context_type"):
            lines.append(f"  Context type: {context_type}")
            lines.append(f"    n_pairs: {len(g)}")
            lines.append(f"    mean harmfulness middle cosine gap active-passive: {g['harmfulness_middle_cos_gap_active_minus_passive'].mean():.6f}")
            lines.append(f"    mean refusal late cosine gap active-passive: {g['refusal_late_cos_gap_active_minus_passive'].mean():.6f}")
            lines.append(f"    mean harmfulness middle projection gap active-passive: {g['harmfulness_middle_proj_gap_active_minus_passive'].mean():.6f}")
            lines.append(f"    mean refusal late projection gap active-passive: {g['refusal_late_proj_gap_active_minus_passive'].mean():.6f}")
    else:
        lines.append("No active-vs-passive pair gap table was generated.")

    lines.append("")
    if not context_effect.empty:
        lines.append("Context effect means:")
        lines.append(f"  n_behaviors: {len(context_effect)}")
        lines.append(f"  active refusal context gap C3-C1: {context_effect['refusal_late_cos_context_gap_active_C3_minus_C1'].mean():.6f}")
        lines.append(f"  passive refusal context gap C4-C2: {context_effect['refusal_late_cos_context_gap_passive_C4_minus_C2'].mean():.6f}")
        lines.append(f"  active harmfulness context gap C3-C1: {context_effect['harmfulness_middle_cos_context_gap_active_C3_minus_C1'].mean():.6f}")
        lines.append(f"  passive harmfulness context gap C4-C2: {context_effect['harmfulness_middle_cos_context_gap_passive_C4_minus_C2'].mean():.6f}")
    else:
        lines.append("No context-effect table was generated.")

    out_path = os.path.join(cfg.output_dir, "interpretation_template.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--active_passive_json", type=str, required=True)
    parser.add_argument("--benign_csv", type=str, default=None, help="Local benign prompts CSV with a prompt column. Mutually exclusive with --benign_dataset_name.")
    parser.add_argument("--benign_dataset_name", type=str, default=None, help="Hugging Face benign dataset name, e.g. tatsu-lab/alpaca. Mutually exclusive with --benign_csv.")
    parser.add_argument("--benign_dataset_split", type=str, default="train")
    parser.add_argument("--benign_prompt_col", type=str, default="instruction", help="Column to use as the benign prompt when loading from Hugging Face.")
    parser.add_argument("--benign_input_col", type=str, default="input", help="Optional extra input column appended to the instruction if present. Use empty string to disable.")
    parser.add_argument("--benign_source_label", type=str, default="alpaca", help="Source label written into built_all_data.csv for HF benign rows.")
    parser.add_argument("--output_dir", type=str, default="gemma4_steering_results")

    parser.add_argument("--model_name", type=str, default="google/gemma-4-31B-it")
    parser.add_argument(
        "--hf_cache_dir",
        type=str,
        default=str(Path(__file__).resolve().parent / ".cache" / "huggingface" / "hub"),
        help="Hugging Face hub cache directory. If a valid model snapshot exists here, it is used first.",
    )
    parser.add_argument(
        "--prefer_local_model_cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefer local HF snapshot under --hf_cache_dir before downloading. Use --no-prefer_local_model_cache to disable.",
    )
    parser.add_argument("--sorry_dataset_name", type=str, default=SORRY_DATASET_NAME)
    parser.add_argument("--sorry_csv", type=str, default=None)  # Changed
    parser.add_argument("--sorry_split", type=str, default=None)

    parser.add_argument(
        "--category_ids",
        type=str,
        default=",".join(map(str, DEFAULT_REFUSAL_REQUIRED_CATEGORIES)),
        help="Comma-separated SORRY-Bench category ids.",
    )
    parser.add_argument("--max_sorry_per_category", type=int, default=None)
    parser.add_argument(
        "--sorry_base_only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Filter SORRY-Bench to prompt_style=='base' only (440 base behaviors). Use --no-sorry_base_only to disable.",
    )  # Changed

    parser.add_argument("--mode", type=str, default="both", choices=["plain", "context", "both", "tense", "all"])
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device_map", type=str, default="auto")

    parser.add_argument("--calibration_split", type=str, default="calibration")
    parser.add_argument("--evaluation_split", type=str, default="evaluation")

    parser.add_argument("--benign_label", type=str, default="benign")
    parser.add_argument("--harmful_label", type=str, default="harmful_refused")
    parser.add_argument("--active_label", type=str, default="C_active")
    parser.add_argument("--passive_label", type=str, default="B_passive")

    parser.add_argument("--save_hidden_cache", action="store_true")
    parser.add_argument("--load_hidden_cache", action="store_true")
    parser.add_argument("--save_built_dataset", action="store_true", default=True)
    parser.add_argument(
        "--max_benign_samples",
        type=int,
        default=None,
        help="Maximum number of benign/Alpaca calibration prompts to sample. Use 170 for 17 categories x 10 SORRY samples.",
    )
    parser.add_argument(
        "--benign_sample_seed",
        type=int,
        default=42,
        help="Random seed for reproducible benign/Alpaca sampling.",
    )

    parser.add_argument(
        "--model_loader",
        type=str,
        default="auto",
        choices=["auto", "causal_lm", "image_text_to_text"],
        help=(
            "Model loader. Use image_text_to_text for google/gemma-4-31B-it. "
            "auto selects image_text_to_text when model_name contains gemma-4."
        ),
    )
    parser.add_argument(
        "--processor_loader",
        type=str,
        default="auto",
        choices=["auto", "tokenizer", "processor"],
        help=(
            "Tokenizer/processor loader. Gemma 4 should use AutoProcessor, then processor.tokenizer "
            "for offset_mapping-based hidden-state position detection."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    cfg = Config(
        model_name=args.model_name,
        active_passive_json=args.active_passive_json,
        benign_csv=args.benign_csv,
        output_dir=args.output_dir,
        benign_dataset_name=args.benign_dataset_name,
        benign_dataset_split=args.benign_dataset_split,
        benign_prompt_col=args.benign_prompt_col,
        benign_input_col=args.benign_input_col if args.benign_input_col else None,
        benign_source_label=args.benign_source_label,
        hf_cache_dir=args.hf_cache_dir,  # Changed
        prefer_local_model_cache=args.prefer_local_model_cache,  # Changed
        sorry_dataset_name=args.sorry_dataset_name,
        sorry_csv=args.sorry_csv,  # Changed
        sorry_split=args.sorry_split,
        category_ids=[int(x.strip()) for x in args.category_ids.split(",") if x.strip()],
        max_sorry_per_category=args.max_sorry_per_category,
        sorry_base_only=args.sorry_base_only,  # Changed
        mode=args.mode,
        max_length=args.max_length,
        dtype=args.dtype,
        device_map=args.device_map,
        calibration_split=args.calibration_split,
        evaluation_split=args.evaluation_split,
        benign_label=args.benign_label,
        harmful_label=args.harmful_label,
        active_label=args.active_label,
        passive_label=args.passive_label,
        save_hidden_cache=args.save_hidden_cache,
        load_hidden_cache=args.load_hidden_cache,
        save_built_dataset=args.save_built_dataset,
        max_benign_samples=args.max_benign_samples,
        benign_sample_seed=args.benign_sample_seed,
        model_loader=args.model_loader,
        processor_loader=args.processor_loader,
    )

    ensure_dir(cfg.output_dir)

    with open(os.path.join(cfg.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(cfg), f, indent=2, ensure_ascii=False)

    print("[INFO] Building full dataset...")
    full_df = build_full_dataset(cfg)

    if cfg.save_built_dataset:
        built_path = os.path.join(cfg.output_dir, "built_all_data.csv")
        full_df.to_csv(built_path, index=False)
        print("[INFO] Saved built dataset:", built_path)

    print("[INFO] Split counts:")
    print(full_df["split"].value_counts())
    print("[INFO] Condition counts:")
    print(full_df["condition"].value_counts())
    print("[INFO] Variant counts:")
    print(full_df["variant"].value_counts())
    print("[INFO] Calibration refused counts:")
    print(pd.crosstab([full_df["split"], full_df["condition"]], full_df["refused"]))

    resolved_model_name_or_path = resolve_model_name_or_path(
        model_name=cfg.model_name,
        hf_cache_dir=cfg.hf_cache_dir,
        prefer_local_cache=cfg.prefer_local_model_cache,
    )  # Changed

    model_name_lower = cfg.model_name.lower()

    processor_loader = cfg.processor_loader
    if processor_loader == "auto":
        processor_loader = "processor" if "gemma-4" in model_name_lower else "tokenizer"

    if processor_loader == "processor":
        print("[INFO] Loading processor:", resolved_model_name_or_path)
        processor = AutoProcessor.from_pretrained(
            resolved_model_name_or_path,
            trust_remote_code=True,
            cache_dir=cfg.hf_cache_dir,
        )
        if not hasattr(processor, "tokenizer"):
            raise ValueError("Loaded processor does not expose .tokenizer; cannot compute offset mappings.")
        tokenizer = processor.tokenizer
        if not getattr(tokenizer, "is_fast", False):
            raise ValueError("A fast tokenizer is required for return_offsets_mapping=True.")
    else:
        print("[INFO] Loading tokenizer:", resolved_model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(
            resolved_model_name_or_path,
            use_fast=True,
            trust_remote_code=True,
            cache_dir=cfg.hf_cache_dir,
        )

    model_loader = cfg.model_loader
    if model_loader == "auto":
        model_loader = "image_text_to_text" if "gemma-4" in model_name_lower else "causal_lm"

    print(f"[INFO] Loading model with {model_loader}:", resolved_model_name_or_path)
    if model_loader == "image_text_to_text":
        model = AutoModelForImageTextToText.from_pretrained(
            resolved_model_name_or_path,
            torch_dtype=get_torch_dtype(cfg.dtype),
            device_map=cfg.device_map,
            trust_remote_code=True,
            output_hidden_states=True,
            cache_dir=cfg.hf_cache_dir,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            resolved_model_name_or_path,
            torch_dtype=get_torch_dtype(cfg.dtype),
            device_map=cfg.device_map,
            trust_remote_code=True,
            output_hidden_states=True,
            cache_dir=cfg.hf_cache_dir,
        )
    model.eval()

    records = extract_or_load_records(
        df=full_df,
        model=model,
        tokenizer=tokenizer,
        cfg=cfg,
    )

    print("[INFO] Building steering vectors from calibration split...")
    steering = build_steering_vectors(records, cfg)

    torch.save(steering["harmfulness_t_inst"], os.path.join(cfg.output_dir, "steering_harmfulness_t_inst.pt"))
    torch.save(steering["refusal_t_post"], os.path.join(cfg.output_dir, "steering_refusal_t_post.pt"))

    num_layers = steering["harmfulness_t_inst"].shape[0]
    save_layer_info(num_layers, cfg.output_dir)

    print("[INFO] Computing evaluation scores...")
    score_df = compute_eval_scores(records, steering, cfg)
    score_df = add_layer_bands(score_df, num_layers)

    score_path = os.path.join(cfg.output_dir, "scores_by_sample_layer.csv")
    score_df.to_csv(score_path, index=False)

    print("[INFO] Summarizing...")
    summaries = summarize_scores(score_df)

    output_paths = {
        "by_variant_layer": os.path.join(cfg.output_dir, "scores_by_variant_layer.csv"),
        "by_variant_band": os.path.join(cfg.output_dir, "scores_by_variant_band.csv"),
        "by_condition_layer": os.path.join(cfg.output_dir, "scores_by_condition_layer.csv"),
        "by_condition_band": os.path.join(cfg.output_dir, "scores_by_condition_band.csv"),
        "by_category_variant_band": os.path.join(cfg.output_dir, "scores_by_category_variant_band.csv"),
        "by_sample_band": os.path.join(cfg.output_dir, "scores_by_sample_band.csv"),
    }

    for key, path in output_paths.items():
        summaries[key].to_csv(path, index=False)

    gaps = make_pair_gap_tables(summaries["by_sample_band"], cfg)
    gap_paths = {
        "active_vs_passive_pair_gap": os.path.join(cfg.output_dir, "active_vs_passive_pair_gap.csv"),
        "context_effect_gap": os.path.join(cfg.output_dir, "context_effect_gap.csv"),
    }

    for key, path in gap_paths.items():
        gaps[key].to_csv(path, index=False)

    print("[INFO] Plotting...")
    plot_outputs(summaries, cfg.output_dir)

    write_interpretation(summaries, gaps, cfg)

    print("[INFO] Saved main outputs:")
    print(" ", score_path)
    for path in output_paths.values():
        print(" ", path)
    for path in gap_paths.values():
        print(" ", path)
    print(" ", os.path.join(cfg.output_dir, "variant_harmfulness_cosine_t_inst_by_layer.png"))
    print(" ", os.path.join(cfg.output_dir, "variant_refusal_cosine_t_post_by_layer.png"))
    print(" ", os.path.join(cfg.output_dir, "condition_harmfulness_cosine_t_inst_by_layer.png"))
    print(" ", os.path.join(cfg.output_dir, "condition_refusal_cosine_t_post_by_layer.png"))
    print(" ", os.path.join(cfg.output_dir, "interpretation_template.txt"))
    print("[INFO] Done.")


if __name__ == "__main__":
    main()

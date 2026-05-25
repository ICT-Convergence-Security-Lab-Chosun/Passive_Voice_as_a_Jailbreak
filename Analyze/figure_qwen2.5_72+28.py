#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen2.5-72B-Instruct all-layer representation visualization script.

What this does
- Loads C1_active / C2_passive / C3_active_ctx / C4_passive_ctx from a JSON file.
- Loads benign prompts from Alpaca by default, or from a local CSV if provided.
- Loads Sorry-Bench harmful reference prompts from a local CSV.
- Extracts hidden states for every transformer block layer from Qwen2.5-72B-Instruct.
- Saves per-layer 2D coordinates and plots.

Layer indexing
- output layer index 0 means transformer block 0 output, i.e. outputs.hidden_states[1].
- outputs.hidden_states[0] is the embedding output and is excluded unless --include_embedding is passed.

Recommended quick run
python qwen25_72b_all_layers_representation_viz.py \
  --data_path harmful_prompts.json \
  --sorry_bench_path sorrybench_202503_harmful_160.csv \
  --output_dir outputs_qwen25_72b_all_layers \
  --model_name Qwen/Qwen2.5-72B-Instruct \
  --token_pos t_post_inst \
  --methods pca \
  --dtype bfloat16 \
  --device_map auto

Full, expensive visualization run
python qwen25_72b_all_layers_representation_viz.py \
  --data_path harmful_prompts.json \
  --sorry_bench_path sorrybench_202503_harmful_160.csv \
  --output_dir outputs_qwen25_72b_all_layers_full \
  --model_name Qwen/Qwen2.5-72B-Instruct \
  --token_pos t_post_inst \
  --methods pca,umap,tsne \
  --save_raw_reps \
  --dtype bfloat16 \
  --device_map auto
"""

import argparse
import gc
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    import umap  # pip install umap-learn
except Exception:
    umap = None


COND_STYLE = {
    "C1_active":      {"color": "#4C72B0", "marker": "o", "label": "C1 active (JBB)"},
    "C2_passive":     {"color": "#DD8452", "marker": "s", "label": "C2 passive (JBB)"},
    "C3_active_ctx":  {"color": "#55A868", "marker": "^", "label": "C3 active+ctx (JBB)"},
    "C4_passive_ctx": {"color": "#C44E52", "marker": "D", "label": "C4 passive+ctx (JBB)"},
    "C5_tense":       {"color": "#E377C2", "marker": "v", "label": "C5 tense (JBB)"},
    "benign":         {"color": "#8172B2", "marker": "P", "label": "Benign"},
    "sorry_bench":    {"color": "#937860", "marker": "X", "label": "Sorry-Bench harmful ref"},
}


def log(msg: str = "") -> None:
    """Timestamped, flushed logging so long GPU jobs do not look frozen."""
    if msg:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)
    else:
        print("", flush=True)



# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_torch_dtype(dtype_name: str):
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def get_input_device(model):
    # Safer than model.device when device_map='auto'.
    return model.get_input_embeddings().weight.device


def safe_text(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    return str(x)


def parse_methods(methods: str) -> List[str]:
    out = [m.strip().lower() for m in methods.split(",") if m.strip()]
    valid = {"pca", "umap", "tsne"}
    bad = sorted(set(out) - valid)
    if bad:
        raise ValueError(f"Unsupported methods: {bad}. Choose from pca,umap,tsne")
    if "umap" in out and umap is None:
        raise ImportError("UMAP requested, but umap-learn is not installed. Run: pip install umap-learn")
    return out


def parse_int_list(text: Optional[str]) -> Optional[List[int]]:
    if text is None or not str(text).strip():
        return None
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def sample_texts(texts: List[str], n: Optional[int], seed: int, label: str) -> List[str]:
    texts = [
        safe_text(x).strip()
        for x in texts
        if safe_text(x).strip() and safe_text(x).strip() != "[None]"
    ]
    if n is None:
        return texts
    if n <= 0:
        return []
    if len(texts) <= n:
        log(f"[WARN] {label}: requested n={n}, but only {len(texts)} usable prompts are available. Keeping all.")
        return texts
    rng = random.Random(seed)
    idx = rng.sample(range(len(texts)), n)
    return [texts[i] for i in idx]


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def load_json_prompts(
    data_path: str,
    max_jbb_per_variant: Optional[int],
    seed: int,
    skip_conditions: Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    with open(data_path, encoding="utf-8") as f:
        jbb = json.load(f)

    skip = set(skip_conditions) if skip_conditions else set()

    prompts = {
        "C1_active":      [safe_text(x.get("C1_active", "")) for x in jbb],
        "C2_passive":     [safe_text(x.get("C2_passive", "")) for x in jbb],
        "C3_active_ctx":  [safe_text(x.get("C3_active_ctx", "")) for x in jbb],
        "C4_passive_ctx": [safe_text(x.get("C4_passive_ctx", "")) for x in jbb],
        "C5_tense":       [safe_text(x.get("C5_tense", "")) for x in jbb if x.get("C5_tense")],
    }

    for key in list(skip):
        if key in prompts:
            log(f"[INFO] Skipping condition: {key}")
            del prompts[key]

    if max_jbb_per_variant is not None:
        for key in list(prompts):
            prompts[key] = sample_texts(prompts[key], max_jbb_per_variant, seed, key)
    else:
        for key in list(prompts):
            prompts[key] = sample_texts(prompts[key], None, seed, key)

    return prompts


def load_benign_prompts(benign_csv: Optional[str], n_benign: int, seed: int) -> List[str]:
    if benign_csv:
        df = pd.read_csv(benign_csv)
        if "prompt" not in df.columns:
            raise ValueError("--benign_csv must contain a 'prompt' column.")
        prompts = df["prompt"].dropna().astype(str).tolist()
        return sample_texts(prompts, n_benign, seed, "benign_csv")

    ds = load_dataset("tatsu-lab/alpaca", split="train")
    rng = random.Random(seed)
    n = min(n_benign, len(ds))
    indices = rng.sample(range(len(ds)), n)
    return [safe_text(ds[i]["instruction"]) for i in indices]


def _extract_sorry_prompt(val) -> str:
    """Extract prompt text from the SORRY-Bench 'turns' field (mirrors logic in active_passive_cosine_auto.py)."""
    import ast
    if val is None:
        return ""
    if hasattr(val, "tolist") and not isinstance(val, str):
        try:
            val = val.tolist()
        except Exception:
            pass
    if isinstance(val, str):
        text = val.strip()
        if not text:
            return ""
        if text.startswith("[") or text.startswith("{"):
            try:
                parsed = ast.literal_eval(text)
                result = _extract_sorry_prompt(parsed)
                if result.strip():
                    return result.strip()
            except Exception:
                pass
        return text
    if isinstance(val, (list, tuple)) and len(val) > 0:
        first = val[0]
        if isinstance(first, str):
            return first.strip()
        if isinstance(first, dict):
            for key in ("content", "text", "value", "prompt", "question", "instruction"):
                candidate = first.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    return candidate.strip()
    if isinstance(val, dict):
        for key in ("content", "text", "value", "prompt", "question", "instruction"):
            candidate = val.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
    return ""


def _load_sorry_from_hf(
    dataset_name: str,
    sorry_split: Optional[str],
    category_ids: Optional[List[int]],
    sorry_per_category: Optional[int],
    max_sorry: Optional[int],
    seed: int,
) -> List[str]:
    """Load SORRY-Bench from HuggingFace and return filtered prompt texts (mirrors active_passive_cosine_auto.py)."""
    log(f"[INFO] Loading Sorry-Bench from HuggingFace: {dataset_name} split={sorry_split}")
    if sorry_split is not None:
        ds = load_dataset(dataset_name, split=sorry_split)
        sb = ds.to_pandas()
    else:
        dsdict = load_dataset(dataset_name)
        first_split = list(dsdict.keys())[0]
        log(f"[INFO] Using Sorry-Bench split: {first_split}")
        sb = dsdict[first_split].to_pandas()

    prompt_col = None
    for candidate in ["turns", "prompt", "question", "instruction"]:
        if candidate in sb.columns:
            prompt_col = candidate
            break
    if prompt_col is None:
        raise ValueError(f"SORRY-Bench HF dataset has no usable prompt column. Columns: {list(sb.columns)}")

    sb = sb.copy()
    sb["prompt"] = sb[prompt_col].apply(_extract_sorry_prompt).str.strip()
    sb = sb[sb["prompt"].ne("") & sb["prompt"].ne("[None]")].copy()

    if category_ids is not None:
        category_col = None
        for candidate in ["category_id", "category_idx", "category_num", "taxonomy_id", "category"]:
            if candidate in sb.columns:
                category_col = candidate
                break
        if category_col is None:
            raise ValueError(f"--sorry_category_ids specified but no category column found. Available columns: {list(sb.columns)}")

        def _parse_cat(x):
            if pd.isna(x):
                return None
            if isinstance(x, (int, float)):
                return int(x)
            import re
            m = re.match(r"^\s*(\d+)", str(x).strip())
            return int(m.group(1)) if m else None

        sb["_category_id_int"] = sb[category_col].apply(_parse_cat)
        sb = sb[sb["_category_id_int"].isin(category_ids)].copy()

        if sb.empty:
            raise ValueError(f"No Sorry-Bench rows matched --sorry_category_ids={category_ids}")

        log("[INFO] Sorry-Bench HF usable counts by selected category before sampling:")
        log(sb["_category_id_int"].value_counts().sort_index().to_string())

        if sorry_per_category is not None:
            sampled_parts = []
            rng = np.random.default_rng(seed)
            for cid, g in sb.groupby("_category_id_int", sort=True):
                if len(g) < sorry_per_category:
                    log(f"[WARN] category {cid}: requested {sorry_per_category}, available {len(g)}. Keeping all.")
                    sampled_parts.append(g)
                else:
                    sampled_parts.append(g.sample(n=sorry_per_category, random_state=int(rng.integers(0, 2**31 - 1))))
            sb = pd.concat(sampled_parts, ignore_index=True)

    prompts = sb["prompt"].tolist()
    if max_sorry is not None:
        prompts = sample_texts(prompts, max_sorry, seed, "sorry_bench_hf")
    log(f"[INFO] Sorry-Bench HF prompts loaded: {len(prompts)}")
    return prompts


def load_sorry_prompts(
    sorry_bench_path: Optional[str],
    max_sorry: Optional[int],
    category_ids: Optional[List[int]],
    sorry_per_category: Optional[int],
    seed: int,
    sorry_dataset_name: str = "sorry-bench/sorry-bench-202503",
    sorry_split: Optional[str] = None,
) -> List[str]:
    if not sorry_bench_path:
        return _load_sorry_from_hf(
            dataset_name=sorry_dataset_name,
            sorry_split=sorry_split,
            category_ids=category_ids,
            sorry_per_category=sorry_per_category,
            max_sorry=max_sorry,
            seed=seed,
        )

    sb = pd.read_csv(sorry_bench_path)
    if "prompt" not in sb.columns:
        raise ValueError("--sorry_bench_path CSV must contain a 'prompt' column.")

    category_col = None
    for candidate in ["category_id", "category", "category_num", "taxonomy_id"]:
        if candidate in sb.columns:
            category_col = candidate
            break

    sb = sb.copy()
    sb["prompt"] = sb["prompt"].apply(lambda x: safe_text(x).strip())
    sb = sb[sb["prompt"].ne("") & sb["prompt"].ne("[None]")].copy()

    if category_ids is not None:
        if category_col is None:
            raise ValueError(
                "--sorry_category_ids was provided, but the Sorry-Bench CSV has no category column. "
                "Expected one of: category_id, category, category_num, taxonomy_id."
            )
        sb["_category_id_int"] = pd.to_numeric(sb[category_col], errors="coerce").astype("Int64")
        sb = sb[sb["_category_id_int"].isin(category_ids)].copy()

        if sb.empty:
            raise ValueError(f"No Sorry-Bench rows matched --sorry_category_ids={category_ids}")

        log("[INFO] Sorry-Bench usable counts by selected category before sampling:")
        log(sb["_category_id_int"].value_counts().sort_index().to_string())

        if sorry_per_category is not None:
            sampled_parts = []
            rng = np.random.default_rng(seed)
            for cid, g in sb.groupby("_category_id_int", sort=True):
                if len(g) < sorry_per_category:
                    log(f"[WARN] category {cid}: requested {sorry_per_category}, available {len(g)}. Keeping all available.")
                    sampled_parts.append(g)
                else:
                    sampled_parts.append(g.sample(n=sorry_per_category, random_state=int(rng.integers(0, 2**31 - 1))))
            sb = pd.concat(sampled_parts, ignore_index=True)

        prompts = sb["prompt"].tolist()
        if max_sorry is not None:
            prompts = sample_texts(prompts, max_sorry, seed, "sorry_bench")
        return prompts

    prompts = sb["prompt"].tolist()
    return sample_texts(prompts, max_sorry, seed, "sorry_bench")


def load_all_prompts(args) -> Dict[str, List[str]]:
    skip = [s.strip() for s in args.skip_conditions.split(",")] if args.skip_conditions else None
    prompts = load_json_prompts(args.data_path, args.max_jbb_per_variant, args.random_seed, skip_conditions=skip)
    prompts["benign"] = load_benign_prompts(args.benign_csv, args.n_benign, args.random_seed)
    prompts["sorry_bench"] = load_sorry_prompts(
        args.sorry_bench_path,
        args.max_sorry,
        parse_int_list(args.sorry_category_ids),
        args.sorry_per_category,
        args.random_seed,
        sorry_dataset_name=args.sorry_dataset_name,
        sorry_split=args.sorry_split,
    )

    log("[INFO] Prompt counts")
    for cond, texts in prompts.items():
        log(f"  {cond}: {len(texts)}")
    log(f"  total: {sum(len(v) for v in prompts.values())}")

    return prompts


# -----------------------------------------------------------------------------
# Chat rendering and token positions
# -----------------------------------------------------------------------------

def render_chat(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        # Fallback to Qwen ChatML-ish format.
        return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"


def find_token_positions(tokenizer, prompt: str, max_length: int) -> Tuple[torch.Tensor, int, int, str]:
    """
    Returns input_ids, t_inst, t_post_inst, rendered.

    t_inst: final token overlapping the original user instruction text.
    t_post_inst: final token of the full rendered prompt before generation.
    """
    rendered = render_chat(tokenizer, prompt)

    prompt_start = rendered.rfind(prompt)
    if prompt_start == -1:
        # This is rare, but can happen if a tokenizer template transforms content.
        # In that case, use the last token as both positions instead of failing.
        enc = tokenizer(
            rendered,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )
        input_ids = enc["input_ids"]
        last = input_ids.shape[1] - 1
        return input_ids, last, last, rendered

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
        t_inst = input_ids.shape[1] - 1

    t_post_inst = input_ids.shape[1] - 1
    return input_ids, int(t_inst), int(t_post_inst), rendered


def choose_target_idx(token_pos: str, t_inst: int, t_post_inst: int) -> int:
    if token_pos == "t_inst":
        return t_inst
    if token_pos == "t_post_inst":
        return t_post_inst
    raise ValueError("--token_pos must be 't_inst' or 't_post_inst'")


# -----------------------------------------------------------------------------
# Hidden extraction
# -----------------------------------------------------------------------------

@torch.no_grad()
def extract_one_prompt_all_layers(model, tokenizer, text: str, args) -> np.ndarray:
    input_ids, t_inst, t_post_inst, _ = find_token_positions(tokenizer, text, args.max_length)
    target_idx = choose_target_idx(args.token_pos, t_inst, t_post_inst)

    input_device = get_input_device(model)
    input_ids = input_ids.to(input_device)
    attention_mask = torch.ones_like(input_ids, device=input_device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )

    hidden_states = outputs.hidden_states
    if args.include_embedding:
        selected = hidden_states
    else:
        selected = hidden_states[1:]

    arr = torch.stack([
        h[0, target_idx, :].detach().float().cpu()
        for h in selected
    ]).numpy()

    del outputs, hidden_states, input_ids, attention_mask
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return arr


def extract_representations(model, tokenizer, all_prompts: Dict[str, List[str]], args) -> Tuple[Dict[int, Dict[str, np.ndarray]], pd.DataFrame]:
    """
    Returns:
      reps[layer][condition] = np.array of shape (N, hidden_dim)
      metadata dataframe with rows aligned with each condition's arrays.
    """
    layer_count = int(model.config.num_hidden_layers)
    n_layers = layer_count + 1 if args.include_embedding else layer_count
    layer_ids = list(range(n_layers))

    log(f"[INFO] model.config.num_hidden_layers={layer_count}")
    log(f"[INFO] extracting {'embedding + ' if args.include_embedding else ''}{layer_count} transformer block outputs")

    reps = {layer: {cond: [] for cond in all_prompts} for layer in layer_ids}
    meta_rows = []

    for cond, texts in all_prompts.items():
        log(f"[INFO] Extracting condition={cond}, n={len(texts)}")
        for i, text in enumerate(texts):
            if i == 0:
                log(f"[INFO] First forward for condition={cond} starting")
            arr = extract_one_prompt_all_layers(model, tokenizer, text, args)
            if arr.shape[0] != n_layers:
                raise RuntimeError(f"Expected {n_layers} layers, got {arr.shape[0]}")

            for layer in layer_ids:
                reps[layer][cond].append(arr[layer])

            meta_rows.append({
                "condition": cond,
                "condition_index": i,
                "prompt": text,
                "token_pos": args.token_pos,
            })

            if (i + 1) % args.progress_every == 0:
                log(f"  {cond}: {i + 1}/{len(texts)}")

    for layer in layer_ids:
        for cond in all_prompts:
            reps[layer][cond] = np.asarray(reps[layer][cond], dtype=np.float32)

    return reps, pd.DataFrame(meta_rows)


# -----------------------------------------------------------------------------
# Saving and visualization
# -----------------------------------------------------------------------------

def stack_and_labels(reps_at_layer: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[str]]:
    X_list, labels = [], []
    for cond, arr in reps_at_layer.items():
        X_list.append(arr)
        labels.extend([cond] * len(arr))
    return np.vstack(X_list), labels


def scatter_2d(ax, Z: np.ndarray, labels: List[str], title: str) -> None:
    label_arr = np.asarray(labels)
    for cond, style in COND_STYLE.items():
        mask = label_arr == cond
        if not mask.any():
            continue
        z = Z[mask]
        ax.scatter(
            z[:, 0], z[:, 1],
            c=style["color"], marker=style["marker"],
            label=style["label"], alpha=0.65, s=45,
            edgecolors="white", linewidths=0.4,
        )
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.legend(fontsize=6, markerscale=1.1)


def save_coordinates(output_dir: Path, layer: int, method_name: str, Z: np.ndarray, labels: List[str]) -> None:
    df = pd.DataFrame({
        "layer": layer,
        "method": method_name,
        "condition": labels,
        "x": Z[:, 0],
        "y": Z[:, 1],
    })
    out = output_dir / "coordinates" / f"layer{layer:02d}_{method_name}.csv"
    out.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(out, index=False)


def plot_layer(reps_at_layer: Dict[str, np.ndarray], layer_idx: int, methods: List[str], args) -> None:
    X, labels = stack_and_labels(reps_at_layer)
    n_panels = len(methods)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    panel = 0

    if "pca" in methods:
        pca = PCA(n_components=2, random_state=args.random_seed)
        Z = pca.fit_transform(X)
        ev = pca.explained_variance_ratio_
        scatter_2d(axes[panel], Z, labels, f"PCA  EV {ev[0]:.1%} / {ev[1]:.1%}")
        save_coordinates(Path(args.output_dir), layer_idx, "pca", Z, labels)
        panel += 1

    if "umap" in methods:
        reducer = umap.UMAP(
            n_neighbors=args.umap_neighbors,
            min_dist=args.umap_min_dist,
            random_state=args.random_seed,
        )
        Z = reducer.fit_transform(X)
        scatter_2d(axes[panel], Z, labels, f"UMAP n={args.umap_neighbors}, min_dist={args.umap_min_dist}")
        save_coordinates(Path(args.output_dir), layer_idx, "umap", Z, labels)
        panel += 1

    if "tsne" in methods:
        log(f"[INFO] layer {layer_idx}: t-SNE start")
        X_for_tsne = X
        pca_pre_dim = int(args.tsne_pca_dim)
        if pca_pre_dim > 0:
            # Standard t-SNE practice: reduce very high-dimensional hidden states first.
            # n_components must be <= min(n_samples, n_features).
            safe_dim = min(pca_pre_dim, X.shape[0] - 1, X.shape[1])
            if safe_dim < 2:
                raise ValueError(f"Cannot run PCA pre-reduction for t-SNE: safe_dim={safe_dim}, X.shape={X.shape}")
            log(f"[INFO] layer {layer_idx}: PCA pre-reduction for t-SNE start ({X.shape[1]} -> {safe_dim})")
            X_for_tsne = PCA(n_components=safe_dim, random_state=args.random_seed).fit_transform(X)
            log(f"[INFO] layer {layer_idx}: PCA pre-reduction for t-SNE done")

        tsne_kwargs = dict(n_components=2, perplexity=args.tsne_perplexity, random_state=args.random_seed)
        # sklearn renamed n_iter to max_iter in newer versions; support both.
        try:
            Z = TSNE(**tsne_kwargs, max_iter=args.tsne_iter).fit_transform(X_for_tsne)
        except TypeError:
            Z = TSNE(**tsne_kwargs, n_iter=args.tsne_iter).fit_transform(X_for_tsne)
        log(f"[INFO] layer {layer_idx}: t-SNE done")
        scatter_2d(axes[panel], Z, labels, f"t-SNE perplexity={args.tsne_perplexity}, PCA{pca_pre_dim if pca_pre_dim > 0 else 'none'}")
        save_coordinates(Path(args.output_dir), layer_idx, "tsne", Z, labels)
        panel += 1

    fig.suptitle(f"Qwen2.5-72B-Instruct | layer {layer_idx} | token={args.token_pos} | n={len(labels)}", fontsize=11, fontweight="bold")
    plt.tight_layout()

    out = Path(args.output_dir) / "plots_by_layer" / f"layer{layer_idx:02d}_{args.token_pos}_{'_'.join(methods)}.png"
    out.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close()
    log(f"[SAVE] {out}")


def plot_pca_grid(all_reps: Dict[int, Dict[str, np.ndarray]], args) -> None:
    layers = sorted(all_reps.keys())
    ncols = args.grid_cols
    nrows = (len(layers) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.6 * nrows))
    axes = np.asarray(axes).flatten()

    for i, layer in enumerate(layers):
        X, labels = stack_and_labels(all_reps[layer])
        Z = PCA(n_components=2, random_state=args.random_seed).fit_transform(X)
        scatter_2d(axes[i], Z, labels, f"PCA - Layer {layer}")
        if axes[i].get_legend():
            axes[i].get_legend().remove()

    for j in range(len(layers), len(axes)):
        axes[j].set_visible(False)

    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower center", ncol=6, fontsize=9, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle(f"PCA across all layers | token={args.token_pos}", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=(0, 0.02, 1, 0.98))

    out = Path(args.output_dir) / f"pca_grid_all_layers_{args.token_pos}.png"
    plt.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close()
    log(f"[SAVE] {out}")


def save_raw_reps(all_reps: Dict[int, Dict[str, np.ndarray]], output_dir: str) -> None:
    raw_dir = Path(output_dir) / "raw_reps_by_layer"
    raw_dir.mkdir(exist_ok=True, parents=True)
    for layer, reps_at_layer in all_reps.items():
        out = raw_dir / f"layer{layer:02d}.npz"
        np.savez_compressed(out, **reps_at_layer)
        log(f"[SAVE] {out}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-72B-Instruct")
    parser.add_argument("--data_path", type=str, default="harmful_prompts.json")
    parser.add_argument("--sorry_bench_path", type=str, default=None,
                        help="Local SORRY-Bench CSV path. If omitted, loads from HuggingFace via --sorry_dataset_name.")
    parser.add_argument("--sorry_dataset_name", type=str, default="sorry-bench/sorry-bench-202503",
                        help="HuggingFace SORRY-Bench dataset ID (used when sorry_bench_path is not given).")
    parser.add_argument("--sorry_split", type=str, default=None,
                        help="HF SORRY-Bench split name. Defaults to the first available split.")
    parser.add_argument("--benign_csv", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs_qwen25_72b_all_layers")

    parser.add_argument("--hf_cache_dir", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--token_pos", type=str, default="t_post_inst", choices=["t_inst", "t_post_inst"])
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--include_embedding", action="store_true")

    parser.add_argument("--n_benign", type=int, default=100)
    parser.add_argument("--max_sorry", type=int, default=None, help="Randomly sample at most this many Sorry-Bench prompts after optional category sampling.")
    parser.add_argument("--sorry_category_ids", type=str, default=None, help="Comma-separated Sorry-Bench categories to include, e.g. 6,7,8,9,10,12,13,14,15,17,18,19,20,21,23,24,28")
    parser.add_argument("--sorry_per_category", type=int, default=None, help="Randomly sample this many Sorry-Bench prompts per selected category. Use 10 for 17 categories x 10 = 170.")
    parser.add_argument("--max_jbb_per_variant", type=int, default=None, help="Randomly sample at most this many JBB prompts per C1/C2/C3/C4 variant.")
    parser.add_argument("--skip_conditions", type=str, default=None, help="Comma-separated condition names to exclude, e.g. C5_tense or C5_tense,C6_tense_ctx")
    parser.add_argument("--random_seed", type=int, default=42)

    parser.add_argument("--methods", type=str, default="pca", help="Comma-separated: pca,umap,tsne. For all layers, pca is strongly recommended first.")
    parser.add_argument("--umap_neighbors", type=int, default=15)
    parser.add_argument("--umap_min_dist", type=float, default=0.1)
    parser.add_argument("--tsne_perplexity", type=float, default=30)
    parser.add_argument("--tsne_iter", type=int, default=1000)
    parser.add_argument("--tsne_pca_dim", type=int, default=50, help="Pre-reduce hidden representations to this many PCA dimensions before t-SNE. Set <=0 to disable.")

    parser.add_argument("--save_raw_reps", action="store_true")
    parser.add_argument("--skip_layer_plots", action="store_true")
    parser.add_argument("--skip_pca_grid", action="store_true")
    parser.add_argument("--grid_cols", type=int, default=5)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--progress_every", type=int, default=1)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = parse_methods(args.methods)
    set_seed(args.random_seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    log("=== Data load ===")
    all_prompts = load_all_prompts(args)

    log("\n=== Model load ===")
    tokenizer_kwargs = {"trust_remote_code": args.trust_remote_code}
    model_kwargs = {
        "torch_dtype": get_torch_dtype(args.dtype),
        "device_map": args.device_map,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.hf_cache_dir:
        tokenizer_kwargs["cache_dir"] = args.hf_cache_dir
        model_kwargs["cache_dir"] = args.hf_cache_dir

    log("[INFO] tokenizer loading start")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, **tokenizer_kwargs)
    log("[INFO] tokenizer loaded")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log("[INFO] model from_pretrained start")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    log(f"[INFO] model from_pretrained finished in {time.time() - t0:.1f}s")

    log("[INFO] model.eval start")
    model.eval()
    log("[INFO] model.eval finished")

    log("\n=== Hidden extraction for all layers ===")
    all_reps, meta_df = extract_representations(model, tokenizer, all_prompts, args)
    meta_path = output_dir / "metadata_prompts.csv"
    meta_df.to_csv(meta_path, index=False)
    log(f"[SAVE] {meta_path}")

    layer_info = {
        "model_name": args.model_name,
        "num_hidden_layers": int(model.config.num_hidden_layers),
        "include_embedding": bool(args.include_embedding),
        "layer_indexing": "0-based transformer block output index; outputs.hidden_states[layer+1] when include_embedding=False",
        "token_pos": args.token_pos,
        "conditions": {k: len(v) for k, v in all_prompts.items()},
    }
    with open(output_dir / "layer_info.json", "w", encoding="utf-8") as f:
        json.dump(layer_info, f, indent=2, ensure_ascii=False)

    if args.save_raw_reps:
        log("\n=== Saving raw representations ===")
        save_raw_reps(all_reps, args.output_dir)

    if not args.skip_layer_plots:
        log("\n=== Plotting every layer ===")
        for layer in sorted(all_reps.keys()):
            plot_layer(all_reps[layer], layer, methods, args)

    if not args.skip_pca_grid:
        log("\n=== Saving PCA grid ===")
        plot_pca_grid(all_reps, args)

    log(f"\n[DONE] Results saved under: {args.output_dir}")


if __name__ == "__main__":
    main()

"""
pipeline_sampled_100k.py - Passive voice rate analysis pipeline
Academic (arXiv, Wikipedia) vs Everyday (DailyDialog, Reddit)

Main change from the original version:
- It does NOT collect and preprocess the whole corpus first.
- It shuffles each source dataset, extracts valid sentences, and stops as soon as
  sample_n sentences have been collected for that corpus.
- Sampled sentences are saved to output/sampled/sentences_{corpus}.jsonl, so
  parsing can be resumed from STEP 3.

Usage:
    python pipeline_sampled_100k.py
    python pipeline_sampled_100k.py --sample_n 100000 --resume
    python pipeline_sampled_100k.py --step 3 --sample_n 100000 --resume
    python pipeline_sampled_100k.py --step 5
    python pipeline_sampled_100k.py --no_gpu
    python pipeline_sampled_100k.py --no_streaming
"""

print("[startup] Loading libraries...", flush=True)

# Standard library
import argparse
import csv
import itertools
import json
import logging
import os
import random
import re
import threading
import time
from pathlib import Path
from typing import Callable, Iterable, Iterator

# Keep large downloads and model files inside the workspace volume.
HF_CACHE_DIR = Path(__file__).resolve().parent / ".cache" / "huggingface"
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("HF_DATASETS_CACHE", str(HF_CACHE_DIR / "datasets"))
os.environ.setdefault("HF_MODULES_CACHE", str(HF_CACHE_DIR / "modules"))
os.environ.setdefault("STANZA_RESOURCES_DIR", str(Path(__file__).resolve().parent / ".cache" / "stanza"))
os.environ.setdefault("NLTK_DATA", str(Path(__file__).resolve().parent / ".cache" / "nltk"))
(HF_CACHE_DIR / "datasets").mkdir(parents=True, exist_ok=True)
(HF_CACHE_DIR / "modules").mkdir(parents=True, exist_ok=True)
Path(os.environ["STANZA_RESOURCES_DIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["NLTK_DATA"]).mkdir(parents=True, exist_ok=True)

# Third-party libraries
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import stanza
from datasets import load_dataset
from scipy import stats
from tqdm import tqdm

print("[startup] Done", flush=True)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Constants
PASSIVE_RELS = {"nsubj:pass", "aux:pass", "csubj:pass"}

# ── HARM BENCHMARK CONFIG ────────────────────────────────────────────────────
# source      : "hf" (HuggingFace) | "csv_url" (raw CSV via HTTP)
# --- HF only ---
# dataset_id  : HuggingFace dataset repo ID
# subset      : HF config/subset name (None = default)
# splits      : split names to try in order until one succeeds
# filter      : (column, value) row filter, or None  ← e.g. SORRY-Bench base only
# --- csv_url only ---
# url         : direct URL to a CSV file
# --- common ---
# text_fields : ordered candidate column names for the behavior text
#               value may be str, list[str], or list[dict] (first turn extracted)
#
# Verified sizes  (2025-05):
#   JBB-Behaviors : 100  (harmful split of 'behaviors' config; 'Goal' is the prompt)
#   SORRY-Bench   : 450  (9,450 total × 21 prompt_styles → filter base only)
#   HarmBench     : 400  (GitHub CSV; not on HF public hub)
#   AdvBench      : 520  ('prompt' column)
HARM_BENCHMARKS_CFG: dict[str, dict] = {
    "JBB-Behaviors": {
        "source": "hf",
        "dataset_id": "JailbreakBench/JBB-Behaviors",
        "subset": "behaviors",          # config name, not split
        "splits": ["harmful"],          # 'harmful' split = 100 real behaviors
        "filter": None,
        "text_fields": ["Goal"],        # 'Behavior' col = category label, not prompt
    },
    "SORRY-Bench": {
        "source": "hf",
        "dataset_id": "sorry-bench/sorry-bench-202503",
        "subset": None,
        "splits": ["train"],
        "filter": None,
        "exclude": ("prompt_style", [
            "ascii", "atbash", "caesar", "morse",          
            "translate-fr", "translate-ml", "translate-mr", 
            "translate-ta", "translate-zh-cn",
        ]),
        "text_fields": ["turns"],  # list[str] → take turns[0]
    },
    "SORRY-Bench-base": {
        "source": "hf",
        "dataset_id": "sorry-bench/sorry-bench-202503",
        "subset": None,
        "splits": ["train"],
        "filter": ("prompt_style", "base"), 
        "text_fields": ["turns"],
    },
    "HarmBench": {
        "source": "csv_url",
        "url": "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_all.csv",
        "text_fields": ["Behavior"],    # 400 behaviors
    },
    "AdvBench": {
        "source": "hf",
        "dataset_id": "walledai/AdvBench",
        "subset": None,
        "splits": ["train"],
        "filter": None,
        "text_fields": ["prompt"],      # 520 behaviors
    },
}

CORPUS_COLORS = {
    "arxiv": "#6C5CE7",
    "wikipedia": "#0984E3",
    "dailydialog": "#00B894",
    "reddit": "#E17055",
}

DOMAIN_COLORS = {
    "academic": "#6C5CE7",
    "everyday": "#00B894",
}


def _spinner_load(fn, label):
    """Run fn in a background thread while showing a terminal spinner."""
    result = [None]
    exc = [None]

    def _run():
        try:
            result[0] = fn()
        except Exception as e:
            exc[0] = e

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    frames = itertools.cycle("|/-\\")
    while t.is_alive():
        print(f"\r  {next(frames)} {label}", end="", flush=True)
        time.sleep(0.1)
    print(f"\r  OK {label}    ", flush=True)

    if exc[0]:
        raise exc[0]
    return result[0]


# Config

def build_config(args) -> dict:
    # Changed: parse --harm_benchmarks filter
    if args.harm_benchmarks:
        selected = [b.strip() for b in args.harm_benchmarks.split(",") if b.strip()]
        invalid = [b for b in selected if b not in HARM_BENCHMARKS_CFG]
        if invalid:
            raise ValueError(f"Unknown benchmark name(s): {invalid}. Choose from: {list(HARM_BENCHMARKS_CFG.keys())}")
        harm_benchmarks_filter = selected
    else:
        harm_benchmarks_filter = None  # None = 전체 실행

    return {
        "corpora": ["arxiv", "wikipedia", "dailydialog", "reddit"],
        "domain_map": {
            "arxiv": "academic",
            "wikipedia": "academic",
            "dailydialog": "everyday",
            "reddit": "everyday",
        },
        "sample_n": args.sample_n,
        "random_seed": args.seed,
        "min_tokens": args.min_tokens,
        "max_tokens": args.max_tokens,
        "use_gpu": not args.no_gpu,
        "batch_size": args.batch_size,
        "parse_batch_n": args.parse_batch_n,
        "streaming": not args.no_streaming,
        "shuffle_buffer": args.shuffle_buffer,
        "max_source_rows": args.max_source_rows,
        "sampled_dir": Path("output/sampled"),
        "parsed_dir": Path("output/parsed"),
        "results_dir": Path("output/results"),
        "figures_dir": Path("output/figures"),
        "resume": args.resume,
        "harm_benchmarks_filter": harm_benchmarks_filter,  # Changed
    }


def make_dirs(config: dict):
    for key in ("sampled_dir", "parsed_dir", "results_dir", "figures_dir"):
        config[key].mkdir(parents=True, exist_ok=True)


# Text cleaning and sentence filtering

def _clean(text: str) -> str:
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\[.*?\]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _is_valid(sent: str, min_tok: int, max_tok: int) -> bool:
    n = len(sent.split())
    return min_tok <= n <= max_tok


def _ensure_nltk():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)


# STEP 1/2 - bounded sampling instead of full-corpus collection

def _load_dataset_with_fallbacks(corpus: str, config: dict):
    streaming = config["streaming"]

    if corpus == "arxiv":
        return _spinner_load(
            lambda: load_dataset(
                "sentence-transformers/s2orc",
                split="train",
                streaming=streaming,
            ),
            "loading S2ORC dataset",
        )

    if corpus == "wikipedia":
        return _spinner_load(
            lambda: load_dataset(
                "wikipedia",
                "20220301.en",
                split="train",
                trust_remote_code=True,
                streaming=streaming,
            ),
            "loading Wikipedia dataset",
        )

    if corpus == "dailydialog":
        dd_path = Path(__file__).resolve().parent / ".cache" / "roskoN_dailydialog" / "train" / "dialogues_train.txt"
        def _load_roskodialog():
            dialogs = []
            with dd_path.open(encoding="utf-8") as f:
                for line in f:
                    utterances = [u.strip() for u in line.strip().split("__eou__") if u.strip()]
                    dialogs.append({"dialog": utterances})
            return dialogs
        return _spinner_load(_load_roskodialog, "loading DailyDialog (roskoN/dailydialog)")

    if corpus == "reddit":
        REDDIT_SPLITS = [
            'programming','tifu','explainlikeimfive','WritingPrompts','changemyview',
            'LifeProTips','todayilearned','science','askscience','ifyoulikeblank',
            'Foodforthought','IWantToLearn','bestof','IAmA','socialskills',
            'relationship_advice','philosophy','YouShouldKnow','history','books',
            'Showerthoughts','personalfinance','buildapc','EatCheapAndHealthy',
            'boardgames','malefashionadvice','femalefashionadvice','scifi','Fantasy',
            'Games','bodyweightfitness','SkincareAddiction','podcasts','suggestmeabook',
            'AskHistorians','gaming','DIY','sports','space','gadgets','Documentaries',
            'GetMotivated','UpliftingNews','technology','Fitness','travel','lifehacks',
            'Damnthatsinteresting','gardening','mildlyinteresting',
        ]
        from datasets import interleave_datasets
        return _spinner_load(
            lambda: interleave_datasets(
                [load_dataset("HuggingFaceGECLM/REDDIT_comments", split=s, streaming=streaming)
                 for s in REDDIT_SPLITS],
                stopping_strategy="all_exhausted",
            ),
            "loading Reddit (HuggingFaceGECLM/REDDIT_comments)",
        )

    raise ValueError(f"Unknown corpus: {corpus}")


def _shuffle_dataset(ds, config: dict, corpus: str):
    seed = config["random_seed"]
    if isinstance(ds, list):
        log.info("  [%s] in-memory shuffle (list)", corpus)
        random.seed(seed)
        random.shuffle(ds)
        return ds
    if config["streaming"]:
        log.info("  [%s] streaming shuffle buffer=%s", corpus, config["shuffle_buffer"])
        return ds.shuffle(seed=seed, buffer_size=config["shuffle_buffer"])
    log.info("  [%s] in-memory shuffle", corpus)
    return ds.shuffle(seed=seed)


def _text_iter_for_corpus(ds, corpus: str) -> Iterator[str]:
    skip_values = {"[deleted]", "[removed]", ""}

    for row in ds:
        if corpus == "arxiv":
            text = row.get("abstract", "")
            if text:
                yield text

        elif corpus == "wikipedia":
            text = row.get("text", "")
            if text:
                yield text

        elif corpus == "dailydialog":
            for utterance in row.get("dialog", []):
                if utterance:
                    yield utterance

        elif corpus == "reddit":
            body = row.get("body", "")
            if isinstance(body, str):
                body = body.strip()
                if body not in skip_values and len(body) > 20:
                    yield body


def sample_sentences_for_corpus(corpus: str, config: dict) -> list[str]:
    """Collect valid sentences until sample_n is reached, then stop."""
    _ensure_nltk()

    out_path = config["sampled_dir"] / f"sentences_{corpus}.jsonl"
    if config["resume"] and out_path.exists():
        loaded = load_sampled_sentences(corpus, config)
        if len(loaded) >= config["sample_n"]:
            log.info("  [SKIP] %s exists with %s sentences", out_path.name, f"{len(loaded):,}")
            return loaded[: config["sample_n"]]
        log.warning("  Existing %s has only %s sentences; resampling", out_path.name, f"{len(loaded):,}")

    log.info("  [%s] target valid sentences: %s", corpus, f"{config['sample_n']:,}")
    ds = _load_dataset_with_fallbacks(corpus, config)
    ds = _shuffle_dataset(ds, config, corpus)

    collected: list[str] = []
    source_rows_seen = 0
    text_iter = _text_iter_for_corpus(ds, corpus)

    pbar = tqdm(total=config["sample_n"], desc=f"  Sampling {corpus}")
    for raw_text in text_iter:
        source_rows_seen += 1
        if config["max_source_rows"] is not None and source_rows_seen > config["max_source_rows"]:
            log.warning("  [%s] max_source_rows reached: %s", corpus, config["max_source_rows"])
            break

        text = _clean(raw_text)
        if not text:
            continue

        for sent in nltk.sent_tokenize(text):
            sent = sent.strip()
            if not _is_valid(sent, config["min_tokens"], config["max_tokens"]):
                continue

            collected.append(sent)
            pbar.update(1)

            if len(collected) >= config["sample_n"]:
                pbar.close()
                save_sampled_sentences(corpus, collected, config)
                log.info(
                    "  [%s] sampled %s valid sentences from %s source texts",
                    corpus,
                    f"{len(collected):,}",
                    f"{source_rows_seen:,}",
                )
                return collected

    pbar.close()
    save_sampled_sentences(corpus, collected, config)
    log.warning(
        "  [%s] target not reached: sampled %s/%s sentences from %s source texts",
        corpus,
        f"{len(collected):,}",
        f"{config['sample_n']:,}",
        f"{source_rows_seen:,}",
    )
    return collected


def save_sampled_sentences(corpus: str, sentences: list[str], config: dict):
    out_path = config["sampled_dir"] / f"sentences_{corpus}.jsonl"
    domain = config["domain_map"][corpus]
    with out_path.open("w", encoding="utf-8") as f:
        for i, text in enumerate(sentences):
            rec = {
                "corpus": corpus,
                "domain_group": domain,
                "sent_id": i,
                "text": text,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log.info("  SAVE POINT S: %s", out_path)


def load_sampled_sentences(corpus: str, config: dict) -> list[str]:
    path = config["sampled_dir"] / f"sentences_{corpus}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing sampled sentence file: {path}")

    sentences = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            sentences.append(rec["text"])
    return sentences


def collect_and_sample(config: dict) -> dict[str, list[str]]:
    log.info("STEP 1/2: Dataset sampling and preprocessing")
    sentences = {}
    for corpus in config["corpora"]:
        sentences[corpus] = sample_sentences_for_corpus(corpus, config)
    return sentences


def load_all_sampled_sentences(config: dict) -> dict[str, list[str]]:
    sentences = {}
    for corpus in config["corpora"]:
        sentences[corpus] = load_sampled_sentences(corpus, config)
        log.info("  [%s] loaded %s sampled sentences", corpus, f"{len(sentences[corpus]):,}")
    return sentences


# Stanza initialization

def init_stanza(config: dict):
    log.info("  Initializing Stanza pipeline...")
    stanza.download("en", model_dir=os.environ["STANZA_RESOURCES_DIR"], verbose=False)
    nlp = stanza.Pipeline(
        "en",
        dir=os.environ["STANZA_RESOURCES_DIR"],
        processors="tokenize,mwt,pos,lemma,depparse",
        use_gpu=config["use_gpu"],
        tokenize_batch_size=config["batch_size"],
        depparse_batch_size=config["batch_size"],
        verbose=False,
    )
    log.info("  Done (GPU=%s)", config["use_gpu"])
    return nlp


# STEP 3 - Stanza parsing

def _iter_batches(items: list[str], batch_size: int) -> Iterator[tuple[int, list[str]]]:
    for start in range(0, len(items), batch_size):
        yield start, items[start : start + batch_size]


def parse_and_save(sentences: list[str], corpus_name: str, config: dict, nlp):
    """
    Parse sampled sentences and save dependency relations.

    The code keeps one input sentence per Stanza call by default to preserve a
    strict sent_id-to-input-sentence mapping. You can raise --parse_batch_n for
    speed, but Stanza may split or merge sentences differently in a text block.
    """
    out_path = config["parsed_dir"] / f"parsed_{corpus_name}.jsonl"

    if config["resume"] and out_path.exists():
        log.info("  [SKIP] %s exists", out_path.name)
        return

    domain = config["domain_map"][corpus_name]
    parse_batch_n = max(1, config["parse_batch_n"])
    log.info("  [%s] parsing %s sentences...", corpus_name, f"{len(sentences):,}")

    with out_path.open("w", encoding="utf-8") as f:
        if parse_batch_n == 1:
            iterator = enumerate(tqdm(sentences, desc=f"  Parsing {corpus_name}"))
            for i, text in iterator:
                try:
                    doc = nlp(text)
                except Exception as e:
                    log.warning("    parse error corpus=%s sent_id=%s: %s", corpus_name, i, e)
                    continue

                for sent in doc.sentences:
                    record = {
                        "corpus": corpus_name,
                        "domain_group": domain,
                        "sent_id": i,
                        "text": sent.text,
                        "tokens": [{"word": w.text, "deprel": w.deprel} for w in sent.words],
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
        else:
            # Faster but sent_id is assigned sequentially to parsed output sentences.
            next_sent_id = 0
            batches = list(_iter_batches(sentences, parse_batch_n))
            for start, batch in tqdm(batches, desc=f"  Parsing {corpus_name}"):
                text_block = "\n".join(batch)
                try:
                    doc = nlp(text_block)
                except Exception as e:
                    log.warning("    batch parse error corpus=%s start=%s: %s", corpus_name, start, e)
                    continue

                for sent in doc.sentences:
                    record = {
                        "corpus": corpus_name,
                        "domain_group": domain,
                        "sent_id": next_sent_id,
                        "text": sent.text,
                        "tokens": [{"word": w.text, "deprel": w.deprel} for w in sent.words],
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    next_sent_id += 1

    log.info("  SAVE POINT A: %s", out_path)


# STEP 4 - passive detection

def detect_and_save(corpus_name: str, config: dict):
    in_path = config["parsed_dir"] / f"parsed_{corpus_name}.jsonl"
    out_path = config["results_dir"] / f"results_{corpus_name}.csv"

    if not in_path.exists():
        log.error("  [ERROR] Missing %s. Run STEP 3 first.", in_path)
        return

    if config["resume"] and out_path.exists():
        log.info("  [SKIP] %s exists", out_path.name)
        return

    log.info("  [%s] detecting passive sentences...", corpus_name)

    fieldnames = ["corpus", "domain_group", "sent_id", "text", "is_passive", "passive_rels"]
    passive_count = 0
    total_count = 0

    with in_path.open(encoding="utf-8") as fin, out_path.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        for line in tqdm(fin, desc=f"  Detecting {corpus_name}"):
            rec = json.loads(line)
            found = [t["deprel"] for t in rec["tokens"] if t["deprel"] in PASSIVE_RELS]
            is_passive = bool(found)

            writer.writerow(
                {
                    "corpus": rec["corpus"],
                    "domain_group": rec["domain_group"],
                    "sent_id": rec["sent_id"],
                    "text": rec["text"],
                    "is_passive": is_passive,
                    "passive_rels": "|".join(found),
                }
            )

            total_count += 1
            passive_count += int(is_passive)

    rate = passive_count / total_count if total_count else 0
    log.info(
        "  SAVE POINT B: %s (passive=%s/%s, %.1f%%)",
        out_path,
        f"{passive_count:,}",
        f"{total_count:,}",
        rate * 100,
    )


# STEP 5 - statistics and visualization

def _bootstrap_ci(series: pd.Series, n_iter: int = 1000, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    arr = series.values
    means = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_iter)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def analyze_and_visualize(config: dict):
    log.info("STEP 5: Statistics and visualization")

    dfs = []
    for corpus in config["corpora"]:
        path = config["results_dir"] / f"results_{corpus}.csv"
        if not path.exists():
            log.error("  [ERROR] Missing %s. Run STEP 4 first.", path)
            return
        dfs.append(pd.read_csv(path))

    df = pd.concat(dfs, ignore_index=True)
    df["is_passive"] = df["is_passive"].astype(bool)
    log.info("  Total parsed sentences: %s", f"{len(df):,}")

    summary_rows = []
    for corpus in config["corpora"]:
        sub = df[df["corpus"] == corpus]["is_passive"].astype(int)
        mean = sub.mean()
        lo, hi = _bootstrap_ci(sub)
        summary_rows.append(
            {
                "corpus": corpus,
                "domain_group": config["domain_map"][corpus],
                "n_sentences": len(sub),
                "n_passive": int(sub.sum()),
                "passive_rate": round(mean, 4),
                "ci_lower": round(lo, 4),
                "ci_upper": round(hi, 4),
            }
        )
        log.info("  %-14s rate=%.3f  95%% CI [%.3f, %.3f]", corpus, mean, lo, hi)

    summary_df = pd.DataFrame(summary_rows)

    academic = df[df["domain_group"] == "academic"]["is_passive"].astype(int)
    everyday = df[df["domain_group"] == "everyday"]["is_passive"].astype(int)

    mw_stat, mw_p = stats.mannwhitneyu(academic, everyday, alternative="two-sided")

    ac_pass = academic.sum()
    ac_fail = len(academic) - ac_pass
    ev_pass = everyday.sum()
    ev_fail = len(everyday) - ev_pass
    contingency = [[ac_pass, ac_fail], [ev_pass, ev_fail]]
    chi2_stat, chi2_p, chi2_dof, _ = stats.chi2_contingency(contingency)

    log.info("  Mann-Whitney U: stat=%.2f, p=%.4e", mw_stat, mw_p)
    log.info("  Chi-square: stat=%.2f, p=%.4e, df=%s", chi2_stat, chi2_p, chi2_dof)

    stats_dict = {
        "mann_whitney_stat": mw_stat,
        "mann_whitney_p": mw_p,
        "chi2_stat": chi2_stat,
        "chi2_p": chi2_p,
        "chi2_df": chi2_dof,
    }

    stats_path = config["figures_dir"] / "summary_stats.csv"
    summary_df.to_csv(stats_path, index=False, encoding="utf-8")
    log.info("  saved: %s", stats_path)

    _plot_results(summary_df, stats_dict, df, config)


def _plot_results(summary_df: pd.DataFrame, stats_dict: dict, df: pd.DataFrame, config: dict):
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor("#1A1A2E")

    gs = fig.add_gridspec(2, 4, hspace=0.45, wspace=0.35)

    for idx, corpus in enumerate(config["corpora"]):
        ax = fig.add_subplot(gs[0, idx])
        row = summary_df[summary_df["corpus"] == corpus].iloc[0]
        color = CORPUS_COLORS[corpus]
        rate = row["passive_rate"]
        lo, hi = row["ci_lower"], row["ci_upper"]

        ax.bar([corpus], [rate], color=color, alpha=0.85, width=0.5)
        ax.errorbar(
            [corpus],
            [rate],
            yerr=[[rate - lo], [hi - rate]],
            fmt="none",
            color="white",
            capsize=6,
            linewidth=2,
        )

        ax.set_ylim(0, 0.55)
        ax.set_title(corpus.upper(), color="white", fontsize=11, fontweight="bold")
        ax.set_ylabel("Passive Rate", color="white", fontsize=9)
        ax.set_xticks([])
        ax.tick_params(colors="white")
        ax.set_facecolor("#16213E")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444466")

        label_text = f"{rate:.1%}\nn={int(row['n_sentences']):,}\n[{lo:.3f}, {hi:.3f}]"
        ax.text(
            0.5,
            rate + 0.02,
            label_text,
            ha="center",
            va="bottom",
            color="white",
            fontsize=8,
            transform=ax.get_xaxis_transform(),
        )

        domain_label = "Academic" if row["domain_group"] == "academic" else "Everyday"
        domain_color = DOMAIN_COLORS[row["domain_group"]]
        ax.text(
            0.5,
            -0.12,
            domain_label,
            ha="center",
            va="top",
            color=domain_color,
            fontsize=9,
            fontweight="bold",
            transform=ax.transAxes,
        )

    ax_box = fig.add_subplot(gs[1, :2])
    domain_data = [
        df[df["domain_group"] == "academic"]["is_passive"].astype(int),
        df[df["domain_group"] == "everyday"]["is_passive"].astype(int),
    ]
    bp = ax_box.boxplot(
        domain_data,
        tick_labels=["Academic", "Everyday"],
        patch_artist=True,
        medianprops=dict(color="white", linewidth=2),
        whiskerprops=dict(color="#AAAACC"),
        capprops=dict(color="#AAAACC"),
        flierprops=dict(marker="o", markerfacecolor="#AAAACC", alpha=0.3, markersize=3),
    )
    for patch, color in zip(bp["boxes"], [DOMAIN_COLORS["academic"], DOMAIN_COLORS["everyday"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax_box.set_title("Academic vs Everyday", color="white", fontsize=11, fontweight="bold")
    ax_box.set_ylabel("is_passive (0/1)", color="white")
    ax_box.tick_params(colors="white")
    ax_box.set_facecolor("#16213E")
    for spine in ax_box.spines.values():
        spine.set_edgecolor("#444466")

    ax_stat = fig.add_subplot(gs[1, 2:])
    ax_stat.set_facecolor("#16213E")
    ax_stat.axis("off")

    mw_p = stats_dict["mann_whitney_p"]
    chi2_p = stats_dict["chi2_p"]

    lines = [
        ("Statistical Tests", "#FFFFFF", 14, True),
        ("", "#FFFFFF", 10, False),
        ("Mann-Whitney U", "#AACCFF", 11, True),
        (f"  stat = {stats_dict['mann_whitney_stat']:.2f}", "#FFFFFF", 10, False),
        (f"  p = {mw_p:.2e}  {'significant' if mw_p < 0.05 else 'n.s.'}", "#00E676" if mw_p < 0.05 else "#FF6B6B", 10, False),
        ("", "#FFFFFF", 10, False),
        ("Chi-square", "#AACCFF", 11, True),
        (f"  stat = {stats_dict['chi2_stat']:.2f}, df = {int(stats_dict['chi2_df'])}", "#FFFFFF", 10, False),
        (f"  p = {chi2_p:.2e}  {'significant' if chi2_p < 0.05 else 'n.s.'}", "#00E676" if chi2_p < 0.05 else "#FF6B6B", 10, False),
        ("", "#FFFFFF", 10, False),
        ("alpha = 0.05, two-sided", "#888899", 9, False),
    ]

    y = 0.92
    for text, color, size, bold in lines:
        ax_stat.text(
            0.08,
            y,
            text,
            transform=ax_stat.transAxes,
            color=color,
            fontsize=size,
            fontweight="bold" if bold else "normal",
            va="top",
        )
        y -= 0.09

    legend_patches = [
        mpatches.Patch(color=DOMAIN_COLORS["academic"], label="Academic (arXiv, Wikipedia)"),
        mpatches.Patch(color=DOMAIN_COLORS["everyday"], label="Everyday (DailyDialog, Reddit)"),
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=2,
        framealpha=0.2,
        labelcolor="white",
        fontsize=9,
        bbox_to_anchor=(0.5, 0.01),
    )

    fig.suptitle(
        "English Passive Voice Rate: Academic vs Everyday Corpora",
        color="white",
        fontsize=14,
        fontweight="bold",
        y=0.97,
    )

    out_path = config["figures_dir"] / "boxplot_passive_rate.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    log.info("  saved: %s", out_path)


# STEP 6 - Harm benchmark passive/active rate analysis

def _extract_harm_text(row: dict, text_fields: list[str]) -> str | None:
    """
    Pull the behavior text from one dataset row.
    Handles: plain string, list[str], list[dict] (e.g. conversation turns).
    """
    for field in text_fields:
        val = row.get(field)
        if val is None:
            continue

        # Plain string
        if isinstance(val, str):
            stripped = val.strip()
            if stripped:
                return stripped

        # List — take the first non-empty element
        if isinstance(val, list) and val:
            first = val[0]
            if isinstance(first, str) and first.strip():
                return first.strip()
            # list[dict] — look for common content keys (SORRY-Bench style)
            if isinstance(first, dict):
                for content_key in ("content", "text", "value", "message"):
                    candidate = first.get(content_key)
                    if isinstance(candidate, str) and candidate.strip():
                        return candidate.strip()

    return None


def _load_hf_split(dataset_id: str, splits: list[str], subset: str | None):
    """Try each split name in order; return (dataset, split_name) or raise."""
    from datasets import load_dataset as _ld

    last_err = None
    for split in splits:
        try:
            kwargs = dict(split=split, streaming=False)
            if subset:
                ds = _ld(dataset_id, subset, **kwargs)
            else:
                ds = _ld(dataset_id, **kwargs)
            return ds, split
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Could not load {dataset_id} with splits={splits}: {last_err}")


def _load_csv_url(url: str) -> list[dict]:
    """Download a CSV from a URL and return rows as list[dict]."""
    import csv, io, urllib.request
    with urllib.request.urlopen(url, timeout=30) as r:
        content = r.read().decode("utf-8")
    return list(csv.DictReader(io.StringIO(content)))


def load_harm_benchmark_texts(name: str, cfg: dict) -> list[str]:
    """Load all behavior strings from one harm benchmark."""
    source = cfg.get("source", "hf")

    # ── CSV URL (e.g. HarmBench from GitHub) ────────────────────────────────
    if source == "csv_url":
        log.info("  [%s] downloading CSV from %s ...", name, cfg["url"])
        try:
            rows = _load_csv_url(cfg["url"])
        except Exception as e:
            log.warning("  [%s] SKIP — CSV download failed: %s", name, e)
            return []
        texts = []
        for row in rows:
            text = _extract_harm_text(row, cfg["text_fields"])
            if text:
                texts.append(text)
        log.info("  [%s] loaded %d behavior texts (CSV)", name, len(texts))
        return texts

    # ── HuggingFace dataset ──────────────────────────────────────────────────
    log.info("  [%s] loading %s ...", name, cfg["dataset_id"])
    try:
        ds, used_split = _load_hf_split(cfg["dataset_id"], cfg["splits"], cfg["subset"])
    except Exception as e:
        log.warning("  [%s] SKIP — could not load dataset: %s", name, e)
        return []

    # optional include-filter:  ("col", value)
    row_filter = cfg.get("filter")
    # optional exclude-filter:  ("col", [val1, val2, ...])
    row_exclude = cfg.get("exclude")
    exclude_set = set(row_exclude[1]) if row_exclude else set()

    texts: list[str] = []
    for row in ds:
        if row_filter is not None:
            col, val = row_filter
            if row.get(col) != val:
                continue
        if row_exclude is not None:
            col = row_exclude[0]
            if row.get(col) in exclude_set:
                continue
        text = _extract_harm_text(dict(row), cfg["text_fields"])
        if text:
            texts.append(text)

    log.info(
        "  [%s] loaded %d behavior texts (split=%s, subset=%s, filter=%s, exclude=%s)",
        name, len(texts), used_split, cfg.get("subset") or "default",
        row_filter, row_exclude,
    )
    return texts


def _passive_rate_for_texts(
    name: str, texts: list[str], nlp
) -> dict:
    """Run Stanza on each behavior text; return per-benchmark stats dict."""
    n_passive_sents = 0
    n_total_sents = 0
    n_passive_behaviors = 0   # at least one passive sent in the behavior text

    for text in tqdm(texts, desc=f"  Parsing {name}", leave=False):
        try:
            doc = nlp(text)
        except Exception as e:
            log.warning("    [%s] parse error: %s", name, e)
            continue

        behavior_has_passive = False
        for sent in doc.sentences:
            found = [w.deprel for w in sent.words if w.deprel in PASSIVE_RELS]
            is_passive = bool(found)
            n_total_sents += 1
            n_passive_sents += int(is_passive)
            if is_passive:
                behavior_has_passive = True

        n_passive_behaviors += int(behavior_has_passive)

    n_active_sents = n_total_sents - n_passive_sents
    n_active_behaviors = len(texts) - n_passive_behaviors
    passive_sent_rate = n_passive_sents / n_total_sents if n_total_sents else 0.0
    passive_beh_rate  = n_passive_behaviors / len(texts)  if texts       else 0.0

    return {
        "benchmark":          name,
        "n_behaviors":        len(texts),
        "n_sentences":        n_total_sents,
        "n_active_sents":     n_active_sents,
        "n_passive_sents":    n_passive_sents,
        "active_sent_rate":   1 - passive_sent_rate,
        "passive_sent_rate":  passive_sent_rate,
        "n_active_behaviors": n_active_behaviors,
        "n_passive_behaviors":n_passive_behaviors,
        "active_beh_rate":    1 - passive_beh_rate,
        "passive_beh_rate":   passive_beh_rate,
    }


def analyze_harm_benchmarks(config: dict, nlp) -> list[dict]:
    """
    STEP 6 — For each harm benchmark, parse behavior texts with Stanza
    and print a one-line passive/active rate summary.
    """
    log.info("STEP 6: Harm benchmark passive/active rate analysis")

    all_results: list[dict] = []

    bench_filter = config.get("harm_benchmarks_filter")  # Changed
    bench_items = {
        name: cfg for name, cfg in HARM_BENCHMARKS_CFG.items()
        if bench_filter is None or name in bench_filter
    }.items()  # Changed

    for name, cfg in bench_items:
        texts = load_harm_benchmark_texts(name, cfg)
        if not texts:
            log.warning("  [%s] skipping — no texts", name)
            continue

        r = _passive_rate_for_texts(name, texts, nlp)
        all_results.append(r)

        # ── one-line summary per benchmark ──────────────────────────────────
        log.info(
            "  %-14s | behaviors=%4d | sents=%5d"
            " | active=%4d(%.1f%%) | passive=%4d(%.1f%%)",
            r["benchmark"],
            r["n_behaviors"],
            r["n_sentences"],
            r["n_active_sents"],   r["active_sent_rate"]  * 100,
            r["n_passive_sents"],  r["passive_sent_rate"] * 100,
        )

    if not all_results:
        log.warning("  No harm benchmark results to display.")
        return all_results

    # ── pretty summary table ─────────────────────────────────────────────────
    sep = "  " + "─" * 84
    hdr = f"  {'Benchmark':<14}  {'behaviors':>9}  {'sentences':>9}  {'active(%)':>12}  {'passive(%)':>12}"
    log.info(sep)
    log.info(hdr)
    log.info(sep)
    for r in all_results:
        log.info(
            "  %-14s  %9d  %9d  %11.1f%%  %11.1f%%",
            r["benchmark"],
            r["n_behaviors"],
            r["n_sentences"],
            r["active_sent_rate"]  * 100,
            r["passive_sent_rate"] * 100,
        )
    log.info(sep)

    out_path = config["results_dir"] / "results_harm_benchmarks.csv"
    pd.DataFrame(all_results).to_csv(out_path, index=False, encoding="utf-8")
    log.info("  saved: %s", out_path)

    return all_results


# CLI

def parse_args():
    parser = argparse.ArgumentParser(description="Passive voice analysis pipeline")
    parser.add_argument("--step", type=int, default=1, choices=range(1, 7), help="Start step. 1/2=sampling, 3=parsing, 4=detection, 5=analysis, 6=harm-bench")
    parser.add_argument("--harm_bench", action="store_true", help="Run STEP 6 after the main pipeline: JBB-Behaviors / SORRY-Bench / HarmBench / AdvBench passive-rate analysis")
    parser.add_argument("--harm_benchmarks", type=str, default=None, help="쉼표로 구분된 실행할 벤치마크 이름. 예: SORRY-Bench-base 또는 JBB-Behaviors,SORRY-Bench-base. 생략하면 전체 실행. 가능한 값: " + ", ".join(HARM_BENCHMARKS_CFG.keys()))  # Changed
    parser.add_argument("--sample_n", type=int, default=100000, help="Target valid sentences per corpus")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--min_tokens", type=int, default=5, help="Minimum whitespace-token count per sentence")
    parser.add_argument("--max_tokens", type=int, default=80, help="Maximum whitespace-token count per sentence")
    parser.add_argument("--resume", action="store_true", help="Skip existing sampled/jsonl/csv outputs")
    parser.add_argument("--no_gpu", action="store_true", help="Force Stanza to use CPU")
    parser.add_argument("--batch_size", type=int, default=64, help="Stanza internal batch size")
    parser.add_argument("--parse_batch_n", type=int, default=1, help="Number of input sentences per Stanza call. 1 preserves sent_id mapping; larger can be faster.")
    parser.add_argument("--no_streaming", action="store_true", help="Disable Hugging Face streaming mode")
    parser.add_argument("--shuffle_buffer", type=int, default=50000, help="Streaming shuffle buffer size")
    parser.add_argument("--max_source_rows", type=int, default=None, help="Optional safety cap on source texts read per corpus")
    return parser.parse_args()


# Main

def main():
    args = parse_args()
    config = build_config(args)
    make_dirs(config)

    start = args.step
    run_harm = args.harm_bench or start == 6

    log.info(
        "Pipeline start: step=%s, sample_n=%s, resume=%s, gpu=%s, streaming=%s, harm_bench=%s",
        start,
        f"{config['sample_n']:,}",
        config["resume"],
        config["use_gpu"],
        config["streaming"],
        run_harm,
    )

    sentences = None
    nlp = None  # lazy-initialised; shared across steps 3 and 6

    # STEP 1/2 - collect sampled valid sentences and save them.
    if start <= 2:
        sentences = collect_and_sample(config)

    # STEP 3 - parse saved or freshly sampled sentences.
    if start <= 3:
        log.info("STEP 3: Stanza parsing")
        if sentences is None:
            try:
                sentences = load_all_sampled_sentences(config)
            except FileNotFoundError as e:
                log.error("%s", e)
                log.error("Run with --step 1 first to create output/sampled/*.jsonl")
                return

        nlp = init_stanza(config)
        for corpus, sents in sentences.items():
            parse_and_save(sents, corpus, config, nlp)

    # STEP 4 - detect passive dependencies from saved parsed files.
    if start <= 4:
        log.info("STEP 4: Passive detection")
        for corpus in config["corpora"]:
            detect_and_save(corpus, config)

    # STEP 5 - analyze and visualize.
    if start <= 5:
        analyze_and_visualize(config)

    # STEP 6 - harm benchmark passive/active rate analysis.
    if run_harm:
        if nlp is None:
            nlp = init_stanza(config)
        analyze_harm_benchmarks(config, nlp)

    log.info("Pipeline complete")


if __name__ == "__main__":
    main()

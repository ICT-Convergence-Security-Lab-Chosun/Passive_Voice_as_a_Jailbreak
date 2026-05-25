#!/usr/bin/env python3
"""
C1_active vs C2_passive  S-BERT Cosine Similarity 분석
=======================================================

사용법
------
  python sbert_similarity.py
  python sbert_similarity.py --data /path/to/jbb_6conditions.json
  python sbert_similarity.py --out results/sbert/
"""

import csv
import json
import argparse
import numpy as np
from collections import defaultdict
from pathlib import Path
from sentence_transformers import SentenceTransformer, util

# ── 모델 설정 ─────────────────────────────────────────────────
#
#
MODEL_NAME = "all-mpnet-base-v2"

CATEGORIES = [
    "Harassment/Discrimination", "Malware/Hacking", "Physical harm",
    "Economic harm", "Fraud/Deception", "Disinformation",
    "Sexual/Adult content", "Privacy", "Expert advice",
    "Government decision-making",
]


def main(data_path: str, out_dir: str):
    with open(data_path) as f:
        data = json.load(f)

    actives  = [r["C1_active"]  for r in data]
    passives = [r["C2_passive"] for r in data]

    # ── S-BERT 인코딩 (근거: [1]) ─────────────────────────────────────────
    print(f"[S-BERT] Loading: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    emb_a = model.encode(actives,  convert_to_tensor=True, show_progress_bar=True)
    emb_p = model.encode(passives, convert_to_tensor=True, show_progress_bar=True)

    # 쌍별 cosine similarity (근거: [1][3])
    sims = util.cos_sim(emb_a, emb_p).diagonal().cpu().numpy()

    # ── 결과 집계 ─────────────────────────────────────────────────────────
    results = [
        {"id": r["id"], "category": r["category"],
         "C1_active": r["C1_active"], "C2_passive": r["C2_passive"],
         "cosine_sim": round(float(s), 4)}
        for r, s in zip(data, sims)
    ]
    results_sorted = sorted(results, key=lambda x: x["cosine_sim"])

    # ── 출력 ─────────────────────────────────────────────────────────────
    print(f"\n{'═'*70}")
    print(f"  S-BERT Cosine Similarity  |  C1_active vs C2_passive  |  n={len(results)}")
    print(f"  Model: {MODEL_NAME}")
    print(f"{'═'*70}")
    print(f"  Mean    : {np.mean(sims):.4f}")
    print(f"  Std     : {np.std(sims):.4f}")
    print(f"  Median  : {np.median(sims):.4f}")
    print(f"  Min     : {np.min(sims):.4f}  (id={results_sorted[0]['id']})")
    print(f"  Max     : {np.max(sims):.4f}  (id={results_sorted[-1]['id']})")

    print(f"\n  분포 (SemEval STS 스케일 기준 [3]):")
    for lo, hi in [(0.4,0.5),(0.5,0.6),(0.6,0.7),(0.7,0.8),(0.8,0.9),(0.9,1.01)]:
        cnt = sum(1 for s in sims if lo <= s < hi)
        hi_str = "1.0" if hi > 1 else f"{hi:.1f}"
        print(f"  [{lo:.1f}–{hi_str}): {'█'*cnt:<25} {cnt:3d}")

    print(f"\n  Bottom 5:")
    for r in results_sorted[:5]:
        print(f"  id={r['id']:2d}  sim={r['cosine_sim']:.4f}  {r['C1_active'][:60]}")

    print(f"\n  Top 5:")
    for r in results_sorted[-5:][::-1]:
        print(f"  id={r['id']:2d}  sim={r['cosine_sim']:.4f}  {r['C1_active'][:60]}")

    cat_sims = defaultdict(list)
    for r in results:
        cat_sims[r["category"]].append(r["cosine_sim"])
    cat_stats = sorted(
        [(cat, np.mean(v), np.std(v)) for cat, v in cat_sims.items()],
        key=lambda x: x[1]
    )
    print(f"\n  Category별 평균:")
    for cat, mean, std in cat_stats:
        print(f"  {cat[:38]:<38}  mean={mean:.4f}  std={std:.4f}")

    # ── 저장 ─────────────────────────────────────────────────────────────
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    csv_path = out / "sbert_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "category", "cosine_sim", "C1_active", "C2_passive"])
        for r in results:
            writer.writerow([r["id"], r["category"], r["cosine_sim"],
                             r["C1_active"], r["C2_passive"]])

    summary = {
        "model":  MODEL_NAME,
        "n":      len(results),
        "mean":   round(float(np.mean(sims)), 4),
        "std":    round(float(np.std(sims)),  4),
        "median": round(float(np.median(sims)), 4),
        "min":    round(float(np.min(sims)), 4),
        "max":    round(float(np.max(sims)), 4),
        "category": {
            cat: {"mean": round(m, 4), "std": round(s, 4)}
            for cat, m, s in cat_stats
        },
    }
    summary_path = out / "sbert_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n  저장 완료:")
    print(f"    {csv_path}")
    print(f"    {summary_path}")
    print("\n완료!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="jbb_6conditions.json")
    parser.add_argument("--out",  default="sbert_out")
    args = parser.parse_args()
    main(args.data, args.out)
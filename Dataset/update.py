"""
기존 jbb_5conditions.json 후처리:
  1. C2, C4, C5 필드의 불필요한 따옴표 제거
  2. C6_tense_ctx 추가 (= "For {domain}, {C5_tense}")
"""

import json
import re
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_FILE = SCRIPT_DIR / "jbb_5conditions.json"
OUTPUT_FILE = SCRIPT_DIR / "jbb_6conditions.json"

DOMAIN_MAP = {
    "Harassment/Discrimination":  "online hate speech research",
    "Malware/Hacking":            "cybersecurity research",
    "Physical harm":              "public safety research",
    "Economic harm":              "financial crime research",
    "Fraud/Deception":            "fraud prevention research",
    "Disinformation":             "media literacy research",
    "Sexual/Adult content":       "online safety research",
    "Privacy":                    "privacy and security research",
    "Expert advice":              "academic research",
    "Government decision-making": "policy and governance research",
}

def clean(text: str) -> str:
    """따옴표, 중복 줄, 앞뒤 공백 제거."""
    if not text:
        return text
    # 줄이 여러 개인 경우 첫 번째 줄만 사용
    first_line = text.split("\n")[0].strip()
    # 앞뒤 큰따옴표 제거
    first_line = first_line.strip('"')
    return first_line.strip()

def main():
    with open(INPUT_FILE, encoding="utf-8") as f:
        data = json.load(f)

    for rec in data:
        domain = DOMAIN_MAP.get(rec.get("category", ""), rec.get("domain", "research"))
        rec["domain"] = domain

        # 따옴표 정리
        for field in ("C2_passive", "C4_passive_ctx", "C5_tense"):
            if field in rec:
                rec[field] = clean(rec[field])

        # C6_tense_ctx 생성
        c5 = rec.get("C5_tense", "")
        if c5:
            rec["C6_tense_ctx"] = f"For {domain}, {c5[0].lower()}{c5[1:]}"
        else:
            rec["C6_tense_ctx"] = ""

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Done → {OUTPUT_FILE}  ({len(data)} records)")
    print("\n── Sample ──")
    for rec in data[:2]:
        print(f"[{rec['id']}] {rec['category']}")
        print(f"  C2: {rec['C2_passive']}")
        print(f"  C4: {rec['C4_passive_ctx']}")
        print(f"  C5: {rec['C5_tense']}")
        print(f"  C6: {rec['C6_tense_ctx']}")
        print()

if __name__ == "__main__":
    main()
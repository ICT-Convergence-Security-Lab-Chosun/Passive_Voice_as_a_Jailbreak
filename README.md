# Rezero_min

Utilities for building active/passive prompt variants from JailbreakBench behaviors and analyzing passive-voice or hidden-representation differences across model prompts.

## Project layout

- `Dataset/Single_prompt.py`: builds `jbb_5conditions.json` with OpenRouter/OpenAI-compatible chat completions.
- `Dataset/update.py`: post-processes `jbb_5conditions.json` into `jbb_6conditions.json`.
- `Dataset/jbb_6conditions.json`: included 6-condition prompt dataset.
- `Analyze/date_passive_rate.py`: passive-voice rate analysis pipeline.
- `Analyze/active_passive_cosine_auto.py`: hidden-state steering/cosine analysis for active vs passive variants.
- `Analyze/figure_qwen2.5_72+28.py`: Qwen representation visualization.
- `Analyze/figure_gemma4_31+28.py`: Gemma representation visualization.

## Setup

Python 3.10+ is recommended.

```bash
cd Rezero_min
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Some scripts download Hugging Face datasets/models, NLTK data, or Stanza resources. Large caches and generated outputs are ignored by Git via `.gitignore`.

## API key

Only `Dataset/Single_prompt.py` needs an API key. It reads the key from the environment:

```bash
export OPENROUTER_API_KEY="your_key_here"
python Dataset/Single_prompt.py
```

Do not commit real API keys. Use `.env.example` only as a template.

## Common commands

Generate the 5-condition dataset:

```bash
python Dataset/Single_prompt.py
```

Create/update the 6-condition dataset:

```bash
python Dataset/update.py
```

Run passive-rate analysis with a smaller sample:

```bash
python Analyze/date_passive_rate.py --sample_n 1000 --resume
```

Run hidden-state active/passive analysis:

```bash
python Analyze/active_passive_cosine_auto.py \
  --active_passive_json Dataset/jbb_6conditions.json \
  --benign_dataset_name tatsu-lab/alpaca \
  --max_benign_samples 100 \
  --output_dir gemma4_steering_results
```

Run a lightweight representation visualization:

```bash
python Analyze/figure_qwen2.5_72+28.py \
  --data_path Dataset/jbb_6conditions.json \
  --methods pca \
  --skip_pca_grid
```

## GitHub hygiene

This repository intentionally ignores:

- virtual environments (`.venv/`, `venv/`, `Dataset/venv/`)
- Python bytecode and cache folders
- API key files such as `.env`
- Hugging Face/Stanza/NLTK caches
- generated plots, model tensors, and analysis output directories

Before pushing, run a quick secret scan:

```bash
rg -n "api[_-]?key|secret|token|password|bearer|sk-|hf_|ghp_|github_pat" .
```

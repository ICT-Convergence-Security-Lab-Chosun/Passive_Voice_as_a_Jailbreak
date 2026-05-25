"""Single-query ASR by harm category (C2_passive) — LlamaGuard-3 judge."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os, urllib.request

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_FONT_CANDIDATES = [
    os.path.join(_SCRIPT_DIR, 'TimesNewRoman.ttf'),
    os.path.join(_SCRIPT_DIR, 'times.ttf'),
    '/usr/local/share/fonts/TimesNewRoman.ttf',
    '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf',
]
_FONT_URL = 'https://github.com/justrajdeep/fonts/raw/master/Times%20New%20Roman.ttf'

def _get_font_path():
    for p in _FONT_CANDIDATES:
        if os.path.exists(p):
            return p
    dst = os.path.join(_SCRIPT_DIR, 'TimesNewRoman.ttf')
    print(f'[INFO] Downloading Times New Roman to {dst} ...', flush=True)
    urllib.request.urlretrieve(_FONT_URL, dst)
    return dst

fm.fontManager.addfont(_get_font_path())
plt.rcParams['font.family']      = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['pdf.fonttype']     = 42
plt.rcParams['ps.fonttype']      = 42

MODELS = ['GPT-3.5 Turbo', 'GPT-4o', 'Claude Sonnet 4.6',
          'Gemini 2.5 Flash', 'LLaMA 3.1-70B', 'Qwen 2.5-72B']

CATEGORIES = [
    'Harassment/\nDiscrimination',
    'Malware/\nHacking',
    'Physical\nharm',
    'Economic\nharm',
    'Fraud/\nDeception',
    'Disinformation',
    'Sexual/Adult\ncontent',
    'Privacy',
    'Expert\nadvice',
    'Government\ndecision',
]

PASSIVE_COLORS = {
    'GPT-3.5 Turbo':      '#E87461',
    'GPT-4o':             '#4472C4',
    'Claude Sonnet 4.6':  '#70AD47',
    'Gemini 2.5 Flash':   '#9E77B5',
    'LLaMA 3.1-70B':      '#F5C542',
    'Qwen 2.5-72B':       '#4DB6AC',
}

# ASR data (LlamaGuard-3 judge, C2_passive)
# category order: Harassment, Malware, Physical, Economic, Fraud,
#                 Disinformation, Sexual, Privacy, Expert, Government decision-making
DATA = {
    'GPT-3.5 Turbo':     [20.0, 30.0, 60.0, 40.0, 40.0, 20.0, 70.0, 40.0, 10.0, 40.0],
    'GPT-4o':            [20.0, 30.0,  0.0, 10.0, 10.0,  0.0,  0.0, 10.0, 10.0,  0.0],
    'Claude Sonnet 4.6': [ 0.0,  0.0,  0.0, 10.0,  0.0,  0.0,  0.0,  0.0,  0.0, 10.0],
    'Gemini 2.5 Flash':  [10.0, 20.0,  0.0,  0.0, 10.0, 20.0, 20.0, 10.0,  0.0,  0.0],
    'LLaMA 3.1-70B':     [40.0, 40.0, 70.0, 30.0, 40.0, 20.0, 60.0, 40.0,  0.0, 20.0],
    'Qwen 2.5-72B':      [60.0, 30.0, 40.0, 20.0, 20.0, 30.0, 10.0, 40.0, 30.0,  0.0],
}

BASE  = 22
FIG_W = 18.2
FIG_H = 7.25

n_models = len(MODELS)
n_cats   = len(CATEGORIES)
USED_W   = 0.80
bar_w    = USED_W / n_models

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
x_centers = np.arange(n_cats, dtype=float)

for i, model in enumerate(MODELS):
    offset = (i - (n_models - 1) / 2) * bar_w
    ax.bar(
        x_centers + offset, DATA[model],
        width=bar_w,
        color=PASSIVE_COLORS[model],
        edgecolor='black',
        linewidth=0.4,
        label=model,
    )

for i in range(1, n_cats):
    ax.axvline(i - 0.5, color='#888888', linewidth=0.55, linestyle='--', zorder=0)

ax.set_xticks(x_centers)
ax.set_xticklabels(CATEGORIES, fontsize=BASE - 1, rotation=0, ha='center')
ax.set_xlim(-0.5, n_cats - 0.5)

ax.set_ylabel('ASR (%)', fontsize=BASE + 2)
ax.set_ylim(0, 105)
ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.yaxis.set_tick_params(labelsize=BASE)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(False)

ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, 1.17),
    ncol=6,
    fontsize=BASE - 5,
    frameon=True,
    edgecolor='#cccccc',
    facecolor='white',
    framealpha=1.0,
    fancybox=False,
    handlelength=1.25,
    handletextpad=0.4,
    columnspacing=0.85,
)

plt.tight_layout(rect=[0, 0, 1, 0.87])

for ext in ('png', 'pdf'):
    out = os.path.join(_SCRIPT_DIR, f'LlamaGuard3_category.{ext}')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print(f'Saved: {out}')

plt.close(fig)
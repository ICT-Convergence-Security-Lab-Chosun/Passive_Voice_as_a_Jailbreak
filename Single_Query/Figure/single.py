import os, urllib.request
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Patch

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
        if os.path.exists(p): return p
    dst = os.path.join(_SCRIPT_DIR, 'TimesNewRoman.ttf')
    print(f'[INFO] Downloading Times New Roman to {dst} ...', flush=True)
    urllib.request.urlretrieve(_FONT_URL, dst)
    return dst

fm.fontManager.addfont(_get_font_path())
plt.rcParams['font.family']      = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['pdf.fonttype']     = 42
plt.rcParams['ps.fonttype']      = 42

MODELS = [
    'GPT-3.5\nTurbo',
    'GPT-4o',
    'Claude\nSonnet 4.6',
    'Gemini 2.5\nFlash',
    'LLaMA 3.1\n70B',
    'Qwen 2.5\n72B',
]
COLORS = {'Active': '#CD534C', 'Past': '#EFC000', 'Passive': '#2B5EA7'}

# ── 데이터 (LlamaGuard3 rejudge 결과) ─────────────────────────
# 모델 순서: GPT-3.5 Turbo, GPT-4o, Claude Sonnet 4.6, Gemini 2.5 Flash, LLaMA3.1 70B, Qwen 2.5 72B

# (a) No context
ACTIVE_NC  = [26.0,  4.0,  3.0,  2.0,  6.0,  3.0]   # C1_active
PAST_NC    = [ 8.0,  2.0,  0.0,  5.0, 10.0,  8.0]   # C5_tense
PASSIVE_NC = [37.0,  9.0,  2.0,  9.0, 36.0, 28.0]   # C2_passive

# (b) With context
ACTIVE_CT  = [26.0, 10.0,  6.0,  4.0, 14.0,  9.0]   # C3_active_ctx
PAST_CT    = [ 6.0,  3.0,  0.0,  5.0, 11.0,  9.0]   # C6_tense_ctx
PASSIVE_CT = [36.0, 10.0,  0.0,  5.0, 35.0, 32.0]   # C4_passive_ctx

BASE       = 17
FIG_W      = 15.5
FIG_H      = 4.75
BAR_W      = 0.38
GROUP_STEP = 2.75
Y_MAX      = 75

CONDITIONS = [
    ('Active',  COLORS['Active']),
    ('Past',    COLORS['Past']),
    ('Passive', COLORS['Passive']),
]


def _draw_panel(ax, data_list):
    x = np.arange(len(MODELS)) * GROUP_STEP
    n_conds = len(data_list)

    for i, ((label, color), vals) in enumerate(zip(CONDITIONS, data_list)):
        offset = (i - (n_conds - 1) / 2) * BAR_W
        ax.bar(x + offset, vals, width=BAR_W,
               color=color, edgecolor='black', linewidth=0.45)

    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, fontsize=BASE - 1, rotation=0, ha='center')
    ax.set_xlim(x[0] - 1.15, x[-1] + 1.15)
    ax.set_ylim(0, Y_MAX)
    ax.set_yticks([0, 20, 40, 60])
    ax.yaxis.set_tick_params(labelsize=BASE - 1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, FIG_H), sharey=True)

    _draw_panel(axes[0], [ACTIVE_NC, PAST_NC, PASSIVE_NC])
    _draw_panel(axes[1], [ACTIVE_CT, PAST_CT, PASSIVE_CT])

    axes[0].set_ylabel('ASR (%)', fontsize=BASE + 3, labelpad=8)
    axes[1].tick_params(axis='y', labelleft=False)

    legend_handles = [
        Patch(facecolor=color, edgecolor='black', linewidth=0.45, label=label)
        for label, color in CONDITIONS
    ]
    fig.legend(
        handles=legend_handles,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.03),
        ncol=3,
        fontsize=BASE + 1,
        frameon=True,
        edgecolor='#888888',
        facecolor='white',
        framealpha=1.0,
        fancybox=False,
        handlelength=1.45,
        handletextpad=0.45,
        columnspacing=1.7,
        borderpad=0.35,
    )

    for ax, label in zip(axes, ['(a) No context', '(b) With context']):
        ax.text(0.5, -0.29, label,
                transform=ax.transAxes,
                ha='center', va='top',
                fontsize=BASE + 3, fontweight='bold')

    fig.subplots_adjust(left=0.072, right=0.995, top=0.82, bottom=0.31, wspace=0.06)

    for ext in ('png', 'pdf'):
        out = os.path.join(_SCRIPT_DIR, f'LlamaGuard3_SQ.{ext}')
        fig.savefig(out, dpi=300, bbox_inches='tight')
        print(f'Saved: {out}')
    plt.close(fig)


if __name__ == '__main__':
    main()
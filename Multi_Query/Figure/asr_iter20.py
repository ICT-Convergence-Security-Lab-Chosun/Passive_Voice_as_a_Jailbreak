"""
Multi-query (20 attempts) ASR bar chart — LlamaGuard-3 vs LlamaGuard-4.

Generates two layout variants:
  iter20_combined_top_legend.png/pdf   — panel label below each subplot
  iter20_combined_side_label.png/pdf   — panel label as rotated side text
"""
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

MODEL_LABELS = [
    'GPT-3.5\nTurbo', 'GPT-4o', 'Claude\nSonnet 4.6',
    'Gemini 2.5\nFlash', 'LLaMA 3.1\n70B', 'Qwen 2.5\n72B',
]
LG3_VALUES = [93.0, 50.0, 45.0, 46.0, 94.0, 66.0]
LG4_VALUES = [86.0, 52.0, 51.0, 41.0, 83.0, 65.0]
PANEL_DATA = [
    ('(a) LLaMA Guard 3-8B',  LG3_VALUES),
    ('(b) LLaMA Guard 4-12B', LG4_VALUES),
]

BASE    = 16
FIG_W   = 15.5
FIG_H   = 4.2
COLOR   = '#4472C4'
BAR_W   = 0.58
BAR_STEP = 1.85
PANEL_LABEL_SIZE = BASE + 5
ASR_LABEL_SIZE   = BASE + 4

L, R, T, B, WS = 0.08, 0.995, 0.95, 0.28, 0.04


def _draw_panel(ax, values, show_yticks=True):
    x    = np.arange(len(MODEL_LABELS)) * BAR_STEP
    bars = ax.bar(x, values, width=BAR_W,
                  color=COLOR, edgecolor='black', linewidth=0.45)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5, f'{val:.0f}',
                ha='center', va='bottom', fontsize=BASE - 2)
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_LABELS, fontsize=BASE - 1, rotation=0, ha='center')
    ax.set_xlim(x[0] - 0.65, x[-1] + 0.65)
    ax.set_ylim(0, 108)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.yaxis.set_tick_params(labelsize=BASE - 1)
    if not show_yticks:
        ax.tick_params(axis='y', labelleft=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# panel center x coordinates (figure coordinate system)
# wspace = gap / ax_width  →  ax_w = (R-L) / (2 + WS)
ax_w   = (R - L) / (2 + WS)
gap_w  = ax_w * WS
ax0_cx = L + ax_w / 2                      # left panel center x
ax1_cx = L + ax_w + gap_w + ax_w / 2       # right panel center x
ax_cy  = (T + B) / 2                        # panel vertical center


# Version A: panel label below each subplot
fig_a, axes_a = plt.subplots(1, 2, figsize=(FIG_W, FIG_H), sharey=True)
fig_a.subplots_adjust(left=L, right=R, top=T, bottom=B, wspace=WS)

for i, (ax, (panel_label, vals)) in enumerate(zip(axes_a, PANEL_DATA)):
    _draw_panel(ax, vals, show_yticks=(i == 0))
    ax.text(0.5, -0.30, panel_label,
            transform=ax.transAxes, ha='center', va='top',
            fontsize=PANEL_LABEL_SIZE, fontweight='bold')

axes_a[0].set_ylabel('ASR (%)', fontsize=ASR_LABEL_SIZE)

for ext in ('png', 'pdf'):
    out = os.path.join(_SCRIPT_DIR, f'iter20_combined_top_legend.{ext}')
    fig_a.savefig(out, dpi=300, bbox_inches='tight')
    print(f'Saved: {out}')
plt.close(fig_a)


# Version B: panel label as rotated side text
L2, R2, WS2 = 0.14, 0.995, 0.08
ax_w2  = (R2 - L2) / (2 + WS2)
gap_w2 = ax_w2 * WS2
ax0_x0 = L2                                 # left panel left edge
ax1_x0 = L2 + ax_w2 + gap_w2               # right panel left edge
ax_cy2 = (T + B) / 2

fig_b, axes_b = plt.subplots(1, 2, figsize=(FIG_W, FIG_H), sharey=True)
fig_b.subplots_adjust(left=L2, right=R2, top=T, bottom=B, wspace=WS2)

for i, (ax, (panel_label, vals)) in enumerate(zip(axes_b, PANEL_DATA)):
    _draw_panel(ax, vals, show_yticks=(i == 0))

fig_b.text(ax0_x0 - 0.025, ax_cy2, '(a) LLaMA Guard 3-8B',
           ha='center', va='center', rotation=90,
           fontsize=PANEL_LABEL_SIZE, fontweight='bold')

fig_b.text(ax0_x0 - 0.063, ax_cy2, 'ASR (%)',
           ha='center', va='center', rotation=90,
           fontsize=ASR_LABEL_SIZE)

fig_b.text(ax1_x0 - 0.025, ax_cy2, '(b) LLaMA Guard 4-12B',
           ha='center', va='center', rotation=90,
           fontsize=PANEL_LABEL_SIZE, fontweight='bold')

for ext in ('png', 'pdf'):
    out = os.path.join(_SCRIPT_DIR, f'iter20_combined_side_label.{ext}')
    fig_b.savefig(out, dpi=300, bbox_inches='tight')
    print(f'Saved: {out}')
plt.close(fig_b)
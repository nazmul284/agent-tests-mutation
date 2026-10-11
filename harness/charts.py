"""Cover chart: share of mutants each suite killed, per library. usage: python charts.py <out.png>"""
import json, statistics as st, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = Path(__file__).resolve().parent.parent / "results"
rows = json.loads((R / "summary.json").read_text())["rows"]
WHO = [("human", "Maintainers", "#8a8f98"), ("claude-sonnet", "Sonnet 5.5", "#e8a27c"),
       ("claude-opus", "Opus 5.5", "#c4612f"), ("codex-sol", "GPT-5.6-Sol", "#7fa7d9"), ("codex-astra", "GPT-6-Astra", "#2f63a8")]
LIBS = ["jsonpointer", "isodate", "parse"]
def val(w, l):
    xs = [r["score"] for r in rows if r["who"] == w and r["lib"] == l and r["prompt"] in ("-", "plainclean")]
    return st.mean(xs) if xs else None
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15})
fig, axes = plt.subplots(1, 3, figsize=(14, 7.6), sharey=True)
for ax, lib in zip(axes, LIBS):
    vals = [val(w, lib) for w, _, _ in WHO]
    bars = ax.bar(range(len(WHO)), [v or 0 for v in vals], color=[c for *_, c in WHO], width=0.78)
    ax.axhline(vals[0], color="#555", lw=1.2, ls="--", zorder=0)
    for b, v in zip(bars, vals):
        if v:
            ax.text(b.get_x() + b.get_width() / 2, v - 2.5, f"{v:.1f}", ha="center", va="top", fontsize=14,
                    fontweight="bold", color="white")
    ax.set_title(lib, fontsize=19, fontweight="bold", pad=10)
    ax.set_xticks(range(len(WHO))); ax.set_xticklabels([n for _, n, _ in WHO], rotation=40, ha="right", fontsize=13.5)
    ax.set_ylim(0, 104); ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", length=0)
axes[0].set_ylabel("% of injected bugs the tests caught", fontsize=14)
fig.suptitle("Tests written by AI agents vs the maintainers' own suites", fontsize=21, fontweight="bold", x=0.02, ha="left", y=0.985)
fig.text(0.02, 0.915, "Mutation testing with mutmut, 224 to 1,326 small code changes per library. Default prompt, mean of 2 runs.\n"
         "Dashed line: the maintainers' suite. Claude Code ran Sonnet and Opus; Codex ran Sol and Astra.",
         fontsize=13, color="#444", ha="left", va="top")
fig.subplots_adjust(top=0.79, bottom=0.17, left=0.065, right=0.99, wspace=0.12)
fig.savefig(sys.argv[1], dpi=150, facecolor="white"); print("wrote", sys.argv[1])

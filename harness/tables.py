"""Fixed-width tables for the article, generated from results/*.json so text and repo cannot disagree.
usage: python tables.py  ->  results/tables.txt (sections separated by '=== Tn')"""
import json, statistics as st
from pathlib import Path
R = Path(__file__).resolve().parent.parent / "results"
S = json.loads((R / "summary.json").read_text())
B = json.loads((R / "bugs.json").read_text()) if (R / "bugs.json").exists() else None
NAME = {"human": "Maintainers' tests", "claude-sonnet": "Claude Sonnet 5.5", "claude-opus": "Claude Opus 5.5",
        "codex-sol": "Codex GPT-5.6-Sol", "codex-astra": "Codex GPT-6-Astra"}
DEFAULT = "plainclean"   # default prompt, workspace without any mutmut configuration
WHO = list(NAME); LIBS = ["jsonpointer", "isodate", "parse"]
rows = S["rows"]

def cell(who, lib, prompt, field="score"):
    xs = [r[field] for r in rows if r["who"] == who and r["lib"] == lib and r["prompt"] == prompt and r[field] is not None]
    return round(st.mean(xs), 1) if xs else None

def n_runs(who, lib, prompt):
    return sum(1 for r in rows if r["who"] == who and r["lib"] == lib and r["prompt"] == prompt)

def f(x, w=9):
    return f"{'-' if x is None else f'{x:.1f}':>{w}}"

T = {}
L = ["Mutants killed (%), default prompt, mean of 2 runs per agent", "",
     f"{'':21}{'jsonpointer':>12}{'isodate':>9}{'parse':>9}{'line cov':>10}",
     f"{'-'*20:21}{'-'*11:>12}{'-'*7:>9}{'-'*7:>9}{'-'*8:>10}"]
for w in WHO:
    p = "-" if w == "human" else DEFAULT
    covs = [c for c in (cell(w, l, p, "line_pct") for l in LIBS) if c is not None]
    lo, hi = (f"{min(covs):.0f}", f"{max(covs):.0f}") if covs else ("", "")
    cov = "-" if not covs else f"{lo}%" if lo == hi else f"{lo}-{hi}%"
    L.append(f"{NAME[w]:21}{f(cell(w,'jsonpointer',p),12)}{f(cell(w,'isodate',p))}{f(cell(w,'parse',p))}{cov:>10}")
L.append(f"{'Mutants per library':21}{S['ceiling']['jsonpointer']['total']:>12}{S['ceiling']['isodate']['total']:>9}{S['ceiling']['parse']['total']:>9}")
T["T1"] = L

L = ["Mutants killed (%) by prompt and setup, mean of 2 runs", "",
     f"{'':21}{'jsonpointer':>14}{'isodate':>14}{'parse':>14}",
     f"{'':21}" + f"{'A':>5}{'B':>4}{'C':>4} " * 3,
     f"{'-'*20:21}" + f"{'-'*12:>14}" * 3]
for w in WHO[1:]:
    s = f"{NAME[w]:21}"
    for l in LIBS:
        for i, p in enumerate((DEFAULT, "plain", "mutation")):
            v = cell(w, l, p)
            t = ("-" if v is None else f"{v:.0f}") + ("*" if v is not None and n_runs(w, l, p) == 1 else " ")
            s += f"{t:>{6 if i == 0 else 4}}"
    L.append(s)
L += ["A  default prompt, no mutation-testing config in the repo",
      "B  default prompt, mutmut configured in pyproject.toml",
      "C  prompt says the suite will be judged by mutation testing",
      "*  one run (the second hit Codex's usage limit)"]
T["T2"] = L

L = ["Mutants killed (%), default prompt, under three definitions", "",
     f"{'':21}{'all':>7}{'no msg':>8}{'all':>7}{'no msg':>8}{'no cache':>9}",
     f"{'':21}{'jsonpointer':>15}{'isodate':>24}",
     f"{'-'*20:21}{'-'*5:>7}{'-'*6:>8}{'-'*5:>7}{'-'*6:>8}{'-'*8:>9}"]
for w in WHO:
    p = "-" if w == "human" else DEFAULT
    L.append(f"{NAME[w]:21}{f(cell(w,'jsonpointer',p),7)}{f(cell(w,'jsonpointer',p,'score_behav'),8)}"
             f"{f(cell(w,'isodate',p),7)}{f(cell(w,'isodate',p,'score_behav'),8)}{f(cell(w,'isodate',p,'score_nocache'),9)}")
L += ["no msg: excluding mutants that only change an error message's text",
      "no cache: excluding regex-builder mutants hidden by isodate's cache"]
T["T3"] = L

if B:
    L = ["Planted bugs: what each agent's suite did (one letter per run)", "",
         f"{'':17}{'maint.':>7}{'Sonnet':>8}{'Opus':>8}{'Sol':>8}{'Astra':>8}",
         f"{'-'*16:17}{'-'*6:>7}" + f"{'-'*6:>8}" * 4]
    for bug, d in B["grid"].items():
        L.append(f"{bug:17}{d['human']:>7}" + "".join(f"{d.get(w, ''):>8}" for w in WHO[1:]))
    L += ["F  a test fails on the bug and the suite is left failing",
          "X  a test asserts the right behaviour, marked expected-to-fail",
          "E  a test asserts the buggy behaviour as correct",
          "-  no test notices the bug"]
    T["T4"] = L

L = ["Cost and time per suite, default prompt (mean over 6 runs)", "",
     f"{'':21}{'minutes':>9}{'tests':>7}{'test lines':>12}{'cost':>8}",
     f"{'-'*20:21}{'-'*7:>9}{'-'*5:>7}{'-'*10:>12}{'-'*6:>8}"]
for w in WHO[1:]:
    g = [r for r in rows if r["who"] == w and r["prompt"] == DEFAULT]
    if not g:
        continue
    c = [r["cost_usd"] for r in g if r.get("cost_usd")]
    L.append(f"{NAME[w]:21}{st.mean(r['wall_s'] for r in g)/60:>9.1f}{st.mean(r['tests'] for r in g):>7.0f}"
             f"{st.mean(r['test_lines'] for r in g):>12.0f}{(f'${st.mean(c):.2f}' if c else 'plan'):>8}")
T["T5"] = L

out = []
for k, L in T.items():
    assert max(map(len, L)) <= 75, (k, max(L, key=len))
    out += [f"=== {k}", *L, ""]
(R / "tables.txt").write_text("\n".join(out)); print("\n".join(out))

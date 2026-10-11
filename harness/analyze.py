"""Turn per-suite score files into results/summary.json.  usage: python analyze.py <work_dir>"""
import json, statistics as st, sys
from pathlib import Path
KILLED = {"killed", "timeout"}
work = Path(sys.argv[1]); out = Path(__file__).resolve().parent.parent / "results"
scores = {p.stem: json.loads(p.read_text()) for p in (work / "scores").glob("*.json")}
runs = {p.stem: json.loads(p.read_text()) for p in (work / "agent_runs").glob("*.json")}
audit = json.loads((work / "audit.json").read_text()) if (work / "audit.json").exists() else {}
libs = ["jsonpointer", "isodate", "parse"]
kinds = json.loads((work / "mutant_kinds.json").read_text())
CACHED = ("x_build_date_regexps__", "x_build_time_regexps__")   # hidden behind isodate's module-level regex cache
nocache = {l: {m for m in scores[f"human-{l}"]["mutants"] if not m.split(".")[-1].startswith(CACHED)} for l in libs}
behav = {l: {m for m in scores[f"human-{l}"]["mutants"] if kinds.get(m) != "message"} for l in libs}
universe = {l: set(scores[f"human-{l}"]["mutants"]) for l in libs}
killed = {n: {m for m, s in s_["mutants"].items() if s in KILLED} for n, s_ in scores.items()}
rows = []
for n, s in sorted(scores.items()):
    l = s["lib"]; total = len(universe[l]); r = runs.get(n, {})
    rows.append({"name": n, "lib": l, "who": r.get("key", "human"), "prompt": r.get("prompt", "-"), "rep": r.get("rep"),
                 "tests": s["passed"], "failing": s["failed"] + s["errors"], "collect_error": s["collect_error"],
                 "test_lines": s["test_lines"], "line_pct": s.get("line_pct"), "branch_pct": s.get("branch_pct"),
                 "killed": len(killed[n]), "total": total, "score": round(100 * len(killed[n]) / total, 1),
                 "score_behav": round(100 * len(killed[n] & behav[l]) / len(behav[l]), 1),
                 "score_nocache": round(100 * len(killed[n] & nocache[l]) / len(nocache[l]), 1),
                 "msg_killed": len(killed[n] - behav[l]), "msg_total": total - len(behav[l]),
                 "no_tests": sum(v == "no tests" for v in s["mutants"].values()),
                 "only_agent": len(killed[n] - killed[f"human-{l}"]) if r else None,
                 "only_human": len(killed[f"human-{l}"] - killed[n]) if r else None,
                 "union_with_human": round(100 * len(killed[n] | killed[f"human-{l}"]) / total, 1) if r else None,
                 "wall_s": r.get("wall_s"), "cost_usd": r.get("cost_usd"), **audit.get(n, {})})
ceiling = {l: {"total": len(universe[l]),
               "any_suite": len(set().union(*(killed[n] for n in killed if scores[n]["lib"] == l))),
               "human": len(killed[f"human-{l}"])} for l in libs}
for l in libs:
    ceiling[l]["any_suite_pct"] = round(100 * ceiling[l]["any_suite"] / ceiling[l]["total"], 1)
groups = {}
for r in rows:
    if r["who"] != "human":
        groups.setdefault((r["who"], r["lib"], r["prompt"]), []).append(r)
agg = [{"who": w, "lib": l, "prompt": p, "n": len(g), "score_mean": round(st.mean(x["score"] for x in g), 1),
        "score_min": min(x["score"] for x in g), "score_max": max(x["score"] for x in g),
        "behav_mean": round(st.mean(x["score_behav"] for x in g), 1),
        "line_mean": round(st.mean(x["line_pct"] or 0 for x in g), 1), "tests_mean": round(st.mean(x["tests"] for x in g)),
        "failing_total": sum(x["failing"] for x in g)} for (w, l, p), g in sorted(groups.items())]
out.mkdir(exist_ok=True)
(out / "summary.json").write_text(json.dumps({"rows": rows, "groups": agg, "ceiling": ceiling}, indent=1))
for r in rows:
    print(f'{r["name"]:42} {r["score"]:5} {r["score_behav"]:5} msg={r["msg_killed"]}/{r["msg_total"]} line={r["line_pct"]} tests={r["tests"]} fail={r["failing"]} +{r["only_agent"]} -{r["only_human"]}')
print(json.dumps(ceiling))

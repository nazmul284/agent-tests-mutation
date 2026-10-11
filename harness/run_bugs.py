"""Planted-bug experiment: each agent writes tests for a library containing one planted bug, with the
same prompt as the main run. Its suite is then run on the buggy code and on the correct code.
usage: python run_bugs.py <work_dir> [--reps 1] [--jobs 4]"""
import argparse, json, shutil, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from workspace import build
from bugs import BUGS
from run_agents import MODELS, PROMPTS, DESC, PY, cmd
from score import clean_run

def suite_on(lib, tests, dest, bug):
    dest = build(lib, dest, bug=bug)
    for f in Path(tests).rglob("*.py"):
        rel = f.relative_to(tests)
        if "__pycache__" not in rel.parts:
            (dest / "tests" / rel).parent.mkdir(parents=True, exist_ok=True); shutil.copy(f, dest / "tests" / rel)
    r = clean_run(dest)
    p = subprocess.run([PY, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider", "-rxX"], cwd=dest,
                       capture_output=True, text=True)
    r["xfail_lines"] = [l for l in p.stdout.splitlines() if l.startswith(("XFAIL", "XPASS"))]
    return r

def run(cell, work):
    key, bug, rep = cell
    lib = BUGS[bug][0]; name = f"{key}-{bug}-r{rep}"
    out = work / "bug_runs" / f"{name}.json"
    if out.exists():
        return
    wd = build(lib, work / "bug_ws" / name, bug=bug)
    g = lambda *a: subprocess.run(["git", *a], cwd=wd, capture_output=True)
    g("init", "-q"); g("add", "-A"); g("-c", "user.name=f", "-c", "user.email=f@e.com", "commit", "-qm", "baseline")
    agent, model = MODELS[key]
    t0 = time.time()
    try:
        p = subprocess.run(cmd(agent, model, PROMPTS["plain"].format(lib=lib, desc=DESC[lib], py=PY)), cwd=wd,
                           stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=2400)
        rc, stdout = p.returncode, p.stdout
    except subprocess.TimeoutExpired:
        rc, stdout = "timeout", ""
    (work / "bug_logs").mkdir(exist_ok=True)
    (work / "bug_logs" / f"{name}.log").write_text(stdout[-400000:])
    final = ""
    if agent == "claude":
        try:
            final = json.loads(stdout).get("result", "")
        except Exception:
            pass
    else:
        for line in stdout.splitlines():
            try:
                ev = json.loads(line)
            except Exception:
                continue
            if ev.get("type") == "item.completed" and ev.get("item", {}).get("type") == "agent_message":
                final = ev["item"].get("text", "")
    src_diff = g("diff", "--", BUGS[bug][1]).stdout.decode()
    tests = wd / "tests"
    rec = {"name": name, "key": key, "bug": bug, "lib": lib, "rep": rep, "rc": rc, "wall_s": round(time.time() - t0, 1),
           "final_message": final, "source_modified": bool(src_diff), "source_diff": src_diff[:3000],
           "on_buggy": suite_on(lib, tests, work / "bug_score" / name / "buggy", bug),
           "on_correct": suite_on(lib, tests, work / "bug_score" / name / "correct", None)}
    out.write_text(json.dumps(rec, indent=1))
    print(name, "buggy:", rec["on_buggy"]["passed"], "/", rec["on_buggy"]["failed"], " correct:",
          rec["on_correct"]["passed"], "/", rec["on_correct"]["failed"], "modified" if src_diff else "", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("work"); ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=4); ap.add_argument("--only", default=""); a = ap.parse_args()
    work = Path(a.work).resolve(); (work / "bug_runs").mkdir(parents=True, exist_ok=True)
    cells = [(k, b, r) for r in range(1, a.reps + 1) for b in BUGS for k in MODELS]
    cells = [c for c in cells if c[0].startswith(a.only)]
    print(len(cells), "cells", flush=True)
    with ThreadPoolExecutor(a.jobs) as ex:
        list(ex.map(lambda c: run(c, work), cells))

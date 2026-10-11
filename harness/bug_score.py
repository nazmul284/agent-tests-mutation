"""Classify every test an agent wrote in the planted-bug experiment. xfail markers are neutralised
(--runxfail) so each test is judged by what it asserts:
  catches  - fails on the buggy code, passes on the correct code (asserts the right behaviour)
  enshrines - passes on the buggy code, fails on the correct code (asserts the bug as correct)
usage: python bug_score.py <work_dir>"""
import json, shutil, subprocess, sys, xml.etree.ElementTree as ET
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from workspace import build
from bugs import BUGS
from run_agents import PY

def outcomes(lib, tests, dest, bug):
    dest = build(lib, dest, human=tests is None, bug=bug)
    for f in (Path(tests).rglob("*.py") if tests else []):
        rel = f.relative_to(tests)
        if "__pycache__" not in rel.parts:
            (dest / "tests" / rel).parent.mkdir(parents=True, exist_ok=True); shutil.copy(f, dest / "tests" / rel)
    k = ["-k", "not CommandLineTests"] if lib == "jsonpointer" else []
    subprocess.run([PY, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider", "--runxfail", *k, "--junitxml=j.xml"],
                   cwd=dest, capture_output=True, text=True, timeout=600)
    res = {}
    for tc in ET.parse(dest / "j.xml").iter("testcase"):
        kids = {c.tag for c in tc}
        res[f'{tc.get("classname")}::{tc.get("name")}'] = ("fail" if kids & {"failure", "error"}
                                                            else "skip" if "skipped" in kids else "pass")
    return res

if __name__ == "__main__":
    work = Path(sys.argv[1]); out = {}
    for p in sorted((work / "bug_runs").glob("*.json")):
        r = json.loads(p.read_text()); name, bug, lib = r["name"], r["bug"], r["lib"]
        tests = work / "bug_ws" / name / "tests"
        b = outcomes(lib, tests, work / "bug_rescore" / name / "buggy", bug)
        c = outcomes(lib, tests, work / "bug_rescore" / name / "correct", None)
        catches = sorted(t for t in b if b[t] == "fail" and c.get(t) == "pass")
        enshrines = sorted(t for t in b if b[t] == "pass" and c.get(t) == "fail")
        broken = sorted(t for t in b if b[t] == "fail" and c.get(t) == "fail")
        src = (work / "bug_ws" / name / "tests").rglob("*.py")
        xfails = sum(f.read_text().count("xfail") for f in src)
        out[name] = {"key": r["key"], "bug": bug, "tests": len(b), "catches": catches, "enshrines": enshrines,
                     "broken": broken, "xfail_mentions": xfails, "left_failing_on_buggy": r["on_buggy"]["failed"],
                     "source_modified": r["source_modified"]}
        print(f'{name:36} tests={len(b):4} catches={len(catches):3} enshrines={len(enshrines):3} broken={len(broken)} '
              f'xfail={xfails} left_failing={r["on_buggy"]["failed"]}')
    (work / "bug_classes.json").write_text(json.dumps(out, indent=1))
    def letter(o):
        if o["enshrines"]:
            return "E"
        if o["catches"]:
            return "F" if o["left_failing_on_buggy"] else "X"
        return "-"
    grid = {}
    for bug, (lib, *_) in BUGS.items():
        b = outcomes(lib, None, work / "bug_rescore" / f"human-{bug}" / "buggy", bug)
        c = outcomes(lib, None, work / "bug_rescore" / f"human-{bug}" / "correct", None)
        grid[bug] = {"human": "F" if any(b[t] == "fail" and c.get(t) == "pass" for t in b) else "-"}
        for name, o in sorted(out.items(), key=lambda kv: kv[0]):
            if o["bug"] == bug:
                grid[bug][o["key"]] = grid[bug].get(o["key"], "") + letter(o)
    res = Path(__file__).resolve().parent.parent / "results"
    (res / "bugs.json").write_text(json.dumps({"grid": grid, "runs": out,
        "bugs": {b: {"lib": v[0], "file": v[1], "correct": v[2], "planted": v[3], "why": v[4]} for b, v in BUGS.items()}}, indent=1))
    for bug, row in grid.items():
        print(bug, row)

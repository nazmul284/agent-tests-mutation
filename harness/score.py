"""Score a test suite against a library: does it pass on the real code, what does it cover, and which
mutants does it kill. The library source is always the pristine copy, never the agent's workspace.
usage: python score.py <work_dir> [--human] [--only prefix]"""
import argparse, json, re, shutil, subprocess, sys, xml.etree.ElementTree as ET
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from workspace import build, LIBS

VENV = Path(__file__).resolve().parent.parent / ".venv/bin"

def pytest_args(dest):
    txt = (dest / "pyproject.toml").read_text()
    return json.loads(re.search(r"pytest_add_cli_args = (\[.*\])", txt).group(1))

def clean_run(dest):
    """Run the suite on unmutated code; return counts and the node ids that failed."""
    p = subprocess.run([VENV / "python", "-m", "pytest", "tests", *pytest_args(dest), "--junitxml=junit.xml"],
                       cwd=dest, capture_output=True, text=True, timeout=600)
    res = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0, "failed_ids": [], "collect_error": False}
    if not (dest / "junit.xml").exists():
        res["collect_error"] = True; return res
    for tc in ET.parse(dest / "junit.xml").iter("testcase"):
        kids = {c.tag for c in tc}
        if "failure" in kids or "error" in kids:
            mod = tc.get("classname", "").split(".")
            path, cls = [], []
            for i, part in enumerate(mod):   # classname is tests.test_x.Class -> tests/test_x.py::Class
                if (dest / "/".join(mod[:i + 1])).with_suffix(".py").exists():
                    path, cls = mod[:i + 1], mod[i + 1:]
            nid = "/".join(path) + ".py::" + "::".join(cls + [tc.get("name")])
            res["failed_ids"].append(nid); res["failed" if "failure" in kids else "errors"] += 1
            if tc.get("name") == "" or not path:
                res["collect_error"] = True
        elif "skipped" in kids:
            res["skipped"] += 1
        else:
            res["passed"] += 1
    (dest / "junit.xml").unlink()
    return res

def coverage(dest, lib):
    src = [s.rstrip("/").removesuffix(".py") for s in (LIBS[lib]["mutate"] or list(LIBS[lib]["code"]))]
    subprocess.run([VENV / "python", "-m", "coverage", "run", "--branch", f"--source={','.join(src)}",
                    "-m", "pytest", "tests", *pytest_args(dest)], cwd=dest, capture_output=True, timeout=600)
    subprocess.run([VENV / "python", "-m", "coverage", "json", "-o", "cov.json"], cwd=dest, capture_output=True)
    t = json.loads((dest / "cov.json").read_text())["totals"]
    for f in (".coverage", "cov.json"):
        (dest / f).unlink(missing_ok=True)
    return {"line_pct": round(100 * t["covered_lines"] / max(t["num_statements"], 1), 1),
            "branch_pct": round(100 * t["covered_branches"] / max(t["num_branches"], 1), 1)}

def mutate(dest):
    subprocess.run([VENV / "mutmut", "run", "--max-children", "4"], cwd=dest, capture_output=True, timeout=3600)
    out = subprocess.run([VENV / "mutmut", "results", "--all", "true"], cwd=dest, capture_output=True, text=True).stdout
    return dict(l.strip().rsplit(": ", 1) for l in out.splitlines() if ": " in l)

def score(lib, tests_from, dest, human=False):
    dest = build(lib, dest, human)
    if not human:
        for f in Path(tests_from).rglob("*.py"):
            rel = f.relative_to(tests_from)
            if "__pycache__" not in rel.parts:
                (dest / "tests" / rel).parent.mkdir(parents=True, exist_ok=True); shutil.copy(f, dest / "tests" / rel)
    tests = sorted(str(p.relative_to(dest)) for p in (dest / "tests").rglob("*.py"))
    lines = sum(len(p.read_text().splitlines()) for p in (dest / "tests").rglob("test*.py"))
    clean = clean_run(dest)
    rec = {"lib": lib, "test_files": tests, "test_lines": lines, **{k: v for k, v in clean.items()}}
    if clean["collect_error"] or clean["passed"] == 0:
        rec["mutants"] = {}; return rec
    if clean["failed_ids"]:   # score only the tests that pass on the real code
        txt = (dest / "pyproject.toml").read_text()
        extra = "".join(f', "--deselect", {json.dumps(n)}' for n in clean["failed_ids"])
        (dest / "pyproject.toml").write_text(re.sub(r"(pytest_add_cli_args = \[.*)\]", lambda m: m.group(1) + extra + "]", txt))
    rec.update(coverage(dest, lib))
    rec["mutants"] = mutate(dest)
    return rec

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("work"); ap.add_argument("--human", action="store_true")
    ap.add_argument("--only", default=""); a = ap.parse_args(); work = Path(a.work).resolve()
    outdir = work / "scores"; outdir.mkdir(exist_ok=True)
    if a.human:
        jobs = [(f"human-{lib}", lib, None) for lib in LIBS]
    else:
        jobs = [(r["name"], r["lib"], work / "ws" / r["name"] / "tests")
                for r in (json.loads(p.read_text()) for p in sorted((work / "agent_runs").glob("*.json")))]
    for name, lib, src in jobs:
        out = outdir / f"{name}.json"
        if out.exists() or not name.startswith(a.only):
            continue
        rec = {"name": name, **score(lib, src, work / "score_ws" / name, human=a.human)}
        out.write_text(json.dumps(rec, indent=1))
        st = list(rec["mutants"].values())
        print(name, rec["passed"], "passed", rec["failed"] + rec["errors"], "failed",
              {s: st.count(s) for s in set(st)}, rec.get("line_pct"), flush=True)

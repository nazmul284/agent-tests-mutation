"""Ask each agent to write a test suite for each library, starting from a workspace with no tests.
Resumable: a cell with a saved record is skipped.  usage: python run_agents.py <work_dir> [--reps 2] [--jobs 4] [--only prefix]"""
import argparse, json, shutil, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from workspace import build

PY = str(Path(__file__).resolve().parent.parent / ".venv/bin/python")
DESC = {"jsonpointer": "a Python implementation of JSON Pointer, RFC 6901",
        "isodate": "an ISO 8601 date, time and duration parser and formatter",
        "parse": "the opposite of str.format(): it parses strings against format specifications"}
BASE = ("This repository contains {lib}, {desc}. It has no tests. Write a pytest test suite for it in "
        "tests/test_agent.py that would catch bugs in the library. Use only pytest and the standard library. "
        "Do not modify the library's code. Do not download or copy tests from anywhere; write them yourself "
        "from the code. Run them with `{py} -m pytest tests` and make sure they pass.")
MUT = (" The suite will be judged by mutation testing: small changes will be made to the library's code, "
       "such as flipped comparisons, changed constants and deleted statements, and a good suite fails for "
       "as many of those changes as possible.")
PROMPTS = {"plain": BASE, "mutation": BASE + MUT}
MODELS = {"claude-sonnet": ("claude", "claude-sonnet-5-5"), "claude-opus": ("claude", "claude-opus-5-5"),
          "codex-sol": ("codex", "gpt-5.6-sol"), "codex-astra": ("codex", "gpt-6-astra")}

def cmd(agent, model, task, stream=False):
    if agent == "claude":
        return ["claude", "-p", task, "--model", model, "--dangerously-skip-permissions",
                "--setting-sources", "project", "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
                "--no-session-persistence"] + (["--output-format", "stream-json", "--verbose"] if stream
                                               else ["--output-format", "json"])
    return ["codex", "exec", "-m", model, "--dangerously-bypass-approvals-and-sandbox", "--ignore-user-config",
            "--ignore-rules", "--ephemeral", "--skip-git-repo-check", "--json", task]

CLEAN_PYPROJECT = '[tool.pytest.ini_options]\npythonpath = ["."]\n'

def run(cell, work, clean=False):
    """clean=True: the workspace's pyproject.toml carries no mutmut section, and Claude's tool calls are logged."""
    key, lib, prompt, rep = cell
    name = f"{key}-{lib}-{prompt}{'clean' if clean else ''}-r{rep}"
    out = work / "agent_runs" / f"{name}.json"
    if out.exists():
        return
    wd = build(lib, work / "ws" / name)
    if clean:
        (wd / "pyproject.toml").write_text(CLEAN_PYPROJECT)
    g = lambda *a: subprocess.run(["git", *a], cwd=wd, capture_output=True)
    g("init", "-q"); g("add", "-A"); g("-c", "user.name=f", "-c", "user.email=f@e.com", "commit", "-qm", "baseline")
    agent, model = MODELS[key]
    task = PROMPTS[prompt].format(lib=lib, desc=DESC[lib], py=PY)
    t0 = time.time()
    try:
        p = subprocess.run(cmd(agent, model, task, stream=clean), cwd=wd, stdin=subprocess.DEVNULL,
                           capture_output=True, text=True, timeout=2400)
        rc, stdout, stderr = p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        rc, stdout, stderr = "timeout", (e.stdout or b"").decode(errors="replace"), ""
    (work / "logs").mkdir(exist_ok=True)
    (work / "logs" / f"{name}.log").write_text(stdout[-400000:] + "\n--- stderr ---\n" + stderr[-5000:])
    changed = g("status", "--porcelain").stdout.decode().splitlines()
    rec = {"name": name, "agent": agent, "model": model, "key": key, "lib": lib, "prompt": prompt + ("clean" if clean else ""), "rep": rep,
           "rc": rc, "wall_s": round(time.time() - t0, 1), "changed": changed}
    if agent == "claude":
        try:
            j = json.loads(stdout.strip().splitlines()[-1])
            rec["cost_usd"] = j.get("total_cost_usd"); rec["turns"] = j.get("num_turns")
        except Exception:
            pass
    else:
        for line in stdout.splitlines():
            try:
                ev = json.loads(line)
            except Exception:
                continue
            if ev.get("type") == "turn.completed":
                rec["usage"] = ev.get("usage")
    out.write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: rec[k] for k in ("name", "rc", "wall_s")}), len(changed), "changed", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("work"); ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--jobs", type=int, default=4); ap.add_argument("--only", default=None)
    ap.add_argument("--prompts", default="plain,mutation"); ap.add_argument("--clean", action="store_true")
    a = ap.parse_args(); work = Path(a.work).resolve(); (work / "agent_runs").mkdir(parents=True, exist_ok=True)
    cells = [(k, l, p, r) for r in range(1, a.reps + 1) for p in a.prompts.split(",") for l in DESC for k in MODELS]
    if a.only:
        cells = [c for c in cells if "-".join(map(str, c)).startswith(a.only)]
    print(len(cells), "cells", flush=True)
    with ThreadPoolExecutor(a.jobs) as ex:
        list(ex.map(lambda c: run(c, work, a.clean), cells))

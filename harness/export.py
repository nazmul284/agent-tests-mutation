"""Copy everything a reader needs to check the article from the work directory into results/.
usage: python export.py <work_dir>"""
import json, re, shutil, sys
from pathlib import Path
work = Path(sys.argv[1]); R = Path(__file__).resolve().parent.parent / "results"
for sub in ("suites", "planted_bug_suites", "scores", "agent_runs", "bug_runs"):
    shutil.rmtree(R / sub, ignore_errors=True); (R / sub).mkdir(parents=True)
for p in (work / "agent_runs").glob("*.json"):
    shutil.copy(p, R / "agent_runs" / p.name)
    src = work / "ws" / p.stem / "tests"
    if src.exists():
        shutil.copytree(src, R / "suites" / p.stem, ignore=shutil.ignore_patterns("__pycache__"))
for p in (work / "scores").glob("*.json"):
    shutil.copy(p, R / "scores" / p.name)
for p in (work / "bug_runs").glob("*.json"):
    shutil.copy(p, R / "bug_runs" / p.name)
    src = work / "bug_ws" / p.stem / "tests"
    if src.exists():
        shutil.copytree(src, R / "planted_bug_suites" / p.stem, ignore=shutil.ignore_patterns("__pycache__"))
for logs, runs in (("logs", "agent_runs"), ("bug_logs", "bug_runs")):   # raw agent transcripts
    shutil.rmtree(R / logs, ignore_errors=True); (R / logs).mkdir()
    for p in (work / logs).glob("*.log"):
        if (work / runs / f"{p.stem}.json").exists():
            shutil.copy(p, R / logs / p.name)
for f in ("mutant_kinds.json", "audit.json", "bug_classes.json"):
    if (work / f).exists():
        shutil.copy(work / f, R / f)
print("exported to", R)

def final_message(log):
    text = log.split("\n--- stderr ---")[0].strip()
    try:                                   # Claude, --output-format json
        return json.loads(text)["result"]
    except Exception:
        pass
    msg = ""
    for line in text.splitlines():         # Claude stream-json, or Codex --json
        try:
            ev = json.loads(line)
        except Exception:
            continue
        if ev.get("type") == "result":
            msg = ev.get("result", msg)
        if ev.get("type") == "item.completed" and ev.get("item", {}).get("type") == "agent_message":
            msg = ev["item"].get("text", msg)
    return msg

msgs = {p.stem: final_message(p.read_text(errors="replace")) for p in sorted((work / "logs").glob("*.log"))
        if (work / "agent_runs" / f"{p.stem}.json").exists()}
(R / "final_messages.json").write_text(json.dumps(msgs, indent=1, ensure_ascii=False))
print(len(msgs), "final messages,", sum(1 for m in msgs.values() if not m), "empty")

# Scrub local paths so the published results do not carry the machine's directory layout.
subs = [(str(work.resolve()), "<work>"), (str(work), "<work>"),
        (str(Path(__file__).resolve().parent.parent), "<repo>"), (str(Path.home()), "~")]
for p in R.rglob("*"):
    if p.is_file() and p.suffix in (".json", ".py", ".txt", ".md", ".log"):
        t = p.read_text(errors="replace"); n = t
        for a, b in subs:
            n = n.replace(a, b)
        n = n.replace("/private<work>", "<work>")
        n = re.sub(r"(/private)?/tmp/claude-\d+/[^\s\"'\\]*", "<tmp>", n)
        n = re.sub(r"~/\.claude/projects/[^\s\"'\\]*", "<claude-project>", n)
        if n != t:
            p.write_text(n)
print("scrubbed local paths")

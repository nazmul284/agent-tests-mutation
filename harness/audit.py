"""Contamination checks for each agent suite: did the agent reach the network for tests, and how much
of its suite repeats the maintainers' tests line for line.  usage: python audit.py <work_dir>"""
import ast, json, re, sys
from pathlib import Path
SRC = Path(__file__).resolve().parent.parent / "libs_src"
HUMAN = {"jsonpointer": [SRC / "python-json-pointer/tests.py"],
         "isodate": sorted((SRC / "isodate/tests").glob("*.py")),
         "parse": sorted((SRC / "parse/tests").glob("*.py"))}
NET = re.compile(r"\b(curl|wget|git clone|pip download|pip install|urllib\.request|requests\.get|"
                 r"site-packages/[\w.-]*(jsonpointer|isodate|parse)[^\s\"']*test)", re.I)

def norm_lines(paths):
    out = set()
    for p in paths:
        for l in p.read_text(errors="replace").splitlines():
            l = re.sub(r"\s+", "", l)
            if len(l) >= 30 and not l.startswith(("#", "import", "from", "def", "class", '"""')):
                out.add(l)
    return out

def literals(paths):
    out = set()
    for p in paths:
        try:
            tree = ast.parse(p.read_text())
        except SyntaxError:
            continue
        out |= {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str) and len(n.value) >= 8}
    return out

if __name__ == "__main__":
    work = Path(sys.argv[1]); res = {}
    for rp in sorted((work / "agent_runs").glob("*.json")):
        r = json.loads(rp.read_text()); name, lib = r["name"], r["lib"]
        agent_files = [p for p in (work / "ws" / name / "tests").rglob("*.py") if "__pycache__" not in p.parts]
        log = (work / "logs" / f"{name}.log").read_text(errors="replace")
        hits = sorted({m.group(0) for m in NET.finditer(log)})
        al, hl = norm_lines(agent_files), norm_lines(HUMAN[lib])
        alit, hlit = literals(agent_files), literals(HUMAN[lib])
        res[name] = {"net_hits": hits, "agent_lines": len(al), "shared_lines": len(al & hl),
                     "agent_literals": len(alit), "shared_literals": len(alit & hlit)}
        print(name, res[name])
    (work / "audit.json").write_text(json.dumps(res, indent=1))

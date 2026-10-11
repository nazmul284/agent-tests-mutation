"""Classify every mutant by what it changed: an error message, another string literal, or logic.
Reads the mutated sources mutmut writes under mutants/.  usage: python classify.py <work_dir>"""
import ast, difflib, io, json, re, sys, tokenize
from pathlib import Path

def toks(src):
    out = []
    for t in tokenize.generate_tokens(io.StringIO(src).readline):
        if t.type in (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.COMMENT, tokenize.ENDMARKER):
            continue
        out.append((t.type, t.string, t.line))
    return out

def classify(orig, mut):
    a, b = toks(orig), toks(mut)
    sm = difflib.SequenceMatcher(None, [x[:2] for x in a], [x[:2] for x in b], autojunk=False)
    changed, lines = [], []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op != "equal":
            changed += a[i1:i2] + b[j1:j2]; lines += [x[2] for x in a[i1:i2] + b[j1:j2]]
    if not changed:
        return "logic"
    stringy = all(t[0] == tokenize.STRING or t[1] == "None" or t[0] == tokenize.FSTRING_START
                  or t[0] in (getattr(tokenize, "FSTRING_MIDDLE", -1), getattr(tokenize, "FSTRING_END", -1))
                  for t in changed)
    if not stringy:
        return "logic"
    return "message" if any(re.search(r"\b(raise|warn|Error|Exception)\b", l) for l in lines) else "string"

if __name__ == "__main__":
    work = Path(sys.argv[1]); res = {}
    for lib in ["jsonpointer", "isodate", "parse"]:
        mdir = work / "score_ws" / f"human-{lib}" / "mutants"
        for f in mdir.rglob("*.py"):
            if "tests" in f.relative_to(mdir).parts:
                continue
            src = f.read_text(); mod = ".".join(f.relative_to(mdir).with_suffix("").parts).removesuffix(".__init__")
            funcs = {n.name: ast.get_source_segment(src, n) for n in ast.walk(ast.parse(src))
                     if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and "__mutmut_" in n.name}
            for name, body in funcs.items():
                m = re.match(r"(.*)__mutmut_(\d+)$", name)
                if not m or f"{m.group(1)}__mutmut_orig" not in funcs:
                    continue
                orig = funcs[f"{m.group(1)}__mutmut_orig"].replace(f"{m.group(1)}__mutmut_orig", "F", 1)
                res[f"{mod}.{name}"] = classify(orig, body.replace(name, "F", 1))
    (work / "mutant_kinds.json").write_text(json.dumps(res))
    from collections import Counter
    for lib in ["jsonpointer", "isodate", "parse"]:
        print(lib, Counter(v for k, v in res.items() if k.split(".")[0] == lib))

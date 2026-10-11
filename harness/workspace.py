"""Build a clean workspace for a library: its source only (no tests), a pyproject with pytest and
mutmut config, and optionally the maintainers' tests under tests/test_human*. usage:
    python workspace.py <lib> <dest> [--with-human-tests]"""
import shutil, sys
from pathlib import Path
SRC = Path(__file__).resolve().parent.parent / "libs_src"
LIBS = {
    "jsonpointer": {"code": {"jsonpointer.py": "python-json-pointer/jsonpointer.py"},
                    "mutate": ["jsonpointer.py"],
                    "human": {"tests/test_human.py": "python-json-pointer/tests.py"},
                    "pytest_k": "not CommandLineTests"},       # those test a separate CLI script
    "isodate": {"code": {"isodate": "isodate/src/isodate"}, "mutate": ["isodate/"],
                "human": {"tests": "isodate/tests"}, "pytest_k": None},
    "parse": {"code": {"parse.py": "parse/parse.py"} if (SRC / "parse/parse.py").exists() else {"parse": "parse/parse"},
              "mutate": None, "human": {"tests": "parse/tests"}, "pytest_k": None},
}
def build(lib, dest, human=False, bug=None):
    cfg = LIBS[lib]; dest = Path(dest)
    shutil.rmtree(dest, ignore_errors=True); dest.mkdir(parents=True)
    for d, s in cfg["code"].items():
        (shutil.copytree if (SRC / s).is_dir() else shutil.copy)(SRC / s, dest / d)
    if lib == "isodate":   # setuptools_scm normally generates this file at build time
        (dest / "isodate/version.py").write_text('version = "0.0.0.dev0"\n')
    if bug:   # plant one bug from bugs.py
        from bugs import BUGS
        blib, f, old, new, _ = BUGS[bug]; assert blib == lib
        src = (dest / f).read_text(); assert src.count(old) == 1, (bug, src.count(old))
        (dest / f).write_text(src.replace(old, new))
    mutate = cfg["mutate"] or list(cfg["code"])
    (dest / "tests").mkdir(exist_ok=True)
    if human:
        for d, s in cfg["human"].items():
            if (SRC / s).is_dir():
                for f in (SRC / s).iterdir():
                    if f.suffix == ".py" or f.is_dir():
                        (shutil.copytree if f.is_dir() else shutil.copy)(f, dest / "tests" / f.name)
            else:
                shutil.copy(SRC / s, dest / d)
    args = ['"-q"', '"-p"', '"no:cacheprovider"'] + ([f'"-k"', f'"{cfg["pytest_k"]}"'] if cfg["pytest_k"] else [])
    (dest / "pyproject.toml").write_text(
        f'[tool.mutmut]\nsource_paths = {mutate!r}\npytest_add_cli_args_test_selection = ["tests/"]\n'
        f'pytest_add_cli_args = [{", ".join(args)}]\n\n[tool.pytest.ini_options]\npythonpath = ["."]\n'.replace("'", '"'))
    return dest
if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2], "--with-human-tests" in sys.argv); print("built", sys.argv[2])

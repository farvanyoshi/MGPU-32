"""All RTL lints clean under verilator --lint-only -Wall."""

from rtlutil import lint, requires_rtl


@requires_rtl
def test_lint_clean():
    r = lint()
    assert r.returncode == 0, r.stdout + r.stderr
    assert "%Warning" not in r.stdout + r.stderr, r.stdout + r.stderr

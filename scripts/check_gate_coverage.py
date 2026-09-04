#!/usr/bin/env python3
"""Fail if any RELEASE_BLOCKING gate is executed by no workflow.

Why this exists
---------------
The 2026-08-18 audit found 59 RELEASE_BLOCKING gates that no workflow ran. The
fix enumerated the missing test binaries by name in `flagship-gates.yml`. Within
a day, review found the enumeration had already missed four of them
(`sym_op_dense`, `obstacle_vi_slope`, `strang2d_parallel_bit_equal`,
`chernoff1d_parallel_bit_equal`) — a hand-maintained list of ~75 `--test` flags
drifts the moment a gate is added.

So the list is no longer trusted: this script recomputes the invariant from the
contract and the workflow files on every PR, and fails if it is violated.

A test is reachable by the plain `cargo test --workspace --release` job unless
it is blocked by any of:
  * a file-level `#![cfg(...)]` mentioning `slow-tests` (including compound
    forms like `#![cfg(all(feature = "parallel", feature = "slow-tests"))]`),
  * a per-test `#[cfg(...)]` mentioning `slow-tests`,
  * `#[ignore]` / `#[ignore = "..."]`.
A blocked test is still covered if some workflow names its binary via `--test`.

Stdlib only, on purpose: the repo caps dependencies and this must run in a bare
CI step. The YAML reader below handles the flat `- name:/severity:/test_file:`
shape of properties.yaml and nothing more.
"""
from __future__ import annotations

import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTRACT = "contracts/semiflow-core.properties.yaml"
CRATE_ROOTS = ("crates/semiflow/", "crates/semiflow-py/",
               "crates/semiflow-ffi/", "crates/semiflow-wasm/", "")

#: A pointer looks like a path and ends in a source extension.
PATH_RE = re.compile(r"^[\w./{}, -]+\.(rs|py)$")


def read_gates(path):
    """Yield (name, severity, test_file) for every entry in the contract."""
    entry, out = {}, []
    for raw in open(path, encoding="utf-8"):
        line = raw.rstrip("\n")
        m = re.match(r"\s*-\s+name:\s*(.+)$", line)
        if m:
            if entry.get("name"):
                out.append(entry)
            entry = {"name": m.group(1).strip().strip("\"'")}
            continue
        for key in ("severity", "test_file"):
            m = re.match(rf"\s+{key}:\s*(.+)$", line)
            if m:
                entry[key] = m.group(1).strip().strip("\"'")
    if entry.get("name"):
        out.append(entry)
    return out


def scan_tests(path):
    """-> (file_is_slow_gated, {fn_name: blocked_bool})."""
    lines = open(path, encoding="utf-8", errors="replace").read().split("\n")
    file_slow = any(l.strip().startswith("#![cfg(") and "slow-tests" in l
                    for l in lines)
    fns, attrs = {}, []
    for line in lines:
        s = line.strip()
        if s.startswith("#["):
            attrs.append(s)
            continue
        if s.startswith("//") or not s:
            continue
        m = re.match(r"(pub\s+)?(async\s+)?fn\s+([A-Za-z0-9_]+)", s)
        if m and any(a.startswith("#[test") for a in attrs):
            joined = " ".join(attrs)
            fns[m.group(3)] = ("#[ignore" in joined
                               or bool(re.search(r"#\[cfg\([^)]*slow-tests", joined)))
        attrs = []
    return file_slow, fns


def named_binaries():
    """Every test binary any workflow (or xtask test-flagship) runs by name."""
    names = set()
    for wf in glob.glob(".github/workflows/*.yml"):
        for m in re.finditer(r"--test\s+([A-Za-z0-9_]+)", open(wf).read()):
            names.add(m.group(1))
    src = open("xtask/src/main.rs", encoding="utf-8").read()
    block = src[src.index("fn test_flagship"):]
    for m in re.finditer(r'"--test",\s*\n\s*"([A-Za-z0-9_]+)"',
                         block[:block.index("\n}")]):
        names.add(m.group(1))
    return names


def split_specs(field):
    """Split a `test_file` value into specs on commas OUTSIDE brace groups.

    The field holds either a comma-separated list of paths, or one path using
    the brace shorthand `..._d{2,3,4,5}_slope.rs`, or both. Splitting on every
    comma shreds the brace group into `..._d{2`, `3`, `4`, `5}_slope.rs`, none
    of which resolves — which silently dropped G_DDIM and G_MATRIX out of this
    check when it was first written.
    """
    parts, buf, depth = [], "", 0
    for ch in field:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            parts.append(buf); buf = ""
            continue
        buf += ch
    parts.append(buf)
    out = []
    for part in parts:
        out += [p.strip() for p in re.split(r"(?<=\.rs)\s+", part) if p.strip()]
    # Several entries append a prose note to the path, e.g.
    #   "…/g_adjoint_fp_order.rs (sub-gate 1; entry superseded by …)".
    # Splitting leaves the note as a second "spec". A fragment that is not
    # shaped like a path is a comment, not a pointer — drop it, so that a
    # pointer which IS path-shaped but does not resolve can be treated as the
    # error it is (see `unreached`).
    return [p for p in out if PATH_RE.match(p.split("::")[0])]


def expand(spec):
    """`a{1,2}b.rs` -> [a1b.rs, a2b.rs]; anything else -> [itself]."""
    path = spec.split("::")[0]
    m = re.match(r"^(.*)\{([^}]*)\}(.*)$", path)
    if not m:
        return [path]
    return [f"{m.group(1)}{x}{m.group(3)}"
            for x in re.split(r"[,\s]+", m.group(2)) if x]


def locate(rel):
    for root in CRATE_ROOTS:
        if os.path.exists(root + rel):
            return root + rel
    return None


def unreached(gates, names):
    """RELEASE_BLOCKING gates no workflow can execute."""
    cache, bad = {}, []
    for gate in gates:
        if gate.get("severity") != "RELEASE_BLOCKING" or not gate.get("test_file"):
            continue
        verdicts, missing = [], []
        for spec in split_specs(gate["test_file"]):
            fn = spec.split("::")[1].split()[0] if "::" in spec else None
            for rel in expand(spec):
                verdict = _verdict(rel, fn, names, cache)
                if verdict is None:
                    missing.append(rel)      # path-shaped, but no such file
                else:
                    verdicts.append(verdict)
        if missing:
            # A pointer that names a file which does not exist is contract
            # drift, and silently dropping it lets part of a gate go unchecked.
            bad.append((gate["name"], f"{gate['test_file']}   [no such file: "
                                      f"{', '.join(missing)}]"))
        elif not verdicts:
            bad.append((gate["name"], gate["test_file"] + "   [no pointer resolves]"))
        elif any(v == "UNREACHED" for v in verdicts):
            # EVERY pointer must be reachable, not merely one of them. A
            # binding-parity gate names its Rust and Python halves; running only
            # the Python half executes only half the assertion, and "some
            # pointer runs" would score that as covered.
            bad.append((gate["name"], gate["test_file"]))
    return bad


def _verdict(rel, fn, names, cache):
    full = locate(rel)
    if not full or full.endswith(".py") or "/src/" in full:
        return "OK" if full else None
    if full not in cache:
        cache[full] = scan_tests(full)
    file_slow, fns = cache[full]
    if fn and fn in fns:
        blocked = file_slow or fns[fn]
    else:
        # Whole-file pointer: blocked if ANY test in it is blocked, not only if
        # every test is. A file where 1 of 2 tests carries `#[ignore]` is half
        # covered by plain CI, and the ignored half is exactly the part a gate
        # tends to live in.
        blocked = file_slow or any(fns.values())
    if not blocked:
        return "OK"
    stem = os.path.splitext(os.path.basename(full))[0]
    return "OK" if stem in names else "UNREACHED"


def main():
    os.chdir(ROOT)
    bad = unreached(read_gates(CONTRACT), named_binaries())
    if not bad:
        print("check-gate-coverage: PASS — every RELEASE_BLOCKING gate is "
              "executed by some workflow")
        return 0
    print(f"check-gate-coverage: FAIL — {len(bad)} RELEASE_BLOCKING gate(s) "
          f"run in NO workflow:\n")
    for name, tf in bad:
        print(f"  {name:36} {tf}")
    print("\nAdd the binary to a job in .github/workflows/flagship-gates.yml "
          "(or nightly.yml if it runs for hours).")
    return 1


if __name__ == "__main__":
    sys.exit(main())

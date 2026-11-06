#!/usr/bin/env python3
"""Generate `src/normalise/tables.rs` from the reference normaliser's literal tables.

The reference's behaviour lives in 19 functions *and* 33 module-level tables (ABBR, SYMBOLS, UNITS,
ROMAN_*, TITLES, CURCODE, …). Retyping those by hand is where transcription errors would creep in, so
they are extracted with `ast` and emitted as Rust constants. Anything the evaluator cannot handle is
reported instead of silently skipped, so the remaining hand work is exactly the functions.

    python tools/gen_normalise_tables.py --repo /path/to/underdog_woof --out src/normalise/tables.rs

The generated file is written in the extractor's layout; run `cargo fmt` afterwards (the checked-in
`src/normalise/tables.rs` is formatted, so a diff against raw output is expected and is formatting
only).
"""
import argparse
import ast
import pathlib

SKIP = {"LETTER"}  # identity comprehension (c -> c); Rust needs no map for it


class Tup(list):
    """A Python tuple, so the emitter can keep tuples and lists apart."""


class Evaluator:
    """Evaluates the subset of Python used by the table definitions."""

    def __init__(self):
        self.ns = {}
        self.skipped = []

    def eval(self, node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.List):
            return [self.eval(e) for e in node.elts]
        if isinstance(node, ast.Tuple):
            return Tup(self.eval(e) for e in node.elts)
        if isinstance(node, ast.Set):
            return set(self.eval(e) for e in node.elts)
        if isinstance(node, ast.Dict):
            return {self.eval(k): self.eval(v) for k, v in zip(node.keys, node.values)}
        if isinstance(node, ast.Name):
            if node.id not in self.ns:
                raise KeyError(node.id)
            return self.ns[node.id]
        if isinstance(node, ast.BinOp):
            left, right = self.eval(node.left), self.eval(node.right)
            if isinstance(node.op, ast.BitOr):
                return left | right
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Pow):
                return left ** right
            raise NotImplementedError("binop")
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Constant):
                text = fn.value.value
                if fn.attr == "split":
                    sep = self.eval(node.args[0]) if node.args else None
                    return text.split(sep)
                if fn.attr == "join":
                    # "|".join([a, b, ...]) - used to build one long alternation
                    return text.join(self.eval(node.args[0]))
                raise NotImplementedError(f"str.{fn.attr}")
            if isinstance(fn, ast.Name) and fn.id in ("set", "list", "tuple", "sorted"):
                arg = self.eval(node.args[0])
                return {"set": set, "list": list, "tuple": tuple, "sorted": sorted}[fn.id](arg)
            raise NotImplementedError("call")
        if isinstance(node, (ast.DictComp, ast.SetComp, ast.ListComp)):
            return self._comprehension(node)
        raise NotImplementedError(type(node).__name__)

    def _comprehension(self, node):
        # only the pattern actually used: {w + "s" for w in X} / {w: v for w, v in X.items()}
        if len(node.generators) != 1:
            raise NotImplementedError("nested comprehension")
        gen = node.generators[0]
        if not isinstance(gen.target, ast.Name) or gen.ifs:
            raise NotImplementedError("comprehension target")
        source = self.eval(gen.iter)
        name = gen.target.id
        out = []
        for item in source:
            self.ns[name] = item
            out.append(self.eval(node.elt if hasattr(node, "elt") else node.key))
        self.ns.pop(name, None)
        if isinstance(node, ast.DictComp):
            raise NotImplementedError("dict comprehension")
        return set(out) if isinstance(node, ast.SetComp) else out

    def run(self, tree):
        for node in tree.body:
            if isinstance(node, ast.Assign):
                if not (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)):
                    continue
                name = node.targets[0].id
                if name in SKIP:
                    self.skipped.append((name, "identity comprehension"))
                    continue
                try:
                    self.ns[name] = self.eval(node.value)
                except Exception as e:  # noqa: BLE001 - report, never guess
                    self.skipped.append((name, f"{type(e).__name__}: {e}"))
            elif isinstance(node, ast.AugAssign):
                if not isinstance(node.target, ast.Name):
                    continue
                name = node.target.id
                try:
                    self.ns[name] = self.eval(node.value) if isinstance(node.op, ast.Assign) else (
                        self.ns[name] | self.eval(node.value)
                    )
                except Exception as e:  # noqa: BLE001
                    self.skipped.append((name, f"augassign {type(e).__name__}: {e}"))
        return self.ns


def rust_literal(value, indent=4):
    pad = " " * indent
    if isinstance(value, str):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, Tup):
        inner = ", ".join(rust_literal(v, indent) for v in value)
        return f"({inner})"
    if isinstance(value, (list, tuple)):
        inner = ", ".join(rust_literal(v, indent) for v in value)
        return f"&[{inner}]"
    if isinstance(value, (set, frozenset)):
        inner = ", ".join(sorted(rust_literal(v, indent) for v in value))
        return f"&[{inner}]"
    if isinstance(value, dict):
        items = ", ".join(f"({rust_literal(k, indent)}, {rust_literal(v, indent)})" for k, v in value.items())
        return f"&[{items}]"
    raise TypeError(type(value))


def u32_max():
    return 2**32 - 1


def rust_type(value):
    if isinstance(value, str):
        return "&str"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        if value < 0:
            return "i64"
        return "u32" if value <= u32_max() else "u64"
    if isinstance(value, float):
        return "f32"
    if isinstance(value, Tup):
        return "(" + ", ".join(rust_type(v) for v in value) + ")"
    if isinstance(value, (list, tuple)):
        if not value:
            return "&[&str]"
        inner = {rust_type(v) for v in value}
        return f"&[{inner.pop() if len(inner) == 1 else rust_type(value[0])}]"
    if isinstance(value, (set, frozenset)):
        sample = next(iter(value))
        return f"&[{rust_type(sample)}]"
    if isinstance(value, dict):
        k, v = next(iter(value.items()))
        return f"&[({rust_type(k)}, {rust_type(v)})]"
    raise TypeError(type(value))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", default="src/normalise/tables.rs")
    a = ap.parse_args()
    src = pathlib.Path(a.repo, "underdog_woof", "normalize.py").read_text()
    ev = Evaluator()
    ns = ev.run(ast.parse(src))

    lines = [
        "//! Lookup tables extracted from the reference normaliser by `tools/gen_normalise_tables.py`.",
        "//!",
        "//! Derivative content: extracted from Underdog Woof, which is licensed under the Underdog",
        "//! Non-Commercial License 1.0 (see LICENSE and NOTICE at the repository root).",
        "//!",
        "//! Generated — do not edit. Regenerate whenever the reference changes; the counts in the",
        "//! doc comments are what the extractor saw, so a shrinking table is visible in review.",
        "#![allow(dead_code)] // consumed as the rule functions get ported",
        "",
    ]
    emitted = 0
    for name in sorted(ns):
        value = ns[name]
        try:
            doc = f"/// `{name}` ({len(value)} entries)" if isinstance(value, (list, tuple, set, dict)) else f"/// `{name}`"
            lines.append(doc)
            lines.append(f"pub const {name.upper()}: {rust_type(value)} = {rust_literal(value)};")
            lines.append("")
            emitted += 1
        except Exception as e:  # noqa: BLE001
            ev.skipped.append((name, f"emit {type(e).__name__}: {e}"))

    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out).write_text("\n".join(lines))
    print(f"emitted {emitted} tables -> {a.out}")
    if ev.skipped:
        print(f"not extracted ({len(ev.skipped)}):")
        for name, why in ev.skipped:
            print(f"  {name:14s} {why}")


if __name__ == "__main__":
    main()

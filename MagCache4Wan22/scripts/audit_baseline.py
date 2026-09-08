#!/usr/bin/env python3
"""Audit common upstream ancestry and SeaCache's cache-disabled numerical path."""
from __future__ import annotations
import argparse
import ast
import copy
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))
from common import PROJECT, artifact, external_output, index_result, read_json, sha256, validate_source, write_json


def method(path, cls, name):
    tree = ast.parse(Path(path).read_text())
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    return copy.deepcopy(next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == name))


class Baseline(ast.NodeTransformer):
    """Specialize only the explicit disabled SeaCache path and bookkeeping."""
    def visit_FunctionDef(self, node):
        positional = list(zip(node.args.args[-len(node.args.defaults):], node.args.defaults)) if node.args.defaults else []
        required = node.args.args[:len(node.args.args) - len(node.args.defaults)]
        positional = [(a, d) for a, d in positional if not a.arg.startswith("seacache")]
        node.args.args = required + [a for a, _ in positional]
        node.args.defaults = [d for _, d in positional]
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
            node.body = node.body[1:]
        return self.generic_visit(node)

    def visit_If(self, node):
        expr = ast.unparse(node.test)
        if expr == "seacache is None":
            return [self.visit(n) for n in node.body]
        if any(isinstance(n, ast.Name) and n.id in {"seacache", "seacache_config"} for n in ast.walk(node.test)):
            return None
        return self.generic_visit(node)

    def visit_Assign(self, node):
        target = ast.unparse(node.targets[0])
        if target in {"seacache", "runtime_protocol", "previous_model_stage", "model_stage"}:
            return None
        return self.generic_visit(node)

    def visit_Call(self, node):
        node.keywords = [k for k in node.keywords if k.arg is None or not k.arg.startswith("seacache")]
        return self.generic_visit(node)

    def visit_For(self, node):
        if ast.unparse(node.target) == "(step_index, t)":
            node.target.elts[0].id = "_"
        return self.generic_visit(node)


def normalized(function):
    return ast.dump(ast.fix_missing_locations(Baseline().visit(function)), include_attributes=False)


def audit(source, sea_source):
    source, sea_source = Path(source), Path(sea_source)
    provenance = validate_source(source)
    lock = read_json(PROJECT / "upstream_lock.json")["wan22"]
    ancestry = {}
    for name in ("SeaCache4Wan22", "TeaCache4Wan22"):
        other = read_json(PROJECT.parent / name / "upstream_lock.json")["wan22"]
        if other["commit"] != lock["commit"] or any(other["original_file_sha256"][p] != h for p, h in lock["original_file_sha256"].items()):
            raise ValueError(f"baseline ancestry mismatch: {name}")
        ancestry[name] = other["commit"]
    checked = []
    for relative, cls, name in (("wan/text2video.py", "WanT2V", "generate"),
                                ("wan/modules/model.py", "WanModel", "forward")):
        if normalized(method(source / relative, cls, name)) != normalized(method(sea_source / relative, cls, name)):
            raise ValueError(f"disabled SeaCache numerical path differs: {relative}:{cls}.{name}")
        checked.append(f"{cls}.{name}")
    # SeaCache's modulated-norm helper is only a refactor: inline its verified expression.
    native = method(source / "wan/modules/model.py", "WanAttentionBlock", "forward")
    candidate = method(sea_source / "wan/modules/model.py", "WanAttentionBlock", "forward")
    helper = method(sea_source / "wan/modules/model.py", "WanAttentionBlock", "_modulated_norm1")
    if len(helper.body) != 1 or not isinstance(helper.body[0], ast.Return):
        raise ValueError("unexpected modulated norm helper")
    class Inline(ast.NodeTransformer):
        def visit_Assign(self, node):
            if ast.unparse(node.targets[0]) == "x_m":
                if ast.unparse(node.value) != "self._modulated_norm1(x, e)":
                    raise ValueError("unexpected modulated norm invocation")
                return None
            return self.generic_visit(node)
        def visit_Name(self, node):
            return copy.deepcopy(helper.body[0].value) if node.id == "x_m" else node
    if normalized(native) != normalized(Inline().visit(candidate)):
        raise ValueError("WanAttentionBlock math differs")
    dependencies = {}
    for folder in ("wan/modules", "wan/utils", "wan/configs"):
        for path in sorted((source / folder).glob("*.py")):
            relative = path.relative_to(source)
            if str(relative) == "wan/modules/model.py":
                continue
            if sha256(path) != sha256(sea_source / relative):
                raise ValueError(f"baseline dependency mismatch: {relative}")
            dependencies[str(relative)] = sha256(path)
    return dict(status="pass", source=provenance, common_upstream=ancestry,
                original_generate=artifact(source / "generate.py"),
                seacache_source=str(sea_source.resolve()),
                seacache_compared_files={p: artifact(sea_source / p) for p in ("wan/text2video.py", "wan/modules/model.py")},
                equivalent_disabled_cache_methods=checked + ["WanAttentionBlock.forward"],
                identical_dependency_sha256=dependencies,
                scope="Static AST/numerical path audit. Method CLI files contain different cache and timing plumbing. "
                      "Full A14B cross-method baseline tensors/videos have not been compared.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--seacache-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = external_output(args.output)
    result = audit(args.source, args.seacache_source)
    output.parent.mkdir(parents=True, exist_ok=True)
    index_result(output.parent)
    if not (output.parent / "README.md").exists():
        (output.parent / "README.md").write_text("# Baseline source audit\n\nSource hashes and cache-disabled AST comparison; no generated-video measurements.\n")
    write_json(output, result)
    print(result["status"])

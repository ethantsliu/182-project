"""Source logs follow the selected output directory even when legacy logs cannot be used."""
import ast
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("variant", ["t1-cpu-ngram", "t2-cpu-ngram"])
def test_source_log_uses_selected_directory_without_touching_legacy(variant, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "logs").write_text("legacy root-owned location stand-in")
    selected = tmp_path / "session source logs"
    monkeypatch.setenv("TRAIN_LOG_DIR", str(selected))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "triton", SimpleNamespace())
    path = ROOT / "variants" / variant / "track_1_short" / "run_log.py"
    spec = importlib.util.spec_from_file_location("source_log_" + variant.replace("-", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    print0, flush = module.start_run_log(True, "run-id")
    print0("complete source and environment")
    flush()
    assert (selected / "run-id.txt").read_text() == "complete source and environment\n"
    assert f"TRAIN_LOG_PATH={selected / 'run-id.txt'}" in capsys.readouterr().out
    assert (tmp_path / "logs").read_text() == "legacy root-owned location stand-in"


def test_medium_inline_source_log_uses_selected_directory(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "logs").write_text("legacy root-owned location stand-in")
    selected = tmp_path / "session source logs"
    monkeypatch.setenv("TRAIN_LOG_DIR", str(selected))
    path = ROOT / "variants" / "t2-gpu-port" / "train_gpt_medium.py"
    tree = ast.parse(path.read_text())
    # Execute only startup logging, without importing a trainer or initializing a GPU.
    start = next(i for i, node in enumerate(tree.body)
                 if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "logfile" for t in node.targets))
    block = ast.Module(body=tree.body[start:start + 3], type_ignores=[])
    namespace = dict(os=os, master_process=True, args=SimpleNamespace(run_id="run-id"))
    exec(compile(block, str(path), "exec"), namespace)
    namespace["print0"]("complete source and environment")
    assert (selected / "run-id.txt").read_text() == "complete source and environment\n"
    assert f"TRAIN_LOG_PATH={selected / 'run-id.txt'}" in capsys.readouterr().out
    assert (tmp_path / "logs").read_text() == "legacy root-owned location stand-in"

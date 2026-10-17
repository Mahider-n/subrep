"""The native dependency must be initialized and match the parent's Git pin."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from integration import native_metamo


def source_tree(path):
    for package in ("core", "category", "dynamics", "magus", "openpsi"):
        folder = path / package
        folder.mkdir(parents=True)
        (folder / "__init__.py").touch()
    return path


def test_uninitialized_default_explains_submodule_setup(tmp_path, monkeypatch):
    monkeypatch.delenv("METAMO_SOURCE_ROOT", raising=False)
    monkeypatch.setattr(native_metamo, "DEFAULT_METAMO_ROOT", tmp_path / "missing")
    with pytest.raises(ValueError, match="git submodule update --init --recursive"):
        native_metamo.load_metamo()


@pytest.mark.parametrize("entry", ["", "100644 " + "1" * 40 + " 0\tthird_party/metamo-python\n",
                                 "160000 " + "1" * 40 + " 1\tthird_party/metamo-python\n"])
def test_directory_without_registered_gitlink_is_rejected(tmp_path, monkeypatch, entry):
    root = source_tree(tmp_path / "metamo")
    monkeypatch.delenv("METAMO_SOURCE_ROOT", raising=False)
    monkeypatch.setattr(native_metamo, "DEFAULT_METAMO_ROOT", root)
    monkeypatch.setattr(native_metamo.subprocess, "check_output", lambda *a, **k: entry)
    with pytest.raises(ValueError, match="registered|Git submodule"):
        native_metamo.resolve_metamo_root()


@pytest.mark.parametrize("actual", ["1" * 40, "2" * 40])
def test_default_checkout_must_match_registered_revision(tmp_path, monkeypatch, actual):
    root = source_tree(tmp_path / "metamo")
    expected = "1" * 40
    monkeypatch.delenv("METAMO_SOURCE_ROOT", raising=False)
    monkeypatch.setattr(native_metamo, "DEFAULT_METAMO_ROOT", root)
    def git_read(args, **kwargs):
        if "ls-files" in args:
            return f"160000 {expected} 0\tthird_party/metamo-python\n"
        return actual + "\n"
    monkeypatch.setattr(native_metamo.subprocess, "check_output", git_read)
    if actual == expected:
        assert native_metamo.resolve_metamo_root() == root
    else:
        with pytest.raises(ValueError, match="differs from pinned revision"):
            native_metamo.resolve_metamo_root()


def test_external_override_is_explicit_and_argument_wins_over_environment(tmp_path, monkeypatch):
    configured = source_tree(tmp_path / "configured")
    explicit = source_tree(tmp_path / "explicit")
    monkeypatch.setenv("METAMO_SOURCE_ROOT", str(configured))
    def forbidden(*args, **kwargs):
        pytest.fail("An external development override has no parent Git pin")
    monkeypatch.setattr(native_metamo.subprocess, "check_output", forbidden)
    assert native_metamo.resolve_metamo_root() == configured
    assert native_metamo.resolve_metamo_root(explicit) == explicit


def test_conflicting_top_level_package_is_rejected(tmp_path, monkeypatch):
    root = source_tree(tmp_path / "metamo")
    monkeypatch.setitem(sys.modules, "core", SimpleNamespace(__file__=str(tmp_path / "other/core.py")))
    with pytest.raises(ValueError, match="conflicting MetaMo import: core"):
        native_metamo.load_metamo(root)


def test_cli_missing_submodule_fails_before_creating_artifacts(tmp_path, monkeypatch, capsys):
    from demo.run_metamo_phase01 import main
    output = tmp_path / "artifacts"
    monkeypatch.delenv("METAMO_SOURCE_ROOT", raising=False)
    monkeypatch.setattr(native_metamo, "DEFAULT_METAMO_ROOT", tmp_path / "missing")
    monkeypatch.setattr(sys, "argv", ["run_metamo_phase01", "--output", str(output)])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    assert "git submodule update --init --recursive" in capsys.readouterr().err
    assert not output.exists()

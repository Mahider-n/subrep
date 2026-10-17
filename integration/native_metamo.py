"""Load native MetaMo classes from the pinned submodule by default."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SUBMODULE_PATH = Path("third_party/metamo-python")
DEFAULT_METAMO_ROOT = REPOSITORY_ROOT / SUBMODULE_PATH
SUBMODULE_SETUP = "git submodule update --init --recursive"


def _validate_submodule_revision(root):
    """Compare the checkout to the gitlink, including a newly staged gitlink."""
    try:
        staged = subprocess.check_output(
            ["git", "-C", str(REPOSITORY_ROOT), "ls-files", "--stage", "--", str(SUBMODULE_PATH)],
            text=True, stderr=subprocess.PIPE,
        ).splitlines()
        if len(staged) != 1:
            raise ValueError("MetaMo submodule has no single registered Git revision")
        fields = staged[0].split(maxsplit=3)
        if len(fields) != 4 or fields[0] != "160000" or fields[2] != "0":
            raise ValueError("MetaMo dependency must be registered as a Git submodule")
        actual = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.PIPE,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"Cannot verify MetaMo submodule revision. Run `{SUBMODULE_SETUP}`.") from exc
    if actual != fields[1]:
        raise ValueError(
            f"MetaMo checkout {actual} differs from pinned revision {fields[1]}. "
            f"Run `{SUBMODULE_SETUP}`."
        )


def resolve_metamo_root(source_root=None):
    """Explicit path > METAMO_SOURCE_ROOT > the repository's pinned submodule."""
    configured = source_root if source_root is not None else os.environ.get("METAMO_SOURCE_ROOT")
    root = Path(configured if configured is not None else DEFAULT_METAMO_ROOT).resolve()
    packages = ("core", "category", "dynamics", "magus", "openpsi")
    for package in packages:
        if not (root / package / "__init__.py").exists():
            if root == DEFAULT_METAMO_ROOT.resolve():
                raise ValueError(f"MetaMo submodule is not initialized. Run `{SUBMODULE_SETUP}`.")
            raise ValueError(f"invalid MetaMo source root: missing {package}")
    if root == DEFAULT_METAMO_ROOT.resolve():
        _validate_submodule_revision(root)
    return root


def load_metamo(source_root=None):
    root = resolve_metamo_root(source_root)
    packages = ("core", "category", "dynamics", "magus", "openpsi")
    for package in packages:
        for name, module in tuple(sys.modules.items()):
            if name == package or name.startswith(package + "."):
                location = getattr(module, "__file__", None)
                if location is None or not Path(location).resolve().is_relative_to(root):
                    raise ValueError(f"conflicting MetaMo import: {name}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    state = importlib.import_module("core.state")
    bimonad = importlib.import_module("category.bimonad")
    appraisal = importlib.import_module("openpsi.appraisal")
    decision = importlib.import_module("magus.decision")
    return SimpleNamespace(
        Action=state.Action, Stimulus=state.Stimulus, State=state.MotivationalState,
        bimonad=bimonad.MetaMoPseudoBimonad(appraisal.OpenPsiAppraisal(), decision.MagusDecision()),
        source_root=str(root),
    )

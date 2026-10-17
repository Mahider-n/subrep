"""Archive the existing pipeline, then exercise the native MetaMo bridge.

python -m demo.run_metamo_phase01
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, is_dataclass
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import shutil
import subprocess
import tarfile

import numpy as np

from integration.contracts import EvaluationProtocol, ObjectiveSchema
from integration.native_metamo import resolve_metamo_root


ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    def convert(item):
        if is_dataclass(item):
            return asdict(item)
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(type(item).__name__)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False, default=convert) + "\n")


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def archive_sources(output, metamo_root, plan=None):
    """Archive source bytes actually used, including uncommitted bridge files."""
    sources = [*ROOT.glob("integration/*.py"), ROOT / "demo/run_metamo_phase01.py",
               ROOT / "tests/test_metamo_phase01.py", ROOT / ".gitmodules"]
    sources += [ROOT / path for path in git(ROOT, "ls-files").splitlines()
                if Path(path).suffix in {".py", ".txt", ".md"}]
    if metamo_root:
        for package in ("core", "category", "dynamics", "magus", "openpsi"):
            sources += list((metamo_root / package).glob("*.py"))
    checksums = {}
    with tarfile.open(output / "sources.tar.gz", "w:gz") as archive:
        for source in sorted(set(sources)):
            if source.is_file():
                if source.is_relative_to(ROOT):
                    relative = "subrep/" + str(source.relative_to(ROOT))
                else:
                    relative = "metamo/" + str(source.relative_to(metamo_root))
                checksums[relative] = hashlib.sha256(source.read_bytes()).hexdigest()
                archive.add(source, arcname=relative)
    write_json(output / "source_checksums.json", checksums)
    (output / "dependencies.lock.txt").write_text(subprocess.check_output(
        [__import__("sys").executable, "-m", "pip", "freeze", "--all"], text=True))
    shutil.copy2(ROOT / "models/pilot_ppo.pt", output / "pilot_ppo.pt")
    models = {}
    for name in ("pilot_ppo.pt", "generator.pt", "mdn_policy_best.pth"):
        source = ROOT / "models" / name
        if source.exists():
            shutil.copy2(source, output / name)
            models[name] = {"source": "checkpoint", "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
        else:
            models[name] = {"source": "absent"}
    write_json(output / "provenance.json", {
        "python": platform.python_version(), "platform": platform.platform(),
        "subrep_commit": git(ROOT, "rev-parse", "HEAD"),
        "subrep_branch": git(ROOT, "branch", "--show-current"),
        "subrep_worktree": git(ROOT, "status", "--short"),
        "metamo_commit": git(metamo_root, "rev-parse", "HEAD") if metamo_root else None,
        "metamo_worktree": git(metamo_root, "status", "--short") if metamo_root else None,
        "pilot_sha256": hashlib.sha256((output / "pilot_ppo.pt").read_bytes()).hexdigest(),
        "models": models,
        "plan_reference": {"filename": plan.name, "sha256": hashlib.sha256(plan.read_bytes()).hexdigest()} if plan else None,
        "source_archive": "sources.tar.gz", "source_checksums": "source_checksums.json",
        "phase1_mdn": "not_activated_global_certificates_only",
        "motive_mapping": "lander_priority_v1_engineering_default_not_empirically_calibrated",
    })


def reproduce(output):
    from demo import run_full_pipeline as legacy
    from certification.metta_storage import CertificateStore
    from library.skill_library import SkillLibrary
    output.mkdir(parents=True, exist_ok=True)
    replacements = {"CERT_FILE": str(output / "certificates.metta"),
                    "LIBRARY_FILE": str(output / "library.json"),
                    "REPORT_JSON_PATH": str(output / "admission_report.json"),
                    "REPORT_MD_PATH": str(output / "admission_report.md")}
    originals = {name: getattr(legacy, name) for name in replacements}
    baseline_method = legacy.IdlePolicy.run_baseline_episodes
    baseline_stats = {}
    def capture_baseline(self, *args, **kwargs):
        result = baseline_method(self, *args, **kwargs)
        baseline_stats.update(result)
        return result
    environments = []
    env_type = legacy.SubRepEnv
    def capture_env(*args, **kwargs):
        env = env_type(*args, **kwargs)
        environments.append(env)
        return env
    try:
        for name, value in replacements.items():
            setattr(legacy, name, value)
        legacy.IdlePolicy.run_baseline_episodes = capture_baseline
        legacy.SubRepEnv = capture_env
        with (output / "console.log").open("w") as log, redirect_stdout(log), redirect_stderr(log):
            result = legacy.run_pipeline()
    finally:
        for name, value in originals.items():
            setattr(legacy, name, value)
        legacy.IdlePolicy.run_baseline_episodes = baseline_method
        legacy.SubRepEnv = env_type
        for env in environments:
            env.close()
    certificates = CertificateStore()
    certificates.load_from_file(output / "certificates.metta")
    library = SkillLibrary(cert_store=certificates)
    library.load(str(output / "library.json"))
    cert_ids = sorted(c.skill_id for c in certificates.load_all())
    lib_ids = sorted(s.skill_id for s in library.get_admitted_skills())
    if cert_ids != lib_ids:
        raise RuntimeError("phase0_certificate_library_identity_mismatch")
    report = json.loads((output / "admission_report.json").read_text())
    write_json(output / "baseline_raw.json", baseline_stats)
    write_json(output / "result.json", result)
    protocol = {
        "reward_schema": asdict(ObjectiveSchema()), "gamma": legacy.GAMMA,
        "baseline": {"seeds": list(range(legacy.SEED, legacy.SEED + legacy.BASELINE_EPISODES)),
                     "horizon": "environment_termination_or_truncation", "action": 0},
        "candidates": {"max_steps": legacy.MAX_STEPS,
                       "reset_seeds_without_generator_prefilter": [legacy.SEED + 1000 + i
                                                                   for i in range(1, legacy.NUM_EPISODES + 1)]},
        "legacy_certificate_seed_field": legacy.SEED,
        "limitation": "legacy certificates record root seed, not actual candidate reset seed; horizons are unmatched",
        "generator_source": "checkpoint" if (ROOT / "models/generator.pt").exists() else "absent_no_prefilter",
        "mdn_metadata": {k: v for k, v in report.get("summary", report).items()
                         if k.startswith("mdn_") or k in {"derived_weights", "support_values", "support_geometry_feasible"}},
        "certificate_ids": cert_ids, "library_ids": lib_ids, "identity_sets_equal": True,
    }
    write_json(output / "actual_protocol.json", protocol)
    return {key: result[key] for key in ("admitted", "rejected", "cds_pass_count", "pds_pass_count")}


def boundary(output, metamo_root, seeds, horizon):
    from env.lunar_lander_wrapper import SubRepEnv
    from integration.native_metamo import load_metamo
    from integration.registry import PolicyRegistry
    from integration.lander_provider import build_lander_provider
    from integration.evaluation import evaluate
    from integration.persistence import AdmissionStore
    from integration.coordinator import Coordinator
    output.mkdir(parents=True, exist_ok=True)
    native = load_metamo(metamo_root)
    schema = ObjectiveSchema()
    protocol = EvaluationProtocol(schema, tuple(seeds), horizon=horizon,
        dependencies=tuple((name, version(name)) for name in ("mo-gymnasium", "gymnasium", "Box2D", "numpy", "torch")))
    registry = PolicyRegistry(schema)
    baseline_registration, actions = build_lander_provider(native, registry, ROOT / "models/pilot_ppo.pt")
    store = AdmissionStore(output / "state.json", protocol)
    decisions = []
    with (output / "console.log").open("w") as log, redirect_stdout(log), redirect_stderr(log):
        baseline = evaluate(baseline_registration, protocol, SubRepEnv)
        store.set_baseline(baseline)
        for action in actions:
            registration = registry.resolve(action)
            evidence = evaluate(registration, protocol, SubRepEnv)
            decision = store.admit(registration, evidence, epsilon=5.0)
            decisions.append({"action_id": action.id, "behavior_id": registration.skill_id,
                              "evidence": evidence.summary(),
                              **{k: v for k, v in decision.items() if k != "certificate"},
                              "certificate": decision["certificate"].to_dict() if decision["admitted"] else None})
        # Restart verifies persisted evidence/math/identities; screening reattaches
        # executables from the trusted registry before operational selection.
        store = AdmissionStore(output / "state.json", protocol)
        coordinator = Coordinator(registry=registry, store=store, bimonad=native.bimonad)
        state_a = native.State(np.array([.8, .3, .7, .3, .2, .5, .8, .3]), np.full(6, .5))
        state_b = native.State(np.array([.7, .4, .8, .4, .3, .4, .9, .3]), np.full(6, .5))
        stimulus = native.Stimulus(novelty=.2, conduciveness=.1, risk=.2, effort=.3)
        env = SubRepEnv(seed=seeds[0])
        try:
            obs, _ = env.reset(seed=seeds[0])
            selected = coordinator.decide(state_a, state_b, stimulus, obs, actions)
            outcome = coordinator.execute(selected, env, observation=obs, state_a=state_a,
                                          state_b=state_b, stimulus=stimulus, seed=seeds[0])
            # Motive change reuses the same evidence; no evaluation/training here.
            shifted_a = native.State(np.array([.3, .5, .1, .3, .2, 1., .1, .2]), np.array([.5, .5, .5, 1., 0., 0.]))
            shifted = coordinator.decide(shifted_a, shifted_a, stimulus, obs, actions)
            unknown = native.Action("unregistered_controller", np.zeros(8), .5, np.zeros(8))
            empty = coordinator.decide(state_a, state_b, stimulus, obs, [unknown])
            stopped = coordinator.execute(empty, env, observation=obs, state_a=state_a,
                                          state_b=state_b, stimulus=stimulus)
        finally:
            env.close()
    store.export(output)
    summary = {"baseline": baseline.summary(), "protocol": asdict(protocol), "admissions": decisions,
               "library_size": store.library.count(), "rejections": len(store.rejections),
               "normal_selection": asdict(selected.selection) if selected.selection else None,
               "normal_outcome": outcome, "motive_shift_weights": [selected.geometry.weights, shifted.geometry.weights],
               "motive_shift_selected_action": shifted.selection.action_id if shifted.selection else None,
               "empty_set_recovery": asdict(empty.recovery), "stop_outcome": stopped,
               "scope": "phases_0_and_1; global certificates; no MDN activation or decay scheduler"}
    write_json(output / "result.json", summary)
    return {"library_size": summary["library_size"], "selected_action": summary["normal_selection"],
            "empty_set_mode": empty.recovery.mode}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metamo-root", type=Path,
                        help="Development override; defaults to the pinned third_party/metamo-python submodule")
    parser.add_argument("--output", type=Path, default=Path("demo/artifacts/metamo_phase01"))
    parser.add_argument("--phase", choices=("all", "reproduce", "boundary"), default="all")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--plan", type=Path, help="Optional source proposal; record its filename and digest")
    args = parser.parse_args()
    try:
        args.metamo_root = resolve_metamo_root(args.metamo_root)
    except ValueError as exc:
        parser.error(str(exc))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    archive_sources(output, args.metamo_root.resolve(), args.plan)
    result = {}
    if args.phase in {"all", "reproduce"}:
        result["phase0"] = reproduce(output / "phase0")
    if args.phase in {"all", "boundary"}:
        result["phase1"] = boundary(output / "phase1", args.metamo_root, args.seeds, args.horizon)
    write_json(output / "artifact_checksums.json", {
        str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(output.rglob("*")) if path.is_file()
        and path.name != "artifact_checksums.json" and not path.name.endswith(".lock")})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

"""Acceptance tests for measurement and operational boundary."""
from dataclasses import asdict, replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from integration.contracts import (EpisodeEvidence, EvaluationEvidence, EvaluationProtocol,
                                   MotiveAdapter, MotiveGeometry, ObjectiveSchema, validate_action)
from integration.coordinator import BackupApproval, Coordinator, RecoveryPolicy
from integration.evaluation import evaluate
from integration.native_metamo import load_metamo
from integration.persistence import AdmissionStore
from integration.registry import PolicyRegistry


def action(name="good", delta=0.02):
    return SimpleNamespace(id=name, goal_correlations=np.full(8, 0.3), risk_estimate=0.1,
                           delta_g=np.full(8, delta))


def evidence(registration, protocol, objectives=(5, -1)):
    return EvaluationEvidence(registration.skill_id, protocol, tuple(
        EpisodeEvidence(seed, float(sum(objectives)), objectives, protocol.horizon,
                        False, False, "max_steps") for seed in protocol.seeds))


@pytest.fixture
def setup(tmp_path):
    schema = ObjectiveSchema()
    protocol = EvaluationProtocol(schema, (1, 2), horizon=3)
    registry = PolicyRegistry(schema)
    source = tmp_path / "controller.py"
    source.write_text("fixed action controller v1\n")
    def register(candidate, value=0):
        return registry.register(candidate, behavior_version="v1", artifact_paths=[source],
                                 configuration={"action": value}, factory=lambda seed: lambda obs: value)
    idle = register(action("comparison_idle"))
    store = AdmissionStore(tmp_path / "state.json", protocol)
    baseline = evidence(idle, protocol, (0, 0))
    store.set_baseline(baseline)
    return SimpleNamespace(schema=schema, protocol=protocol, registry=registry, store=store,
                           baseline=baseline, source=source, register=register)


@pytest.fixture
def native():
    # Missing native dependencies must fail visibly; CI initializes submodules.
    return load_metamo()


def test_native_classes_come_from_configured_checkout(native):
    import inspect
    from pathlib import Path
    root = Path(native.source_root)
    for component in (native.Action, native.Stimulus, native.State, type(native.bimonad),
                      type(native.bimonad.appraisal), type(native.bimonad.decision)):
        assert Path(inspect.getfile(component)).resolve().is_relative_to(root)


def native_inputs(native):
    state_a = native.State(np.array([.8, .2, .7, .3, .2, .5, .8, .4]), np.full(6, .5))
    state_b = state_a.copy()
    return state_a, state_b, native.Stimulus(.2, .1, .3, .1), np.zeros(8)


def bridge(setup, native, **kwargs):
    return Coordinator(registry=setup.registry, store=setup.store, bimonad=native.bimonad, **kwargs)


class LiveEnv:
    environment_id = "mo-lunar-lander-v3"
    objective_schema_version = "lander_safety_fuel_v2"
    objective_names = ("safety", "fuel")

    def __init__(self, seed=0):
        self.seed = seed
        self.resets = self.steps = 0
        self.closed = False

    def reset(self, seed=None):
        self.seed = seed
        self.resets += 1
        self.steps = 0
        return np.zeros(8), {}

    def step(self, action_id):
        self.steps += 1
        return np.full(8, self.steps), np.array([1., -.1]), False, False, {}

    def close(self):
        self.closed = True


def test_matched_rollouts_reset_policy_and_retain_raw_evidence(setup):
    creations, calls, environments = [], [], []
    candidate = action("stateful")
    def factory(seed):
        creations.append(seed)
        count = 0
        def policy(obs):
            nonlocal count
            count += 1
            calls.append((seed, count))
            return 0
        return policy
    registration = setup.registry.register(candidate, behavior_version="v1", factory=factory,
                                         artifact_paths=[setup.source])
    def env_factory(seed):
        env = LiveEnv(seed)
        environments.append(env)
        return env
    result = evaluate(registration, setup.protocol, env_factory)
    assert creations == [1, 2]
    assert calls == [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]
    assert all(env.closed and env.resets == 1 for env in environments)
    assert [ep.seed for ep in result.episodes] == [1, 2]
    assert all(ep.steps == 3 and ep.stop_reason == "max_steps" for ep in result.episodes)
    assert result.summary()["payoff_std"] == 0
    assert result.mean_payoff == pytest.approx(.9 * (1 + .99 + .99**2))


@pytest.mark.parametrize("field,value", [("horizon", 4), ("gamma", .8), ("seeds", (1, 3))])
def test_noncomparable_evidence_never_certified(setup, field, value):
    registration = setup.register(action())
    incompatible = replace(setup.protocol, **{field: value})
    result = setup.store.admit(registration, evidence(registration, incompatible))
    assert not result["admitted"] and result["reason"] == "evaluation_protocol_mismatch"
    assert setup.store.library.count() == setup.store.certificates.count() == 0
    assert result["rejection_persisted"]


def test_fuel_schema_and_objective_order_enforced_at_boundaries(setup):
    with pytest.raises(ValueError):
        ObjectiveSchema(names=("fuel", "safety"))
    old_schema = replace(setup.schema, version="legacy_inverted_fuel_v1")
    with pytest.raises(ValueError, match="unknown_objective_schema"):
        PolicyRegistry(old_schema)
    old_protocol = replace(setup.protocol, schema=old_schema)
    with pytest.raises(ValueError, match="snapshot_reward_schema_or_protocol_mismatch"):
        AdmissionStore(setup.store.path, old_protocol)
    registration = setup.register(action())
    with pytest.raises(ValueError, match="reward_schema_mismatch"):
        evaluate(registration, old_protocol, LiveEnv)


@pytest.mark.parametrize("objectives,epsilon,gate,margin", [((0, 0), 0, "CDS", 0),
                         ((1, -2), 3, "PDS", 0), ((5, -1), 0, "CDS", 3)])
def test_gate_boundaries_and_idempotent_restart(setup, objectives, epsilon, gate, margin):
    registration = setup.register(action())
    measured = evidence(registration, setup.protocol, objectives)
    result = setup.store.admit(registration, measured, epsilon=epsilon)
    assert result["admitted"] and result["certificate"].gate_type == gate
    assert result["certificate"].admission_margin == margin
    restarted = AdmissionStore(setup.store.path, setup.protocol)
    retry = restarted.admit(registration, measured, epsilon=epsilon)
    assert retry["idempotent"] and retry["certificate"] == result["certificate"]
    assert restarted.library.count() == restarted.certificates.count() == 1
    assert restarted.library.get_skill(registration.skill_id).policy is None


def test_rejected_evidence_survives_restart_without_certificate(setup):
    registration = setup.register(action("rejected"))
    measured = evidence(registration, setup.protocol, (-2, -1))
    result = setup.store.admit(registration, measured)
    assert not result["admitted"] and result["reason"] == "gate_failed"
    restarted = AdmissionStore(setup.store.path, setup.protocol)
    assert restarted.rejections[0]["behavior_id"] == registration.skill_id
    assert restarted.rejections[0]["evidence"]["episodes"][0]["seed"] == 1
    assert restarted.library.count() == restarted.certificates.count() == 0


def test_immutable_admission_cannot_overwrite_evidence(setup):
    registration = setup.register(action())
    original = setup.store.admit(registration, evidence(registration, setup.protocol))
    result = setup.store.admit(registration, evidence(registration, setup.protocol, (6, -1)))
    assert not result["admitted"] and result["reason"] == "immutable_admission_conflict"
    assert setup.store.resolve_certificate(registration) == original["certificate"]


@pytest.mark.parametrize("failure", ["false", "exception"])
def test_library_failure_never_leaves_orphan_certificate(setup, monkeypatch, failure):
    from library.skill_library import SkillLibrary
    registration = setup.register(action())
    original = SkillLibrary.add_skill
    def reject(*args, **kwargs):
        if failure == "exception":
            raise LookupError("injected_library_failure")
        return False
    monkeypatch.setattr(SkillLibrary, "add_skill", reject)
    result = setup.store.admit(registration, evidence(registration, setup.protocol))
    assert not result["admitted"]
    monkeypatch.setattr(SkillLibrary, "add_skill", original)
    restarted = AdmissionStore(setup.store.path, setup.protocol)
    assert restarted.library.count() == restarted.certificates.count() == 0


def test_disk_commit_failure_preserves_previous_snapshot(setup, monkeypatch):
    registration = setup.register(action())
    def broken(data):
        raise OSError("disk unavailable")
    monkeypatch.setattr(setup.store, "_atomic_write", broken)
    result = setup.store.admit(registration, evidence(registration, setup.protocol))
    assert not result["admitted"] and not result["rejection_persisted"]
    restarted = AdmissionStore(setup.store.path, setup.protocol)
    assert restarted.library.count() == setup.store.library.count() == 0


def test_corrupt_certificate_and_evidence_fail_on_restart(setup):
    registration = setup.register(action())
    setup.store.admit(registration, evidence(registration, setup.protocol))
    data = json.loads(setup.store.path.read_text())
    data["entries"][registration.skill_id]["certificate"]["delta_r"] += 1
    setup.store.path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="certificate_manifest_or_evidence_mismatch"):
        AdmissionStore(setup.store.path, setup.protocol)


def test_registry_detects_changed_artifact_and_unknown_actions(setup):
    candidate = action()
    registration = setup.register(candidate)
    with pytest.raises(ValueError, match="unknown_action_id"):
        setup.registry.resolve(action("safe_answer"))
    with pytest.raises(ValueError, match="action_id_already_registered"):
        setup.register(candidate)
    setup.source.write_text("different controller")
    with pytest.raises(ValueError, match="artifact_digest_changed"):
        registration.new_policy(1)


def test_mapping_sensitivity_zero_default_and_strict_downstream(setup):
    adapter = MotiveAdapter(setup.schema, epsilon=2)
    zero = SimpleNamespace(G=np.zeros(8), M=np.zeros(6))
    stimulus = SimpleNamespace(novelty=0., conduciveness=0., risk=0., effort=0.)
    neutral = adapter.project(zero, stimulus)
    assert neutral.weights == (.5, .5) and neutral.zero_priority_default
    safety_state = SimpleNamespace(G=np.array([1., 0, 0, 0, 0, 0, 1., 0]), M=np.zeros(6))
    fuel_state = SimpleNamespace(G=np.array([0, 0, 0, 0, 0, 1., 0, 0]), M=np.zeros(6))
    assert adapter.project(safety_state, stimulus).weights == (1., 0.)
    assert adapter.project(fuel_state, stimulus).weights == (0., 1.)
    assert adapter.project(fuel_state, stimulus).epsilon == 2
    with pytest.raises(ValueError, match="sum to one"):
        replace(neutral, weights=(2, 1))
    for malformed in (np.zeros((8, 1)), np.full(8, np.nan)):
        with pytest.raises(ValueError):
            adapter.project(SimpleNamespace(G=malformed, M=np.zeros(6)), stimulus)
    invalid = action()
    invalid.delta_g = np.zeros((8, 1))
    with pytest.raises(ValueError):
        validate_action(invalid)
    assert len(neutral.context) == 18


def test_every_native_decision_path_receives_only_original_eligible_actions(setup, native, monkeypatch):
    good, bad = action(), action("bad", delta=.9)
    reg_good, reg_bad = setup.register(good), setup.register(bad)
    setup.store.admit(reg_good, evidence(reg_good, setup.protocol))
    setup.store.admit(reg_bad, evidence(reg_bad, setup.protocol, (-2, -1)))
    seen = []
    decide = native.bimonad.decision.decide
    score = native.bimonad.decision.score_candidate
    def spy_decide(state, candidates):
        seen.append(tuple(candidates))
        assert len(candidates) == 1 and candidates[0] is good
        return decide(state, candidates)
    def spy_score(state, candidate):
        assert candidate is good
        return score(state, candidate)
    monkeypatch.setattr(native.bimonad.decision, "decide", spy_decide)
    monkeypatch.setattr(native.bimonad.decision, "score_candidate", spy_score)
    original_a, original_b, stimulus, obs = native_inputs(native)
    a_before, b_before = original_a.G.copy(), original_b.M.copy()
    result = bridge(setup, native).decide(original_a, original_b, stimulus, obs, [bad, good])
    assert result.action is good and result.eligible_actions == (good,)
    assert len(seen) >= 2  # Includes native stability diagnostics, not only consensus.
    assert np.array_equal(original_a.G, a_before) and np.array_equal(original_b.M, b_before)
    assert ("bad", "no_certificate") in result.rejected
    assert setup.store.library.get_skill(reg_good.skill_id).policy is not None


def test_current_epsilon_blocks_historically_admitted_pds_before_magus(setup, native, monkeypatch):
    candidate = action()
    registration = setup.register(candidate)
    assert setup.store.admit(registration, evidence(registration, setup.protocol, (1, -2)), epsilon=3)["admitted"]
    def forbidden(*args):
        pytest.fail("MAGUS must not receive an empty or rejected set")
    monkeypatch.setattr(native.bimonad, "step", forbidden)
    result = bridge(setup, native).decide(*native_inputs(native), [candidate])
    assert result.no_selection and result.recovery.mode == "stop"
    assert result.rejected == (("good", "current_epsilon_exceeded"),)


def test_subrep_owner_is_exclusive_and_preserves_action_identity(setup, native, monkeypatch):
    first, second = action("a"), action("b")
    for candidate in (first, second):
        registration = setup.register(candidate)
        setup.store.admit(registration, evidence(registration, setup.protocol))
    def forbidden(*args):
        pytest.fail("MAGUS cannot replace SubRep's winner")
    monkeypatch.setattr(native.bimonad.decision, "decide", forbidden)
    monkeypatch.setattr(native.bimonad.decision, "score_candidate", forbidden)
    result = bridge(setup, native, owner="subrep_selects").decide(*native_inputs(native), [second, first])
    assert result.action is first
    assert result.selection.eligible_ids == ("b", "a")


def test_normal_execution_preserves_live_environment_and_links_feedback(setup, native):
    candidate = action()
    registration = setup.register(candidate)
    setup.store.admit(registration, evidence(registration, setup.protocol))
    coordinator = bridge(setup, native)
    state_a, state_b, stimulus, obs = native_inputs(native)
    result = coordinator.decide(state_a, state_b, stimulus, obs, [candidate])
    env = LiveEnv()
    outcome = coordinator.execute(result, env, observation=obs, state_a=state_a,
                                  state_b=state_b, stimulus=stimulus)
    assert outcome["status"] == "completed" and outcome["mode"] == "normal"
    assert env.resets == 0 and env.steps == 3
    assert outcome["decision_reference"] == result.selection.selection_id
    assert outcome["certificate_id"] == result.selection.certificate_id
    assert outcome["persisted"]
    restarted = AdmissionStore(setup.store.path, setup.protocol)
    assert restarted.events[-1]["execution_id"] == outcome["execution_id"]
    restarted_coordinator = Coordinator(registry=setup.registry, store=restarted, bimonad=native.bimonad)
    with pytest.raises(ValueError, match="cycle_id_already_used"):
        restarted_coordinator.decide(state_a, state_b, stimulus, obs, [candidate], cycle_id=result.cycle_id)
    with pytest.raises(ValueError, match="cycle_already_executed"):
        coordinator.execute(result, env, observation=obs, state_a=state_a, state_b=state_b, stimulus=stimulus)


@pytest.mark.parametrize("change", ["observation", "motives", "artifact", "descriptor", "cancel"])
def test_execution_rechecks_freshness_identity_and_cancellation(setup, native, change):
    candidate = action()
    registration = setup.register(candidate)
    setup.store.admit(registration, evidence(registration, setup.protocol))
    coordinator = bridge(setup, native)
    a, b, stimulus, obs = native_inputs(native)
    result = coordinator.decide(a, b, stimulus, obs, [candidate])
    if change == "observation":
        obs[0] += 1
    elif change == "motives":
        a.G[0] -= .1
    elif change == "artifact":
        setup.source.write_text("changed")
    elif change == "descriptor":
        candidate.delta_g[0] += .1
    env = LiveEnv()
    outcome = coordinator.execute(result, env, observation=obs, state_a=a, state_b=b,
                                  stimulus=stimulus, cancelled=change == "cancel")
    assert outcome["status"] == "stopped" and env.steps == env.resets == 0


def test_backup_requires_explicit_approval_identity_scope_and_authority(setup, native):
    backup = setup.register(action("backup"))
    approval = BackupApproval("approval-fixture", "test-host", "validation-evidence-fixture",
                              backup.skill_id, setup.protocol.identity, (-1.,) * 8, (1.,) * 8)
    recovery = RecoveryPolicy(backup_action_id="backup", approval=approval, authority="test-host", budget_seconds=5)
    coordinator = bridge(setup, native, recovery=recovery)
    a, b, stimulus, obs = native_inputs(native)
    result = coordinator.decide(a, b, stimulus, obs, [])
    assert result.no_selection and result.recovery.mode == "backup"
    env = LiveEnv()
    outcome = coordinator.execute(result, env, observation=obs, state_a=a, state_b=b, stimulus=stimulus)
    assert outcome["mode"] == "backup" and outcome["status"] == "completed"
    assert outcome["approval_id"] == approval.approval_id and outcome["certificate_id"] is None
    assert env.steps == 3 and env.resets == 0
    recovery.authority = "different-authority"
    blocked = coordinator.decide(a, b, stimulus, obs, [])
    assert blocked.recovery.mode == "stop"
    recovery.authority = "test-host"
    outside = coordinator.decide(a, b, stimulus, np.full(8, 2.), [])
    assert outside.recovery.mode == "stop"


def test_expired_recovery_stops_and_never_defaults_to_idle(setup, native):
    recovery = RecoveryPolicy(budget_seconds=0)
    coordinator = bridge(setup, native, recovery=recovery)
    a, b, stimulus, obs = native_inputs(native)
    result = coordinator.decide(a, b, stimulus, obs, [])
    assert result.recovery.mode == "stop" and result.recovery.reason == "recovery_budget_exhausted"
    env = LiveEnv()
    outcome = coordinator.execute(result, env, observation=obs, state_a=a, state_b=b, stimulus=stimulus)
    assert outcome["status"] == "stopped" and env.steps == env.resets == 0


def test_motive_shift_reuses_certificates_without_evaluation(setup, native):
    candidate = action()
    registration = setup.register(candidate)
    setup.store.admit(registration, evidence(registration, setup.protocol))
    before = setup.store.resolve_certificate(registration)
    coordinator = bridge(setup, native)
    a, b, stimulus, obs = native_inputs(native)
    first = coordinator.decide(a, b, stimulus, obs, [candidate])
    a.G[5] = b.G[5] = 1.
    a.G[0] = b.G[0] = .3
    second = coordinator.decide(a, b, stimulus, obs, [candidate])
    assert first.geometry.weights != second.geometry.weights
    assert setup.store.resolve_certificate(registration) == before
    assert first.action is second.action is candidate


def test_invalid_native_state_returns_structured_stop(setup, native):
    a, b, stimulus, obs = native_inputs(native)
    a.G = np.zeros((8, 1))
    result = bridge(setup, native).decide(a, b, stimulus, obs, [])
    assert result.no_selection and result.recovery.mode == "stop"


def test_wrong_live_reward_orientation_is_blocked_before_execution(setup, native):
    candidate = action()
    registration = setup.register(candidate)
    setup.store.admit(registration, evidence(registration, setup.protocol))
    coordinator = bridge(setup, native)
    a, b, stimulus, obs = native_inputs(native)
    selected = coordinator.decide(a, b, stimulus, obs, [candidate])
    env = LiveEnv()
    env.objective_schema_version = "legacy_inverted_fuel_v1"
    outcome = coordinator.execute(selected, env, observation=obs, state_a=a, state_b=b, stimulus=stimulus)
    assert outcome["stop_reason"] == "execution_environment_schema_mismatch"
    assert env.steps == env.resets == 0


def test_expensive_backup_loading_cannot_start_after_deadline(setup, native):
    clock = [0.0]
    def slow_factory(seed):
        clock[0] += 1
        return lambda obs: 0
    backup = setup.registry.register(action("backup"), behavior_version="v1",
                                    artifact_paths=[setup.source], factory=slow_factory)
    approval = BackupApproval("approval", "host", "validation-fixture", backup.skill_id,
                              setup.protocol.identity, (-1.,) * 8, (1.,) * 8)
    recovery = RecoveryPolicy(backup_action_id="backup", approval=approval, authority="host",
                              budget_seconds=.25, clock=lambda: clock[0])
    coordinator = bridge(setup, native, recovery=recovery)
    a, b, stimulus, obs = native_inputs(native)
    selected = coordinator.decide(a, b, stimulus, obs, [])
    assert selected.recovery.mode == "backup"
    env = LiveEnv()
    outcome = coordinator.execute(selected, env, observation=obs, state_a=a, state_b=b, stimulus=stimulus)
    assert outcome["stop_reason"] == "recovery_budget_exhausted_before_execution"
    assert env.steps == env.resets == 0


def test_current_budget_can_admit_a_pds_with_a_larger_historical_budget(setup, native):
    from integration.contracts import MotiveAdapter
    candidate = action()
    registration = setup.register(candidate)
    assert setup.store.admit(registration, evidence(registration, setup.protocol, (1, -2)), epsilon=5)["admitted"]
    coordinator = bridge(setup, native, adapter=MotiveAdapter(setup.schema, epsilon=3))
    result = coordinator.decide(*native_inputs(native), [candidate])
    assert result.action is candidate and result.selection


def test_magus_ties_preserve_provider_input_order(setup, native):
    first, second = action("a"), action("b")
    for candidate in (first, second):
        registration = setup.register(candidate)
        setup.store.admit(registration, evidence(registration, setup.protocol))
    result = bridge(setup, native).decide(*native_inputs(native), [second, first])
    assert result.action is second and result.selection.eligible_ids == ("b", "a")

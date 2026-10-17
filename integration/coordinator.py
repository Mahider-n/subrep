"""Screen once, select through exactly one owner, and handle empty eligibility.

The host supplies native MetaMo objects and owns environment/state progression.
Rejected candidates have no execution path through this boundary.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import time
import uuid

import numpy as np

from env.skill_executor import SkillExecutor
from .contracts import (MotiveAdapter, digest, identifier, nonnegative, validate_action,
                        validate_environment, validate_state, validate_stimulus, vector)
from .persistence import now


@dataclass(frozen=True)
class NoSelection:
    cycle_id: str
    context_id: str
    reasons: tuple[tuple[str, str], ...]
    deadline: float
    recovery_id: str


@dataclass(frozen=True)
class BackupApproval:
    """Host-provided validation evidence and a declarative observation scope.

    Supplying an approval is an explicit deployment configuration action. The
    bridge checks its identity, protocol, authority and bounds; it does not infer
    a safety guarantee from an approval label or approve the idle baseline.
    """

    approval_id: str
    authority: str
    evidence_reference: str
    behavior_id: str
    protocol_id: str
    observation_low: tuple[float, ...]
    observation_high: tuple[float, ...]

    def __post_init__(self):
        for field in ("approval_id", "authority", "evidence_reference", "behavior_id", "protocol_id"):
            identifier(getattr(self, field), field)
        low = vector(self.observation_low, 8, "backup_scope_low")
        high = vector(self.observation_high, 8, "backup_scope_high")
        if np.any(np.asarray(low) > high):
            raise ValueError("invalid backup observation scope")
        object.__setattr__(self, "observation_low", low)
        object.__setattr__(self, "observation_high", high)

    def applies(self, observation):
        obs = np.asarray(vector(observation, 8, "observation"))
        return bool(np.all(obs >= self.observation_low) and np.all(obs <= self.observation_high))


@dataclass(frozen=True)
class RecoveryDecision:
    recovery_id: str
    cycle_id: str
    context_id: str
    mode: str
    reason: str
    behavior_id: str | None = None
    action_id: str | None = None
    approval_id: str | None = None
    deadline: float | None = None


class RecoveryPolicy:
    def __init__(self, *, backup_action_id=None, approval: BackupApproval | None = None,
                 authority=None, stop_mode="stop", budget_seconds=0.25, clock=time.monotonic):
        if stop_mode not in {"stop", "escalation"}:
            raise ValueError("invalid configured stop policy")
        self.budget_seconds = nonnegative(budget_seconds, "recovery_budget")
        self.backup_action_id, self.approval, self.authority = backup_action_id, approval, authority
        self.stop_mode, self.clock = stop_mode, clock

    def recover(self, no_selection: NoSelection, observation, registry, protocol):
        decision = dict(recovery_id=no_selection.recovery_id, cycle_id=no_selection.cycle_id,
                        context_id=no_selection.context_id, deadline=no_selection.deadline)
        reason = "no_applicable_validated_backup"
        if self.clock() >= no_selection.deadline:
            reason = "recovery_budget_exhausted"
        elif self.backup_action_id and self.approval:
            try:
                registration = registry.get(self.backup_action_id)
                approval = self.approval
                if (approval.authority != self.authority
                        or approval.behavior_id != registration.skill_id
                        or approval.protocol_id != protocol.identity
                        or registration.schema != protocol.schema
                        or not approval.applies(observation)):
                    raise ValueError("backup_not_approved_or_applicable")
                registration.verify()
                if self.clock() >= no_selection.deadline:
                    raise ValueError("recovery_budget_exhausted")
                return RecoveryDecision(**decision, mode="backup", reason="validated_backup",
                                        behavior_id=registration.skill_id,
                                        action_id=registration.action_id, approval_id=approval.approval_id)
            except (ValueError, OSError) as exc:
                reason = str(exc)
        return RecoveryDecision(**decision, mode=self.stop_mode, reason=reason)


@dataclass(frozen=True)
class SelectionDecision:
    selection_id: str
    cycle_id: str
    context_id: str
    action_id: str
    behavior_id: str
    certificate_id: str
    owner: str
    eligible_ids: tuple[str, ...]


@dataclass
class CycleDecision:
    cycle_id: str
    snapshot_id: str
    geometry: object | None
    eligible_actions: tuple
    rejected: tuple[tuple[str, str], ...]
    selection: SelectionDecision | None = None
    no_selection: NoSelection | None = None
    recovery: RecoveryDecision | None = None
    action: object | None = None
    target_a: object | None = None
    target_b: object | None = None
    consensus_target: object | None = None
    inputs: tuple | None = None


class Coordinator:
    def __init__(self, *, registry, store, bimonad, adapter=None,
                 owner="subrep_filter_then_magus", recovery=None):
        if owner not in {"subrep_filter_then_magus", "subrep_selects"}:
            raise ValueError("invalid_selection_owner")
        if registry.schema != store.protocol.schema:
            raise ValueError("registry_store_schema_mismatch")
        self.registry, self.store, self.bimonad = registry, store, bimonad
        self.adapter = adapter or MotiveAdapter(registry.schema)
        if self.adapter.schema != registry.schema:
            raise ValueError("adapter_registry_schema_mismatch")
        self.owner = owner
        self.recovery = recovery or RecoveryPolicy()
        self._cycles = {}
        self._executed = set()
        self._seen_cycle_ids = {event["cycle_id"] for event in store.events
                                if event.get("type") == "decision"}

    def _snapshot(self, state_a, state_b, stimulus, observation, candidates):
        ga, ma = validate_state(state_a)
        gb, mb = validate_state(state_b)
        signals = validate_stimulus(stimulus)
        obs = vector(observation, 8, "observation")
        context = self.bimonad.parallel_merge(state_a.copy(), state_b.copy())
        geometry = self.adapter.project(context, stimulus)
        actions = []
        for action in candidates:
            try:
                validate_action(action)
                actions.append({"id": action.id,
                                "correlations": vector(action.goal_correlations, 8, "correlations"),
                                "delta_g": vector(action.delta_g, 8, "delta_g"),
                                "risk": float(action.risk_estimate)})
            except (ValueError, TypeError, AttributeError):
                actions.append({"invalid_action": str(getattr(action, "id", "<invalid>"))})
        identity = digest({"a": ga + ma, "b": gb + mb, "stimulus": signals,
                           "observation": obs, "actions": actions,
                           "geometry": geometry.identity, "owner": self.owner,
                           "protocol": self.store.protocol.identity})
        return geometry, identity

    def _empty(self, cycle, snapshot, reasons, geometry=None, inputs=None, stop_reason="invalid_context"):
        recovery_id = uuid.uuid4().hex
        no_selection = NoSelection(cycle, snapshot, tuple(reasons),
                                   self.recovery.clock() + self.recovery.budget_seconds, recovery_id)
        if inputs is not None:
            recovery = self.recovery.recover(no_selection, inputs[3], self.registry, self.store.protocol)
        else:
            recovery = RecoveryDecision(recovery_id, cycle, snapshot, self.recovery.stop_mode,
                                        stop_reason, deadline=no_selection.deadline)
        return CycleDecision(cycle, snapshot, geometry, (), tuple(reasons),
                             no_selection=no_selection, recovery=recovery, inputs=inputs)

    def decide(self, state_a, state_b, stimulus, observation, candidates, *, cycle_id=None):
        cycle = cycle_id or uuid.uuid4().hex
        identifier(cycle, "cycle_id")
        if cycle in self._seen_cycle_ids:
            raise ValueError("cycle_id_already_used")
        candidates = tuple(candidates)
        try:
            geometry, snapshot = self._snapshot(state_a, state_b, stimulus, observation, candidates)
            inputs = (state_a.copy(), state_b.copy(), stimulus, tuple(observation), candidates)
        except (ValueError, TypeError, AttributeError) as exc:
            result = self._empty(cycle, "invalid", [("context", str(exc))])
            return self._record(result)
        eligible, rejected, certs, registrations = [], [], {}, {}
        ids = [getattr(a, "id", None) for a in candidates]
        for action in candidates:
            action_id = str(getattr(action, "id", "<invalid>"))
            try:
                if ids.count(getattr(action, "id", None)) != 1:
                    raise ValueError("ambiguous_action_id")
                registration = self.registry.resolve(action)
                cert = self.store.resolve_certificate(registration)
                if cert.delta_r + min(cert.delta_n) < -geometry.epsilon:
                    raise ValueError("current_epsilon_exceeded")
                policy = registration.new_policy(self.store.protocol.seeds[0])
                if not self.store.library.register_policy(registration.skill_id, policy):
                    raise ValueError("unresolved_library_entry")
                eligible.append(action)
                certs[action.id], registrations[action.id] = cert, registration
            except Exception as exc:
                rejected.append((action_id, str(exc)))
        for action_id, reason in rejected:
            try:
                registration = self.registry.get(action_id)
                behavior_id = registration.skill_id
            except ValueError:
                behavior_id = None
            try:
                self.store.record_rejection(reason, action_id=action_id, behavior_id=behavior_id,
                                            scope="runtime", context_id=snapshot)
            except (OSError, ValueError, RuntimeError):
                result = self._empty(cycle, snapshot, [("store", "rejection_persistence_failed")],
                                     geometry, inputs)
                return self._record(result)
        if not eligible:
            return self._record(self._empty(cycle, snapshot,
                                           rejected or [("candidates", "empty_candidate_set")], geometry, inputs))
        if self.owner == "subrep_filter_then_magus":
            _, target_a = self.bimonad.step(state_a.copy(), stimulus, eligible)
            _, target_b = self.bimonad.step(state_b.copy(), stimulus, eligible)
            chosen, consensus = self.bimonad.consensus_transition(
                state_a.copy(), state_b.copy(), stimulus, eligible)
        else:
            def ranking(action):
                cert = certs[action.id]
                score = cert.delta_r + float(np.dot(geometry.weights, cert.delta_n))
                return (-score, registrations[action.id].skill_id)
            chosen = min(eligible, key=ranking)
            context_a = self.bimonad._decision_context(state_a.copy(), stimulus)
            context_b = self.bimonad._decision_context(state_b.copy(), stimulus)
            target_a = self.bimonad._state_from_delta(context_a, chosen.delta_g)
            target_b = self.bimonad._state_from_delta(context_b, chosen.delta_g)
            consensus = self.bimonad.parallel_merge(target_a, target_b)
        if not any(chosen is action for action in eligible):
            raise RuntimeError("selector_returned_unscreened_action")
        registration, cert = registrations[chosen.id], certs[chosen.id]
        selection = SelectionDecision(uuid.uuid4().hex, cycle, snapshot, chosen.id,
                                      registration.skill_id, digest(cert.to_dict()), self.owner,
                                      tuple(action.id for action in eligible))
        result = CycleDecision(cycle, snapshot, geometry, tuple(eligible), tuple(rejected),
                               selection=selection, action=chosen, target_a=target_a,
                               target_b=target_b, consensus_target=consensus, inputs=inputs)
        return self._record(result)

    def _record(self, result):
        event = {"type": "decision", "timestamp": now(), "cycle_id": result.cycle_id,
                 "snapshot_id": result.snapshot_id,
                 "geometry": asdict(result.geometry) if result.geometry else None,
                 "eligible_ids": [a.id for a in result.eligible_actions],
                 "rejected": list(result.rejected),
                 "selection": asdict(result.selection) if result.selection else None,
                 "no_selection": asdict(result.no_selection) if result.no_selection else None,
                 "recovery": asdict(result.recovery) if result.recovery else None}
        try:
            self.store.record_event(event)
        except (OSError, ValueError, RuntimeError):
            result = self._empty(result.cycle_id, result.snapshot_id,
                                 [("store", "decision_persistence_failed")],
                                 stop_reason="decision_persistence_failed")
        self._cycles[result.cycle_id] = result
        self._seen_cycle_ids.add(result.cycle_id)
        return result

    def execute(self, decision, env, *, observation, state_a, state_b, stimulus,
                seed=None, max_steps=None, cancelled=False):
        """Execute from the current observation; never reset the operational env.

        The host must serialize environment advancement and supply a fresh
        observation/context. Return explicit stale/cancelled/stop outcomes.
        """
        if self._cycles.get(decision.cycle_id) is not decision:
            raise ValueError("unknown_decision")
        if decision.cycle_id in self._executed:
            raise ValueError("cycle_already_executed")
        mode = "normal" if decision.selection else (
            "backup" if decision.recovery and decision.recovery.mode == "backup" else "stop")
        reference = decision.selection.selection_id if decision.selection else decision.recovery.recovery_id
        outcome = {"type": "execution_outcome", "execution_id": uuid.uuid4().hex,
                   "timestamp": now(), "cycle_id": decision.cycle_id,
                   "context_id": decision.snapshot_id, "mode": mode, "decision_reference": reference,
                   "seed": seed, "protocol_id": self.store.protocol.identity,
                   "behavior_id": None, "certificate_id": None, "approval_id": None}
        started = time.monotonic()
        execution_started = False
        try:
            if cancelled:
                raise ValueError("cancelled")
            if mode == "stop":
                raise ValueError(decision.recovery.reason)
            _, fresh = self._snapshot(state_a, state_b, stimulus, observation, decision.inputs[4])
            if fresh != decision.snapshot_id:
                raise ValueError("stale_context_rescreen_required")
            action_id = decision.selection.action_id if mode == "normal" else decision.recovery.action_id
            registration = self.registry.get(action_id)
            registration.verify()
            outcome["behavior_id"] = registration.skill_id
            if mode == "normal":
                cert = self.store.resolve_certificate(registration)
                if (registration.skill_id != decision.selection.behavior_id
                        or digest(cert.to_dict()) != decision.selection.certificate_id):
                    raise ValueError("selection_identity_changed")
                outcome["certificate_id"] = decision.selection.certificate_id
            else:
                recovered = self.recovery.recover(decision.no_selection, observation,
                                                  self.registry, self.store.protocol)
                if recovered != decision.recovery:
                    raise ValueError("backup_expired_or_no_longer_applicable")
                outcome["approval_id"] = recovered.approval_id
            policy = registration.new_policy(seed if seed is not None else self.store.protocol.seeds[0])
            if mode == "backup" and self.recovery.clock() >= decision.no_selection.deadline:
                raise ValueError("recovery_budget_exhausted_before_execution")
            horizon = self.store.protocol.horizon if max_steps is None else max_steps
            if type(horizon) is not int or horizon <= 0 or horizon > self.store.protocol.horizon:
                raise ValueError("execution_horizon_exceeds_contract")
            validate_environment(env, self.store.protocol.schema)
            executor = SkillExecutor(env, policy, gamma=self.store.protocol.gamma, max_steps=horizon)
            execution_started = True
            payoff, objectives, terminated = executor.run_episode(initial_obs=np.asarray(observation))
            outcome.update(payoff=payoff, objectives=list(map(float, objectives)), terminated=terminated,
                           truncated=executor.last_run_info["truncated"], steps=executor.last_run_info["steps"],
                           stop_reason=executor.last_run_info["stop_reason"], status="completed")
        except Exception as exc:
            outcome.update(status="failed" if execution_started else "stopped",
                           stop_reason=str(exc), error_type=type(exc).__name__)
        outcome["duration_seconds"] = time.monotonic() - started
        self._executed.add(decision.cycle_id)
        try:
            self.store.record_event(outcome)
            outcome["persisted"] = True
        except (ValueError, OSError, RuntimeError):
            outcome["persisted"] = False
        return outcome

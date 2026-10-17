"""Strict measurement and motivational contracts for the initial bridge."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math

import numpy as np


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def vector(value, size: int, name: str, *, low=None, high=None) -> tuple[float, ...]:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (size,) or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be a finite vector with shape ({size},)")
    if low is not None and np.any(arr < low):
        raise ValueError(f"{name} must be >= {low}")
    if high is not None and np.any(arr > high):
        raise ValueError(f"{name} must be <= {high}")
    return tuple(float(x) for x in arr)


def nonnegative(value, name: str) -> float:
    if np.asarray(value).shape != ():
        raise ValueError(f"{name} must be a scalar")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def identifier(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


@dataclass(frozen=True)
class ObjectiveSchema:
    """Only the corrected two-objective environment mode is activated here."""

    version: str = "lander_safety_fuel_v2"
    environment: str = "mo-lunar-lander-v3"
    names: tuple[str, ...] = ("safety", "fuel")
    units: tuple[str, ...] = ("environment_reward", "negative_engine_cost")
    reward_mapping: str = "safety=raw[0]+raw[1];fuel=raw[2]+raw[3]"
    scalarization: str = "sum_objectives_v1"

    def __post_init__(self):
        object.__setattr__(self, "names", tuple(self.names))
        object.__setattr__(self, "units", tuple(self.units))
        for field in ("version", "environment", "reward_mapping", "scalarization"):
            identifier(getattr(self, field), field)
        if self.names != ("safety", "fuel") or len(self.units) != 2:
            raise ValueError("Phase 1 requires ordered environment objectives (safety, fuel)")

    @property
    def identity(self):
        return digest(asdict(self))


@dataclass(frozen=True)
class EvaluationProtocol:
    schema: ObjectiveSchema
    seeds: tuple[int, ...] = (42, 43, 44, 45, 46)
    gamma: float = 0.99
    horizon: int = 200
    aggregation: str = "episode_mean_population_std_v1"
    reset_policy: str = "fresh_factory_and_seed_each_episode_v1"
    dependencies: tuple[tuple[str, str], ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "seeds", tuple(self.seeds))
        object.__setattr__(self, "dependencies", tuple(sorted(tuple(x) for x in self.dependencies)))
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be a nonempty unique sequence")
        if any(type(seed) is not int or seed < 0 for seed in self.seeds):
            raise ValueError("seeds must be nonnegative integers")
        if type(self.horizon) is not int or self.horizon <= 0:
            raise ValueError("horizon must be a positive integer")
        if not math.isfinite(self.gamma) or not 0 <= self.gamma <= 1:
            raise ValueError("gamma must be in [0, 1]")
        if self.aggregation != "episode_mean_population_std_v1":
            raise ValueError("unsupported aggregation")
        if self.reset_policy != "fresh_factory_and_seed_each_episode_v1":
            raise ValueError("unsupported policy reset protocol")

    @property
    def identity(self):
        return digest(asdict(self))

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        data["schema"] = ObjectiveSchema(**data["schema"])
        return cls(**data)


@dataclass(frozen=True)
class EpisodeEvidence:
    seed: int
    payoff: float
    objectives: tuple[float, float]
    steps: int
    terminated: bool
    truncated: bool
    stop_reason: str

    def __post_init__(self):
        object.__setattr__(self, "objectives", vector(self.objectives, 2, "objectives"))
        if not math.isfinite(self.payoff):
            raise ValueError("payoff must be finite")
        if type(self.seed) is not int or type(self.steps) is not int or self.steps <= 0:
            raise ValueError("invalid episode seed or length")
        if type(self.terminated) is not bool or type(self.truncated) is not bool:
            raise ValueError("termination flags must be booleans")
        if self.stop_reason not in {"terminated", "truncated", "max_steps"}:
            raise ValueError("invalid stop reason")


@dataclass(frozen=True)
class EvaluationEvidence:
    behavior_id: str
    protocol: EvaluationProtocol
    episodes: tuple[EpisodeEvidence, ...]

    def __post_init__(self):
        identifier(self.behavior_id, "behavior_id")
        object.__setattr__(self, "episodes", tuple(self.episodes))
        if tuple(ep.seed for ep in self.episodes) != self.protocol.seeds:
            raise ValueError("evidence must contain every declared seed in protocol order")
        for ep in self.episodes:
            if ep.steps > self.protocol.horizon:
                raise ValueError("episode exceeds protocol horizon")
            if not np.isclose(ep.payoff, sum(ep.objectives), rtol=1e-5, atol=1e-4):
                raise ValueError("payoff disagrees with sum_objectives_v1")

    @property
    def identity(self):
        return digest(asdict(self))

    @property
    def mean_payoff(self):
        return float(np.mean([ep.payoff for ep in self.episodes]))

    @property
    def mean_objectives(self):
        return np.mean([ep.objectives for ep in self.episodes], axis=0)

    def summary(self):
        return {
            "evidence_id": self.identity,
            "mean_payoff": self.mean_payoff,
            "mean_objectives": self.mean_objectives.tolist(),
            "payoff_std": float(np.std([ep.payoff for ep in self.episodes])),
            "objective_std": np.std([ep.objectives for ep in self.episodes], axis=0).tolist(),
            "episodes": len(self.episodes),
        }

    @classmethod
    def from_dict(cls, data):
        return cls(data["behavior_id"], EvaluationProtocol.from_dict(data["protocol"]),
                   tuple(EpisodeEvidence(**ep) for ep in data["episodes"]))


def validate_state(state):
    return (vector(state.G, 8, "G", low=0, high=1),
            vector(state.M, 6, "M", low=0, high=1))


def validate_environment(env, schema: ObjectiveSchema):
    if (getattr(env, "environment_id", None) != schema.environment
            or getattr(env, "objective_schema_version", None) != schema.version
            or getattr(env, "objective_names", None) != schema.names):
        raise ValueError("execution_environment_schema_mismatch")


def validate_stimulus(stimulus):
    values = vector([stimulus.novelty, stimulus.conduciveness, stimulus.risk, stimulus.effort],
                    4, "stimulus")
    if not (-1 <= values[1] <= 1) or any(not 0 <= values[i] <= 1 for i in (0, 2, 3)):
        raise ValueError("stimulus requires novelty/risk/effort in [0,1], conduciveness in [-1,1]")
    return values


def validate_action(action):
    identifier(action.id, "Action.id")
    vector(action.goal_correlations, 8, "goal_correlations", low=-1, high=1)
    vector(action.delta_g, 8, "delta_g", low=-1, high=1)
    risk = nonnegative(action.risk_estimate, "risk_estimate")
    if risk > 1:
        raise ValueError("risk_estimate must be <= 1")


@dataclass(frozen=True)
class MotiveGeometry:
    context: tuple[float, ...]
    schema_id: str
    weights: tuple[float, float]
    epsilon: float
    mapping_version: str = "lander_priority_v1"
    budget_version: str = "fixed_deficit_v1"
    context_version: str = "metamo_context18_pre_appraisal_consensus_v1"
    zero_priority_default: bool = False

    def __post_init__(self):
        object.__setattr__(self, "context", vector(self.context, 18, "context"))
        weights = vector(self.weights, 2, "weights", low=0)
        if not np.isclose(sum(weights), 1.0, atol=1e-9, rtol=0):
            raise ValueError("weights must sum to one; downstream normalization is forbidden")
        object.__setattr__(self, "weights", weights)
        nonnegative(self.epsilon, "epsilon")

    @property
    def identity(self):
        return digest(asdict(self))


class MotiveAdapter:
    """Interpretable engineering mapping; coefficients are not empirically calibrated.

    safety_priority = G[individuation] + G[help] + G[ethics]
                      + mean(M[threshold], M[securing]) + stimulus.risk
    fuel_priority = G[self] + M[resolution] + stimulus.effort

    Fuel means conserving a negative engine-cost return. Efficiency is represented
    by self-maintenance, deliberation and effort; epsilon is an independent fixed
    configuration, never inferred from action risk. 
    """

    def __init__(self, schema: ObjectiveSchema, epsilon: float = 0):
        self.schema = schema
        self.epsilon = nonnegative(epsilon, "epsilon")

    def project(self, state, stimulus) -> MotiveGeometry:
        goals, mods = validate_state(state)
        signals = validate_stimulus(stimulus)
        safety = goals[0] + goals[2] + goals[6] + (mods[4] + mods[5]) / 2 + signals[2]
        fuel = goals[5] + mods[3] + signals[3]
        total = safety + fuel
        weights = (safety / total, fuel / total) if total else (0.5, 0.5)
        return MotiveGeometry(goals + mods + signals, self.schema.identity, weights,
                             self.epsilon, zero_priority_default=not bool(total))

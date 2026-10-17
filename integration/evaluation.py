"""Matched-seed evidence collection in a separate evaluation environment."""
from __future__ import annotations

import random
import numpy as np

from env.skill_executor import SkillExecutor
from .contracts import EpisodeEvidence, EvaluationEvidence, EvaluationProtocol, validate_environment
from .registry import OBJECTIVE_REGISTRY, PolicyRegistration


def evaluate(registration: PolicyRegistration, protocol: EvaluationProtocol, env_factory):
    if protocol.schema != registration.schema:
        raise ValueError("reward_schema_mismatch")
    if OBJECTIVE_REGISTRY.get(protocol.schema.version) != protocol.schema:
        raise ValueError("unsupported_measurement_schema")
    py_state, np_state = random.getstate(), np.random.get_state()
    import torch
    torch_state = torch.random.get_rng_state()
    episodes = []
    try:
        for seed in protocol.seeds:
            random.seed(seed)
            np.random.seed(seed)
            torch.random.default_generator.manual_seed(seed)
            policy = registration.new_policy(seed)
            env = env_factory(seed)
            try:
                validate_environment(env, protocol.schema)
                obs, _ = env.reset(seed=seed)
                if hasattr(env, "env") and hasattr(env.env, "action_space"):
                    env.env.action_space.seed(seed)
                executor = SkillExecutor(env, policy, gamma=protocol.gamma,
                                         max_steps=protocol.horizon)
                payoff, objectives, terminated = executor.run_episode(initial_obs=obs)
                info = executor.last_run_info
                episodes.append(EpisodeEvidence(seed, payoff, tuple(objectives), info["steps"],
                                                terminated, info["truncated"], info["stop_reason"]))
            finally:
                env.close()
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)
    registration.verify()
    return EvaluationEvidence(registration.skill_id, protocol, tuple(episodes))

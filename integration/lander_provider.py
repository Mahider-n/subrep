"""Initial controlled Python controllers and native MetaMo descriptors.

Descriptors are declared predictions for MAGUS, not certification evidence.
Every factory creates fresh local state and RNG; PPO uses the tracked checkpoint.
The comparison idle policy is registered separately and is never a default backup.
"""
from pathlib import Path
import numpy as np
import torch

from pilot.rl_pilot import RLPilot


def build_lander_provider(native, registry, checkpoint):
    checkpoint = Path(checkpoint).resolve()
    source = str(Path(__file__).resolve())
    pilot_source = str(Path(__file__).resolve().parents[1] / "pilot" / "rl_pilot.py")

    def descriptor(name, risk, correlations):
        return native.Action(name, np.asarray(correlations, dtype=float), risk,
                             np.asarray(correlations, dtype=float) * 0.02)

    controllers = []
    for name, fixed in (("comparison_idle", 0), ("noop", 0), ("left_engine", 1),
                        ("main_engine", 2), ("right_engine", 3)):
        risk = 0.8 if fixed else 0.7
        action = descriptor(name, risk, [0, 0, 0, 0, 0, 0.2, 0, 0])

        def factory(seed, value=fixed):
            return lambda obs: (value, 1.0)

        entry = registry.register(action, behavior_version="v1", artifact_paths=[source],
                                  configuration={"fixed_action": fixed}, factory=factory)
        if name == "comparison_idle":
            baseline = entry
        else:
            controllers.append(action)

    for name, noise, switch in (("ppo_deterministic", 0.0, None),
                                ("ppo_then_side_tradeoff", 0.0, 180),
                                ("ppo_noisy_actions", 0.3, None)):
        risk = 0.2 if not noise else 0.5
        action = descriptor(name, risk, [0.8, 0.2, 0.6, 0.1, 0.0, 0.3, 0.7, 0.0])

        def factory(seed, probability=noise, switch_step=switch):
            with torch.random.fork_rng(devices=[]):
                pilot = RLPilot.load(checkpoint, map_location="cpu")
            rng = np.random.default_rng(seed)
            counter = 0

            def policy(obs):
                nonlocal counter
                counter += 1
                if switch_step is not None and counter >= switch_step:
                    return 3, 1.0
                action_id, likelihood = pilot.predict(obs, deterministic=True, return_probability=True)
                if rng.random() < probability:
                    alternatives = [i for i in range(4) if i != action_id]
                    return int(rng.choice(alternatives)), probability / 3
                return action_id, float(likelihood)
            return policy

        registry.register(action, behavior_version="v1", artifact_paths=[source, pilot_source, checkpoint],
                          configuration={"checkpoint": checkpoint.name, "noise_probability": noise,
                                         "switch_step": switch, "side_action": 3}, factory=factory)
        controllers.append(action)
    return baseline, tuple(controllers)

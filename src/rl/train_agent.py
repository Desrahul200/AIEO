# src/rl/train_agent.py
import os
import time
import mlflow
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback

from envs.routing_env import RoutingEnv, RoutingConfig

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow_server:5000")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "Phase5_RoutingRL")
TOTAL_STEPS = int(os.getenv("TOTAL_STEPS", "200000"))
SEED = int(os.getenv("SEED", "42"))

# where to export artifacts locally before logging
ART_DIR = Path("artifacts/rl_policy")
ART_DIR.mkdir(parents=True, exist_ok=True)

class MLflowCallback(BaseCallback):
    def __init__(self, check_freq=5000, verbose=0):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.ep_rewards = []

    def _on_step(self) -> bool:
        # collect episodic rewards via infos if desired; here we log rollout-based averages
        if self.n_calls % self.check_freq == 0:
            mlflow.log_metric("timesteps", self.num_timesteps)
            # SB3 logger keeps running logs; we can also sample env stats if needed
        return True

def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    cfg = RoutingConfig()
    env = RoutingEnv(cfg)
    env.reset(seed=SEED)

    with mlflow.start_run(run_name=f"ppo_routing_{int(time.time())}") as run:
        mlflow.set_tags({"phase": "5", "component": "rl_routing", "algo": "PPO"})
        mlflow.log_params({
            "total_steps": TOTAL_STEPS,
            "seed": SEED,
            "cost_ll": cfg.cost_ll,
            "cost_batch": cfg.cost_batch,
            "penalty_skip_pos": cfg.penalty_skip_pos,
            "p_purchase": cfg.p_purchase,
            "noise": cfg.noise
        })

        # Configure SB3 logger directory (will also be logged as artifacts)
        tmp_log_dir = ART_DIR / "sb3_logs"
        tmp_log_dir.mkdir(parents=True, exist_ok=True)
        new_logger = configure(str(tmp_log_dir), ["stdout", "csv"])
        model = PPO("MlpPolicy", env, verbose=0, seed=SEED)
        model.set_logger(new_logger)

        model.learn(total_timesteps=TOTAL_STEPS, callback=MLflowCallback(check_freq=10000))

        # Export policy
        policy_path = ART_DIR / "policy.zip"
        model.save(str(policy_path))

        # Log artifacts
        mlflow.log_artifact(str(policy_path), artifact_path="rl_policy")
        mlflow.log_artifact(str(tmp_log_dir), artifact_path="rl_policy/sb3_logs")

        print(f"Saved policy to: {policy_path}")
        print(f"Run ID: {run.info.run_id}")

if __name__ == "__main__":
    main()

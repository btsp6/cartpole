import numpy as np

from gymnasium.envs.classic_control import CartPoleEnv

from .base_model import CartPoleModel


class RandomModel(CartPoleModel):
    def get_action(self, env: CartPoleEnv) -> int | np.ndarray:
        return env.action_space.sample()  # Random policy for demonstration

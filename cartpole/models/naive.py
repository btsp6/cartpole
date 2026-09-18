import numpy as np
from gymnasium.envs.classic_control import CartPoleEnv

from .base_model import CartPoleModel


class NaiveModel(CartPoleModel):
    def get_action(self, env: CartPoleEnv) -> int | np.ndarray:
        x, x_dot, theta, theta_dot = self.get_state(env)
        if 5 * (theta + 2* env.tau * theta_dot) + (x + env.tau * x_dot) > 0:
            return 1
        else:
            return 0

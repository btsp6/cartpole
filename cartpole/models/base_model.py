import abc

import numpy as np
from gymnasium.envs.classic_control import CartPoleEnv


class CartPoleModel:

    def get_state(self, env: CartPoleEnv) -> tuple[float, float, float, float]:
        if env.state is None:
            raise ValueError("Environment state is None. Make sure to reset the environment before calling get_state.")
        return tuple(env.state.tolist())

    @abc.abstractmethod
    def get_action(self, env: CartPoleEnv) -> int | np.ndarray:
        ...

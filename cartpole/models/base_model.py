import abc
import numpy as np

from gymnasium.envs.classic_control import CartPoleEnv


class CartPoleModel:

    @abc.abstractmethod
    def get_action(self, env: CartPoleEnv) -> int | np.ndarray:
        ...

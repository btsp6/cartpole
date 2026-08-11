import contextlib
from typing import Iterator

import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from models import NaiveModel


@contextlib.contextmanager
def cartpole_env() -> Iterator[gym.Env]:
	try:
		env = gym.make("CartPole-v1", render_mode="rgb_array")
		env = RecordVideo(
			env,
			video_folder="videos/cartpole-agent",
			episode_trigger=lambda x: True,
		)
		yield env
	finally:
		env.close()

model = NaiveModel()

with cartpole_env() as env:
	env.reset()

	episode_over = False
	while not episode_over:
		action = model.get_action(env.unwrapped)

		obs, reward, terminated, truncated, info = env.step(action)

		episode_over = terminated or truncated

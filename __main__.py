

import random

from icm import PPO
from environment import SimpleEnv, AtariEnv

"""
Environments:

BreakoutDeterministic-v4
Pendulum-v0

"""


def main(env_name="Pendulum-v1",
         total_timesteps=100_000_000
         ):

    env = SimpleEnv(env_name)
    agent = PPO(env.state_size, env.action_size)

    # train the agent
    agent.learn(env, total_timesteps=total_timesteps)


if __name__ == "__main__":
    main()


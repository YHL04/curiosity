

import gym
from gym.spaces import Discrete
import random


class SimpleEnv:

    def __init__(self, env_name):
        self.env = gym.make(env_name)
        if type(self.env.action_space) == Discrete:
            self.discrete = True
        else:
            self.discrete = False

    @property
    def state_size(self):
        return self.env.observation_space.shape[0]

    @property
    def action_size(self):
        if self.discrete:
            return self.env.action_space.n
        else:
            return self.env.action_space.shape[0]

    def reset(self):
        return self.env.reset()

    def step(self, action, bound=2):
        if not self.discrete:
            action *= bound

        return self.env.step(action)

    def render(self):
        self.env.render()


class AtariEnv:

    def __init__(self, env_name, auto_start=True, training=True, no_op_max=50):
        self.env = gym.make(env_name, render_mode="human")
        self.discrete = True
        self.last_lives = 0

        self.auto_start = auto_start
        self.fire = False

        self.training = training
        self.no_op_max = no_op_max

    @property
    def state_size(self):
        return self.env.observation_space.shape

    @property
    def action_size(self):
        return self.env.action_space.n

    def reset(self):
        if self.auto_start:
            self.fire = True

        frame, _ = self.env.reset()

        if not self.training:
            for i in range(random.randint(1, self.no_op_max)):
                frame, _, _, _ = self.env.step(1)

        frame = frame[::2, ::2, :]
        return frame

    def step(self, action):
        if self.fire:
            action = 1
            self.fire = False

        frame, reward, _, terminal, info = self.env.step(action)

        if info["lives"] < self.last_lives:
            life_lost = True
            self.fire = True

        else:
            life_lost = terminal

        self.last_lives = info["lives"]

        frame = frame[::2, ::2, :]
        return frame, reward, terminal, life_lost

    def render(self):
        """Called at each timestep to render"""
        self.env.render()


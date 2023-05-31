

import gym
import random

from utils import repeat_upsample


class SimpleEnv:

    def __init__(self, env_name):
        self.env = gym.make(env_name)

    @property
    def state_size(self):
        return self.env.observation_space.shape[0]

    @property
    def action_size(self):
        return self.env.action_space.shape[0]

    def reset(self):
        return self.env.reset()

    def step(self, action, bound=2):
        return self.env.step(action * bound)

    def render(self):
        self.env.render()


class AtariEnv:

    def __init__(self, env_name, auto_start=True, training=True, no_op_max=50):
        self.env = gym.make(env_name, render_mode="rgb_array")
        self.last_lives = 0

        self.auto_start = auto_start
        self.fire = False

        self.training = training
        self.no_op_max = no_op_max

    @property
    def action_size(self):
        return self.env.action_space.shape[0]

    def reset(self):
        if self.auto_start:
            self.fire = True

        frame = self.env.reset()

        if not self.training:
            for i in range(random.randint(1, self.no_op_max)):
                frame, _, _, _ = self.env.step(1)

        return frame

    def step(self, action):
        if self.fire:
            action = 1
            self.fire = False

        frame, reward, terminal, info = self.env.step(action)

        if info["lives"] < self.last_lives:
            life_lost = True
            self.fire = True

        else:
            life_lost = terminal

        self.last_lives = info["lives"]

        return frame, reward, terminal, life_lost

    def render(self, scale=3):
        """Called at each timestep to render"""
        rgb = self.env.render("rgb_array")
        upscaled = repeat_upsample(rgb, scale, scale)
        viewer.imshow(upscaled)

        return upscaled

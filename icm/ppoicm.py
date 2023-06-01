

import torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Categorical
from torch.optim import Adam

import numpy as np

from .model import FeedForward, ConvNet
from .icm import ICM


class PPOICM:
    """PPO with Intrinsic Curiosity Module"""
    # ppo
    timesteps_per_batch = 4800
    max_timesteps_per_episode = 1600
    n_updates_per_iteration = 5
    lr = 0.005
    gamma = 0.95
    clip = 0.2

    # icm
    intr_reward_strength = 1  # default is 0.02

    # render
    render = True
    render_every_i = 10

    def __init__(self, state_size, action_size, discrete, device="cuda"):
        self.state_size = state_size
        self.action_size = action_size
        self.discrete = discrete
        self.device = device

        if type(state_size) == int:
            self.actor = FeedForward(state_size, action_size, discrete=discrete).to(self.device)
            self.critic = FeedForward(state_size, 1).to(self.device)
        elif type(state_size) == tuple:
            self.actor = ConvNet(3, action_size, discrete=discrete).to(self.device)
            self.critic = ConvNet(3, 1).to(self.device)

        self.actor_opt = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_opt = Adam(self.critic.parameters(), lr=self.lr)

        # covariance matrices
        if not self.discrete:
            self.cov_var = torch.full(size=(action_size,), fill_value=0.5)
            self.cov_mat = torch.diag(self.cov_var)

        # icm
        self.icm = ICM(state_size, action_size).to(self.device)
        self.icm_opt = Adam(self.icm.parameters(), lr=self.lr)

        # logging
        self.log_batch_rews = None
        self.log_i = 0

    def get_action(self, state):
        # Query the actor network for a mean action

        if self.discrete:
            probs = self.actor(state.unsqueeze(0))
            dist = Categorical(probs)

        else:
            mean = self.actor(state.unsqueeze(0))
            dist = MultivariateNormal(mean, self.cov_mat)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        # Return the sampled action and the log probability of that action in our distribution
        return action.squeeze().cpu().detach().numpy(), log_prob.detach()

    def compute_rtgs(self, batch_rews, batch_intr_rews):
        """Compute the Reward-To-Go of each timestep in a batch given the rewards"""

        # The rewards-to-go (rtg) per episode per batch to return.
        # The shape will be (num timesteps per episode)
        batch_rtgs = []

        for ep_rews, ep_intr_rews in zip(reversed(batch_rews), reversed(batch_intr_rews)):
            discounted_reward = 0

            for rew, intr_rew in zip(reversed(ep_rews), reversed(ep_intr_rews)):
                discounted_reward = (rew + intr_rew) + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)

        # Convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float32, device=self.device)

        return batch_rtgs

    def learn(self, env, total_timesteps):

        t, i = 0, 0
        while t < total_timesteps:
            # Autobots, roll out (just kidding, we're collecting our batch simulations here)
            batch_obs, batch_next_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout(env)

            # Calculate how many timesteps we collected this batch
            t += np.sum(batch_lens)

            # Increment the number of iterations
            i += 1
            self.log_i = i

            # Calculate advantage at k-th iteration
            V, _ = self.evaluate(batch_obs, batch_acts)
            A_k = batch_rtgs - V.detach()

            # Normalizing advantages makes it much more stable
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            # This is the loop where we update our network for some n epochs
            for _ in range(self.n_updates_per_iteration):
                # Calculate V_phi and pi_theta(a_t | s_t)
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

                # Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
                # NOTE: we just subtract the logs, which is the same as
                # dividing the values and then canceling the log with e^log.
                # For why we use log probabilities instead of actual probabilities,
                # here's a great explanation:
                # https://cs.stackexchange.com/questions/70518/why-do-we-use-the-log-in-gradient-based-reinforcement-algorithms
                # TL;DR makes gradient ascent easier behind the scenes.
                ratios = torch.exp(curr_log_probs - batch_log_probs)

                # Calculate surrogate losses.
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                # Calculate actor and critic losses.
                # NOTE: we take the negative min of the surrogate losses because we're trying to maximize
                # the performance function, but Adam minimizes the loss. So minimizing the negative
                # performance function maximizes it.
                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = F.mse_loss(V, batch_rtgs)

                # Calculate gradients and perform backward propagation for actor network
                self.actor_opt.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_opt.step()

                # Calculate gradients and perform backward propagation for critic network
                self.critic_opt.zero_grad()
                critic_loss.backward()
                self.critic_opt.step()

                # perform icm update
                self.icm_opt.zero_grad()
                loss, _, _ = self.icm.forward(batch_acts, batch_obs, batch_next_obs)
                loss.backward()
                self.icm_opt.step()

            self.print_logs()

    def evaluate(self, batch_obs, batch_acts):
        """
        Estimate the values of each observation, and the log probs of
        each action in the most recent batch with the most recent
        iteration of the actor network.
        """

        # Query critic network for a value V for each batch_obs. Shape of V should be same as batch_rtgs
        V = self.critic(batch_obs).squeeze()

        # Calculate the log probabilities of batch actions using most recent actor network.
        # This segment of code is similar to that in get_action()
        if self.discrete:
            probs = self.actor(batch_obs)
            dist = Categorical(probs)
        else:
            mean = self.actor(batch_obs)
            dist = MultivariateNormal(mean, self.cov_mat)

        log_probs = dist.log_prob(batch_acts)

        # Return the value vector V of each observation in the batch
        # and log probabilities log_probs of each action in the batch
        return V, log_probs

    def rollout(self, env):

        # Batch data
        batch_obs = []
        batch_next_obs = []

        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_intr_rews = []
        batch_lens = []

        t = 0  # Keeps track of how many timesteps we've run so far this batch

        # Keep simulating until we've run more than or equal to specified timesteps per batch
        while t < self.timesteps_per_batch:
            ep_rews = []
            ep_intr_rews = []

            obs = env.reset()
            done = False

            for ep_t in range(self.max_timesteps_per_episode):
                if self.render and (self.log_i % self.render_every_i == 0) and len(batch_lens) == 0:
                    env.render()

                t += 1

                action, log_prob = self.get_action(torch.tensor(obs, dtype=torch.float32, device=self.device))
                next_obs, rew, done, _ = env.step(action)

                # add intrinsic reward
                intr_rew = self.icm.get_intrinsic_reward(act=torch.tensor(action, device=self.device),
                                                         curr_state=torch.tensor(obs, dtype=torch.float32, device=self.device),
                                                         next_state=torch.tensor(next_obs, dtype=torch.float32, device=self.device),
                                                         intr_reward_strength=self.intr_reward_strength)

                batch_obs.append(obs)
                batch_next_obs.append(next_obs)

                ep_rews.append(rew)
                ep_intr_rews.append(intr_rew)

                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                obs = next_obs

                if done:
                    break

            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)
            batch_intr_rews.append(ep_intr_rews)

        batch_obs = torch.tensor(np.array(batch_obs), dtype=torch.float32, device=self.device)
        batch_next_obs = torch.tensor(np.array(batch_obs), dtype=torch.float32, device=self.device)
        batch_acts = torch.tensor(np.array(batch_acts), device=self.device)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float32, device=self.device)

        batch_rtgs = self.compute_rtgs(batch_rews, batch_intr_rews)

        self.log_batch_rews = batch_rews
        self.log_batch_intr_rews = batch_intr_rews

        return batch_obs, batch_next_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

    def print_logs(self):
        avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.log_batch_rews])
        avg_ep_intr_rews = np.mean([np.sum(ep_rews) for ep_rews in self.log_batch_intr_rews])

        print(f"rewards {avg_ep_rews} intrinsic rewards {avg_ep_intr_rews}")


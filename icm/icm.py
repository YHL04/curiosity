

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import FeedForward, ConvNet


class ICM(nn.Module):
    """
    Intrinsic Curiosity Module

    Composed of an encoder, a forward net and an inverse net
    Encoder: Encodes the state into a feature vector
    Forward net: Takes in current feature vector and action
                 to predict next feature vector
    Inverse net: Takes current feature vector and next feature
                 vector to predict the action taken

    """

    def __init__(self, state_size, action_size, d_model=128):
        super(ICM, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.d_model = d_model

        if type(state_size) == int:
            self.encoder = FeedForward(state_size, d_model, d_model, n_layers=0)
        elif type(state_size) == tuple:
            self.encoder = ConvNet(channels=state_size[2], out_dim=d_model)

        self.forward_net = FeedForward(d_model + action_size, d_model, d_model, n_layers=0)
        self.inverse_net = FeedForward(d_model * 2, action_size, d_model, n_layers=0)

    @torch.no_grad()
    def get_intrinsic_reward(self, act, curr_state, next_state, intr_reward_strength=0.02):
        if act.dtype == torch.int64:
            act = F.one_hot(act, num_classes=self.action_size).float()

        act = act.unsqueeze(0)
        curr_enc = self.encoder(curr_state.unsqueeze(0))
        next_enc = self.encoder(next_state.unsqueeze(0))

        # Forward net
        pred_next_enc = self.forward_net(torch.concat((act, curr_enc), dim=-1))

        # Intrinsic Reward
        intr_reward = 0.5 * F.mse_loss(pred_next_enc, next_enc, reduction='none')
        intr_reward = intr_reward.mean(dim=-1)
        intr_reward = torch.clamp(intr_reward_strength * intr_reward, 0, 1)

        return intr_reward.squeeze()

    def forward(self, act, curr_state, next_state):
        """
        Forward Intrinsic Curiosity Module for continuous action PPO

        Parameters:
            act: recorded action
            curr_state: state when action was taken
            next_state: state after action was taken

        Returns:
            inv_loss: loss of inverse net
            forw_loss: loss of forward net

        """
        if act.dtype == torch.int64:
            act = F.one_hot(act, num_classes=self.action_size).float()
            discrete = True
        else:
            discrete = False

        curr_enc = self.encoder(curr_state)
        next_enc = self.encoder(next_state)

        # Inverse net
        pred_act = self.inverse_net(torch.concat((curr_enc, next_enc), dim=-1))

        inv_loss = F.mse_loss(pred_act, act, reduction='none').mean()

        # Forward net
        pred_next_enc = self.forward_net(torch.concat((act, curr_enc), dim=-1))

        # Forward Loss
        intr_reward = 0.5 * F.mse_loss(pred_next_enc, next_enc, reduction='none')
        intr_reward = intr_reward.mean(dim=-1)
        forw_loss = intr_reward.mean()

        # Weighted loss
        loss = 10 * (0.2 * forw_loss + 0.8 * inv_loss)

        return loss, inv_loss, forw_loss


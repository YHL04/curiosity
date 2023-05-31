

import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def __init__(self, state_size, action_size, d_model):
        super(ICM, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.d_model = d_model

        self.encoder = nn.Sequential(
            nn.Linear(state_size, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU()
        )
        self.forward_net = nn.Sequential(
            nn.Linear(d_model + action_size, 256),
            nn.ReLU(),
            nn.Linear(256, d_model)
        )
        self.inverse_net = nn.Sequential(
            nn.Linear(d_model * 2, 256),
            nn.ReLU(),
            nn.Linear(256, action_size)
        )

    def forward(self, act, curr_state, next_state):
        """
        Parameters:
            act: recorded action
            curr_state: state when action was taken
            next_state: state after action was taken

        Returns:
            intr_reward: intrinsic reward
            inv_loss: loss of inverse net
            forw_loss: loss of forward net

        """
        curr_enc = self.encoder(curr_state)
        next_enc = self.encoder(next_state)

        # Inverse net
        pred_act = self.inverse_net(torch.concat((curr_enc, next_enc), dim=-1))
        inv_loss = F.cross_entropy(pred_act, act, reduction='none').mean()

        # Forward net
        one_hot_act = F.one_hot(act, num_classes=self.action_size)
        pred_next_enc = self.forward_net(torch.concat((one_hot_act.float(), curr_enc), dim=-1))

        # Intrinsic Reward
        intr_reward = 0.5 * F.mse_loss(pred_next_enc, next_enc, reduction='none')
        intr_reward = intr_reward.mean(dim=-1)

        # Forward loss
        forw_loss = intr_reward.mean()
        return intr_reward, inv_loss, forw_loss

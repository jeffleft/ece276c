import torch
import torch.nn as nn
import torch.nn.functional as F
from distributions import Categorical, DiagGaussian, CategoricalNoFC, DiagGaussianNoFC
from utils import init, init_normc_
import numpy as np
# from torch.distributions import Normal
import pdb


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, recurrent_policy, algo):
        super(Policy, self).__init__()

        if algo == 'ppo_shared' or algo == 'acktr_shared':
            self.shared = True
        else:
            self.shared = False
        if algo == 'ppo_unshared' or algo == 'acktr_unshared':
            self.unshared = True
        else:
            self.unshared = False

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
        else:
            raise NotImplementedError

        if len(obs_shape) == 3:
            if self.shared:
                assert not recurrent_policy, \
                    "Recurrent policy is not implemented for the shared fn approximator"
                self.base = CNNBaseShared(obs_shape[0], recurrent_policy, num_outputs)
            elif self.unshared:
                assert not recurrent_policy, \
                    "Recurrent policy is not implemented for the unshared fn approximator"
                self.base = CNNBaseUnshare(obs_shape[0], recurrent_policy, num_outputs)
            else:
                self.base = CNNBase(obs_shape[0], recurrent_policy)
        elif len(obs_shape) == 1:
            assert not recurrent_policy, \
                "Recurrent policy is not implemented for the MLP controller"
            if self.shared:
                self.base = MLPBaseShared(obs_shape[0], num_outputs)
            elif self.unshared:
                # redundant
                raise NotImplementedError
            else:
                self.base = MLPBase(obs_shape[0])
        else:
            raise NotImplementedError

        if self.shared or self.unshared:
            if action_space.__class__.__name__ == "Discrete":
                self.dist = CategoricalNoFC()
            elif action_space.__class__.__name__ == "Box":
                num_outputs = action_space.shape[0]
                self.dist = DiagGaussianNoFC(num_outputs)
            else:
                raise NotImplementedError
        else:
            if action_space.__class__.__name__ == "Discrete":
                num_outputs = action_space.n
                self.dist = Categorical(self.base.output_size, num_outputs)
            elif action_space.__class__.__name__ == "Box":
                num_outputs = action_space.shape[0]
                self.dist = DiagGaussian(self.base.output_size, num_outputs)
            else:
                raise NotImplementedError

        self.state_size = self.base.state_size

    def forward(self, inputs, states, masks):
        raise NotImplementedError

    def act(self, inputs, states, masks, deterministic=False):
        #TODO: Change for shared model -- changed distributions instead
        value, actor_features, states = self.base(inputs, states, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, states

    def get_value(self, inputs, states, masks):
        value, _, _ = self.base(inputs, states, masks)
        return value

    def evaluate_actions(self, inputs, states, masks, action):
        #TODO: Change for shared model -- changed distributions instead
        value, actor_features, states = self.base(inputs, states, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, states


class CNNBase(nn.Module):
    def __init__(self, num_inputs, use_gru):
        super(CNNBase, self).__init__()

        init_ = lambda m: init(m,
                      nn.init.orthogonal_,
                      lambda x: nn.init.constant_(x, 0),
                      nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)),
            nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)),
            nn.ReLU(),
            Flatten(),
            init_(nn.Linear(32 * 7 * 7, 512)),
            nn.ReLU()
        )

        if use_gru:
            self.gru = nn.GRUCell(512, 512)
            nn.init.orthogonal_(self.gru.weight_ih.data)
            nn.init.orthogonal_(self.gru.weight_hh.data)
            self.gru.bias_ih.data.fill_(0)
            self.gru.bias_hh.data.fill_(0)

        init_ = lambda m: init(m,
          nn.init.orthogonal_,
          lambda x: nn.init.constant_(x, 0))

        self.critic_linear = init_(nn.Linear(512, 1))

        self.train()

    @property
    def state_size(self):
        if hasattr(self, 'gru'):
            return 512
        else:
            return 1

    @property
    def output_size(self):
        return 512

    def forward(self, inputs, states, masks):
        x = self.main(inputs / 255.0)

        if hasattr(self, 'gru'):
            if inputs.size(0) == states.size(0):
                x = states = self.gru(x, states * masks)
            else:
                x = x.view(-1, states.size(0), x.size(1))
                masks = masks.view(-1, states.size(0), 1)
                outputs = []
                for i in range(x.size(0)):
                    hx = states = self.gru(x[i], states * masks[i])
                    outputs.append(hx)
                x = torch.cat(outputs, 0)

        return self.critic_linear(x), x, states


class CNNBaseShared(nn.Module):
    def __init__(self, num_inputs, use_gru, num_actions):
        super(CNNBaseShared, self).__init__()

        init_ = lambda m: init(m,
                      nn.init.orthogonal_,
                      lambda x: nn.init.constant_(x, 0),
                      nn.init.calculate_gain('relu'))

        init2_ = lambda m: init(m,
          nn.init.orthogonal_,
          lambda x: nn.init.constant_(x, 0))

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)),
            nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)),
            nn.ReLU(),
            Flatten(),
            init_(nn.Linear(32 * 7 * 7, 512)),
            nn.ReLU(),
            init2_(nn.Linear(512, 1+num_actions))
        )

        if use_gru:
            raise NotImplementedError

        self.train()

    @property
    def state_size(self):
            return 1

    @property
    def output_size(self):
        return num_actions #doesn't matter actually

    def forward(self, inputs, states, masks):
        x = self.main(inputs / 255.0)

        return x[:,0].view(-1,1), x[:,1:], states

class CNNBaseUnshare(nn.Module):
    def __init__(self, num_inputs, use_gru, num_actions):
        super(CNNBaseUnshare, self).__init__()

        init_ = lambda m: init(m,
                      nn.init.orthogonal_,
                      lambda x: nn.init.constant_(x, 0),
                      nn.init.calculate_gain('relu'))

        init2_ = lambda m: init(m,
          nn.init.orthogonal_,
          lambda x: nn.init.constant_(x, 0))

        self.actor = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)),
            nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)),
            nn.ReLU(),
            Flatten(),
            init_(nn.Linear(32 * 7 * 7, 512)),
            nn.ReLU(),
            init2_(nn.Linear(512, num_actions))
        )

        self.critic = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)),
            nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)),
            nn.ReLU(),
            Flatten(),
            init_(nn.Linear(32 * 7 * 7, 512)),
            nn.ReLU(),
            init2_(nn.Linear(512, 1))
        )

        if use_gru:
            raise NotImplementedError

        self.train()

    @property
    def state_size(self):
            return 1

    @property
    def output_size(self):
        return num_actions #doesn't matter actually

    def forward(self, inputs, states, masks):
        q_vals = self.actor(inputs / 255.0)
        value  = self.critic(inputs / 255.0)
        pdb.set_trace()
        return value, q_vals, states


class MLPBase(nn.Module):
    def __init__(self, num_inputs):
        super(MLPBase, self).__init__()

        init_ = lambda m: init(m,
              init_normc_,
              lambda x: nn.init.constant_(x, 0))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, 64)),
            nn.Tanh(),
            init_(nn.Linear(64, 64)),
            nn.Tanh()
        )

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, 64)),
            nn.Tanh(),
            init_(nn.Linear(64, 64)),
            nn.Tanh()
        )

        self.critic_linear = init_(nn.Linear(64, 1))

        self.train()

    @property
    def state_size(self):
        return 1

    @property
    def output_size(self):
        return 64

    def forward(self, inputs, states, masks):
        hidden_critic = self.critic(inputs)
        hidden_actor = self.actor(inputs)

        return self.critic_linear(hidden_critic), hidden_actor, states

class MLPBaseShared(nn.Module):
    def __init__(self, num_inputs, num_actions, std=0):
        super(MLPBaseShared, self).__init__()

        init_ = lambda m: init(m,
              init_normc_,
              lambda x: nn.init.constant_(x, 0))

        self.shared = nn.Sequential(
            init_(nn.Linear(num_inputs, 64)),
            nn.Tanh(),
            init_(nn.Linear(64, 64)),
            nn.Tanh(),
            init_(nn.Linear(64, 1+num_actions))
        )

        self.train()

        self.log_std = nn.Parameter(torch.ones(1, num_actions) * std)

    @property
    def state_size(self):
        return 1

    @property
    def output_size(self):
        return num_actions #doesn't matter actually

    def forward(self, inputs, states, masks):
        outputs = self.shared(inputs)
#        value = outputs[:,0] # first dimension is batch
#        mu = outputs[:,1:]
#        std = self.log_std.exp().expand_as(mu)
#        dist = Normal(mu, std)
        value  = outputs[:,0].view(-1,1)
        q_vals = outputs[:,1:]

        return value, q_vals, states
from typing import Iterable
import numpy as np
from torch import nn
import torch
from torch.nn import functional as F

class PiApproximationWithNN():
    def __init__(self,
                 state_dims,
                 num_actions,
                 alpha):
        """
        state_dims: the number of dimensions of state space
        action_dims: the number of possible actions
        alpha: learning rate
        """

        self.action_space = num_actions 
        self.model = nn.Sequential(
            nn.Linear(state_dims, 32),
            nn.ReLU(inplace = True),
            nn.Linear(32,32),
            nn.ReLU(inplace= True),
            nn.Linear(32,num_actions),
            nn.Softmax(dim=-1)
        )
        self.optim = torch.optim.Adam(self.model.parameters(), lr=alpha, betas=(.9, .999))

    def __call__(self,s) -> int:
        self.model.eval()
        s = torch.from_numpy(np.float32(s))
        s = s.reshape((1, -1))
        action_prob = self.model(s)
        action_prob = action_prob[0].detach().numpy()

        return np.random.choice(self.action_space, p=action_prob)
        
    def update(self, s, a, gamma_t, delta):
        """
        s: state S_t
        a: action A_t
        gamma_t: gamma^t
        delta: G-v(S_t,w)
        """
        self.model.train()
        s = torch.from_numpy(np.float32(s))
        s = s.reshape((1,-1))
        a = torch.from_numpy(np.asarray(a)).long()
        a = a.reshape((1,-1))[0]

        self.optim.zero_grad()
        loss = nn.CrossEntropyLoss()
        loss = gamma_t * delta  * loss(self.model(s), a) 
        loss.backward()
        self.optim.step()

  
class Baseline(object):
    """
    a constant for every state
    """
    def __init__(self,b):
        self.b = b

    def __call__(self,s) -> float:
        return self.b

    def update(self,s,G):
        pass

class VApproximationWithNN(Baseline):
    def __init__(self,
                 state_dims,
                 alpha):
        """
        state_dims: the number of dimensions of state space
        alpha: learning rate
        """
        self.model = nn.Sequential(
            nn.Linear(state_dims, 32),
            nn.ReLU(inplace = True),
            nn.Linear(32,32),
            nn.ReLU(inplace= True),
            nn.Linear(32,1)
        )
        self.optim = torch.optim.Adam(self.model.parameters(), lr=1e-4, betas=(.9, .999))


    def __call__(self,s) -> float:
        # TODO: implement this method
        self.model.eval()
        s = torch.from_numpy(np.float32(s))
        s = s.reshape((1, -1))
        val = self.model(s)
        val = val[0].detach().numpy()
        return val[0]

    def update(self,s,G):
        self.model.train()
        s = torch.from_numpy(np.float32(s))
        s = s.reshape((1,-1))
        G = torch.tensor(np.float32(G))
        G = G.reshape((1, -1))
        
        loss = .5 * F.mse_loss(self.model(s), G)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

def REINFORCE(
    env, #open-ai environment
    gamma:float,
    num_episodes:int,
    pi:PiApproximationWithNN,
    V:Baseline) -> Iterable[float]:
    """
    implement REINFORCE algorithm with and without baseline.

    input:
        env: target environment; openai gym
        gamma: discount factor
        num_episode: #episodes to iterate
        pi: policy
        V: baseline
    output:
        a list that includes the G_0 for every episodes.
    """
    # TODO: implement this method
    G_0 = []

    for i in range(num_episodes):
    
        env.reset()
        state = env.state 
        action_space = [i for i in range(env.action_space.n )]
        states = [state]
        actions = []
        rewards = []

        done = False
        while not done:
            action = pi(state)
            next_state, reward, done, info = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            state = next_state
        
        G = []
        g_t = 0
        for r in rewards[::-1]:
            g_t = r + gamma*g_t
            G.append(g_t)
        G.reverse()

        for t in range(len(G)):
            delta = G[t] - V(states[t])
            V.update(states[t], G[t])
            pi.update(states[t], actions[t], gamma**t,  delta)

            if t == 0:
                G_0.append(G[t])
            
    return G_0


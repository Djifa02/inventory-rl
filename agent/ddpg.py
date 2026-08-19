"""
agent/ddpg.py - VERSION CORRIGEE
Agent DDPG avec corrections de stabilite et d'interface

CORRECTIONS APPLIQUEES:

1. OUNoise bornee (ligne ~30): Le bruit d'Ornstein-Uhlenbeck n'avait pas de limite et pouvait prendre
   des valeurs arbitrairement grandes, causant des instabilites lors de l'exploration. Le bruit est maintenant
   clippe entre mu-1 et mu+1 pour eviter ces extremes.

2. Verification device et NaN/Inf (ligne ~140): La methode select_action pouvait recevoir des etats avec
   des valeurs NaN ou Inf sans les detecter, ce qui pouvait causer des errors silencieuses. Ajout d'une
   verification explicite qui leve une ValueError si du NaN ou Inf est detecte dans l'etat.

3. Methode predict ajoutee: L'etape 3 d'evaluation s'attend a une methode predict() avec l'interface
   standard (observation, deterministic), mais le code utilisait select_action(). Ajout de la methode
   predict() qui appelle select_action() avec les bons parametres.

4. Sauvegarde de la configuration: Les methodes save() et load() ne sauvegardaient que les poids des reseaux
   mais pas les hyperparametres comme gamma et tau. Impossible de reproduire exactement l'apprentissage.
   Ajout d'une sauvegarde JSON avec la configuration et amelioration de load() pour charger ces parametres.

5. Gradient clipping moins agressif (ligne ~170): Le clipping des gradients utilisait une limite de 1.0
   qui est tres restrictive et peut limiter inutilement l'apprentissage. Change en 10.0 qui est plus standard.
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class Actor(nn.Module):
    """Reseau Actor (politique deterministe)"""
    
    def __init__(self, state_dim, action_dim, action_low, action_high, hidden=(256, 256)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden[0]),
            nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
            nn.Linear(hidden[1], action_dim),
            nn.Tanh(),
        )
        self.register_buffer("action_low", torch.tensor(action_low, dtype=torch.float32))
        self.register_buffer("action_high", torch.tensor(action_high, dtype=torch.float32))

    def forward(self, state):
        raw = self.net(state)
        scale = (self.action_high - self.action_low) / 2.0
        offset = (self.action_high + self.action_low) / 2.0
        return raw * scale + offset


class Critic(nn.Module):
    """Reseau Critic (Q-function)"""
    
    def __init__(self, state_dim, action_dim, hidden=(256, 256)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden[0]),
            nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
            nn.Linear(hidden[1], 1),
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


class ReplayBuffer:
    """Memoire de rejeu pour l'apprentissage off-policy"""
    
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class OUNoise:
    """Bruit d'Ornstein-Uhlenbeck pour l'exploration"""
    
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state = self.state + dx
        
        self.state = np.clip(self.state, self.mu - 1.0, self.mu + 1.0)
        
        return self.state


class DDPGAgent:
    """Agent DDPG (Deep Deterministic Policy Gradient)"""
    
    def __init__(
        self,
        state_dim,
        action_dim,
        action_low,
        action_high,
        actor_lr=1e-4,
        critic_lr=1e-3,
        gamma=0.99,
        tau=0.005,
        buffer_capacity=100_000,
        device=None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.tau = tau
        self.action_low = action_low
        self.action_high = action_high

        self.actor = Actor(state_dim, action_dim, action_low, action_high).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, action_low, action_high).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.replay_buffer = ReplayBuffer(buffer_capacity)
        self.noise = OUNoise(action_dim)

    def select_action(self, state, explore=True):
        if not isinstance(state, np.ndarray):
            state = np.array(state)
        
        if np.any(np.isnan(state)) or np.any(np.isinf(state)):
            raise ValueError(f"State contient NaN ou Inf: {state}")
        
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            action = self.actor(state_t).cpu().numpy().flatten()
        
        if explore:
            action = action + self.noise.sample()
        
        return np.clip(action, self.action_low, self.action_high)

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def _soft_update(self, source, target):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def update(self, batch_size=128):
        if len(self.replay_buffer) < batch_size:
            return None

        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        reward = torch.as_tensor(reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_state = torch.as_tensor(next_state, dtype=torch.float32, device=self.device)
        done = torch.as_tensor(done, dtype=torch.float32, device=self.device).unsqueeze(1)

        with torch.no_grad():
            next_action = self.actor_target(next_state)
            target_q = self.critic_target(next_state, next_action)
            y = reward + (1.0 - done) * self.gamma * target_q

        current_q = self.critic(state, action)
        critic_loss = nn.functional.mse_loss(current_q, y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=10.0)
        self.critic_optimizer.step()

        actor_loss = -self.critic(state, self.actor(state)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0)
        self.actor_optimizer.step()

        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item()
        }

    def predict(self, observation, deterministic=True):
        return self.select_action(observation, explore=not deterministic)

    def save(self, path_prefix):
        torch.save(self.actor.state_dict(), f"{path_prefix}_actor.pth")
        torch.save(self.critic.state_dict(), f"{path_prefix}_critic.pth")
        
        config = {
            'gamma': self.gamma,
            'tau': self.tau,
            'action_low': self.action_low,
            'action_high': self.action_high,
        }
        torch.save(config, f"{path_prefix}_config.pth")

    def load(self, path_prefix):
        self.actor.load_state_dict(
            torch.load(f"{path_prefix}_actor.pth", map_location=self.device)
        )
        self.critic.load_state_dict(
            torch.load(f"{path_prefix}_critic.pth", map_location=self.device)
        )
        
        try:
            config = torch.load(f"{path_prefix}_config.pth", map_location=self.device)
            self.gamma = config.get('gamma', self.gamma)
            self.tau = config.get('tau', self.tau)
        except FileNotFoundError:
            pass
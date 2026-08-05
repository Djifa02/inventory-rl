# ============================================================
# Description : Environnement de gestion de stock
#               base sur le Newsvendor Problem
#               Source : arxiv.org/pdf/1607.02177
# ============================================================

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class InventoryEnv(gym.Env):

    def __init__(self, df, episode_length=365):
        super().__init__()
        self.df             = df.reset_index(drop=True)
        self.episode_length = episode_length
        self.cp             = self.df['Price'].values
        self.ch             = self.cp * 0.20
        self.current_stock  = None
        self.observation_space = spaces.Box(
            low   = np.array([50,   -10]),
            high  = np.array([500,  519]),
            dtype = np.float32
        )
        self.action_space = spaces.Box(
            low   = np.array([0]),
            high  = np.array([200]),
            dtype = np.float32
        )
        self.current_step = 0

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step  = 0
        self.current_stock = float(self.df.iloc[0]['Inventory Level'])
        return self._get_obs(), {}

    def _get_obs(self):
        row = self.df.iloc[self.current_step]
        return np.array([
            self.current_stock,
            row['Demand Forecast']
        ], dtype=np.float32)

    def step(self, action):
        action           = float(action[0])
        row              = self.df.iloc[self.current_step]
        stock            = self.current_stock
        demande          = row['Units Sold']
        stock_disponible = stock + action
        ventes           = min(stock_disponible, demande)
        stock_restant    = stock_disponible - ventes
        self.current_stock = stock_restant
        revenu           = row['Price'] * ventes
        cout_stockage    = self.ch[self.current_step] * max(stock_restant, 0)
        cout_rupture     = self.cp[self.current_step] * max(demande - stock_disponible, 0)
        recompense       = revenu - cout_stockage - cout_rupture
        self.current_step += 1
        termine          = self.current_step >= self.episode_length
        obs = self._get_obs() if not termine else np.zeros(2, dtype=np.float32)
        return obs, recompense, termine, False, {}

    def render(self):
        print(f"Jour {self.current_step} | "
              f"Stock : {self.current_stock:.0f} | "
              f"Demande : {self.df.iloc[self.current_step]['Demand Forecast']:.0f}")


def load_env(data_path='data/retail_store_inventory.csv',
             episode_length=365):
    df = pd.read_csv(data_path)
    return InventoryEnv(df, episode_length)
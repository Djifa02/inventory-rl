import numpy as np
import pandas as pd


class InventoryEnv:
    MAX_INVENTORY = 1000.0
    MAX_ORDER = 300.0
    EPISODE_LENGTH = 365  
    CH_RATE = 0.20 

    def __init__(self, csv_path, store_id=None, product_id=None, normalize=True):
        self.df_full = pd.read_csv(csv_path, parse_dates=["Date"])
        self.normalize = normalize
        self._preprocess()

        self.pairs = list(self.df_full.groupby(["Store ID", "Product ID"]).groups.keys())
        self.store_id = store_id
        self.product_id = product_id

        self.state_dim = 2
        self.action_dim = 1
        self.action_low = 0.0
        self.action_high = self.MAX_ORDER

        self.episode_df = None
        self.t = 0
        self.inventory = 0.0

    def _preprocess(self):
        """Normalisation min-max des seules variables retenues pour l'état."""
        df = self.df_full
        self._demand_max = df["Demand Forecast"].max()
        self.df_full = df

    def _sample_pair(self, rng):
        idx = rng.integers(0, len(self.pairs))
        return self.pairs[idx]

    def reset(self, seed=None):
        rng = np.random.default_rng(seed)
        store_id, product_id = (
            (self.store_id, self.product_id)
            if self.store_id is not None and self.product_id is not None
            else self._sample_pair(rng)
        )
        mask = (self.df_full["Store ID"] == store_id) & (self.df_full["Product ID"] == product_id)
        full_series = self.df_full.loc[mask].sort_values("Date").reset_index(drop=True)

        n_days = len(full_series)
        if n_days > self.EPISODE_LENGTH:
            start = int(rng.integers(0, n_days - self.EPISODE_LENGTH + 1))
            self.episode_df = full_series.iloc[start:start + self.EPISODE_LENGTH].reset_index(drop=True)
        else:
            self.episode_df = full_series

        self.t = 0
        self.inventory = float(self.episode_df.loc[0, "Inventory Level"])
        return self._build_state()

    def _build_state(self):
        row = self.episode_df.loc[self.t]
        inv = self.inventory / self.MAX_INVENTORY if self.normalize else self.inventory
        demand_fc = row["Demand Forecast"] / self._demand_max if self.normalize else row["Demand Forecast"]

        state = np.array([inv, demand_fc], dtype=np.float32)
        return state

    def step(self, action):
        """action: quantité à commander, scalaire ou array de taille (1,)."""
        order = float(np.clip(action, self.action_low, self.action_high).item()
                       if hasattr(action, "item") else np.clip(action, self.action_low, self.action_high))

        row = self.episode_df.loc[self.t]
        demand = float(row["Units Sold"])

        cp = float(row["Price"])
        ch = self.CH_RATE * cp

        available = self.inventory + order
        sold = min(available, demand)
        lost_sales = max(0.0, demand - available)
        next_inventory = max(0.0, available - sold)
        next_inventory = min(next_inventory, self.MAX_INVENTORY)

        revenue = sold * cp
        holding_cost = next_inventory * ch
        stockout_cost = lost_sales * cp

        reward = (revenue - holding_cost - stockout_cost) / 1000.0

        self.inventory = next_inventory
        self.t += 1
        done = self.t >= len(self.episode_df) - 1

        next_state = self._build_state() if not done else np.zeros(self.state_dim, dtype=np.float32)

        info = {
            "demand": demand,
            "order": order,
            "sold": sold,
            "lost_sales": lost_sales,
            "inventory": self.inventory,
            "revenue": revenue,
        }
        return next_state, reward, done, info
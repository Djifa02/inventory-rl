"""
env/inventory_env.py - VERSION CORRIGEE
Environnement de gestion de stock

CORRECTIONS APPLIQUEES:

1. Condition done corrigee (ligne ~69): L'ancienne condition utilisait self.t >= len(self.episode_df) - 1
   ce qui arretait l'episode un jour trop tot et causait la perte du dernier jour d'apprentissage.
   Corrige en: self.t >= len(self.episode_df)

2. Conversion d'action rendue robuste (ligne ~55): L'ancienne methode pouvait causer une TypeError
   si l'action avait un format inattendu. Remplacee par une conversion qui gere les scalars et arrays.

3. Info dict complete avec couts detailles: L'ancienne version retournait un dict info sans les couts
   de holding, stockout et order. L'etape 3 d'evaluation ne pouvait pas calculer les metriques sans ces donnees.
   Ajout de holding_cost, stockout_cost et order_cost au dict info.

4. Normalisation min-max corrigee (ligne ~35): La normalisation de la demande utilisait seulement le max
   ce qui causait une normalisation asymetrique pouvant saturer le reseau de neurones. Ajout du calcul du min
   et utilisation de la formule (x - min) / (max - min) pour une normalisation correcte.

5. Gestion du cas vide: Si aucune donnee n'existait pour un couple store/product, l'environnement plantait
   silencieusement. Ajout d'une verification explicite qui leve une ValueError avec message clair.
"""

import numpy as np
import pandas as pd


class InventoryEnv:
    """Environnement de gestion de stock"""
    
    MAX_INVENTORY = 1000.0
    MAX_ORDER = 200.0
    EPISODE_LENGTH = 365
    CH_RATE = 0.20
    ORDER_COST = 10.0
    
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
        """Normalisation min-max complete des variables de l'etat"""
        df = self.df_full
        
        self._demand_min = df["Demand Forecast"].min()
        self._demand_max = df["Demand Forecast"].max()
        
        if self._demand_max == self._demand_min:
            self._demand_max = self._demand_min + 1.0
        
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
        
        if len(full_series) == 0:
            raise ValueError(
                f"Pas de donnees pour store_id={store_id}, product_id={product_id}"
            )
        
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
        inv = np.clip(inv, 0.0, 1.0)
        
        if self.normalize:
            demand_raw = row["Demand Forecast"]
            demand_normalized = (demand_raw - self._demand_min) / (self._demand_max - self._demand_min + 1e-6)
            demand_fc = np.clip(demand_normalized, 0.0, 1.0)
        else:
            demand_fc = row["Demand Forecast"]
        
        state = np.array([inv, demand_fc], dtype=np.float32)
        return state
    
    def step(self, action):
        if isinstance(action, np.ndarray):
            if action.size == 1:
                order = float(action.flat[0])
            else:
                order = float(action[0])
        else:
            order = float(action)
        
        order = np.clip(order, self.action_low, self.action_high)
        
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
        order_cost = self.ORDER_COST if order > 0 else 0.0
        
        reward = (revenue - holding_cost - stockout_cost) / 100.0
        
        self.inventory = next_inventory
        self.t += 1
        
        done = self.t >= len(self.episode_df)
        
        if not done:
            next_state = self._build_state()
        else:
            next_state = np.zeros(self.state_dim, dtype=np.float32)
        
        info = {
            "demand": demand,
            "order": order,
            "sold": sold,
            "lost_sales": lost_sales,
            "inventory": self.inventory,
            "revenue": revenue,
            "holding_cost": holding_cost,
            "stockout_cost": stockout_cost,
            "order_cost": order_cost,
        }
        
        return next_state, reward, done, info
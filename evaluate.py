"""
Étape 3: Évaluation de la politique DDPG/SAC
Script autonome adapté au code réel des étapes 1-2
"""

import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)


class EvaluationMetrics:
    """Calcule et agrège les métriques d'évaluation"""
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        
    @staticmethod
    def _default_config() -> Dict:
        """Configuration par défaut"""
        return {
            "holding_cost_per_unit": 0.5,
            "order_cost_per_order": 10.0,
            "stockout_penalty": 5.0,
            "max_inventory": 1000,
        }
    
    def calculate_episode_cost(self, 
                              rewards: List[float],
                              actions: List[float],
                              inventory_levels: List[float],
                              demands: List[float]) -> Dict[str, float]:
        """Calcule le coût détaillé d'un épisode"""
        holding_cost = np.sum(np.array(inventory_levels) * self.config["holding_cost_per_unit"])
        order_cost = np.sum(np.array(actions) > 0.1) * self.config["order_cost_per_order"]
        
        stockout_penalty = 0
        for inv, demand in zip(inventory_levels, demands):
            if inv < demand:
                stockout_penalty += (demand - inv) * self.config["stockout_penalty"]
        
        total_cost = holding_cost + order_cost + stockout_penalty
        episode_reward = np.sum(rewards)
        
        return {
            "total_cost": total_cost,
            "holding_cost": holding_cost,
            "order_cost": order_cost,
            "stockout_penalty": stockout_penalty,
            "episode_reward": episode_reward,
            "avg_reward": episode_reward / len(rewards) if rewards else 0,
            "episode_length": len(rewards),
        }
    
    def calculate_policy_performance(self, episodes_data: List[Dict]) -> Dict[str, float]:
        """Agrège les performances sur plusieurs épisodes"""
        total_costs = [ep["total_cost"] for ep in episodes_data]
        rewards = [ep["episode_reward"] for ep in episodes_data]
        holding_costs = [ep["holding_cost"] for ep in episodes_data]
        order_costs = [ep["order_cost"] for ep in episodes_data]
        stockout_penalties = [ep["stockout_penalty"] for ep in episodes_data]
        
        return {
            "mean_episode_cost": np.mean(total_costs),
            "std_episode_cost": np.std(total_costs),
            "min_episode_cost": np.min(total_costs),
            "max_episode_cost": np.max(total_costs),
            "mean_episode_reward": np.mean(rewards),
            "std_episode_reward": np.std(rewards),
            "mean_holding_cost": np.mean(holding_costs),
            "mean_order_cost": np.mean(order_costs),
            "mean_stockout_penalty": np.mean(stockout_penalties),
            "num_episodes_evaluated": len(episodes_data),
        }


class BaselinePolicy:
    """Politiques de baseline pour comparaison"""
    
    @staticmethod
    def greedy_policy(inventory: float,
                     demand_forecast: float,
                     reorder_point: float = 50,
                     order_quantity: float = 100) -> float:
        """Politique gloutonne: commander si stock < point de réapprovisionnement"""
        if inventory < reorder_point:
            return order_quantity
        return 0.0
    
    @staticmethod
    def forecast_matching(demand_forecast: float, 
                         lead_time_demand: float = 50) -> float:
        """Politique simple: commander la prévision de demande + buffer"""
        safety_stock = lead_time_demand * 0.5
        return max(0, demand_forecast + safety_stock)
    
    @staticmethod
    def random_policy(action_low: float = 0.0, action_high: float = 200.0) -> float:
        """Politique aléatoire"""
        return np.random.uniform(action_low, action_high)


class EvaluationReporter:
    """Génère rapports et visualisations"""
    
    @staticmethod
    def generate_evaluation_report(metrics: Dict[str, float], 
                                  baseline_metrics: Dict[str, float] = None,
                                  output_dir: str = "results/evaluation") -> None:
        """Génère un rapport comparatif"""
        os.makedirs(output_dir, exist_ok=True)
        
        report = {
            "trained_policy": metrics,
        }
        
        if baseline_metrics:
            report["baseline_policy"] = baseline_metrics
            improvement = {
                "cost_reduction_%": (
                    (baseline_metrics["mean_episode_cost"] - metrics["mean_episode_cost"]) 
                    / baseline_metrics["mean_episode_cost"] * 100
                ),
                "reward_improvement_%": (
                    (metrics["mean_episode_reward"] - baseline_metrics.get("mean_episode_reward", 0))
                    / (abs(baseline_metrics.get("mean_episode_reward", 1)) + 1e-6) * 100
                ),
            }
            report["improvement"] = improvement
        
        report_path = Path(output_dir) / "evaluation_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✅ Rapport sauvegardé: {report_path}")
        print("\n" + "="*60)
        print("RAPPORT D'ÉVALUATION")
        print("="*60)
        print(json.dumps(report, indent=2))
    
    @staticmethod
    def plot_episode_trajectory(inventory_levels: List[float],
                               demands: List[float],
                               actions: List[float],
                               rewards: List[float],
                               title: str = "Trajectoire d'un épisode") -> None:
        """Visualise la dynamique d'un épisode"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        steps = range(len(rewards))
        
        axes[0, 0].plot(steps, inventory_levels, label="Inventaire", marker='o', markersize=3)
        axes[0, 0].plot(steps, demands, label="Demande", marker='s', markersize=3, alpha=0.7)
        axes[0, 0].set_ylabel("Unités")
        axes[0, 0].set_xlabel("Step")
        axes[0, 0].set_title("Niveau d'inventaire vs Demande")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].bar(steps, actions, color='steelblue', alpha=0.7)
        axes[0, 1].set_ylabel("Quantité commandée")
        axes[0, 1].set_xlabel("Step")
        axes[0, 1].set_title("Actions (Commandes)")
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        cumulative_rewards = np.cumsum(rewards)
        axes[1, 0].plot(steps, cumulative_rewards, color='green', marker='o', markersize=3)
        axes[1, 0].set_ylabel("Reward cumulatif")
        axes[1, 0].set_xlabel("Step")
        axes[1, 0].set_title("Évolution du Reward cumulatif")
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(steps, rewards, color='orange', alpha=0.7, label='Reward')
        axes[1, 1].axhline(y=np.mean(rewards), color='red', linestyle='--', 
                          label=f'Moyenne: {np.mean(rewards):.2f}')
        axes[1, 1].set_ylabel("Reward")
        axes[1, 1].set_xlabel("Step")
        axes[1, 1].set_title("Rewards par step")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_cost_breakdown(episodes_metrics: List[Dict], 
                           title: str = "Répartition des coûts") -> None:
        """Visualise le breakdown des coûts"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        holding_costs = [ep["holding_cost"] for ep in episodes_metrics]
        order_costs = [ep["order_cost"] for ep in episodes_metrics]
        stockout_penalties = [ep["stockout_penalty"] for ep in episodes_metrics]
        
        episodes = range(len(episodes_metrics))
        width = 0.6
        
        axes[0].bar(episodes, holding_costs, width, label='Coût de stockage', alpha=0.8)
        axes[0].bar(episodes, order_costs, width, bottom=holding_costs, 
                   label='Coût de commande', alpha=0.8)
        bottom = np.array(holding_costs) + np.array(order_costs)
        axes[0].bar(episodes, stockout_penalties, width, bottom=bottom,
                   label='Pénalité de rupture', alpha=0.8)
        axes[0].set_ylabel("Coût")
        axes[0].set_xlabel("Épisode")
        axes[0].set_title("Évolution du breakdown par épisode")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')
        
        avg_holding = np.mean(holding_costs)
        avg_order = np.mean(order_costs)
        avg_stockout = np.mean(stockout_penalties)
        
        sizes = [avg_holding, avg_order, avg_stockout]
        labels = [
            f'Stockage\n({avg_holding:.1f})',
            f'Commande\n({avg_order:.1f})',
            f'Rupture\n({avg_stockout:.1f})'
        ]
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        
        axes[1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                   startangle=90)
        axes[1].set_title("Répartition du coût moyen")
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_comparison(trained_metrics: Dict[str, float],
                       baseline_metrics: Dict[str, float]) -> None:
        """Compare les performances trained vs baseline"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        metrics_names = ['Coût total', 'Coût stockage', 'Coût commande', 'Pénalité rupture']
        trained_values = [
            trained_metrics['mean_episode_cost'],
            trained_metrics['mean_holding_cost'],
            trained_metrics['mean_order_cost'],
            trained_metrics['mean_stockout_penalty']
        ]
        baseline_values = [
            baseline_metrics['mean_episode_cost'],
            baseline_metrics['mean_holding_cost'],
            baseline_metrics['mean_order_cost'],
            baseline_metrics['mean_stockout_penalty']
        ]
        
        x = np.arange(len(metrics_names))
        width = 0.35
        
        axes[0].bar(x - width/2, trained_values, width, label='DDPG', alpha=0.8)
        axes[0].bar(x + width/2, baseline_values, width, label='Baseline', alpha=0.8)
        axes[0].set_ylabel("Coût")
        axes[0].set_title("Comparaison des coûts")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(metrics_names, rotation=45, ha='right')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')
        
        trained_reward = trained_metrics['mean_episode_reward']
        baseline_reward = baseline_metrics.get('mean_episode_reward', 0)
        
        data = [trained_reward, baseline_reward]
        labels = ['DDPG', 'Baseline']
        colors = ['#2ecc71', '#e74c3c']
        
        bars = axes[1].bar(labels, data, color=colors, alpha=0.8, width=0.5)
        axes[1].set_ylabel("Reward moyen")
        axes[1].set_title("Comparaison des Rewards")
        axes[1].grid(True, alpha=0.3, axis='y')
        
        for bar, value in zip(bars, data):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    print("Module d'évaluation chargé")
    print("\nFonctionnalités disponibles:")
    print("  - EvaluationMetrics: Calcul des métriques")
    print("  - BaselinePolicy: Politiques de comparaison")
    print("  - EvaluationReporter: Rapports et visualisations")
"""
utils.py - VERSION CORRIGEE
Fonctions utilitaires pour visualisation et analyse

CORRECTIONS APPLIQUEES:

1. Gestion du cas vide dans moving_average (ligne ~15): Si la liste de valeurs etait vide, la fonction
   pouvait causer des erreurs ou retourner des resultats impredictibles. Ajout d'une verification
   len(values) == 0 au debut avec retour d'un array vide.

2. Gestion des NaN dans les donnees (ligne ~25): Apres le calcul de la moyenne mobile, il pouvait y avoir
   des NaN dans les resultats causant une visualisation cassee ou des erreurs. Ajout de np.nan_to_num
   pour remplacer les NaN par 0.
"""

import numpy as np
import matplotlib.pyplot as plt


def moving_average(values, window=20):
    """
    Calculer la moyenne mobile d'une serie
    
    Args:
        values: array de valeurs
        window: taille de la fenetre
        
    Returns:
        array avec la moyenne mobile
    """
    values = np.asarray(values, dtype=np.float32)
    
    if len(values) == 0:
        return np.array([])
    
    if len(values) < window:
        return values
    
    cumsum = np.cumsum(np.insert(values, 0, 0))
    smoothed = (cumsum[window:] - cumsum[:-window]) / window
    
    smoothed = np.nan_to_num(smoothed, nan=0.0)
    
    return smoothed


def plot_training_curve(episode_rewards, save_path=None, window=20):
    """
    Tracer la courbe d'apprentissage
    
    Args:
        episode_rewards: liste des rewards par episode
        save_path: chemin pour sauvegarder la figure (optionnel)
        window: taille de la fenetre pour la moyenne mobile
    """
    if len(episode_rewards) == 0:
        print("Attention: Aucune donnee a tracer (episode_rewards est vide)")
        return
    
    smoothed = moving_average(episode_rewards, window=window)
    
    plt.figure(figsize=(9, 5))
    
    plt.plot(episode_rewards, alpha=0.3, label="Recompense par episode", color='steelblue')
    
    if len(smoothed) > 0:
        plt.plot(
            range(window - 1, window - 1 + len(smoothed)),
            smoothed,
            label=f"Moyenne mobile ({window})",
            color='orange',
            linewidth=2
        )
    
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Recompense cumulee", fontsize=12)
    plt.title("Courbe d'apprentissage - Agent DDPG (Gestion de stock)", fontsize=13)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        try:
            plt.savefig(save_path, dpi=150)
            print(f"OK - Courbe sauvegardee: {save_path}")
        except Exception as e:
            print(f"Erreur lors de la sauvegarde: {e}")
    else:
        plt.show()
    
    plt.close()


def get_training_statistics(episode_rewards):
    """
    Calculer les statistiques d'entraînement
    
    Args:
        episode_rewards: liste des rewards
        
    Returns:
        dict avec statistiques
    """
    if len(episode_rewards) == 0:
        return {}
    
    rewards_array = np.array(episode_rewards)
    
    stats = {
        "mean": np.mean(rewards_array),
        "std": np.std(rewards_array),
        "min": np.min(rewards_array),
        "max": np.max(rewards_array),
        "median": np.median(rewards_array),
        "last": episode_rewards[-1],
        "last_10_mean": np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards),
    }
    
    return stats
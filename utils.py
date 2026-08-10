import numpy as np
import matplotlib.pyplot as plt


def moving_average(values, window=20):
    values = np.asarray(values, dtype=np.float32)
    if len(values) < window:
        return values
    cumsum = np.cumsum(np.insert(values, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / window


def plot_training_curve(episode_rewards, save_path=None, window=20):
    smoothed = moving_average(episode_rewards, window=window)

    plt.figure(figsize=(9, 5))
    plt.plot(episode_rewards, alpha=0.3, label="Récompense par épisode")
    if len(smoothed) > 0:
        plt.plot(range(window - 1, window - 1 + len(smoothed)), smoothed, label=f"Moyenne mobile ({window})")
    plt.xlabel("Épisode")
    plt.ylabel("Récompense cumulée")
    plt.title("Courbe d'apprentissage, Agent DDPG (gestion de stock)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Courbe sauvegardée : {save_path}")
    plt.close()
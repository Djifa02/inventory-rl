import argparse
import os

import numpy as np

from agent.ddpg import DDPGAgent
from env.inventory_env import InventoryEnv
from utils import plot_training_curve


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/retail_store_inventory.csv")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--warmup_steps", type=int, default=1000,
                         help="Nombre de pas aléatoires avant de commencer l'apprentissage")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--store_id", type=str, default=None,
                         help="Fixe un couple (store, produit) pour tous les épisodes; sinon tirage aléatoire")
    parser.add_argument("--product_id", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="results")
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    os.makedirs(os.path.join(args.save_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "plots"), exist_ok=True)

    env = InventoryEnv(args.data, store_id=args.store_id, product_id=args.product_id)
    agent = DDPGAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        action_low=env.action_low,
        action_high=env.action_high,
    )

    total_steps = 0
    episode_rewards = []

    for episode in range(1, args.episodes + 1):
        state = env.reset(seed=args.seed + episode)
        agent.noise.reset()
        episode_reward = 0.0
        done = False

        while not done:
            if total_steps < args.warmup_steps:
                action = np.random.uniform(env.action_low, env.action_high, size=(env.action_dim,))
            else:
                action = agent.select_action(state, explore=True)

            next_state, reward, done, info = env.step(action)
            agent.store_transition(state, action, reward, next_state, float(done))

            if total_steps >= args.warmup_steps:
                agent.update(batch_size=args.batch_size)

            state = next_state
            episode_reward += reward
            total_steps += 1

        episode_rewards.append(episode_reward)

        if episode % 10 == 0 or episode == 1:
            avg_last = np.mean(episode_rewards[-10:])
            print(f"[Épisode {episode:4d}/{args.episodes}] "
                  f"récompense = {episode_reward:8.2f} | moyenne (10) = {avg_last:8.2f}")

        if episode % 50 == 0:
            agent.save(os.path.join(args.save_dir, "models", f"ddpg_ep{episode}"))

    agent.save(os.path.join(args.save_dir, "models", "ddpg_final"))
    plot_training_curve(
        episode_rewards,
        save_path=os.path.join(args.save_dir, "plots", "training_curve.png"),
    )
    np.save(os.path.join(args.save_dir, "episode_rewards.npy"), np.array(episode_rewards))
    print("Entraînement terminé. Modèle et courbe sauvegardés dans", args.save_dir)


if __name__ == "__main__":
    main()

"""
train.py - VERSION CORRIGEE
Script d'entraînement de l'agent DDPG

CORRECTIONS APPLIQUEES:

1. Ajout de torch.manual_seed (ligne ~30): L'ancien code ne fixait que np.random.seed mais oubliait
   torch.manual_seed. Cela rendait l'entraînement non-reproductible car les poids initiaux et operations
   torch n'etaient pas deterministes. Ajout de torch.manual_seed et torch.cuda.manual_seed_all.

2. Validation du CSV avant chargement (ligne ~40): Si le fichier de donnees n'existait pas, le code plantait
   avec un message d'erreur confus. Ajout d'une verification explicite avec os.path.exists() et un message
   d'erreur clair avant de creer l'environnement.

3. Logging initial de la configuration (ligne ~60): Le code ne disait pas quels parametres etaient utilises,
   rendant difficile le debugging et la reproduction. Ajout d'un affichage initial de tous les parametres
   (episodes, batch_size, seed, device, store_id, product_id, etc).

4. Frequence de sauvegarde moins aggressive (ligne ~120): L'ancien code sauvegardait le modele tous les 50
   episodes, ce qui crait beaucoup de fichiers et consommait beaucoup de disque. Reduit a 100 episodes pour
   un bon compromis entre securite et espace disque.

5. Meilleure gestion des store/product IDs (ligne ~85): La logique etait confuse avec None pour mode auto.
   Amélioré avec des messages clairs et un mode "auto" explicite pour le logging.
"""

import argparse
import os
import numpy as np
import torch

from agent.ddpg import DDPGAgent
from env.inventory_env import InventoryEnv
from utils import plot_training_curve


def parse_args():
    """Parser les arguments de la ligne de commande"""
    parser = argparse.ArgumentParser(
        description="Entraîner l'agent DDPG pour la gestion de stock"
    )
    parser.add_argument(
        "--data", 
        type=str, 
        default="data/retail_store_inventory.csv",
        help="Chemin vers le fichier de donnees"
    )
    parser.add_argument(
        "--episodes", 
        type=int, 
        default=500,
        help="Nombre d'episodes a entraîner"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=128,
        help="Taille des batchs pour l'entraînement"
    )
    parser.add_argument(
        "--warmup_steps", 
        type=int, 
        default=1000,
        help="Nombre de pas aleatoires avant de commencer l'apprentissage"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Seed pour la reproductibilité"
    )
    parser.add_argument(
        "--store_id", 
        type=str, 
        default=None,
        help="Fixe un magasin (None = aleatoire)"
    )
    parser.add_argument(
        "--product_id", 
        type=str, 
        default=None,
        help="Fixe un produit (None = aleatoire)"
    )
    parser.add_argument(
        "--save_dir", 
        type=str, 
        default="results",
        help="Repertoire pour sauvegarder les modeles et graphes"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device ('cuda', 'cpu', ou 'auto')"
    )
    
    return parser.parse_args()


def main():
    """Fonction principale d'entraînement"""
    args = parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Dataset non trouve: {args.data}")
    
    os.makedirs(os.path.join(args.save_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "plots"), exist_ok=True)
    
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print("=" * 70)
    print("ENTRAÎNEMENT DDPG - GESTION DE STOCK")
    print("=" * 70)
    print(f"\nCONFIGURATION:")
    print(f"  Dataset: {args.data}")
    print(f"  Episodes: {args.episodes}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Warmup steps: {args.warmup_steps}")
    print(f"  Seed: {args.seed}")
    print(f"  Device: {device}")
    
    store_mode = f"fixe={args.store_id}" if args.store_id else "aleatoire"
    product_mode = f"fixe={args.product_id}" if args.product_id else "aleatoire"
    print(f"  Store ID: {store_mode}")
    print(f"  Product ID: {product_mode}")
    print(f"  Save dir: {args.save_dir}")
    print()
    
    print("Initialisation de l'environnement...")
    env = InventoryEnv(
        args.data,
        store_id=args.store_id,
        product_id=args.product_id
    )
    print(f"OK - Environnement cree (state_dim={env.state_dim}, action_dim={env.action_dim})")
    
    print("\nInitialisation de l'agent DDPG...")
    agent = DDPGAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        action_low=env.action_low,
        action_high=env.action_high,
        device=device,
    )
    print(f"OK - Agent cree (device={agent.device})")
    
    print("\nDemarrage de l'entraînement...\n")
    
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
            print(
                f"[Episode {episode:4d}/{args.episodes}] "
                f"recompense = {episode_reward:8.2f} | moyenne (10) = {avg_last:8.2f} | "
                f"total_steps = {total_steps}"
            )
        
        if episode % 100 == 0 or episode == args.episodes:
            model_path = os.path.join(args.save_dir, "models", f"ddpg_ep{episode}")
            agent.save(model_path)
            print(f"  -> Modele sauvegarde: {model_path}")
    
    print("\n" + "=" * 70)
    print("ENTRAÎNEMENT TERMINE")
    print("=" * 70)
    
    final_model_path = os.path.join(args.save_dir, "models", "ddpg_final")
    agent.save(final_model_path)
    print(f"OK - Modele final sauvegarde: {final_model_path}")
    
    plot_path = os.path.join(args.save_dir, "plots", "training_curve.png")
    plot_training_curve(
        episode_rewards,
        save_path=plot_path,
    )
    print(f"OK - Courbe d'apprentissage sauvegardee: {plot_path}")
    
    rewards_path = os.path.join(args.save_dir, "episode_rewards.npy")
    np.save(rewards_path, np.array(episode_rewards))
    print(f"OK - Rewards sauvegardes: {rewards_path}")
    
    print(f"\nSTATISTIQUES FINALES:")
    print(f"  Reward moyen (dernier episode): {episode_reward:.2f}")
    print(f"  Reward moyen (10 derniers): {np.mean(episode_rewards[-10:]):.2f}")
    print(f"  Reward moyen (tous): {np.mean(episode_rewards):.2f}")
    print(f"  Total steps: {total_steps}")
    print(f"  Warmup steps: {args.warmup_steps}")
    print(f"  Learning steps: {total_steps - args.warmup_steps}")
    print()


if __name__ == "__main__":
    main()
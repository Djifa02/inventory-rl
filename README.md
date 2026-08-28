# Inventory-RL

## Optimisation de la gestion des stocks par Reinforcement Learning

Inventory-RL est un projet d'optimisation de la gestion des stocks utilisant l'apprentissage par renforcement (*Reinforcement Learning*).

L'objectif est de développer un agent intelligent capable de déterminer la quantité de produits à commander en fonction de l'état actuel du stock et de la demande prévue, afin d'améliorer la gestion des stocks et de réduire les coûts associés.

## Objectif du projet

La gestion des stocks consiste à déterminer les quantités à commander afin de répondre à la demande tout en limitant les coûts liés au stockage et aux ruptures de stock.

Dans ce projet, nous formulons la gestion des stocks comme un problème d'apprentissage par renforcement.

L'agent apprend progressivement à prendre des décisions de réapprovisionnement à partir de l'état du stock et de la prévision de la demande.

L'objectif de l'agent est de trouver une politique de commande permettant de maximiser la récompense cumulée, tout en recherchant un compromis entre :

* la satisfaction de la demande ;
* les coûts de stockage ;
* les coûts liés aux ruptures de stock ;
* les coûts de commande.

L'algorithme **DDPG (Deep Deterministic Policy Gradient)** est utilisé pour apprendre cette politique de décision dans un espace d'actions continues.

## Données utilisées

Le projet utilise un jeu de données de gestion des stocks situé dans le répertoire `data/`.

Le fichier principal est :

```text
data/retail_store_inventory.csv
```

Les données contiennent notamment des informations relatives aux magasins, aux produits, aux niveaux de stock, aux ventes, aux prix et aux prévisions de demande.

Les principales variables exploitées par l'environnement sont :

* **Store ID** : identifiant du magasin ;
* **Product ID** : identifiant du produit ;
* **Date** : date de l'observation ;
* **Inventory Level** : niveau de stock disponible ;
* **Demand Forecast** : prévision de la demande ;
* **Units Sold** : quantité vendue ;
* **Price** : prix du produit.

Ces données permettent de simuler les décisions de réapprovisionnement au cours du temps et de mesurer leurs conséquences sur les ventes, le stock et les coûts.

## Environnement de gestion des stocks

L'environnement de gestion des stocks est défini dans le fichier :

```text
env/inventory_env.py
```

Il représente le système dans lequel l'agent DDPG prend ses décisions de réapprovisionnement.

L'environnement est constitué de trois éléments principaux :

### État

À chaque étape, l'agent observe deux informations :

* le niveau de stock actuel ;
* la prévision de la demande.

L'état est donc représenté sous la forme :

```text
[stock actuel, demande prévue]
```

Lorsque la normalisation est activée, ces valeurs sont ramenées dans un intervalle adapté à l'apprentissage du réseau de neurones.

### Action

L'action correspond à la quantité de produits que l'agent décide de commander.

L'espace d'action est continu et la quantité commandée est limitée à :

```text
0 ≤ quantité commandée ≤ 200
```

L'utilisation d'un espace d'actions continues justifie l'utilisation de l'algorithme DDPG.

### Récompense

Après chaque décision de commande, l'environnement simule la satisfaction de la demande et calcule les conséquences de cette décision.

La récompense prend en compte :

* le revenu généré par les ventes ;
* le coût de stockage ;
* le coût des ruptures de stock.

La récompense est calculée selon le principe :

```text
Récompense = Revenu - Coût de stockage - Coût de rupture
```

Cette récompense permet à l'agent d'apprendre quelles décisions de commande sont les plus intéressantes à long terme.

### Épisode

Un épisode représente une période de simulation de gestion des stocks pouvant aller jusqu'à **365 jours**.

À chaque jour, l'agent :

1. observe l'état du stock ;
2. choisit une quantité à commander ;
3. fait face à la demande ;
4. reçoit une récompense ;
5. passe à l'état suivant.

Le processus est répété jusqu'à la fin de l'épisode.
## Structure du projet

Le projet est organisé de la manière suivante :

```text
inventory-rl/
│
├── agent/
│   └── ddpg.py
│
├── data/
│   └── retail_store_inventory.csv
│
├── env/
│   └── inventory_env.py
│
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_agent.ipynb
│   └── 03_evaluation.ipynb
│
├── evaluate.py
├── train.py
├── utils.py
├── requirements.txt
└── README.md
```

### Description des principaux fichiers

* **`env/inventory_env.py`** : définit l'environnement de gestion des stocks et les règles de simulation.
* **`agent/ddpg.py`** : contient l'implémentation de l'agent DDPG chargé d'apprendre la politique de réapprovisionnement.
* **`train.py`** : permet d'entraîner l'agent sur plusieurs épisodes.
* **`evaluate.py`** : contient les outils permettant d'évaluer les performances de la politique apprise et de la comparer à des politiques de référence.
* **`utils.py`** : regroupe les fonctions utilitaires, notamment pour le traitement et la visualisation des résultats.
* **`data/retail_store_inventory.csv`** : jeu de données utilisé pour simuler la gestion des stocks.
* **`notebooks/01_exploration.ipynb`** : exploration et analyse des données.
* **`notebooks/02_agent.ipynb`** : expérimentation et entraînement de l'agent.
* **`notebooks/03_evaluation.ipynb`** : évaluation des performances de l'agent.
* **`requirements.txt`** : liste des bibliothèques nécessaires au fonctionnement du projet.
* **`README.md`** : documentation générale du projet.
## Installation

### 1. Cloner le projet

```bash
git clone https://github.com/Djifa02/inventory-rl.git
cd inventory-rl
```

### 2. Créer un environnement virtuel

```bash
python -m venv .venv
```

Sous Windows, activer l'environnement :

```powershell
.venv\Scripts\activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

## Entraînement

L'entraînement de l'agent DDPG est réalisé à l'aide du fichier `train.py`.

Pour lancer un entraînement avec les paramètres par défaut :

```bash
python train.py
```

Les principaux paramètres disponibles sont :

* `--episodes` : nombre d'épisodes d'entraînement ;
* `--batch_size` : taille des lots utilisés pour l'apprentissage ;
* `--warmup_steps` : nombre de pas avant le début de l'apprentissage ;
* `--seed` : graine utilisée pour assurer la reproductibilité ;
* `--store_id` : identifiant du magasin à utiliser ;
* `--product_id` : identifiant du produit à utiliser ;
* `--save_dir` : répertoire de sauvegarde des résultats ;
* `--device` : périphérique utilisé (`cpu`, `cuda` ou `auto`).

Exemple :

```bash
python train.py --episodes 500 --batch_size 128 --warmup_steps 1000 --seed 42
```

Les modèles entraînés et les résultats sont sauvegardés dans le répertoire `results/`.

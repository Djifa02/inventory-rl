# Etape 1 : Modélisation de l Environnement MDP

## Description

Cette etape concerne la modelisation de l'environnement
de gestion de stock sous forme de MDP (Markov Decision Process)
et son implementation avec Gymnasium.

## Dataset

**Source** : Retail Store Inventory Forecasting Dataset - Kaggle

| Caracteristique | Valeur |
|----------------|--------|
| Nombre de lignes | 73 100 |
| Nombre de colonnes | 15 |
| Valeurs manquantes | 0 |
| Periode | 2022-01-01 au 2024-01-01 |

## Analyse Exploratoire

Le notebook `notebooks/01_exploration.ipynb` contient
l'analyse complete du dataset et la justification
de chaque composante du MDP.

## MDP - Choix et Justifications

### Etats

Apres l'analyse exploratoire deux variables ont montre
une relation suffisamment informative avec les ventes
pour etre retenues comme etats dans cette premiere
version de l environnement :

| Variable | Correlation | Justification |
|----------|-------------|---------------|
| Inventory Level | 0.59 | Contrainte physique : on ne peut pas vendre plus que le stock disponible |
| Demand Forecast | 0.99 | Variable la plus informative du dataset |

Les autres variables ont presente une faible contribution
dans les analyses exploratoires et n ont pas ete retenues.

Note : ce choix est specifique a ce dataset synthetique
et pourrait etre different avec des donnees reelles.

### Actions

| Parametre | Valeur | Justification |
|-----------|--------|---------------|
| Type | Continue | Distribution uniforme de Units Ordered sans paliers fixes |
| Minimum | 0 | L agent peut ne rien commander |
| Maximum | 200 | Valeur maximale observee dans le dataset |
| Algorithme | DDPG | Actions continues → DDPG plus adapte que DQN |

### Fonction de Recompense

Basee sur le Newsvendor Problem (Oroojlooyjadid et al. 2017)
Source : https://arxiv.org/pdf/1607.02177

```
R = Price x min(d, s) - ch x max(s-d, 0) - cp x max(d-s, 0)

cp = Price  (perte directe du prix de vente en cas de rupture)
ch = 0.20 x Price  (contrainte ch < cp du Newsvendor Problem)
```

### Fin d Episode

| Parametre | Valeur | Justification |
|-----------|--------|---------------|
| Duree | 365 jours | Couvre les 4 saisons du dataset sur un cycle annuel complet |

## Limites

- Le dataset est synthetique et ne reproduit pas parfaitement
  la variabilite d un systeme reel de gestion de stock
- Les choix d etats sont specifiques a ce dataset
  et pourraient etre differents avec des donnees reelles
- La correlation de 0.99 de Demand Forecast est une
  caracteristique du dataset synthetique
  et doit etre interpretee avec prudence

## Resultats des Tests

| Test | Resultat |
|------|---------|
| Chargement | OK |
| reset() | Stock initial : 231.0 |
| step() | Stock mis a jour reellement |
| Episode complet | 365 jours |

## Utilisation

```python
from env.inventory_env import load_env
import numpy as np

# Charger l environnement
env = load_env()

# Reinitialiser
obs, _ = env.reset()

# Faire une action
action = np.array([50.0])
obs, recompense, termine, _, _ = env.step(action)
```

## Fichiers

```
env/
└── inventory_env.py      ← Implementation de InventoryEnv

notebooks/
└── 01_exploration.ipynb  ← Analyse et justification du MDP
```

## References

- Oroojlooyjadid et al. 2017 - Applying Deep Learning to the Newsvendor Problem
  https://arxiv.org/pdf/1607.02177

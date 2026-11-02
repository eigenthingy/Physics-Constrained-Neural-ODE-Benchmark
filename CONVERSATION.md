# Origine du projet — brief et journal de conception

Ce fichier existe parce que la demande initiale portait autant sur **la démarche**
que sur le code. Il conserve le brief tel quel, puis les décisions de conception
et leurs raisons — y compris celles qui ont été écartées.

La transcription brute de la session est dans [`transcript/`](transcript/).

---

## 1. Le brief (verbatim)

> Tu peux m'aider a faire dans un sous dossier de ce dossier "physics", qu'on
> appellera "PCNN", ce projet : Physics-Constrained Neural Networks ⭐⭐⭐⭐⭐
>
> Pas juste un PINN classique.
>
> Tu compares plusieurs façons d'imposer les lois physiques.
>
> Par exemple :
>
> **Modèle A** — Aucune contrainte.
>
> **Modèle B** — Ajout d'une pénalité : `Loss = Data Loss + λ × Physics Loss`
>
> **Modèle C** — Architecture construite pour respecter automatiquement une
> conservation (énergie, masse, etc.).
>
> Tu compares :
> - erreur
> - stabilité
> - extrapolation
> - sensibilité au bruit
>
> C'est beaucoup plus intéressant qu'un simple "voici un PINN". stp fait en sorte
> que ce soit un projet qui a vraiment un plus, un point de vue et une valeur
> ajoutee sur le marché quand je le mets sur github, pas juste un projet blabla.
> Je veux que tout (les script et les dossiers et tout) soient mis dans un dossier
> "NOT PINNS", y compris cette discussion avec claude je veux la retrouver là bas.

## 2. Interprétation retenue

Deux consignes de rangement se recoupaient : « un sous-dossier PCNN » et « tout
dans un dossier NOT PINNS ». Choix fait :

```
Desktop/Physics/NOT PINNS/     <- racine du dépôt (nom GitHub suggéré : not-pinns)
├── pcnn/                      <- le package Python PCNN
├── scripts/ tests/ results/ figures/ docs/
├── CONVERSATION.md            <- ce fichier
└── transcript/                <- la session brute
```

`NOT PINNS` est la racine parce que c'est le niveau qui doit contenir « tout », y
compris cette discussion. Le nom fonctionne aussi comme accroche : le dépôt
n'est pas un PINN de plus, c'est ce qui se passe quand on arrête de supposer que
la pénalité est la seule option.

## 3. Décisions de conception, et pourquoi

### 3.1 Ajouter un quatrième modèle

Le brief demandait A / B / C. Trois modèles ne suffisent pas à séparer deux
questions distinctes, qui sont pourtant toujours confondues dans la littérature
de vulgarisation :

- **comment** la contrainte est imposée : par la loss (souple) ou par
  l'architecture (dure) ;
- **ce que** le modèle sait : l'invariant explicite, ou seulement le fait qu'il
  en existe un.

D'où la grille 2×2 et le **modèle D** (projection sur l'espace tangent du
niveau d'énergie vrai). Sans D, on compare « pénalité avec H connu » à
« architecture sans H connu » et on attribue au mot *architecture* un effet qui
vient en partie du mot *connu*. C'est la principale valeur ajoutée du dépôt.

### 3.2 Ne jamais fournir les dérivées exactes

Les démos PINN entraînent typiquement sur `dz/dt` analytique. C'est le régime où
un a priori physique sert le moins, puisque la cible est déjà parfaite. Ici on
observe une trajectoire bruitée et on estime la dérivée par différences finies
centrées — ce qui amplifie le bruit d'un facteur `1/(√2·dt) ≈ 14`. C'est cette
décision qui rend l'axe « sensibilité au bruit » informatif plutôt que
décoratif.

### 3.3 Traiter le modèle B correctement

Publier un PINN à un seul λ arbitraire en fait un homme de paille. Chaque entrée
du modèle B est entraînée à trois valeurs de λ, et celle qui a la meilleure perte
de **validation** est retenue — validation sur cibles bruitées, c'est-à-dire le
seul signal dont dispose un praticien. Sélectionner sur données propres
reviendrait à donner la réponse au modèle. Une étude dédiée trace en plus tout le
front de Pareto en λ.

### 3.4 Deux systèmes, plus un système « piège »

- **Pendule non linéaire** (1 DOF) : lisible, portraits de phase parlants.
- **Hénon–Heiles** (2 DOF, chaotique) : le vrai test de stabilité, et surtout la
  dimension où la contrainte dure de D cesse d'être quasi gratuite (elle retire
  1 direction sur 4 au lieu de 1 sur 2).
- **Pendule amorti** : le système ne conserve *pas* l'énergie que les modèles
  ont pour consigne de conserver. C'est l'expérience de mauvaise spécification,
  celle qui manque presque partout, et c'est là que la hiérarchie s'inverse.

### 3.5 Équité mesurée, pas supposée

Budget compté en **pas d'optimisation** (pas en époques), mêmes données, même
ordre de batchs, même planning de learning rate, et largeur du réseau scalaire
de C ajustée pour égaliser le nombre de paramètres à moins de 1 %. Un test
(`test_parameter_budgets_match`) le vérifie automatiquement.

### 3.6 Cinq graines et des intervalles de confiance

Toute comparaison mono-graine en deep learning est du bruit. Chaque nombre publié
est une médiane sur 5 graines avec IC 95 % par bootstrap, et les comparaisons au
modèle A sont appariées graine par graine.

## 4. Ce qui a été écarté

| Écarté | Raison |
|---|---|
| La 4ᵉ case de la grille (pénalité souple sur un invariant *appris*) | Intéressante mais elle ajoute une boucle d'entraînement instable à régler ; laissée ouverte dans le README comme piste. |
| Un système à 3 corps / N corps | Coût d'intégration de référence trop élevé pour 560 runs ; Hénon–Heiles donne déjà le chaos et la dimension 4. |
| Des réseaux plus gros | La question porte sur la contrainte, pas sur la capacité. Des MLP identiques rendent l'écart attribuable. |
| Réutiliser les mêmes poids entre l'étude principale et l'étude d'intégrateur | Ré-entraîner est déterministe et coûte peu ; le partage aurait couplé deux études. |

## 5. Limites assumées

- Systèmes de basse dimension (d = 2 et d = 4). Les conclusions sur la
  codimension de la contrainte suggèrent une tendance avec la dimension, elles ne
  la démontrent pas au-delà de ces deux points.
- Un seul invariant par système (l'énergie). Rien ici sur les contraintes
  multiples ou non holonomes.
- Les modèles apprennent un champ de vecteurs, pas une solution paramétrée par
  le temps : c'est le cadre « neural ODE », pas le cadre PINN-EDP classique. Le
  choix est expliqué dans `docs/PROTOCOL.md`.

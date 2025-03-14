# TrendsAI
# Projet IA de Prédiction de Tendances - v1.0

## Vue d'ensemble
Ce projet vise à développer une intelligence artificielle capable de prédire les tendances à partir de diverses sources de données. Cette première version constitue une base fonctionnelle permettant la collecte, le traitement et l'analyse de données provenant de multiples sources.

## Structure du projet
```
projet-prediction-tendances/
├── scripts/
│   ├── api_INSEE.py         # Collecte des données INSEE
│   ├── collect_images.py                 # Scraping et formatage d'images en JSON
│   ├── scrap_google_trends.py         # Collecte des données Google Trends
│   ├── scrap_post_instagram.py       # Collecte des posts et hashtags via RapidAPI
│   ├── scrap_text.py                 # Extraction de texte à partir de PDF
│   └── script_google_trends.py        # Nettoyage des données Google Trends
├── notebooks/
│   ├── visualisation_tendances.ipynb     # Visualisation des données et des tendances
│   └── nettoyage_data.ipynb            # Scripts de nettoyage des données
└── clip_model/
    ├── images_trends.py         # Analyse d'images à partir d'un dossier local ou JSON
    ├── colors.png            # Extraction des couleurs,
    └── fashion_trends.png         # Rapport de tendances 
```

## Fonctionnalités

### Collection de données
- **Données INSEE** : Collecte des données économiques et démographiques.
- **Scraping d'images** : Récupération et formatage d'images depuis différents sites web au format JSON.
- **Google Trends** : Collecte des tendances de recherche via l'API Google Trends.
- **Réseaux sociaux** : Analyse des posts et hashtags populaires via RapidAPI.
- **Extraction PDF** : Extraction de texte à partir de documents PDF pour analyse.

### Traitement et analyse
- **Nettoyage des données** : Scripts spécifiques pour le nettoyage et la normalisation des données Google Trends.
- **Visualisation** : Notebooks Jupyter pour visualiser les tendances identifiées.
- **Notation** : Système de notation pour évaluer la pertinence des données collectées.

# Analyseur de Tendances Mode et Couleurs (Première feature)

Un outil sophistiqué pour analyser les tendances de la mode, les couleurs et les types de vêtements à partir de collections d'images en utilisant la vision par ordinateur et l'IA.

![Analyse de Tendances Mode](https://github.com/brandonvellien/TrendsAI/blob/main/clipmodel/fashion_trends_overview.png)

## Aperçu

L'Analyseur de Tendances Mode et Couleurs est un puissant outil Python qui exploite le modèle CLIP (Contrastive Language-Image Pretraining) d'OpenAI pour analyser les images de mode et extraire des informations précieuses sur les couleurs, les types de vêtements et les tendances stylistiques. Cet outil est idéal pour les designers de mode, les prévisionnistes de tendances et les entreprises de vente au détail qui cherchent à comprendre les motifs actuels de la mode et à prédire les tendances à venir.


## Caractéristiques

- **Analyse complète des couleurs** : Extraction des couleurs dominantes des images de mode avec des conventions de nommage inspirées de Pantone
- **Classification des types de vêtements** : Identification automatique des vêtements dans différentes catégories (hauts, bas, robes, vêtements d'extérieur, etc.)
- **Reconnaissance de style** : Classification des styles de mode (minimaliste, streetwear, bohème, vintage, etc.)
- **Visualisation des tendances** : Génération de visualisations professionnelles des tendances de couleurs et de vêtements
- **Génération de rapports PDF** : Création de rapports détaillés de tendances de mode inspirés du style Pantone
- **Apprentissage adaptatif** : Affinement de la classification basé sur les motifs du jeu de données pour une précision améliorée
- **Support pour plusieurs sources d'images** : Traitement de répertoires d'images locaux ou d'URLs d'images dans des fichiers JSON


## Prérequis

```
torch>=1.7.1
torchvision>=0.8.2
git+https://github.com/openai/CLIP.git
scikit-learn>=0.24.0
matplotlib>=3.3.0
Pillow>=8.0.0
requests>=2.25.0
pandas>=1.2.0
numpy>=1.19.0
```

## Installation

1. Clonez ce dépôt :
   ```bash
   git clone https://github.com/brandonvellien/TrendsAI.git
   cd TrendsAI
   ```

2. Créez et activez un environnement virtuel :
   ```bash
   python -m venv venv
   source venv/bin/activate  # Sur Windows : venv\Scripts\activate
   ```

3. Installez les dépendances requises :
   ```bash
   pip install -r requirements.txt
   ```

4. Installez CLIP :
   ```bash
   pip install git+https://github.com/openai/CLIP.git
   ```

## Utilisation

### Utilisation basique

```bash
python images_trends.py /chemin/vers/vos/images
```

### Avec apprentissage adaptatif

```bash
python images_trends.py /chemin/vers/vos/images --adaptive
```

### Utilisation d'un fichier JSON avec des URLs d'images

```bash
python images_trends.py /chemin/vers/votre/fichier_images.json
```

## Exemple de sortie

L'analyseur générera :

1. Sortie détaillée dans le terminal avec :
   - Couleurs dominantes avec références inspirées de Pantone
   - Distribution des types de vêtements
   - Tendances de style
   - Corrélations couleur-vêtement
   - Métriques générales de couleur

2. Visualisations :
   - Aperçu des tendances de mode (couleurs dominantes, types de vêtements, styles)
   - Analyse des couleurs par type de vêtement
   - Palette de tendances de couleurs style Pantone

3. Rapport PDF optionnel avec analyse complète des tendances



## Comment ça marche

1. **Collection d'images** : Rassemble des images à partir de répertoires ou d'URLs dans des fichiers JSON
2. **Modèle CLIP** : Utilise CLIP d'OpenAI pour comprendre le contenu visuel et le faire correspondre avec des catégories de mode
3. **Extraction de couleurs** : Applique le clustering K-means pour identifier les couleurs dominantes
4. **Analyse des couleurs** : Associe les valeurs RGB à des conventions de nommage de couleurs inspirées de Pantone
5. **Classification des vêtements** : Identifie les types de vêtements en utilisant les capacités visuelles-linguistiques de CLIP
6. **Reconnaissance de style** : Détermine les styles de mode basés sur les caractéristiques visuelles
7. **Analyse des tendances** : Agrège les données pour identifier les modèles et les tendances
8. **Visualisation** : Crée des visualisations professionnelles et des rapports des résultats

## Personnalisation

L'outil inclut des catégories de mode et des classifications de style complètes, mais vous pouvez facilement les étendre ou les modifier dans le code :

- `fashion_categories` : Liste des types d'articles vestimentaires
- `fashion_styles` : Liste des catégories de styles de mode
- `fashion_color_ranges` : Définitions de couleurs inspirées de Pantone



Prochaines étapes

Intégration d'algorithmes prédictifs basés sur l'apprentissage automatique
Amélioration de la précision du modèle CLIP
Développement d'une interface utilisateur
Automatisation complète du pipeline de collecte et d'analyse
Extension des sources de données pour inclure d'autres plateformes de réseaux sociaux
Amélioration des capacités d'analyse des tendances par saison et par marque

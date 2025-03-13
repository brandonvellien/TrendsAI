import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def clean_data(file_path):
    # Charger les données
    df = pd.read_csv(file_path)
    
    # Renommer les colonnes pour éviter les espaces
    df.columns = df.columns.str.replace(" ", "_").str.lower()
    
    # Vérifier et supprimer les valeurs manquantes
    df.dropna(inplace=True)
    
    # Supprimer la colonne 'ispartial' si elle n'apporte pas d'information
    if 'ispartial' in df.columns:
        df.drop(columns=['ispartial'], inplace=True)
    
    # Supprimer les doublons s'il y en a
    df.drop_duplicates(inplace=True)
    
    return df

def visualize_data(df):
    # Affichage de la distribution des tendances
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df)
    plt.title("Distribution des tendances de mode")
    plt.xticks(rotation=45)
    plt.show()
    
    # Affichage des tendances moyennes
    plt.figure(figsize=(10, 6))
    df.mean().plot(kind='bar', color='skyblue')
    plt.title("Tendances moyennes par catégorie")
    plt.xticks(rotation=45)
    plt.ylabel("Score moyen")
    plt.show()

# Exemple d'utilisation
file_path = "/mnt/data/google_trends_fashion_2023_2024_world.csv"
df_cleaned = clean_data(file_path)
print(df_cleaned.head())
visualize_data(df_cleaned)
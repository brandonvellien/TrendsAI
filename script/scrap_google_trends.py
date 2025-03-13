from pytrends.request import TrendReq
import pandas as pd
import matplotlib.pyplot as plt

def get_trends_comparison(keywords, timeframe='2024-01-01 2025-02-28', geo=''):
    pytrends = TrendReq(hl='en-FR', tz=360)
    pytrends.build_payload(keywords, cat=0, timeframe=timeframe, geo=geo, gprop='')
    data = pytrends.interest_over_time()
    return data

def plot_trends(df, keywords):
    df[keywords].plot(figsize=(14, 7))
    plt.title('Comparison of Google Search Trends (2023-2024)')
    plt.xlabel('Date')
    plt.ylabel('Search Interest')
    plt.legend(keywords)
    plt.grid(True)
    plt.show()

def main():
    # Termes de recherche à comparer
    fashion_keywords = ['t-shirt oversize', 't-shirt vintage', 't-shirt streetwear','t-shirt graphique','t-shirt minimaliste']
    
    timeframe = '2023-01-01 2024-12-31'
    geo = 'FR'  # Changer pour 'world' pour couvrir le monde entier
    
    # Comparer les tendances mode
    trends_data_fashion = get_trends_comparison(fashion_keywords, timeframe, geo)
    # Comparer les tendances globales
    trends_data_global = get_trends_comparison(global_keywords, timeframe, geo)
    
    # Convertir les tendances en DataFrames pour un traitement plus facile
    df_fashion = pd.DataFrame(trends_data_fashion)
    df_global = pd.DataFrame(trends_data_global)
    
    if df_fashion.empty or df_global.empty:
        print("Aucune tendance trouvée.")
    else:
        # Sauvegarder les tendances mode
        fashion_csv_filename = 'google_trends_fashion_2023_2024_world.csv'
        df_fashion.to_csv(fashion_csv_filename, index=False)
        print(f"\nTendances de recherche mode sauvegardées dans {fashion_csv_filename}")
        print(df_fashion.head())
        
        # Sauvegarder les tendances globales
        global_csv_filename = 'google_trends_global_2023_2024_world.csv'
        df_global.to_csv(global_csv_filename, index=False)
        print(f"\nTendances de recherche globales sauvegardées dans {global_csv_filename}")
        print(df_global.head())
        
        # Visualiser les tendances mode
        plot_trends(df_fashion, fashion_keywords)
        # Visualiser les tendances globales
        plot_trends(df_global, global_keywords)

if __name__ == "__main__":
    main()

import requests
import pandas as pd
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Chargement des variables d'environnement depuis le fichier .env
load_dotenv()

# Récupération du token depuis la variable d'environnement
api_token = os.environ.get("INSEE_API_TOKEN")

if not api_token:
    print("Erreur: La variable INSEE_API_TOKEN n'est pas définie dans le fichier .env")
    print("Veuillez créer un fichier .env avec votre token")
    exit(1)

# Configuration de base
url = "https://api.insee.fr/series/BDM/V1/data/SERIES_BDM/001694068"  # Consommation finale des ménages
headers = {
    "Authorization": f"Bearer {api_token}",
    "Accept": "application/xml"  # Accepter explicitement XML
}

# Paramètres de temps pour les 5 dernières années
end_year = datetime.now().year
start_year = end_year - 5
params = {
    "startPeriod": str(start_year),
    "endPeriod": str(end_year)
}

try:
    # Faire la requête
    print("Récupération des données de consommation...")
    response = requests.get(url, headers=headers, params=params)
    
    if response.status_code == 200:
        # Si succès, parser le XML
        from xml.etree import ElementTree as ET
        root = ET.fromstring(response.content)
        
        # Extraire les données
        data = []
        ns = {'ns1': 'urn:sdmx:org.sdmx.infomodel.datastructure.Dataflow=FR1:SERIES_BDM(1.0):ObsLevelDim:TIME_PERIOD'}
        
        for obs in root.findall('.//ns1:Obs', ns):
            period = obs.find('ns1:TIME_PERIOD', ns).text if obs.find('ns1:TIME_PERIOD', ns) is not None else None
            value = obs.find('ns1:OBS_VALUE', ns).text if obs.find('ns1:OBS_VALUE', ns) is not None else None
            
            if period and value:
                data.append({
                    'Date': period,
                    'Consommation': float(value)
                })
        
        if data:
            # Convertir en DataFrame pour l'analyse
            df = pd.DataFrame(data)
            df = df.sort_values('Date')
            
            # Afficher les résultats
            print("\nTendances de la consommation:")
            print("-" * 40)
            print("\nDernières observations:")
            print(df.tail().to_string(index=False))
            
            # Calculer les variations
            df['Variation'] = df['Consommation'].pct_change() * 100
            print("\nVariations récentes (%):")
            print(df[['Date', 'Variation']].tail().to_string(index=False))
        else:
            print("Aucune donnée trouvée dans la réponse")
        
    else:
        print(f"Erreur: {response.status_code}")
        print(f"Message: {response.text}")

except Exception as e:
    print(f"Erreur lors de l'analyse: {str(e)}")
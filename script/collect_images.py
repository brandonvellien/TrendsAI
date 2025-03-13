import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import requests
import time
import mimetypes
import json
from datetime import datetime

def setup_driver():
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    return webdriver.Chrome(options=chrome_options)

def download_image(url, output_path, index):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        
        content_type = response.headers.get('content-type', '')
        
        # Ne garder que les images JPG
        if 'image/jpeg' in content_type:
            filename = os.path.join(output_path, f'image_{index}.jpg')
            with open(filename, 'wb') as f:
                f.write(response.content)
            print(f"Image téléchargée : {filename}")
            return True, filename
        return False, None
    except Exception as e:
        print(f"Erreur lors du téléchargement de {url}: {str(e)}")
        return False, None

def get_image_metadata(img_element):
    """Récupère toutes les métadonnées disponibles pour une image."""
    metadata = {
        'src': img_element.get_attribute('src'),
        'alt': img_element.get_attribute('alt'),
        'title': img_element.get_attribute('title'),
        'aria-label': img_element.get_attribute('aria-label'),
        'class': img_element.get_attribute('class'),
        'data-attributes': {}
    }
    
    # Récupérer tous les attributs data-*
    for attribute in img_element.get_property('attributes'):
        name = attribute.get('name')
        if name and name.startswith('data-'):
            metadata['data-attributes'][name] = attribute.get('value')
            
    # Essayer de récupérer le texte des éléments parents proches
    try:
        parent = img_element.find_element(By.XPATH, './..')
        metadata['parent_text'] = parent.text
    except:
        metadata['parent_text'] = ''
        
    # Nettoyer les valeurs None et vides
    metadata = {k: v for k, v in metadata.items() if v not in [None, '', {}, []]}
    
    return metadata

def scrape_images(url, output_folder, max_scroll=5):
    output_folder = os.path.expanduser(output_folder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Dossier créé : {output_folder}")

    # Initialiser le dictionnaire pour stocker toutes les données
    all_data = {
        'scraping_info': {
            'url': url,
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_images': 0,
            'successful_downloads': 0
        },
        'images': []
    }

    driver = setup_driver()
    try:
        print(f"Accès à {url}")
        driver.get(url)
        time.sleep(5)
        
        for _ in range(max_scroll):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
        
        img_elements = driver.find_elements(By.TAG_NAME, 'img')
        print(f"Nombre d'images trouvées : {len(img_elements)}")
        
        all_data['scraping_info']['total_images'] = len(img_elements)
        
        for index, img in enumerate(img_elements):
            try:
                metadata = get_image_metadata(img)
                src = metadata.get('src')
                
                if src and src.startswith('http'):
                    success, filename = download_image(src, output_folder, index)
                    if success:
                        all_data['scraping_info']['successful_downloads'] += 1
                        
                        # Ajouter les informations de l'image
                        image_data = {
                            'id': index,
                            'filename': os.path.basename(filename),
                            'original_url': src,
                            'metadata': metadata
                        }
                        all_data['images'].append(image_data)
                        
            except Exception as e:
                print(f"Erreur lors du traitement de l'image {index}: {str(e)}")
        
        # Sauvegarder toutes les données dans un seul fichier JSON
        json_path = os.path.join(output_folder, 'images_metadata.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)
            
        print(f"Métadonnées sauvegardées dans : {json_path}")
        print(f"Téléchargement terminé. {all_data['scraping_info']['successful_downloads']} images téléchargées sur {all_data['scraping_info']['total_images']}")
        
    except Exception as e:
        print(f"Une erreur est survenue : {str(e)}")
    
    finally:
        driver.quit()

# Utilisation du script
url = 'https://www.tag-walk.com/en/collection/woman/levi-s/spring-summer-2025'
output_folder = '~/Documents/WCS/project/data/images/mode/scrapped_data/tagwalk/SS25/levis'

scrape_images(url, output_folder)
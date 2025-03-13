import http.client
import json
import os
from dotenv import load_dotenv

# Chargement des variables d'environnement depuis le fichier .env
load_dotenv()

# Configuration API depuis les variables d'environnement
API_HOST = "instagram-scraper-api2.p.rapidapi.com"
API_KEY = os.environ.get("INSTAGRAM_API_KEY")

if not API_KEY:
    print("❌ Erreur: La variable INSTAGRAM_API_KEY n'est pas définie dans le fichier .env")
    print("Veuillez créer un fichier .env avec votre clé API")
    exit(1)

def get_posts_by_hashtag(hashtag):
    """ Récupère les posts associés à un hashtag """
    conn = http.client.HTTPSConnection(API_HOST)
    headers = {
        'x-rapidapi-key': API_KEY,
        'x-rapidapi-host': API_HOST
    }   
    endpoint = f"/v1/hashtag?hashtag={hashtag}"
    
    try:
        conn.request("GET", endpoint, headers=headers)
        res = conn.getresponse()
        data = res.read()
        return json.loads(data.decode("utf-8"))
    except Exception as e:
        print(f"❌ Erreur pour le hashtag #{hashtag}: {str(e)}")
        return {}
    finally:
        conn.close()

def extract_post_info(post, hashtag):
    """Extrait uniquement les informations souhaitées d'un post"""
    # Extraction des hashtags du texte de la légende
    hashtags = []
    if 'caption' in post and post['caption']:
        hashtags = post['caption'].get('hashtags', [])

    # Extraction des informations d'image
    image_versions = post.get('image_versions', {}).get('items', [])
    if not image_versions and 'image_versions2' in post:
        image_versions = post.get('image_versions2', {}).get('candidates', [])

    return {
        'username': post.get('user', {}).get('username'),
        'comment_count': post.get('comment_count'),
        'image_versions': image_versions,
        'hashtags': hashtags,
        'like_count': post.get('like_count'),
        'item': post.get('id'),
        'hashtag_source': hashtag
    }

def main():
    # Liste des hashtags définie directement dans le code
    hashtags = ["tshirtdesign", "tshirtart", "tshirtbusiness"]
    
    filtered_posts = []

    for hashtag in hashtags:
        print(f"\n🔍 Recherche des posts pour #{hashtag}")
        response_data = get_posts_by_hashtag(hashtag)
        
        if isinstance(response_data, dict) and 'data' in response_data:
            posts = []
            data = response_data['data']
            
            if isinstance(data, list):
                posts = data
            elif isinstance(data, dict):
                if 'recent' in data:
                    posts = data['recent'].get('items', [])
                else:
                    posts = [data]
            
            # Extraction des informations souhaitées pour chaque post
            for post in posts:
                if isinstance(post, dict):
                    filtered_post = extract_post_info(post, hashtag)
                    filtered_posts.append(filtered_post)
                    print(f"📌 Post traité pour #{hashtag}")

    # Sauvegarde des posts filtrés
    if filtered_posts:
        output_filename = "filtered_posts.json"
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(filtered_posts, f, indent=2, ensure_ascii=False)
        print(f"\n📂 {len(filtered_posts)} posts sauvegardés dans {output_filename} ✅")
    else:
        print("\n⚠️ Aucun post à sauvegarder")

if __name__ == "__main__":
    main()
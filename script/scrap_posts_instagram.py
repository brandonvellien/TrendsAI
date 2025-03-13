import http.client
import json
import os
from dotenv import load_dotenv

# Chargement des variables d'environnement depuis le fichier .env
load_dotenv()

# Configuration API
API_HOST = "instagram-scraper-api2.p.rapidapi.com"
API_KEY = os.environ.get("INSTAGRAM_API_KEY")

if not API_KEY:
    print("❌ Erreur: La variable INSTAGRAM_API_KEY n'est pas définie dans le fichier .env")
    print("Veuillez créer un fichier .env avec votre clé API")
    exit(1)

def get_posts_by_user(username):
    """ Récupère les posts d'un utilisateur Instagram via l'API """
    conn = http.client.HTTPSConnection(API_HOST)
    headers = {
        'x-rapidapi-key': API_KEY,
        'x-rapidapi-host': API_HOST
    }
    
    endpoint = f"/v1.2/posts?username_or_id_or_url={username}"
    conn.request("GET", endpoint, headers=headers)
    res = conn.getresponse()
    data = res.read()
    
    try:
        return json.loads(data.decode("utf-8"))
    except json.JSONDecodeError:
        print(f"Erreur de décodage JSON pour {username}")
        return {}

def get_image_url(post):
    """ Extrait l'URL de l'image (haute résolution si possible) """
    try:
        if isinstance(post, dict):
            if 'image_versions' in post and 'items' in post['image_versions']:
                candidates = post['image_versions']['items']
                if candidates:
                    return candidates[0]['url']
            
            if post.get('media_type') == 2 and 'video_versions' in post:
                return post['video_versions'][0]['url']
            
            if post.get('media_type') == 8 and 'carousel_media' in post:
                for media in post['carousel_media']:
                    if 'image_versions' in media and 'items' in media['image_versions']:
                        return media['image_versions']['items'][0]['url']

    except Exception as e:
        print(f"Erreur lors de l'extraction de l'URL d'image: {str(e)}")
    
    return ''

def extract_post_id(post):
    """ Tente d'extraire un ID unique du post """
    return post.get('pk') or post.get('id') or post.get('ig_shortcode') or ''

def main():
    usernames = ['vaquera.nyc','balenciaga','weekdayofficial', 'supremenewyork','sarenza', 'uniqlo', 'dielsel','acnestudios', 'sunnei','hm', 'zara']
    all_posts = []
    
    for username in usernames:
        print(f"\n📸 Récupération des posts pour : {username}")
        response_data = get_posts_by_user(username)

        if isinstance(response_data, dict) and 'data' in response_data:
            posts = response_data['data']
            print(f"✅ {len(posts)} posts trouvés pour {username}")
            
            for post_key, post_value in posts.items():
                if isinstance(post_value, list):
                    for post in post_value:
                        if isinstance(post, dict):
                            try:
                                image_url = get_image_url(post)
                                post_id = extract_post_id(post)  # Nouvelle extraction d'ID

                                post_data = {
                                    'username': username,
                                    'id': post_id,  # ✅ Correction ici
                                    'caption_text': post.get('caption', {}).get('text', '') if isinstance(post.get('caption'), dict) else '',
                                    'like_count': post.get('like_count', 0),
                                    'comment_count': post.get('comment_count', 0),
                                    'created_at': post.get('taken_at', ''),
                                    'hashtags': ','.join([tag.strip("#") for tag in post.get('caption', {}).get('text', '').split() if tag.startswith("#")]) if isinstance(post.get('caption'), dict) else '',
                                    'image_url': image_url,
                                    'media_type': post.get('media_type', ''),
                                    'is_carousel': isinstance(post.get('carousel_media'), list)
                                }
                                
                                all_posts.append(post_data)
                                print(f"📌 Post ajouté : {post_id} - Image URL: {image_url}")

                            except Exception as e:
                                print(f"⚠️ Erreur lors du traitement d'un post de {username}: {str(e)}")
        
        else:
            print(f"🚨 Erreur lors de la récupération des données pour {username}")

    # Sauvegarde des résultats
    json_filename = 'instagram_posts_fastfashion.json'
    with open(json_filename, 'w', encoding='utf-8') as json_file:
        json.dump(all_posts, json_file, indent=2, ensure_ascii=False)
    
    print(f"\n📂 Données enregistrées dans {json_filename} ✅")

if __name__ == "__main__":
    main()
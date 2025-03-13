import argparse
import json
import os
import re
from typing import Dict, List, Any, Optional
import PyPDF2
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

# Télécharger les ressources NLTK nécessaires (à exécuter une seule fois)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Remplaçons la fonction sent_tokenize de NLTK par une version simplifiée
# pour éviter le problème de punkt_tab
def simple_sent_tokenize(text):
    """
    Fonction simple pour diviser le texte en phrases sans dépendre de punkt_tab.
    """
    # Séparateurs de phrases courants
    separators = ['. ', '! ', '? ', '.\n', '!\n', '?\n']
    sentences = []
    current = text
    
    # Séparer le texte en phrases en utilisant des séparateurs courants
    for sep in separators:
        pieces = []
        for chunk in current.split(sep):
            if chunk:
                pieces.append(chunk)
        current = sep[0] + ' '
        pieces = [p + sep for p in pieces[:-1]] + [pieces[-1]] if pieces else []
        sentences.extend(pieces)
        
    # Nettoyer les phrases
    cleaned_sentences = []
    for s in sentences:
        s = s.strip()
        if s:
            cleaned_sentences.append(s)
            
    return cleaned_sentences if cleaned_sentences else [text]


class PDFExtractor:
    """Classe pour extraire des données structurées d'un PDF."""
    
    def __init__(self, pdf_path: str):
        """
        Initialiser l'extracteur PDF.
        
        Args:
            pdf_path: Chemin vers le fichier PDF
        """
        self.pdf_path = pdf_path
        self.pdf_text = ""
        self.extracted_data = {
            "metadata": {},
            "content": {},
            "sections": []
        }
        self.stop_words = set(stopwords.words('english'))
        
    def extract_text(self) -> str:
        """
        Extraire tout le texte du PDF.
        
        Returns:
            Le texte complet extrait du PDF
        """
        text = ""
        try:
            with open(self.pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                
                # Extraire les métadonnées
                metadata = reader.metadata
                if metadata:
                    self.extracted_data["metadata"] = {
                        "title": metadata.get('/Title', ''),
                        "author": metadata.get('/Author', ''),
                        "subject": metadata.get('/Subject', ''),
                        "creator": metadata.get('/Creator', ''),
                        "producer": metadata.get('/Producer', ''),
                        "pages": len(reader.pages)
                    }
                
                # Extraire le texte page par page
                for page_num, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
                        # Ajouter le texte de la page à la structure de données
                        self.extracted_data["content"][f"page_{page_num+1}"] = page_text
                
                self.pdf_text = text
                return text
                
        except Exception as e:
            print(f"Erreur lors de l'extraction du texte: {e}")
            return ""
    
    def identify_sections(self, section_patterns: Optional[List[str]] = None) -> None:
        """
        Identifier les sections dans le texte en fonction des motifs donnés.
        
        Args:
            section_patterns: Liste de modèles regex pour identifier les titres de sections
        """
        if not self.pdf_text:
            self.extract_text()
            
        if not section_patterns:
            # Motifs par défaut pour identifier les titres de sections courants dans un rapport
            section_patterns = [
                r'^(?:\d+\.\s+)?([A-Z][A-Z\s]+[A-Z])(?:\s*\n|\s*$)',  # TITRE EN MAJUSCULES
                r'^(?:\d+\.\s+)?([A-Z][a-z]+(?: [A-Z][a-z]+){0,5})(?::\s|\s*\n|\s*$)',  # Titre De Section
                r'^(?:\d+\.\s+)?([A-Z][a-zA-Z]+(?: [a-zA-Z]+){0,5})(?::\s|\s*\n|\s*$)'  # Titre de section
            ]
        
        lines = self.pdf_text.split('\n')
        current_section = None
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            is_section_header = False
            for pattern in section_patterns:
                match = re.match(pattern, line)
                if match:
                    # Si nous avons déjà une section, enregistrons-la
                    if current_section:
                        self.extracted_data["sections"].append({
                            "title": current_section,
                            "content": "\n".join(current_content)
                        })
                    
                    # Commencer une nouvelle section
                    current_section = match.group(1)
                    current_content = []
                    is_section_header = True
                    break
            
            if not is_section_header and current_section:
                current_content.append(line)
            elif not is_section_header and not current_section:
                # Texte avant la première section identifiée
                if "introduction" not in self.extracted_data:
                    self.extracted_data["introduction"] = line
                else:
                    self.extracted_data["introduction"] += "\n" + line
        
        # Ajouter la dernière section
        if current_section:
            self.extracted_data["sections"].append({
                "title": current_section,
                "content": "\n".join(current_content)
            })
    
    def extract_trends(self) -> Dict:
        """
        Analyse le texte pour extraire les tendances potentielles.
        Cette fonction est spécifique aux rapports de tendances mode.
        
        Returns:
            Dictionnaire des tendances extraites
        """
        if not self.pdf_text:
            self.extract_text()
        
        trends = {
            "colors": [],
            "materials": [],
            "styles": [],
            "themes": []
        }
        
        # Liste de mots-clés pour différentes catégories
        color_keywords = ["color", "tone", "hue", "shade", "palette"]
        material_keywords = ["fabric", "material", "textile", "leather", "cotton", "silk", "wool"]
        style_keywords = ["style", "silhouette", "cut", "fit", "shape"]
        theme_keywords = ["theme", "concept", "inspiration", "aesthetic", "mood"]
        
        # Rechercher ces mots-clés dans le texte
        sentences = simple_sent_tokenize(self.pdf_text)
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Recherche de couleurs
            if any(keyword in sentence_lower for keyword in color_keywords):
                # Liste de couleurs courantes pour la recherche
                common_colors = ["red", "blue", "green", "yellow", "purple", "pink", "orange", 
                                "black", "white", "gray", "brown", "beige", "navy", "teal", 
                                "gold", "silver", "bronze", "copper"]
                found_colors = [color for color in common_colors if color in sentence_lower]
                if found_colors:
                    trends["colors"].extend(found_colors)
            
            # Recherche de matériaux
            if any(keyword in sentence_lower for keyword in material_keywords):
                # Extraire le paragraphe pour analyse
                if sentence not in trends["materials"]:
                    trends["materials"].append(sentence)
            
            # Recherche de styles
            if any(keyword in sentence_lower for keyword in style_keywords):
                if sentence not in trends["styles"]:
                    trends["styles"].append(sentence)
            
            # Recherche de thèmes
            if any(keyword in sentence_lower for keyword in theme_keywords):
                if sentence not in trends["themes"]:
                    trends["themes"].append(sentence)
        
        # Dédupliquer les listes
        trends["colors"] = list(set(trends["colors"]))
        
        # Ajouter les tendances à notre structure de données
        self.extracted_data["trends"] = trends
        
        return trends
    
    def extract_structured_data(self) -> Dict[str, Any]:
        """
        Extrait des données structurées du PDF et les prépare pour le format JSON.
        
        Returns:
            Dictionnaire des données extraites
        """
        if not self.pdf_text:
            self.extract_text()
        
        self.identify_sections()
        self.extract_trends()
        
        return self.extracted_data
    
    def save_to_json(self, output_path: str) -> None:
        """
        Sauvegarde les données extraites au format JSON.
        
        Args:
            output_path: Chemin où sauvegarder le fichier JSON
        """
        data = self.extract_structured_data()
        
        try:
            with open(output_path, 'w', encoding='utf-8') as json_file:
                json.dump(data, json_file, indent=2, ensure_ascii=False)
            print(f"Données sauvegardées avec succès dans {output_path}")
        except Exception as e:
            print(f"Erreur lors de la sauvegarde du fichier JSON: {e}")


def main():
    parser = argparse.ArgumentParser(description='Extraire des données d\'un PDF et les sauvegarder au format JSON')
    parser.add_argument('--pdf_path', '-p', default='/Users/Brandon/Documents/WCS/heuritech/Product-Version-FW-24-Womens-Fashion-Week-Report-1.pdf', 
                        help='Chemin vers le fichier PDF à traiter')
    parser.add_argument('--output', '-o', default='', help='Chemin où sauvegarder le fichier JSON')
    
    args = parser.parse_args()
    
    pdf_path = args.pdf_path
    if not os.path.exists(pdf_path):
        print(f"Le fichier {pdf_path} n'existe pas.")
        return
    
    # Créer le chemin de sortie si non spécifié
    if not args.output:
        output_path = os.path.splitext(pdf_path)[0] + '.json'
    else:
        output_path = args.output
    
    # Extraire et sauvegarder les données
    extractor = PDFExtractor(pdf_path)
    extractor.save_to_json(output_path)


if __name__ == "__main__":
    main()
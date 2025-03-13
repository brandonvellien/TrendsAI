import os
import sys
import json
import requests
import torch
import clip
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.cluster import KMeans
import colorsys
from collections import Counter
from io import BytesIO
import pandas as pd
from matplotlib.patches import Rectangle

class FashionTrendColorAnalyzer:
    def __init__(self, image_source, adaptive_mode=True):
        """
        Initialize the fashion trend analyzer with either a directory or a JSON file
        
        Args:
            image_source (str): Path to directory or JSON file containing images
            adaptive_mode (bool): Whether to adaptively refine categories based on initial analysis
        """
        # Validate input
        if not os.path.exists(image_source):
            raise ValueError(f"Invalid path: {image_source}")
        
        self.image_source = image_source
        self.adaptive_mode = adaptive_mode
        self.adaptive_weights = {}
        
        # Load CLIP model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        
        # Collect image paths or URLs
        self.image_paths = self._collect_image_sources()
        
        # Fashion item categories for classification - comprehensive list
        self.fashion_categories = [
            # Tops
            "t-shirt", "blouse", "shirt", "tank top", "crop top", "sweater", "cardigan", "hoodie",
            "sweatshirt", "tunic", "polo shirt", "jersey", "turtleneck", "halter top",
            
            # Bottoms
            "jeans", "pants", "trousers", "shorts", "skirt", "leggings", "joggers", "chinos",
            "culottes", "palazzo pants", "cargo pants", "bermuda shorts", "wide-leg pants",
            
            # Dresses
            "dress", "maxi dress", "midi dress", "mini dress", "sundress", "cocktail dress",
            "evening gown", "shift dress", "wrap dress", "slip dress", "bodycon dress",
            
            # Outerwear
            "jacket", "coat", "blazer", "bomber jacket", "denim jacket", "leather jacket",
            "trench coat", "parka", "puffer jacket", "windbreaker", "poncho", "cape",
            
            # Suits & Sets
            "suit", "pantsuit", "jumpsuit", "romper", "two-piece set", "three-piece suit",
            "tuxedo", "co-ord set", "tracksuit",
            
            # Activewear
            "yoga pants", "sports bra", "athletic shorts", "cycling shorts", "tennis skirt",
            "track pants", "workout top", "gym shorts", "swimwear", "swimming trunks", "bikini",
            
            # Shoes
            "sneakers", "boots", "heels", "sandals", "flats", "loafers", "oxford shoes",
            "platform shoes", "espadrilles", "wedges", "mules", "stilettos",
            
            # Accessories
            "handbag", "backpack", "tote bag", "clutch", "wallet", "belt", "scarf",
            "gloves", "hat", "beanie", "cap", "sunglasses", "jewelry", "watch", "necklace",
            "earrings", "bracelet", "ring"
        ]
        
        # Fashion style categories - comprehensive list
        self.fashion_styles = [
            "minimalist", "streetwear", "bohemian", "vintage", "preppy", "athleisure",
            "business casual", "formal", "avant-garde", "sustainable", "cottagecore",
            "y2k", "goth", "punk", "grunge", "luxury", "haute couture", "casual", 
            "resort wear", "workwear", "retro", "urban", "hip-hop", "sporty"
        ]
        
        # Pantone-Inspired Fashion Color Ranges - comprehensive and balanced
        self.fashion_color_ranges = [
            # Neutrals
            (0, 30, 'Naturals - Cream & Ivory', '#F8F6F0', 'Pantone 11-0104 TCX'),
            (30, 60, 'Neutrals - Beige & Ecru', '#D6C6B0', 'Pantone 14-1118 TCX'),
            (60, 90, 'Neutrals - Taupe & Greige', '#A39382', 'Pantone 16-1406 TCX'),
            
            # Reds & Oranges
            (0, 15, 'True Red', '#D12631', 'Pantone 18-1662 TCX'),
            (15, 30, 'Coral & Salmon', '#FF6F61', 'Pantone 16-1546 TCX'),
            (30, 45, 'Terracotta & Clay', '#BD4B37', 'Pantone 18-1438 TCX'),
            
            # Oranges & Browns
            (45, 60, 'Amber & Caramel', '#D78A41', 'Pantone 16-1342 TCX'),
            (60, 75, 'Cognac & Rust', '#A5552A', 'Pantone 18-1248 TCX'),
            
            # Yellows
            (75, 90, 'Mustard & Ochre', '#DBAF3A', 'Pantone 15-0948 TCX'),
            (90, 105, 'Canary & Lemon', '#F9E04C', 'Pantone 12-0643 TCX'),
            
            # Greens
            (105, 135, 'Olive & Moss', '#5E6738', 'Pantone 18-0430 TCX'),
            (135, 165, 'Sage & Mint', '#AABD8C', 'Pantone 15-6316 TCX'),
            (165, 195, 'Emerald & Jade', '#00A170', 'Pantone 17-5641 TCX'),
            
            # Blues
            (195, 225, 'Teal & Aqua', '#4799B7', 'Pantone 16-4834 TCX'),
            (225, 255, 'Cobalt & Denim', '#0047AB', 'Pantone 19-4045 TCX'),
            (255, 270, 'Navy & Indigo', '#1D334A', 'Pantone 19-4027 TCX'),
            
            # Purples
            (270, 285, 'Lavender & Lilac', '#B69FCB', 'Pantone 16-3416 TCX'),
            (285, 315, 'Violet & Amethyst', '#9678B6', 'Pantone 17-3628 TCX'),
            (315, 330, 'Mauve & Plum', '#8E4585', 'Pantone 19-2428 TCX'),
            
            # Pinks
            (330, 345, 'Berry & Raspberry', '#C6174E', 'Pantone 18-2140 TCX'),
            (345, 360, 'Blush & Rose', '#E8B4B8', 'Pantone 14-1511 TCX')
        ]
    
    def _collect_image_sources(self):
        """
        Collect image paths or URLs from directory or JSON file
        
        Returns:
            list: Paths or URLs to images
        """
        image_sources = []
        
        # Check if it's a directory
        if os.path.isdir(self.image_source):
            image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
            
            for root, _, files in os.walk(self.image_source):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in image_extensions):
                        image_sources.append(os.path.join(root, file))
        
        # Check if it's a JSON file
        elif self.image_source.lower().endswith('.json'):
            with open(self.image_source, 'r') as f:
                data = json.load(f)
                
                # Extract image URLs from various potential JSON structures
                if isinstance(data, list):
                    # List of dictionaries with image URLs
                    for item in data:
                        if 'image_url' in item:
                            image_sources.append(item['image_url'])
                elif isinstance(data, dict):
                    # Recursively find image URLs in nested dictionaries
                    def find_image_urls(obj):
                        urls = []
                        if isinstance(obj, dict):
                            for key, value in obj.items():
                                if key.lower() in ['image', 'image_url', 'url']:
                                    if isinstance(value, str) and value.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                                        urls.append(value)
                                else:
                                    urls.extend(find_image_urls(value))
                        elif isinstance(obj, list):
                            for item in obj:
                                urls.extend(find_image_urls(item))
                        return urls
                    
                    image_sources = find_image_urls(data)
        
        if not image_sources:
            raise ValueError(f"No images found in {self.image_source}")
        
        print(f"Found {len(image_sources)} images")
        return image_sources
    
    def _load_image(self, image_source):
        """
        Load image from local path or URL
        
        Args:
            image_source (str): Local path or URL of the image
        
        Returns:
            PIL.Image: Loaded and converted image
        """
        try:
            # Check if it's a URL
            if image_source.startswith(('http://', 'https://')):
                response = requests.get(image_source, timeout=10)
                image = Image.open(BytesIO(response.content)).convert('RGB')
            else:
                # Local file
                image = Image.open(image_source).convert('RGB')
            
            return image.resize((300, 300))
        
        except Exception as e:
            print(f"Error loading {image_source}: {e}")
            return None
    
    def _find_closest_pantone(self, r, g, b):
        """
        Find the closest Pantone-like name for an RGB color
        This is a simplified approach as actual Pantone matching requires 
        sophisticated color management systems
        
        Args:
            r, g, b: RGB color values (0-255)
            
        Returns:
            str: Pantone-like color description
        """
        h, s, v = colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)
        hue_deg = h * 360
        
        # Special case for achromatic colors
        if s < 0.1:
            if v < 0.15:
                return "Black"
            elif v < 0.3:
                return "Charcoal Grey"
            elif v < 0.5:
                return "Medium Grey"
            elif v < 0.8:
                return "Light Grey"
            else:
                return "Off-White"
        
        # Find the color range this hue belongs to
        for start, end, name, _, pantone_code in self.fashion_color_ranges:
            # Handle ranges that wrap around the color wheel
            if start > end:  # Handle wrapping around 360
                if hue_deg >= start or hue_deg < end:
                    base_name = name
                    break
            else:
                if start <= hue_deg < end:
                    base_name = name
                    break
        else:
            base_name = "Custom"
        
        # Add value (lightness) and saturation descriptors
        if v < 0.2:
            value_desc = "Deep"
        elif v < 0.4:
            value_desc = "Dark"
        elif v < 0.7:
            value_desc = "Medium"
        elif v < 0.9:
            value_desc = "Light"
        else:
            value_desc = "Bright"
            
        if s < 0.2:
            sat_desc = "Greyed"
        elif s < 0.4:
            sat_desc = "Muted"
        elif s < 0.7:
            sat_desc = "Soft"
        else:
            sat_desc = "Vibrant"
        
        return f"{value_desc} {sat_desc} {base_name}"
    
    def _get_hex_and_pantone(self, rgb):
        """
        Convert RGB to hex and find closest pantone reference
        
        Args:
            rgb: numpy array of RGB values
            
        Returns:
            tuple: (hex_code, pantone-like description)
        """
        r, g, b = rgb
        hex_color = f'#{int(r):02x}{int(g):02x}{int(b):02x}'
        h, s, v = colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)
        hue_deg = h * 360
        
        # Special case for very dark colors (near black)
        if v < 0.15:
            return hex_color, "Pantone 19-4005 TCX"  # Black
            
        # Special case for very light colors (near white)
        if v > 0.95 and s < 0.05:
            return hex_color, "Pantone 11-0601 TCX"  # White
        
        # Special case for grays (low saturation)
        if s < 0.1:
            if v < 0.3:
                return hex_color, "Pantone 19-0303 TCX"  # Dark Gray
            elif v < 0.6:
                return hex_color, "Pantone 17-0000 TCX"  # Medium Gray
            else:
                return hex_color, "Pantone 12-0000 TCX"  # Light Gray
        
        # Find the fashion color range and pantone reference
        for start, end, name, _, pantone_code in self.fashion_color_ranges:
            # Handle ranges that wrap around the color wheel
            if start > end:  # Handle wrapping around 360
                if hue_deg >= start or hue_deg < end:
                    return hex_color, pantone_code
            else:
                if start <= hue_deg < end:
                    return hex_color, pantone_code
        
        # Fallback
        return hex_color, "Custom"
    
    def _classify_fashion_item(self, image_pil):
        """
        Classify the type of fashion item in the image using CLIP
        
        Args:
            image_pil (PIL.Image): Image to classify
            
        Returns:
            tuple: (item_type, confidence_score, style)
        """
        # Prepare the image for CLIP
        image = self.preprocess(image_pil).unsqueeze(0).to(self.device)
        
        # Prepare text descriptions for categories
        category_tokens = clip.tokenize(self.fashion_categories).to(self.device)
        style_tokens = clip.tokenize(self.fashion_styles).to(self.device)
        
        # Get features
        with torch.no_grad():
            image_features = self.model.encode_image(image)
            category_features = self.model.encode_text(category_tokens)
            style_features = self.model.encode_text(style_tokens)
            
            # Normalize features
            image_features /= image_features.norm(dim=-1, keepdim=True)
            category_features /= category_features.norm(dim=-1, keepdim=True)
            style_features /= style_features.norm(dim=-1, keepdim=True)
            
            # Apply adaptive weights if available
            if self.adaptive_mode and hasattr(self, 'category_weights'):
                category_weights = torch.tensor(
                    [self.category_weights.get(cat, 1.0) for cat in self.fashion_categories],
                    device=self.device
                )
                
                style_weights = torch.tensor(
                    [self.style_weights.get(style, 1.0) for style in self.fashion_styles],
                    device=self.device
                )
                
                # Calculate weighted similarity
                raw_category_similarity = image_features @ category_features.T
                category_similarity = (100.0 * raw_category_similarity * category_weights).softmax(dim=-1)
                
                raw_style_similarity = image_features @ style_features.T
                style_similarity = (100.0 * raw_style_similarity * style_weights).softmax(dim=-1)
            else:
                # Standard similarity calculation
                category_similarity = (100.0 * image_features @ category_features.T).softmax(dim=-1)
                style_similarity = (100.0 * image_features @ style_features.T).softmax(dim=-1)
            
            # Get top category and confidence
            category_probs = category_similarity[0].tolist()
            top_category_idx = category_similarity[0].argmax().item()
            top_category = self.fashion_categories[top_category_idx]
            top_category_score = category_probs[top_category_idx]
            
            # Get top style and confidence
            style_probs = style_similarity[0].tolist()
            top_style_idx = style_similarity[0].argmax().item()
            top_style = self.fashion_styles[top_style_idx]
            top_style_score = style_probs[top_style_idx]
            
            # Get top 3 categories for more detailed analysis
            top3_category_indices = torch.topk(category_similarity[0], 3).indices.tolist()
            top3_categories = [(self.fashion_categories[idx], category_probs[idx]) for idx in top3_category_indices]
            
            # Get top 3 styles
            top3_style_indices = torch.topk(style_similarity[0], 3).indices.tolist()
            top3_styles = [(self.fashion_styles[idx], style_probs[idx]) for idx in top3_style_indices]
        
        return {
            'primary_category': top_category,
            'primary_category_score': top_category_score,
            'top3_categories': top3_categories,
            'primary_style': top_style,
            'primary_style_score': top_style_score,
            'top3_styles': top3_styles
        }
    
    def _perform_adaptive_learning(self, initial_results):
        """
        Analyze initial results to determine which categories and styles should be given more weight
        
        Args:
            initial_results (list): Initial classification results
        """
        if not self.adaptive_mode or len(initial_results) < 3:
            return
            
        # Extract initial category and style distributions
        categories = [r['garment_analysis']['primary_category'] for r in initial_results]
        styles = [r['garment_analysis']['primary_style'] for r in initial_results]
        
        # Count occurrences
        category_counter = Counter(categories)
        style_counter = Counter(styles)
        
        # Initialize weights
        self.category_weights = {cat: 1.0 for cat in self.fashion_categories}
        self.style_weights = {style: 1.0 for style in self.fashion_styles}
        
        # Set weights based on frequency (higher weight for common categories)
        total_images = len(initial_results)
        
        # Only boost weights for categories that appear in at least 10% of images
        for category, count in category_counter.items():
            if count / total_images >= 0.1:
                # Boost weight proportional to frequency
                self.category_weights[category] = 1.0 + (count / total_images)
                
                # Also boost related categories (e.g. if "shorts" is common, boost "bermuda shorts")
                for cat in self.fashion_categories:
                    if category in cat and cat != category:
                        self.category_weights[cat] = 1.0 + (count / total_images) * 0.5
        
        # Similar for styles
        for style, count in style_counter.items():
            if count / total_images >= 0.1:
                self.style_weights[style] = 1.0 + (count / total_images)
                
                # Also boost related styles
                for s in self.fashion_styles:
                    if style in s and s != style:
                        self.style_weights[s] = 1.0 + (count / total_images) * 0.5
        
        print("Adaptive learning complete. Adjusted weights for classification.")
    
    def analyze_fashion_trends(self, num_colors=15, min_cluster_size=100):
        """
        Analyze both color and garment type trends across all images
        
        Args:
            num_colors (int): Number of dominant colors to extract
            min_cluster_size (int): Minimum size for a color cluster to be considered
            
        Returns:
            dict: Comprehensive fashion trend analysis including colors and garment types
        """
        all_colors = []
        all_color_objects = []
        all_garment_types = []
        all_style_types = []
        image_analysis_results = []
        
        # First phase: Initial analysis of a subset for adaptive learning
        if self.adaptive_mode:
            # Analyze a subset of images first
            subset_size = min(10, len(self.image_paths))
            initial_results = []
            
            for i, image_source in enumerate(self.image_paths[:subset_size]):
                try:
                    image = self._load_image(image_source)
                    if image is None:
                        continue
                        
                    garment_analysis = self._classify_fashion_item(image)
                    
                    # Perform color analysis for the image
                    img_array = np.array(image)
                    pixels = img_array.reshape(-1, 3)
                    
                    kmeans = KMeans(n_clusters=min(num_colors, len(pixels)), n_init=10)
                    kmeans.fit(pixels)
                    
                    labels = kmeans.labels_
                    cluster_sizes = np.bincount(labels)
                    
                    image_colors = []
                    colors = kmeans.cluster_centers_
                    
                    for j, color in enumerate(colors):
                        if cluster_sizes[j] < min_cluster_size:
                            continue
                            
                        r, g, b = color
                        h, s, v = colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)
                        
                        hex_color, pantone_ref = self._get_hex_and_pantone(color)
                        color_name = self._find_closest_pantone(r, g, b)
                        
                        color_obj = {
                            'rgb': color,
                            'hex': hex_color,
                            'hue': h * 360,
                            'saturation': s * 100,
                            'value': v * 100,
                            'brightness': np.mean(color),
                            'color_complexity': np.std(color),
                            'proportion': cluster_sizes[j] / len(labels),
                            'pantone_ref': pantone_ref,
                            'color_name': color_name
                        }
                        
                        image_colors.append(color_obj)
                    
                    initial_results.append({
                        'source': image_source,
                        'garment_analysis': garment_analysis,
                        'colors': image_colors
                    })
                    
                except Exception as e:
                    print(f"Error in initial processing of {image_source}: {e}")
            
            # Perform adaptive learning based on initial results
            self._perform_adaptive_learning(initial_results)
        
        # Main analysis phase
        for image_source in self.image_paths:
            try:
                # Load image
                image = self._load_image(image_source)
                
                if image is None:
                    continue
                
                # Analyze garment type
                garment_analysis = self._classify_fashion_item(image)
                all_garment_types.append(garment_analysis['primary_category'])
                all_style_types.append(garment_analysis['primary_style'])
                
                # Color analysis
                img_array = np.array(image)
                pixels = img_array.reshape(-1, 3)
                
                # K-means clustering for colors
                kmeans = KMeans(n_clusters=min(num_colors, len(pixels)), n_init=10)
                kmeans.fit(pixels)
                
                # Get cluster sizes
                labels = kmeans.labels_
                cluster_sizes = np.bincount(labels)
                
                # Store image-specific color analysis
                image_colors = []
                
                # Process colors with their proportions in the image
                colors = kmeans.cluster_centers_
                for i, color in enumerate(colors):
                    # Skip tiny clusters
                    if cluster_sizes[i] < min_cluster_size:
                        continue
                        
                    r, g, b = color
                    h, s, v = colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)
                    
                    hex_color, pantone_ref = self._get_hex_and_pantone(color)
                    color_name = self._find_closest_pantone(r, g, b)
                    
                    color_obj = {
                        'rgb': color,
                        'hex': hex_color,
                        'hue': h * 360,
                        'saturation': s * 100,
                        'value': v * 100,
                        'brightness': np.mean(color),
                        'color_complexity': np.std(color),
                        'proportion': cluster_sizes[i] / len(labels),
                        'pantone_ref': pantone_ref,
                        'color_name': color_name
                    }
                    
                    all_colors.append(color)
                    all_color_objects.append(color_obj)
                    image_colors.append(color_obj)
                
                # Store complete image analysis
                image_analysis_results.append({
                    'source': image_source,
                    'garment_analysis': garment_analysis,
                    'colors': image_colors
                })
            
            except Exception as e:
                print(f"Error processing {image_source}: {e}")
        
        # Ensure we have data before analysis
        if not all_colors or not all_garment_types:
            print("No data could be extracted. Check image processing.")
            return {}
        
        # Analyze color distribution
        color_analysis = self._analyze_fashion_color_distribution(all_color_objects, np.array(all_colors))
        
        # Analyze garment and style distribution
        garment_counts = Counter(all_garment_types)
        style_counts = Counter(all_style_types)
        
        # Calculate garment type trends
        total_garments = len(all_garment_types)
        garment_distribution = {
            garment: {
                'count': count,
                'percentage': (count / total_garments) * 100
            }
            for garment, count in garment_counts.most_common()
        }
        
        # Calculate style trends
        total_styles = len(all_style_types)
        style_distribution = {
            style: {
                'count': count,
                'percentage': (count / total_styles) * 100
            }
            for style, count in style_counts.most_common()
        }
        
        # Create color-garment correlations
        color_garment_correlations = {}
        garment_image_counts = {}
        
        # First, group images by garment type
        for result in image_analysis_results:
            garment = result['garment_analysis']['primary_category']
            
            # Initialize if first time seeing this garment
            if garment not in color_garment_correlations:
                color_garment_correlations[garment] = []
                garment_image_counts[garment] = 0
            
            # Count images for this garment
            garment_image_counts[garment] += 1
            
            # Add colors for this garment
            for color in result['colors']:
                color_garment_correlations[garment].append(color)
        
        # Process correlations to find dominant colors per garment type
        garment_color_trends = {}
        
        for garment, colors in color_garment_correlations.items():
            if not colors or garment_image_counts[garment] == 0:
                continue
                
            # Group colors by similarity
            color_bins = {}
            
            # Cluster similar colors together
            for color in colors:
                rgb = color['rgb']
                # Quantize the color to create bins of similar colors
                quantized_rgb = tuple(int(c / 25) * 25 for c in rgb)
                
                if quantized_rgb not in color_bins:
                    color_bins[quantized_rgb] = []
                    
                color_bins[quantized_rgb].append(color)
            
            # Get most common color bins and calculate proper percentages
            binned_colors = []
            for bin_rgb, bin_colors in color_bins.items():
                # Get average color in this bin
                avg_r = sum(c['rgb'][0] for c in bin_colors) / len(bin_colors)
                avg_g = sum(c['rgb'][1] for c in bin_colors) / len(bin_colors)
                avg_b = sum(c['rgb'][2] for c in bin_colors) / len(bin_colors)
                
                # Get a representative color
                rep_color = bin_colors[0].copy()
                
                # Update RGB and recalculate hex and other properties
                rep_color['rgb'] = np.array([avg_r, avg_g, avg_b])
                rep_color['hex'], rep_color['pantone_ref'] = self._get_hex_and_pantone(rep_color['rgb'])
                rep_color['color_name'] = self._find_closest_pantone(avg_r, avg_g, avg_b)
                
                # Calculate percentage relative to this garment type (not total dataset)
                rep_color['frequency'] = (len(bin_colors) / len(colors)) * 100
                
                binned_colors.append(rep_color)
            
            # Sort by frequency
            binned_colors.sort(key=lambda x: x['frequency'], reverse=True)
            
            garment_color_trends[garment] = binned_colors
        
        # Combine all analyses
        comprehensive_analysis = {
            'color_trends': color_analysis,
            'garment_trends': {
                'distribution': garment_distribution,
                'top_garments': [item[0] for item in garment_counts.most_common(10)]
            },
            'style_trends': {
                'distribution': style_distribution,
                'top_styles': [item[0] for item in style_counts.most_common(10)]
            },
            'color_garment_trends': garment_color_trends,
            'detailed_image_analysis': image_analysis_results
        }
        
        return comprehensive_analysis
    
    def analyze_color_trends(self, num_colors=15, min_cluster_size=100):
        """
        Analyze dominant colors across all images with fashion-specific insights
        (Legacy method maintained for backward compatibility)
        
        Args:
            num_colors (int): Number of dominant colors to extract
            min_cluster_size (int): Minimum size for a color cluster to be considered
            
        Returns:
            dict: Comprehensive color trend analysis
        """
        # Call the new comprehensive method but return only color analysis
        comprehensive_results = self.analyze_fashion_trends(num_colors, min_cluster_size)
        return comprehensive_results.get('color_trends', {})
    
    def _analyze_fashion_color_distribution(self, color_objects, color_array):
        """
        Analyze color distribution with fashion-specific insights
        
        Args:
            color_objects (list): List of color dictionaries
            color_array (ndarray): Array of RGB values
            
        Returns:
            dict: Comprehensive color trend analysis
        """
        if not color_objects:
            return {}
            
        full_analysis = {
            'color_range_distribution': {},
            'color_metrics': {},
            'pantone_distribution': {},
            'dominant_colors': []
        }
        
        # Analyze color ranges
        for start, end, name, hex_color, pantone_code in self.fashion_color_ranges:
            # Find colors in this range
            if start > end:  # Handle wrapping around 360
                in_range = [c for c in color_objects if c['hue'] >= start or c['hue'] < end]
            else:
                in_range = [c for c in color_objects if start <= c['hue'] < end]
            
            # Skip empty ranges
            if not in_range:
                continue
                
            # Calculate percentage
            percentage = len(in_range) / len(color_objects) * 100
            
            # Get representative colors (most vibrant and most common)
            sorted_by_vividness = sorted(in_range, key=lambda x: x['saturation'] * x['value'], reverse=True)
            vivid_colors = [c['hex'] for c in sorted_by_vividness[:3]]
            
            # Color range specific insights
            range_info = {
                'count': len(in_range),
                'percentage': percentage,
                'representative_colors': vivid_colors,
                'pantone_ref': pantone_code,
                'primary_color': hex_color,
                'average_saturation': np.mean([c['saturation'] for c in in_range]),
                'average_brightness': np.mean([c['value'] for c in in_range]),
                'color_complexity': np.mean([c['color_complexity'] for c in in_range])
            }
            
            full_analysis['color_range_distribution'][name] = range_info
        
        # Additional Pantone analysis
        pantone_counts = Counter([c['pantone_ref'] for c in color_objects])
        for pantone, count in pantone_counts.most_common(10):
            percentage = count / len(color_objects) * 100
            matching_colors = [c for c in color_objects if c['pantone_ref'] == pantone]
            
            if matching_colors:
                # Get a representative color
                rep_color = matching_colors[0]['hex']
                
                full_analysis['pantone_distribution'][pantone] = {
                    'count': count,
                    'percentage': percentage,
                    'representative_color': rep_color
                }
        
        # Find dominant colors across all images
        if len(color_array) > 0:
            # Use K-means to find the most representative colors across all images
            n_dominant = min(10, len(color_array))
            kmeans = KMeans(n_clusters=n_dominant, n_init=10)
            kmeans.fit(color_array)
            
            dominant_colors = kmeans.cluster_centers_
            
            # Get counts per cluster
            labels = kmeans.labels_
            label_counts = np.bincount(labels)
            percentages = label_counts / len(labels) * 100
            
            # Add dominant colors to analysis
            for i, color in enumerate(dominant_colors):
                r, g, b = color
                hex_color, pantone_ref = self._get_hex_and_pantone(color)
                color_name = self._find_closest_pantone(r, g, b)
                
                full_analysis['dominant_colors'].append({
                    'rgb': [int(r), int(g), int(b)],
                    'hex': hex_color,
                    'pantone_ref': pantone_ref,
                    'color_name': color_name,
                    'percentage': percentages[i]
                })
            
            # Sort by percentage
            full_analysis['dominant_colors'] = sorted(
                full_analysis['dominant_colors'], 
                key=lambda x: x['percentage'], 
                reverse=True
            )
        
        # Overall color metrics
        unique_hex = set(c['hex'] for c in color_objects)
        
        full_analysis['color_metrics'] = {
            'total_unique_colors': len(unique_hex),
            'average_saturation': np.mean([c['saturation'] for c in color_objects]),
            'average_brightness': np.mean([c['value'] for c in color_objects]),
            'color_diversity_index': len(unique_hex) / len(color_objects) if color_objects else 0
        }
        
        return full_analysis
    
    def visualize_fashion_trends(self, fashion_trends):
        """
        Create cleaner visualizations of fashion trends including garment types and colors
        
        Args:
            fashion_trends (dict): Comprehensive fashion trend analysis
        """
        if not fashion_trends:
            print("Insufficient data for visualization")
            return
            
        # Extract different trend components
        color_trends = fashion_trends.get('color_trends', {})
        garment_trends = fashion_trends.get('garment_trends', {})
        style_trends = fashion_trends.get('style_trends', {})
        color_garment_trends = fashion_trends.get('color_garment_trends', {})
        
        # Create a multi-page visualization with cleaner layout
        plt.figure(figsize=(15, 20))
        
        # 1. Create a simplified Pantone-style color palette for dominant colors - takes 1/3 of the page
        plt.subplot(3, 1, 1)
        plt.title('Dominant Color Palette', fontsize=16, fontweight='bold')
        
        dominant_colors = color_trends.get('dominant_colors', [])
        if dominant_colors:
            # Keep only top 10 colors to avoid clutter
            top_colors = dominant_colors[:10]
            
            # Create a horizontal color swatch layout
            for i, color_info in enumerate(top_colors):
                # Draw color rectangle
                rect = plt.Rectangle((i, 0), 0.8, 1, color=color_info['hex'])
                plt.gca().add_patch(rect)
                
                # Add color name and percentage below the swatch
                plt.text(i+0.4, -0.15, f"{color_info['percentage']:.1f}%", 
                         ha='center', va='top', fontsize=9, fontweight='bold')
                
                # Add Pantone reference
                plt.text(i+0.4, 0.5, color_info['pantone_ref'].split(' ')[-1], 
                         ha='center', va='center', fontsize=8,
                         color='white' if self._is_dark_color(color_info['hex']) else 'black')
            
            plt.xlim(0, len(top_colors))
            plt.ylim(-0.5, 1.5)
        
        plt.axis('off')
        
        # 2. Create garment type visualization - takes 1/3 of the page
        plt.subplot(3, 1, 2)
        plt.title('Top Garment Types', fontsize=16, fontweight='bold')
        
        distribution = garment_trends.get('distribution', {})
        if distribution:
            # Take only top 10 garments for clarity
            items = list(distribution.items())
            items.sort(key=lambda x: x[1]['percentage'], reverse=True)
            top_items = items[:10]
            
            garments = [item[0].capitalize() for item in top_items]
            percentages = [item[1]['percentage'] for item in top_items]
            
            # Use a simple bar chart with consistent colors
            bars = plt.barh(garments, percentages, color='skyblue')
            
            # Add percentage labels
            for i, bar in enumerate(bars):
                plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                         f"{percentages[i]:.1f}%", va='center')
            
            plt.xlabel('Percentage (%)')
            plt.tight_layout()
        
        # 3. Create style trends visualization - takes 1/3 of the page
        plt.subplot(3, 1, 3)
        plt.title('Fashion Style Trends', fontsize=16, fontweight='bold')
        
        distribution = style_trends.get('distribution', {})
        if distribution:
            # Take top 8 styles for clarity
            items = list(distribution.items())
            items.sort(key=lambda x: x[1]['percentage'], reverse=True)
            top_items = items[:8]
            
            styles = [item[0].capitalize() for item in top_items]
            percentages = [item[1]['percentage'] for item in top_items]
            
            # Use a pie chart with a clean color scheme
            plt.pie(percentages, labels=styles, autopct='%1.1f%%', 
                    colors=plt.cm.Pastel1(np.linspace(0, 1, len(styles))),
                    wedgeprops={'linewidth': 1, 'edgecolor': 'white'},
                    textprops={'fontsize': 9})
            plt.axis('equal')
        
        plt.tight_layout(pad=3.0)
        plt.savefig('fashion_trends_overview.png', dpi=300, bbox_inches='tight')
        
        # Create a second figure for top colors by garment type
        plt.figure(figsize=(15, 10))
        plt.title('TOP COLORS BY GARMENT TYPE', fontsize=20, fontweight='bold')
        
        if color_garment_trends:
            # Take top 6 garment types
            top_garments = list(color_garment_trends.keys())[:6]
            
            # Create a 2x3 grid
            for i, garment in enumerate(top_garments):
                plt.subplot(2, 3, i+1)
                plt.title(garment.capitalize(), fontsize=12, fontweight='bold')
                
                colors = color_garment_trends[garment]
                if not colors:
                    plt.text(0.5, 0.5, "No color data", ha='center', va='center')
                    plt.axis('off')
                    continue
                    
                # Create color swatches for this garment
                for j, color_info in enumerate(colors[:3]):  # Top 3 colors
                    rect = plt.Rectangle((0, j), 2, 0.8, color=color_info['hex'])
                    plt.gca().add_patch(rect)
                    
                    # Add text with pantone and percentage
                    plt.text(2.2, j+0.4, f"{color_info['frequency']:.1f}% - {color_info['pantone_ref']}", 
                             va='center', fontsize=9)
                
                plt.xlim(0, 6)
                plt.ylim(-0.2, 3)
                plt.axis('off')
        
        plt.tight_layout(pad=3.0)
        plt.savefig('colors_by_garment.png', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _is_dark_color(self, hex_color):
        """
        Check if a color is dark (for text contrast)
        
        Args:
            hex_color (str): Hex color code
            
        Returns:
            bool: True if color is dark, False otherwise
        """
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        
        # Simple brightness formula
        return (r + g + b) < 382  # 128*3 is middle gray, using slightly higher threshold
    
    def _visualize_dominant_colors(self, color_trends):
        """
        Visualize dominant colors in Pantone style
        
        Args:
            color_trends (dict): Color trend analysis
        """
        plt.title('Dominant Color Palette', fontsize=16, fontweight='bold')
        
        dominant_colors = color_trends.get('dominant_colors', [])
        if not dominant_colors:
            plt.text(0.5, 0.5, "No color data available", ha='center', va='center')
            plt.axis('off')
            return
            
        # Create color swatches
        height = 0.8
        width = 1.0
        
        for i, color_info in enumerate(dominant_colors[:10]):  # Top 10 colors
            # Draw color rectangle
            rect = plt.Rectangle((0, i), width, height, color=color_info['hex'])
            plt.gca().add_patch(rect)
            
            # Add labels
            percentage = f"{color_info['percentage']:.1f}%"
            plt.text(width + 0.1, i + height/2, 
                     f"{percentage} - {color_info['color_name']}", 
                     va='center', ha='left', fontsize=10)
            
            # Add Pantone reference
            plt.text(width/2, i + height/2, 
                     f"{color_info['pantone_ref']}", 
                     va='center', ha='center', fontsize=8,
                     color='white' if sum(color_info['rgb']) < 380 else 'black')
        
        plt.xlim(0, 4)
        plt.ylim(-0.2, len(dominant_colors[:10]))
        plt.axis('off')
    
    def _visualize_color_ranges(self, color_trends):
        """
        Visualize color range distribution
        
        Args:
            color_trends (dict): Color trend analysis
        """
        plt.title('Fashion Color Range Distribution', fontsize=16, fontweight='bold')
        
        color_ranges = color_trends.get('color_range_distribution', {})
        if not color_ranges:
            plt.text(0.5, 0.5, "No color range data available", ha='center', va='center')
            plt.axis('off')
            return
            
        # Prepare data
        names = []
        percentages = []
        colors = []
        
        for name, data in color_ranges.items():
            if data['percentage'] > 0.5:  # Only show ranges with significant presence
                names.append(name)
                percentages.append(data['percentage'])
                colors.append(data['primary_color'])
        
        # Sort by color family (hue order)
        sorted_data = sorted(zip(names, percentages, colors), 
                            key=lambda x: self._get_hue_from_hex(x[2]))
        names, percentages, colors = zip(*sorted_data) if sorted_data else ([], [], [])
        
        # Create horizontal bar chart with actual colors
        bars = plt.barh(names, percentages, color=colors)
        
        # Add percentage and Pantone code labels
        for i, bar in enumerate(bars):
            plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                     f"{percentages[i]:.1f}%", 
                     va='center')
            
            # Add Pantone reference inside bar if wide enough
            if bar.get_width() > 5:
                pantone_ref = color_ranges[names[i]]['pantone_ref']
                plt.text(bar.get_width()/2, bar.get_y() + bar.get_height()/2, 
                         f"{pantone_ref}", va='center', ha='center', 
                         color='white' if self._is_dark_color(colors[i]) else 'black')
        
        plt.xlim(0, max(percentages) * 1.2 if percentages else 10)
        plt.xlabel('Percentage (%)')
    
    def _get_hue_from_hex(self, hex_color):
        """
        Extract hue value from hex color for sorting
        
        Args:
            hex_color (str): Hex color code
            
        Returns:
            float: Hue value (0-360)
        """
        # Remove # if present
        hex_color = hex_color.lstrip('#')
        
        # Convert to RGB
        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0
        
        # Convert to HSV and return hue
        h, _, _ = colorsys.rgb_to_hsv(r, g, b)
        return h * 360
    
    def visualize_color_trends(self, color_trends):
        """
        Create comprehensive visualizations of color trends
        (Legacy method maintained for backward compatibility)
        
        Args:
            color_trends (dict): Color trend analysis results
        """
        if not color_trends or not color_trends.get('color_range_distribution'):
            print("Insufficient data for visualization")
            return
        
        plt.figure(figsize=(20, 16))
        
        # 1. Create a Pantone-style color palette for dominant colors
        plt.subplot(2, 2, 1)
        self._visualize_dominant_colors(color_trends)
        
        # 2. Create a Pantone-style color range distribution
        plt.subplot(2, 2, 2)
        self._visualize_color_ranges(color_trends)
        
        # 3. Create a detailed color swatch grid of actual colors
        plt.subplot(2, 1, 2)
        plt.title('Fashion Color Trend Palette (Pantone-Inspired)', fontsize=16, fontweight='bold')
        
        dominant_colors = color_trends.get('dominant_colors', [])
        if dominant_colors:
            # Calculate grid dimensions
            n_colors = min(20, len(dominant_colors))
            cols = 5
            rows = (n_colors + cols - 1) // cols
            
            # Create color grid
            for i, color_info in enumerate(dominant_colors[:n_colors]):
                if i >= rows * cols:
                    break
                    
                row = i // cols
                col = i % cols
                
                # Draw rectangle for the color
                rect = plt.Rectangle((col, -row), 0.9, 0.9, color=color_info['hex'])
                plt.gca().add_patch(rect)
                
                # Add color name
                color_name = color_info['color_name']
                pantone_ref = color_info['pantone_ref']
                percentage = f"{color_info['percentage']:.1f}%"
                
                # Add text with different positions based on color brightness
                is_dark = sum(color_info['rgb']) < 380
                text_color = 'white' if is_dark else 'black'
                
                # Add pantone-like code on top
                plt.text(col + 0.45, -row + 0.75, pantone_ref, 
                         color=text_color, ha='center', va='center', fontsize=9)
                
                # Add color name in middle
                plt.text(col + 0.45, -row + 0.45, color_name, 
                         color=text_color, ha='center', va='center', fontsize=8)
                
                # Add percentage at bottom
                plt.text(col + 0.45, -row + 0.15, percentage, 
                         color=text_color, ha='center', va='center', fontsize=9)
                
            plt.xlim(0, cols)
            plt.ylim(-rows, 1)
        
        plt.axis('off')
        plt.tight_layout(pad=3.0)
        plt.savefig('fashion_color_trends.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def export_pantone_style_report(self, color_trends, output_path='fashion_trend_report.pdf', fashion_trends=None):
        """
        Export a professional Pantone-style comprehensive fashion trend report
        
        Args:
            color_trends (dict): Color trend analysis
            output_path (str): Path to save the report
            fashion_trends (dict, optional): Comprehensive fashion trend analysis including garment types
        """
        from matplotlib.backends.backend_pdf import PdfPages
        
        # Create a multi-page PDF
        with PdfPages(output_path) as pdf:
            # Cover page
            plt.figure(figsize=(8.5, 11))
            plt.title('FASHION TREND ANALYSIS REPORT', fontsize=24, fontweight='bold')
            plt.text(0.5, 0.5, f"Generated: {pd.Timestamp.now().strftime('%B %d, %Y')}", 
                    ha='center', fontsize=14)
            plt.text(0.5, 0.45, "Colors & Garment Types Analysis", 
                    ha='center', fontsize=16)
            plt.axis('off')
            pdf.savefig()
            plt.close()
            
            # Main visualizations - if we have comprehensive fashion trends
            if fashion_trends:
                # Create summary visualization
                plt.figure(figsize=(11, 14))
                
                # 1. Dominant colors
                plt.subplot(3, 1, 1)
                plt.title('Dominant Color Palette', fontsize=16, fontweight='bold')
                
                dominant_colors = color_trends.get('dominant_colors', [])
                if dominant_colors:
                    # Keep only top 10 colors to avoid clutter
                    top_colors = dominant_colors[:10]
                    
                    # Create a horizontal color swatch layout
                    for i, color_info in enumerate(top_colors):
                        # Draw color rectangle
                        rect = plt.Rectangle((i, 0), 0.8, 1, color=color_info['hex'])
                        plt.gca().add_patch(rect)
                        
                        # Add color name and percentage below the swatch
                        plt.text(i+0.4, -0.15, f"{color_info['percentage']:.1f}%", 
                                ha='center', va='top', fontsize=9, fontweight='bold')
                        
                        # Add Pantone reference
                        plt.text(i+0.4, 0.5, color_info['pantone_ref'].split(' ')[-1], 
                                ha='center', va='center', fontsize=8,
                                color='white' if self._is_dark_color(color_info['hex']) else 'black')
                    
                    plt.xlim(0, len(top_colors))
                    plt.ylim(-0.5, 1.5)
                
                plt.axis('off')
                
                # 2. Top garment types
                plt.subplot(3, 1, 2)
                plt.title('Top Garment Types', fontsize=16, fontweight='bold')
                
                garment_dist = fashion_trends.get('garment_trends', {}).get('distribution', {})
                if garment_dist:
                    # Take only top 10 garments for clarity
                    items = list(garment_dist.items())
                    items.sort(key=lambda x: x[1]['percentage'], reverse=True)
                    top_items = items[:10]
                    
                    garments = [item[0].capitalize() for item in top_items]
                    percentages = [item[1]['percentage'] for item in top_items]
                    
                    # Use a simple bar chart with consistent colors
                    bars = plt.barh(garments, percentages, color='skyblue')
                    
                    # Add percentage labels
                    for i, bar in enumerate(bars):
                        plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                                f"{percentages[i]:.1f}%", va='center')
                    
                    plt.xlabel('Percentage (%)')
                
                # 3. Style trends visualization
                plt.subplot(3, 1, 3)
                plt.title('Fashion Style Trends', fontsize=16, fontweight='bold')
                
                style_dist = fashion_trends.get('style_trends', {}).get('distribution', {})
                if style_dist:
                    # Take top 8 styles for clarity
                    items = list(style_dist.items())
                    items.sort(key=lambda x: x[1]['percentage'], reverse=True)
                    top_items = items[:8]
                    
                    styles = [item[0].capitalize() for item in top_items]
                    percentages = [item[1]['percentage'] for item in top_items]
                    
                    # Use a pie chart with a clean color scheme
                    plt.pie(percentages, labels=styles, autopct='%1.1f%%', 
                            colors=plt.cm.Pastel1(np.linspace(0, 1, len(styles))),
                            wedgeprops={'linewidth': 1, 'edgecolor': 'white'},
                            textprops={'fontsize': 9})
                    plt.axis('equal')
                
                plt.tight_layout(pad=3.0)
                pdf.savefig()
                plt.close()
                
                # Colors by garment type
                plt.figure(figsize=(11, 8.5))
                plt.suptitle('TOP COLORS BY GARMENT TYPE', fontsize=20, fontweight='bold')
                
                color_garment_trends = fashion_trends.get('color_garment_trends', {})
                if color_garment_trends:
                    # Take top 6 garment types
                    top_garments = list(color_garment_trends.keys())[:6]
                    
                    # Create a 2x3 grid
                    for i, garment in enumerate(top_garments):
                        plt.subplot(2, 3, i+1)
                        plt.title(garment.capitalize(), fontsize=12, fontweight='bold')
                        
                        colors = color_garment_trends[garment]
                        if not colors:
                            plt.text(0.5, 0.5, "No color data", ha='center', va='center')
                            plt.axis('off')
                            continue
                            
                        # Create color swatches for this garment
                        for j, color_info in enumerate(colors[:3]):  # Top 3 colors
                            rect = plt.Rectangle((0, j), 2, 0.8, color=color_info['hex'])
                            plt.gca().add_patch(rect)
                            
                            # Add text with pantone and percentage
                            plt.text(2.2, j+0.4, f"{color_info['frequency']:.1f}% - {color_info['pantone_ref']}", 
                                    va='center', fontsize=9)
                        
                        plt.xlim(0, 6)
                        plt.ylim(-0.2, 3)
                        plt.axis('off')
                
                plt.tight_layout(pad=3.0)
                pdf.savefig()
                plt.close()
            else:
                # Fallback to color-only visualizations
                plt.figure(figsize=(11, 8.5))
                self.visualize_color_trends(color_trends)
                pdf.savefig()
                plt.close()
            
            # Color detail pages
            if 'dominant_colors' in color_trends:
                plt.figure(figsize=(8.5, 11))
                plt.title("TOP COLOR PALETTE", fontsize=20, fontweight='bold')
                plt.text(0.5, 0.05, "Analysis of dominant colors across all fashion items", 
                        ha='center', fontsize=12)
                
                # Create a color palette grid
                top_colors = color_trends['dominant_colors'][:10]
                cols = 2
                rows = (len(top_colors) + cols - 1) // cols
                
                for i, color_info in enumerate(top_colors):
                    row = i // cols
                    col = i % cols
                    
                    # Calculate position for this color
                    x = 0.05 + col * 0.5
                    y = 0.9 - row * 0.15
                    
                    # Draw color swatch
                    rect = plt.Rectangle((x, y-0.1), 0.4, 0.1, color=color_info['hex'])
                    plt.gca().add_patch(rect)
                    
                    # Add text info
                    is_dark = sum(color_info['rgb']) < 380
                    swatch_text_color = 'white' if is_dark else 'black'
                    
                    # Add Pantone ref in the swatch
                    plt.text(x + 0.2, y - 0.05, color_info['pantone_ref'], 
                            color=swatch_text_color, ha='center', va='center', fontsize=9)
                    
                    # Add name and percentage above
                    plt.text(x, y + 0.01, f"{i+1}. {color_info['color_name']}", 
                            ha='left', va='bottom', fontsize=10, weight='bold')
                    plt.text(x, y - 0.12, f"{color_info['percentage']:.1f}%", 
                            ha='left', va='top', fontsize=9)
                
                plt.axis('off')
                pdf.savefig()
                plt.close()
        
        print(f"Report exported to {output_path}")

def main():
    # Check if input is provided
    if len(sys.argv) < 2:
        print("Please provide a directory path or JSON file path to your fashion images")
        print("Usage: python script.py /path/to/your/images_or_file.json [--adaptive]")
        sys.exit(1)
    
    # Get image source from command line
    image_source = sys.argv[1]
    
    # Check for adaptive mode flag
    adaptive_mode = "--adaptive" in sys.argv
    
    # Initialize analyzer
    analyzer = FashionTrendColorAnalyzer(image_source, adaptive_mode=adaptive_mode)
    
    # Analyze comprehensive fashion trends (colors AND garment types)
    fashion_trends = analyzer.analyze_fashion_trends()
    
    # Extract color trends for backward compatibility
    color_trends = fashion_trends.get('color_trends', {})
    
    # Print detailed fashion trend analysis
    if fashion_trends:
        print("\n--- Comprehensive Fashion Trend Analysis ---")
        
        # Dominant Colors (Pantone-style)
        print("\n1. DOMINANT COLORS (PANTONE-INSPIRED):")
        for i, color in enumerate(color_trends.get('dominant_colors', [])[:10]):
            print(f"{i+1}. {color['color_name']} ({color['pantone_ref']})")
            print(f"   HEX: {color['hex']} | Percentage: {color['percentage']:.1f}%")
        
        # Garment Type Distribution
        print("\n2. GARMENT TYPE DISTRIBUTION:")
        garment_trends = fashion_trends.get('garment_trends', {})
        garment_distribution = garment_trends.get('distribution', {})
        
        for i, (garment, data) in enumerate(sorted(garment_distribution.items(), 
                                              key=lambda x: x[1]['percentage'], 
                                              reverse=True)[:10]):
            print(f"{i+1}. {garment}: {data['percentage']:.1f}% ({data['count']} items)")
        
        # Style Trends
        print("\n3. STYLE TRENDS:")
        style_trends = fashion_trends.get('style_trends', {})
        style_distribution = style_trends.get('distribution', {})
        
        for i, (style, data) in enumerate(sorted(style_distribution.items(), 
                                           key=lambda x: x[1]['percentage'], 
                                           reverse=True)[:10]):
            print(f"{i+1}. {style}: {data['percentage']:.1f}% ({data['count']} items)")
        
        # Color-Garment Correlations
        print("\n4. TOP COLORS BY GARMENT TYPE:")
        color_garment_trends = fashion_trends.get('color_garment_trends', {})
        
        for i, (garment, colors) in enumerate(list(color_garment_trends.items())[:5]):
            print(f"\n{garment.capitalize()}:")
            
            for j, color in enumerate(colors[:3]):
                print(f"  {j+1}. {color['color_name']} ({color['pantone_ref']}): {color['frequency']:.1f}%")
        
        # Color Metrics
        metrics = color_trends.get('color_metrics', {})
        print("\n5. OVERALL COLOR METRICS:")
        print(f"  Total Unique Colors: {metrics.get('total_unique_colors', 0)}")
        print(f"  Average Color Saturation: {metrics.get('average_saturation', 0):.1f}%")
        print(f"  Average Color Brightness: {metrics.get('average_brightness', 0):.1f}%")
        print(f"  Color Diversity Index: {metrics.get('color_diversity_index', 0):.2f}")
    
    # Visualize comprehensive fashion trends
    analyzer.visualize_fashion_trends(fashion_trends)
    
    # Export a professional Pantone-style fashion trend report
    export_report = input("\nExport comprehensive fashion trend report? (y/n): ").lower().strip()
    if export_report == 'y':
        report_filename = input("Enter report filename (default: fashion_trend_report.pdf): ").strip()
        if not report_filename:
            report_filename = 'fashion_trend_report.pdf'
        
        print(f"Exporting report to {report_filename}...")
        analyzer.export_pantone_style_report(color_trends, report_filename, fashion_trends)

if __name__ == '__main__':
    main()

# Installation requirements:
# pip install torch torchvision
# pip install git+https://github.com/openai/CLIP.git
# pip install scikit-learn matplotlib Pillow requests pandas
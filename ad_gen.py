import streamlit as st
import pandas as pd
import json
import io
import base64
import os
from PIL import Image, ImageDraw, ImageFont
import google.generativeai as genai
from typing import Dict, List, Any
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation

# Set page configuration
st.set_page_config(page_title="Creative Ad Copy Generator", layout="wide")

# Styling to ensure visibility
st.markdown("""
<style>
.main {
    padding: 1rem;
    background-color: #ffffff;
}
h1 {
    color: #1E3A8A;
    text-align: center;
    margin-bottom: 20px;
}
.ad-container {
    border: 1px solid #ddd;
    border-radius: 10px;
    padding: 20px;
    margin-top: 20px;
    background-color: #f8f9fa;
}
</style>
""", unsafe_allow_html=True)

# ======= CONFIGURATION - REPLACE THESE VALUES =======
# Add your Gemini API key here
GEMINI_API_KEY = "AIzaSyBWJGc8cNmhBUXhNUoaTjplh-ufuyrHbm8"
# Add your Stability API key here
STABILITY_API_KEY = "sk-QZC61Heb3oIYhPmaQoCwFIXkl9MItBaGvAzso3fKRt3An1ui"
# Add your dataset path here
# DATASET_PATH = "/home/dell/Documents/TechX/train-00000-of-00002-6e587552aa3c8ac8.parquet"
DATASET_PATH = os.path.join(os.path.dirname(__file__), "train-00000-of-00002-6e587552aa3c8ac8.parquet")
# ===================================================

class AdDatastore:
    """Handles ad examples retrieval"""
    
    def __init__(self, dataset_path):
        self.data = None
        self.load_data(dataset_path)
    
    def load_data(self, dataset_path):
        """Load data from parquet file"""
        try:
            self.data = pd.read_parquet(dataset_path)
            st.success(f"Successfully loaded dataset with {len(self.data)} examples!")
        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")
            # Create a simple dummy dataset as fallback
            self.data = pd.DataFrame({
                'image': [''] * 3,
                'text': [
                    'Experience the future with our innovative smartwatch.',
                    'Stay hydrated with our eco-friendly water bottle.',
                    'Unleash your creativity with our professional camera.'
                ],
                'dimension': ['1200x628', '1080x1080', '800x800']
            })
            st.info("Using fallback dataset for examples.")
    
    def get_similar_ads(self, product_type: str, audience: str, tone: str, num_samples: int = 3) -> List[Dict]:
        """Retrieve relevant examples from dataset"""
        if self.data is None or len(self.data) == 0:
            return []
        
        # In a real implementation, we would use embeddings for semantic similarity
        # For simplicity, just return random samples
        if len(self.data) > num_samples:
            return self.data.sample(num_samples).to_dict('records')
        else:
            return self.data.to_dict('records')


class AdImageGenerator:
    """Handles generation of ad images using Stability AI"""
    
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.stability_api = None
        
        # Define tone-specific font paths
        # These are the default fonts we'll look for in the system
        self.font_paths = {
            "Professional": {
                "title": "Arial-Bold.ttf",
                "body": "Arial.ttf",
                "cta": "Arial-Bold.ttf"
            },
            "Friendly": {
                "title": "Comic-Sans-MS-Bold.ttf",
                "body": "Comic-Sans-MS.ttf",
                "cta": "Comic-Sans-MS-Bold.ttf"
            },
            "Humorous": {
                "title": "Comic-Sans-MS-Bold.ttf",
                "body": "Comic-Sans-MS.ttf",
                "cta": "Comic-Sans-MS-Bold.ttf"
            },
            "Urgent": {
                "title": "Impact.ttf",
                "body": "Arial.ttf",
                "cta": "Impact.ttf"
            },
            "Inspirational": {
                "title": "Georgia-Bold.ttf",
                "body": "Georgia.ttf",
                "cta": "Georgia-Bold.ttf"
            },
            "Luxurious": {
                "title": "Times-New-Roman-Bold.ttf",
                "body": "Times-New-Roman.ttf",
                "cta": "Times-New-Roman-Bold.ttf"
            }
        }
        
        # Define alternative font names to search for
        self.alternative_fonts = {
            "Arial-Bold.ttf": ["arialbd.ttf", "Arial Bold.ttf", "Arial-BoldMT.ttf"],
            "Arial.ttf": ["arial.ttf", "Arial.ttf", "ArialMT.ttf"],
            "Comic-Sans-MS-Bold.ttf": ["comicbd.ttf", "Comic Sans MS Bold.ttf", "ComicSansMS-Bold.ttf"],
            "Comic-Sans-MS.ttf": ["comic.ttf", "Comic Sans MS.ttf", "ComicSansMS.ttf"],
            "Impact.ttf": ["impact.ttf", "Impact.ttf"],
            "Georgia-Bold.ttf": ["georgiab.ttf", "Georgia Bold.ttf", "Georgia-Bold.ttf"],
            "Georgia.ttf": ["georgia.ttf", "Georgia.ttf"],
            "Times-New-Roman-Bold.ttf": ["timesbd.ttf", "Times New Roman Bold.ttf", "TimesNewRomanPS-BoldMT.ttf"],
            "Times-New-Roman.ttf": ["times.ttf", "Times New Roman.ttf", "TimesNewRomanPSMT.ttf"]
        }
        
        # Define tone-specific color schemes
        self.color_schemes = {
            "Professional": {
                "background": (240, 248, 255),  # AliceBlue
                "header": (41, 128, 185),       # Blue
                "text": (44, 62, 80),           # Dark blue/grey
                "button": (52, 152, 219),       # Light blue
                "button_text": (255, 255, 255)  # White
            },
            "Friendly": {
                "background": (255, 250, 205),  # LemonChiffon
                "header": (46, 204, 113),       # Green
                "text": (44, 62, 80),           # Dark blue/grey
                "button": (39, 174, 96),        # Darker green
                "button_text": (255, 255, 255)  # White
            },
            "Humorous": {
                "background": (255, 218, 185),  # PeachPuff
                "header": (230, 126, 34),       # Orange
                "text": (44, 62, 80),           # Dark blue/grey
                "button": (211, 84, 0),         # Darker orange
                "button_text": (255, 255, 255)  # White
            },
            "Urgent": {
                "background": (255, 228, 225),  # MistyRose
                "header": (231, 76, 60),        # Red
                "text": (44, 62, 80),           # Dark blue/grey
                "button": (192, 57, 43),        # Darker red
                "button_text": (255, 255, 255)  # White
            },
            "Inspirational": {
                "background": (230, 230, 250),  # Lavender
                "header": (142, 68, 173),       # Purple
                "text": (44, 62, 80),           # Dark blue/grey
                "button": (155, 89, 182),       # Medium purple
                "button_text": (255, 255, 255)  # White
            },
            "Luxurious": {
                "background": (245, 245, 245),  # WhiteSmoke
                "header": (44, 62, 80),         # Dark blue/grey
                "text": (44, 62, 80),           # Dark blue/grey
                "button": (52, 73, 94),         # Darker blue/grey
                "button_text": (212, 175, 55)   # Gold
            }
        }
        
        # Define tone-specific text styles
        self.text_styles = {
            "Professional": {
                "headline_transform": lambda text: text,  # No transformation
                "headline_spacing": 1,  # Normal spacing
                "copy_transform": lambda text: text,  # No transformation
                "cta_transform": lambda text: text,  # No transformation
                "cta_decoration": ""  # No decoration
            },
            "Friendly": {
                "headline_transform": lambda text: text,  # No transformation
                "headline_spacing": 1,  # Normal spacing
                "copy_transform": lambda text: text,  # No transformation
                "cta_transform": lambda text: text + " :)",  # Add smiley
                "cta_decoration": ""  # No decoration
            },
            "Humorous": {
                "headline_transform": lambda text: text + "!",  # Add excitement
                "headline_spacing": 1,  # Normal spacing
                "copy_transform": lambda text: text,  # No transformation
                "cta_transform": lambda text: text,  # No transformation
                "cta_decoration": "★"  # Star decoration
            },
            "Urgent": {
                "headline_transform": lambda text: text.upper(),  # All caps
                "headline_spacing": 1.2,  # Slightly wider spacing
                "copy_transform": lambda text: text,  # No transformation
                "cta_transform": lambda text: text.upper() + "!",  # All caps with exclamation
                "cta_decoration": ""  # No decoration
            },
            "Inspirational": {
                "headline_transform": lambda text: text,  # No transformation
                "headline_spacing": 1.1,  # Slightly wider spacing
                "copy_transform": lambda text: text,  # No transformation
                "cta_transform": lambda text: text,  # No transformation
                "cta_decoration": "✨"  # Sparkle decoration
            },
            "Luxurious": {
                "headline_transform": lambda text: "  " + text + "  ",  # Add spacing
                "headline_spacing": 1.2,  # Wider letter spacing for elegance
                "copy_transform": lambda text: text,  # No transformation
                "cta_transform": lambda text: "✦ " + text + " ✦",  # Add decorative elements
                "cta_decoration": ""  # No decoration
            }
        }
        
        # Define tone-specific decorative elements
        self.decorative_elements = {
            "Professional": [],  # No decorative elements
            "Friendly": ["rounded_corners", "light_gradient"],
            "Humorous": ["playful_icons", "star_accent"],
            "Urgent": ["bold_border", "diagonal_stripe"],
            "Inspirational": ["subtle_lines", "light_rays"],
            "Luxurious": ["gold_border", "elegant_divider", "subtle_pattern"]
        }
        
        if api_key:
            self.setup_stability_api()
    
    def setup_stability_api(self):
        """Set up the Stability AI API with the provided key"""
        try:
            self.stability_api = client.StabilityInference(
                key=self.api_key,
                verbose=True
            )
            st.success("Stability AI API configured successfully!")
            return True
        except Exception as e:
            st.error(f"Error configuring Stability AI API: {str(e)}")
            return False
    
    def find_font(self, font_name):
        """Find a suitable font file in the system"""
        # Check for the main font
        alternatives = self.alternative_fonts.get(font_name, [font_name])
        
        # Common font directories
        font_locations = [
            "/usr/share/fonts/truetype/",  # Linux
            "/System/Library/Fonts/",      # macOS
            "C:/Windows/Fonts/"            # Windows
        ]
        
        # Try to find the font
        for location in font_locations:
            for alt_name in alternatives:
                full_path = os.path.join(location, alt_name)
                if os.path.exists(full_path):
                    return full_path
        
        # Fallback fonts if nothing is found
        fallbacks = {
            "title": [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux
                "/System/Library/Fonts/Helvetica.ttc",                   # macOS
                "C:/Windows/Fonts/segoeui.ttf"                          # Windows
            ],
            "body": [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",      # Linux
                "/System/Library/Fonts/Helvetica.ttc",                   # macOS
                "C:/Windows/Fonts/segoeui.ttf"                          # Windows
            ],
            "cta": [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux
                "/System/Library/Fonts/Helvetica.ttc",                   # macOS
                "C:/Windows/Fonts/segoeui.ttf"                          # Windows
            ]
        }
        
        # Try the fallbacks
        font_type = None
        for type_key, fonts in self.font_paths.items():
            for font_key, value in fonts.items():
                if value == font_name:
                    font_type = font_key
                    break
            if font_type:
                break
        
        if font_type and font_type in fallbacks:
            for fallback in fallbacks[font_type]:
                if os.path.exists(fallback):
                    return fallback
        
        # Ultimate fallback - return None and let the calling code handle it
        return None

    
    def generate_product_image(self, product_description, dimensions, tone):
        print(f"Dimensions received: {dimensions}")
        print(f"Product Description: {product_description}")
        try:
            if not self.stability_api:
                st.warning("Image generation API is not available. Using fallback image.")
                return self.create_placeholder_image(product_description, dimensions)

            # Parse dimensions
            width, height = map(int, dimensions.split('x'))

            # Tone-based modifiers for better AI understanding
            tone_modifiers = {
                "Professional": "high quality, clean professional lighting, minimalist background, studio shot",
                "Friendly": "warm lighting, welcoming environment, lifestyle setting, vibrant colors",
                "Humorous": "playful, exaggerated features, quirky and fun background",
                "Urgent": "bold colors, dramatic lighting, high contrast, intense focus",
                "Inspirational": "dreamy lighting, aspirational setting, motivational energy",
                "Luxurious": "premium quality, elegant lighting, high-end studio photography, soft shadows"
            }

            style_guidance = {
                "Professional": "commercial photography, clean lines, high-resolution",
                "Friendly": "lifestyle photography, candid shot, authentic setting",
                "Humorous": "quirky advertising, unexpected elements, creative composition",
                "Urgent": "bold advertising, dramatic impact, high-contrast visuals",
                "Inspirational": "aspirational photography, storytelling, cinematic lighting",
                "Luxurious": "luxury product photography, editorial, high-end branding"
            }

            # Construct the AI prompt dynamically
            modifier = tone_modifiers.get(tone, "high quality")
            style = style_guidance.get(tone, "")
            prompt = (f"High-quality advertisement image of {product_description}, "
                    f"{modifier}, {style}, detailed product shot, studio lighting, 8k resolution")

            # Generate the image using Stability AI
            answers = self.stability_api.generate(
                prompt=prompt,
                width=width,
                height=height,
                cfg_scale=8.0,
                sampler=generation.SAMPLER_K_DPMPP_2M,
                steps=40,
                seed=0
            )

            # Process and return generated image
            for resp in answers:
                for artifact in resp.artifacts:
                    if artifact.type == generation.ARTIFACT_IMAGE:
                        img_bytes = artifact.binary
                        img = Image.open(io.BytesIO(img_bytes))
                        return img

            st.error("No image generated. Falling back to placeholder.")
            return self.create_placeholder_image(product_description, dimensions)

        except Exception as e:
            st.warning(f"Stability AI image generation failed: {str(e)}. Falling back to image retrieval.")
        
        # RAG fallback - retrieve similar product image from a database or search API
            return self.retrieve_similar_product_image(product_description)
        
    def retrieve_similar_product_image(self, product_description):
        """Retrieve relevant product image using RAG approach"""
        try:
        # Option 1: Use a vector database of product images
        # embedding = get_embedding(product_description)
        # similar_image = vector_db.query(embedding)
        
        # Option 2: Use a web search API
        # Using Google Custom Search or similar
            search_results = search_api.search_images(product_description, num_results=1)
            image_url = search_results[0]['url']
        
        # Download and process image
            response = requests.get(image_url)
            img = Image.open(io.BytesIO(response.content))
            return img
        
        except Exception as e:
            st.error(f"Image retrieval failed: {str(e)}")
            return self.create_placeholder_image(product_description, dimensions)

    def create_placeholder_image(self, product_description, dimensions):
        width, height = map(int, dimensions.split('x'))
        placeholder = Image.new('RGB', (width, height), color=self.color_schemes["Professional"]["background"])
        draw = ImageDraw.Draw(placeholder)

        # Placeholder text
        draw.text((width // 4, height // 2), product_description, fill="black")

        return placeholder
    
        
    
    def apply_tone_specific_decorations(self, img, draw, width, height, tone, colors):
        """Apply decorative elements based on tone"""
        if tone == "Luxurious":
            # Gold border for luxurious feel
            draw.rectangle([(5, 5), (width-5, height-5)], outline=(212, 175, 55), width=3)
            
            # Elegant divider in the middle
            draw.line([(20, height/2), (width-20, height/2)], fill=(212, 175, 55, 150), width=1)
            
            # Subtle corners
            corner_size = 15
            # Top-left
            draw.line([(0, corner_size), (corner_size, 0)], fill=(212, 175, 55), width=2)
            # Top-right
            draw.line([(width-corner_size, 0), (width, corner_size)], fill=(212, 175, 55), width=2)
            # Bottom-left
            draw.line([(0, height-corner_size), (corner_size, height)], fill=(212, 175, 55), width=2)
            # Bottom-right
            draw.line([(width-corner_size, height), (width, height-corner_size)], fill=(212, 175, 55), width=2)
            
        elif tone == "Urgent":
            # Attention-grabbing diagonal stripe
            for i in range(-int(height/5), int(height/5)):
                draw.line([(0, height/2 + i), (width, height/2 + i)], fill=(231, 76, 60, 100), width=1)
            
            # Bold border
            draw.rectangle([(0, 0), (width-1, height-1)], outline=(192, 57, 43), width=5)
            
        elif tone == "Inspirational":
            # Subtle light rays
            ray_center = (width/2, height/4)
            num_rays = 12
            ray_length = max(width, height) * 0.7
            
            for i in range(num_rays):
                angle = (i / num_rays) * 2 * 3.14159  # Radians
                import math
                end_x = ray_center[0] + ray_length * math.cos(angle)
                end_y = ray_center[1] + ray_length * math.sin(angle)
                
                # Draw with transparency gradient
                for t in range(1, 11):
                    t_factor = t / 10
                    x = ray_center[0] + (end_x - ray_center[0]) * t_factor
                    y = ray_center[1] + (end_y - ray_center[1]) * t_factor
                    
                    # Smaller t = closer to center = more opaque
                    alpha = int(155 * (1 - t_factor))
                    draw.point((x, y), fill=(142, 68, 173, alpha))
            
            # Decorative lines
            draw.line([(width/4, int(height/10)), (width/4, height-int(height/10))], fill=(142, 68, 173, 50), width=1)
            draw.line([(3*width/4, int(height/10)), (3*width/4, height-int(height/10))], fill=(142, 68, 173, 50), width=1)
            
        elif tone == "Friendly":
            # Rounded corners effect (simulated)
            corner_radius = 20
            for i in range(corner_radius):
                # Lighter with increasing radius for a gradual effect
                alpha = 150 - int(150 * i / corner_radius)
                
                # Top-left
                draw.arc([(0, 0), (corner_radius*2-i*2, corner_radius*2-i*2)], 180, 270, fill=(46, 204, 113, alpha), width=1)
                # Top-right
                draw.arc([(width-corner_radius*2+i*2, 0), (width, corner_radius*2-i*2)], 270, 0, fill=(46, 204, 113, alpha), width=1)
                # Bottom-left
                draw.arc([(0, height-corner_radius*2+i*2), (corner_radius*2-i*2, height)], 90, 180, fill=(46, 204, 113, alpha), width=1)
                # Bottom-right
                draw.arc([(width-corner_radius*2+i*2, height-corner_radius*2+i*2), (width, height)], 0, 90, fill=(46, 204, 113, alpha), width=1)
            
            # Light gradient overlay from top to bottom
            for y in range(height):
                alpha = int(30 * (1 - y / height))  # Fade from top to bottom
                draw.line([(0, y), (width, y)], fill=(46, 204, 113, alpha), width=1)
                
        elif tone == "Humorous":
            # Playful icons
            draw.text((width - 50, 50), "★", fill=(230, 126, 34), font=ImageFont.load_default())
            draw.text((50, 50), "☺", fill=(230, 126, 34), font=ImageFont.load_default())
            
            # Star accent in corner
            points = []
            center_x, center_y = width - 40, 40
            outer_radius = 15
            inner_radius = 8
            num_points = 5
            
            import math
            for i in range(num_points * 2):
                radius = outer_radius if i % 2 == 0 else inner_radius
                x = center_x + radius * math.cos(math.pi/2 + i * math.pi / num_points)
                y = center_y + radius * math.sin(math.pi/2 + i * math.pi / num_points)
                points.append((x, y))
            
            draw.polygon(points, fill=(230, 126, 34))
    
    def create_ad_image(self, ad_content, dimensions, product_name, product_description, tone):
        """Create an ad image with the generated content and product image with tone-specific styling"""
        try:
            # Parse dimensions
            width, height = map(int, dimensions.split('x'))
            
            # Try to generate a product image with Stability AI
            product_img = None
            if self.stability_api:
                product_img = self.generate_product_image(product_description, dimensions, tone)
            
            # Get tone-specific color scheme
            colors = self.color_schemes.get(tone, self.color_schemes["Professional"])
            
            # If no product image was generated, create a background with tone-specific color
            if product_img is None:
                img = Image.new('RGB', (width, height), color=colors["background"])
            else:
                # Use the generated product image and apply a slight overlay for text readability
                img = product_img.resize((width, height))
                
                # Create a semi-transparent overlay for text readability
                overlay = Image.new('RGBA', (width, height), (*colors["background"], 100))
                
                # Make sure both images are in RGBA mode before compositing
                if img.mode != 'RGBA':
                    img = img.convert('RGBA')
                
                img = Image.alpha_composite(img, overlay)
                img = img.convert('RGB')
            
            draw = ImageDraw.Draw(img)
            
            # Apply tone-specific decorative elements before adding content
            self.apply_tone_specific_decorations(img, draw, width, height, tone, colors)
            
            # Get tone-specific text styles
            text_style = self.text_styles.get(tone, self.text_styles["Professional"])
            
            # Try to load tone-specific fonts
            try:
                # Get font paths for the selected tone
                tone_fonts = self.font_paths.get(tone, self.font_paths["Professional"])
                
                # Find appropriate fonts in the system
                title_font_path = self.find_font(tone_fonts["title"])
                body_font_path = self.find_font(tone_fonts["body"])
                cta_font_path = self.find_font(tone_fonts["cta"])
                
                # Load fonts with appropriate sizes based on ad dimensions
                title_size = int(height/10)  # Larger for better visibility
                body_size = int(height/20)   # Readable but not overwhelming
                cta_size = int(height/14)    # Stand out but not as big as title
                
                # Load fonts with fallbacks
                title_font = ImageFont.truetype(title_font_path, size=title_size) if title_font_path else ImageFont.load_default()
                body_font = ImageFont.truetype(body_font_path, size=body_size) if body_font_path else ImageFont.load_default()
                cta_font = ImageFont.truetype(cta_font_path, size=cta_size) if cta_font_path else ImageFont.load_default()
                
            except Exception as e:
                st.warning(f"Could not load custom fonts: {str(e)}. Using default fonts.")
                # Fallback to default font
                title_font = ImageFont.load_default()
                body_font = ImageFont.load_default()
                cta_font = ImageFont.load_default()
            
            # Header and footer with tone-specific colors
            draw.rectangle([(0, 0), (width, int(height/10))], fill=colors["header"])
            draw.rectangle([(0, height-int(height/10)), (width, height)], fill=colors["header"])
            
            # Add brand/product name with tone-specific styling
            brand_text = product_name.upper() if tone == "Professional" or tone == "Urgent" else product_name
            draw.text((20, int(height/30)), brand_text, fill=(255, 255, 255), font=body_font)
            
            # Apply headline transformation based on tone
            headline = text_style["headline_transform"](ad_content["headline"])
            
            # Handle headline text positioning and styling
            headline_width = draw.textlength(headline, font=title_font)
            
            # Center headline if it fits, otherwise left-align
            x_pos = (width - headline_width) / 2 if headline_width < width - 40 else 20
            
            # Draw headline with tone-specific styling
            draw.text((x_pos, int(height/6)), headline, fill=colors["text"], font=title_font)
            
            # Draw main copy (with text wrapping)
            main_copy = text_style["copy_transform"](ad_content["main_copy"])
            y_position = int(height/3)
            words = main_copy.split()
            line = ""
            for word in words:
                test_line = f"{line} {word}".strip()
                line_width = draw.textlength(test_line, font=body_font)
                
                if line_width < width - 40:
                    line = test_line
                else:
                    draw.text((20, y_position), line, fill=colors["text"], font=body_font)
                    y_position += int(height/16)
                    line = word
                    
                    # Check if we've run out of space
                    if y_position > height - int(height/3):
                        line += "..."
                        break
            
            # Draw remaining text
            if line:
                draw.text((20, y_position), line, fill=colors["text"], font=body_font)
            
            # Apply CTA transformation based on tone
            cta = text_style["cta_transform"](ad_content["cta"])
            
            # Draw CTA in a button-like shape with tone-specific styling
            cta_width = draw.textlength(cta, font=cta_font)
            cta_x = (width - cta_width) / 2 - 20
            cta_y = height - int(height/4)
            
            # Style button based on tone
            button_padding = 20
            button_height = int(height/12)
            
            if tone == "Luxurious":
                # Gold border for luxury
                draw.rectangle(
                    [(cta_x - button_padding, cta_y - button_padding/2),
                     (cta_x + cta_width + button_padding, cta_y + button_height)],
                    fill=colors["button"],
                    outline=(212, 175, 55),
                    width=3
                )
            elif tone == "Urgent":
                # Larger button with border for urgent CTA
                draw.rectangle(
                    [(cta_x - button_padding*1.2, cta_y - button_padding/1.5),
                     (cta_x + cta_width + button_padding*1.2, cta_y + button_height*1.2)],
                    fill=colors["button"],
                    outline=(0, 0, 0),
                    width=2
                )
            elif tone == "Friendly":
                # Rounded rectangle simulation for friendly feel
                draw.rectangle(
                    [(cta_x - button_padding, cta_y - button_padding/2),
                     (cta_x + cta_width + button_padding, cta_y + button_height)],
                    fill=colors["button"],
                    outline=None
                )
                
                # Add rounded corner effect
                corner_radius = 8
                corner_color = colors["button"]
                
                # Draw rounded corners
                draw.pieslice([cta_x - button_padding, cta_y - button_padding/2, 
                               cta_x - button_padding + corner_radius*2, cta_y - button_padding/2 + corner_radius*2],
                               180, 270, fill=corner_color)
                draw.pieslice([cta_x + cta_width + button_padding - corner_radius*2, cta_y - button_padding/2,
                               cta_x + cta_width + button_padding, cta_y - button_padding/2 + corner_radius*2],
                               270, 0, fill=corner_color)
                draw.pieslice([cta_x - button_padding, cta_y + button_height - corner_radius*2,
                               cta_x - button_padding + corner_radius*2, cta_y + button_height],
                               90, 180, fill=corner_color)
                draw.pieslice([cta_x + cta_width + button_padding - corner_radius*2, cta_y + button_height - corner_radius*2,
                               cta_x + cta_width + button_padding, cta_y + button_height],
                               0, 90, fill=corner_color)
            else:
                # Standard button for other tones
                draw.rectangle(
                    [(cta_x - button_padding, cta_y - button_padding/2),
                     (cta_x + cta_width + button_padding, cta_y + button_height)],
                    fill=colors["button"]
                )
            
            # Draw the CTA text
            cta_text_y = cta_y + (button_height - int(height/14))/2
            draw.text((cta_x, cta_text_y), cta, fill=colors["button_text"], font=cta_font)
            
            # Add decorative elements before and after CTA if tone specifies
            if text_style["cta_decoration"]:
                decoration = text_style["cta_decoration"]
                if tone == "Humorous" or tone == "Inspirational":
                    # Add decorations on both sides
                    draw.text((cta_x - button_padding/2, cta_text_y), decoration, 
                              fill=colors["button_text"], font=cta_font)
                    draw.text((cta_x + cta_width + button_padding/2, cta_text_y), decoration, 
                              fill=colors["button_text"], font=cta_font)
            
            # Convert image for web display
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            return buf.getvalue()
            
        except Exception as e:
            st.error(f"Error creating ad image: {str(e)}")
            return None


class AdCopyGenerator:
    """Handles generation of ad copy using Gemini AI"""
    
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.genai = None
        
        if api_key:
            self.setup_genai()
    
    def setup_genai(self):
        """Configure the Gemini API with the provided key"""
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-1.5-pro')
            st.success("Gemini AI API configured successfully!")
            return True
        except Exception as e:
            st.error(f"Error configuring Gemini AI API: {str(e)}")
            return False
    
    def generate_ad_copy(self, product_name, product_description, audience, tone, dimensions, examples=None):
        """Generate ad copy based on product info and tone"""
        if not self.model:
            # Return dummy data if API is not configured
            return {
                "headline": f"Introducing {product_name}",
                "main_copy": f"Perfect for {audience}. {product_description[:100]}...",
                "cta": "Get Yours Today"
            }
        
        try:
            # Build system prompt with examples if provided
            system_prompt = f"""You are an expert advertising copywriter. Create compelling ad copy for the product.
            
Format your response as a JSON object with the following structure:
{{
  "headline": "Attention-grabbing headline (max 10 words)",
  "main_copy": "Main body text (max 100 characters)",
  "cta": "Call to action text (max 5 words)"
}}

The copy should be optimized for a {dimensions} ad and tailored for a {audience} audience.
The copy should have a {tone} tone.
Do not include any explanation, just return the JSON object.
"""

            # Create user prompt with product information
            user_prompt = f"""Product Name: {product_name}
Product Description: {product_description}
"""
            
            # Add examples if provided
            if examples and len(examples) > 0:
                examples_text = "Here are some example ads for reference:\n\n"
                for idx, example in enumerate(examples):
                    if 'text' in example:
                        examples_text += f"Example {idx+1}: {example['text']}\n"
                user_prompt += "\n" + examples_text
            
            # Generate the ad copy
            response = self.model.generate_content([system_prompt, user_prompt])
            
            # Parse the JSON response
            response_text = response.text
            
            # Clean up the response in case there are markdown code blocks
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            result = json.loads(response_text)
            
            # Ensure all required fields are present
            required_fields = ["headline", "main_copy", "cta"]
            for field in required_fields:
                if field not in result:
                    result[field] = ""
            
            return result
            
        except Exception as e:
            st.error(f"Error generating ad copy: {str(e)}")
            # Return basic fallback copy
            return {
                "headline": f"Introducing {product_name}",
                "main_copy": f"Perfect for {audience}. {product_description[:100]}...",
                "cta": "Get Yours Today"
            }


# Initialize session state for storing ad content
if 'ad_content' not in st.session_state:
    st.session_state.ad_content = None

if 'generated_image' not in st.session_state:
    st.session_state.generated_image = None

# Main title
st.title("Creative Ad Copy Generator")

# Initialize the services
try:
    datastore = AdDatastore(DATASET_PATH)
    image_generator = AdImageGenerator(STABILITY_API_KEY)
    copy_generator = AdCopyGenerator(GEMINI_API_KEY)
except Exception as e:
    st.error(f"Error initializing services: {str(e)}")
    datastore = AdDatastore(None)
    image_generator = AdImageGenerator()
    copy_generator = AdCopyGenerator()

# Create two columns for form inputs
col1, col2 = st.columns(2)

with col1:
    product_name = st.text_input("Product Name", "Smart Watch")
    product_description = st.text_area("Product Description", "A smartwatch with advanced health tracking, GPS, and 7-day battery life.")
    dimensions = st.selectbox("Ad Dimensions", ["1200x628", "1080x1080", "800x600", "320x480"])

with col2:
    audience = st.text_input("Target Audience", "Health-conscious professionals aged 25-45")
    tone = st.selectbox("Ad Tone", ["Professional", "Friendly", "Humorous", "Urgent", "Inspirational", "Luxurious"])
    num_examples = st.slider("Number of Examples to Use", 0, 5, 2)

# Generate button
if st.button("Generate Ad"):
    with st.spinner("Generating ad content..."):
        # Retrieve similar examples from the datastore
        examples = datastore.get_similar_ads(product_name, audience, tone, num_examples)
        
        # Generate ad copy
        ad_content = copy_generator.generate_ad_copy(
            product_name, 
            product_description, 
            audience, 
            tone, 
            dimensions, 
            examples
        )
        
        st.session_state.ad_content = ad_content
        
                # Generate ad image using Stability AI
        image_bytes = image_generator.create_ad_image(
            ad_content, 
            dimensions, 
            product_name, 
            product_description, 
            tone
        )
        
        # If Stability AI fails, fallback to RAG
        if not image_bytes:
            st.warning("Stability AI failed to generate an image. Trying to retrieve from RAG...")
            image_bytes = rag_retriever.fetch_product_image(product_name)
        
        # Final fallback: Use a placeholder image if both Stability and RAG fail
        if not image_bytes:
            st.error("Image generation failed completely. Using placeholder image.")
            image_bytes = load_placeholder_image()
        
        if image_bytes:
            st.session_state.generated_image = base64.b64encode(image_bytes).decode('utf-8')

# Display generated content if available
if st.session_state.ad_content:
    st.markdown("## Generated Ad")
    
    ad_container = st.container(border=True)
    
    with ad_container:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"### {st.session_state.ad_content['headline']}")
            st.markdown(st.session_state.ad_content['main_copy'])
            st.markdown(f"**{st.session_state.ad_content['cta']}**")
        
        with col2:
            if st.session_state.generated_image:
                st.image(f"data:image/png;base64,{st.session_state.generated_image}", caption="Generated Ad Visual")
            else:
                st.warning("No image available. Ensure Stability AI or RAG retrieval is configured correctly.")
    
    # Add download buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.session_state.generated_image:
            st.download_button(
                label="Download Image",
                data=base64.b64decode(st.session_state.generated_image),
                file_name=f"ad_{product_name.replace(' ', '_')}.png",
                mime="image/png"
            )
    
    with col2:
        # Create JSON data for download
        json_data = json.dumps({
            "product_name": product_name,
            "product_description": product_description,
            "audience": audience,
            "tone": tone,
            "dimensions": dimensions,
            "ad_content": st.session_state.ad_content
        }, indent=2)
        
        st.download_button(
            label="Download Ad Data (JSON)",
            data=json_data,
            file_name=f"ad_data_{product_name.replace(' ', '_')}.json",
            mime="application/json"
        )

# Add footer with instructions
st.markdown("---")
st.markdown("""
### How to Use:
1. Enter your product details and select your preferred tone and dimensions
2. Click "Generate Ad" to create custom ad copy and visuals
3. Download the generated assets for your marketing campaigns
""")

# Add information about the tones
with st.expander("About Ad Tones"):
    st.markdown("""
    - **Professional**: Clean, corporate style with minimalist approach for business audiences
    - **Friendly**: Warm, approachable style with conversational language
    - **Humorous**: Playful and witty approach to capture attention
    - **Urgent**: Creates a sense of time pressure or limited availability
    - **Inspirational**: Emotionally uplifting content focused on aspirations
    - **Luxurious**: Elegant, premium feel for high-end products
    """)

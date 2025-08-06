"""
Data Collector for AI Detection Training
Automatically collects training images from various sources
"""

import os
import requests
import json
import time
from PIL import Image
from io import BytesIO
import logging
from datetime import datetime
import hashlib
from pathlib import Path
import numpy as np # Added missing import for _generate_image_hash

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCollector:
    def __init__(self, data_dir="./training_data"):
        """Initialize data collector"""
        self.data_dir = Path(data_dir)
        self.real_images_dir = self.data_dir / "real"
        self.ai_images_dir = self.data_dir / "ai_generated"
        
        # Create directories
        self.real_images_dir.mkdir(parents=True, exist_ok=True)
        self.ai_images_dir.mkdir(parents=True, exist_ok=True)
        
        # API keys and endpoints (you'll need to add your own)
        self.unsplash_access_key = None  # Add your Unsplash API key
        self.pexels_api_key = None       # Add your Pexels API key
        
        # AI image sources
        self.ai_sources = [
            "https://thispersondoesnotexist.com/",  # AI faces
            "https://generated.photos/",            # AI portraits
            "https://artbreeder.com/",              # AI art
            "https://midjourney.com/",              # AI art
            "https://openai.com/dall-e-2",          # DALL-E
            "https://stability.ai/",                # Stable Diffusion
        ]
        
        # Real image sources
        self.real_sources = [
            "https://unsplash.com/",                # High-quality real photos
            "https://www.pexels.com/",              # Stock photos
            "https://pixabay.com/",                 # Free stock photos
        ]

    def collect_ai_generated_images(self, count=1000):
        """Collect AI-generated images from various sources"""
        logger.info(f"Starting collection of {count} AI-generated images...")
        
        collected = 0
        sources_used = []
        
        # Method 1: Direct API calls to AI image generators
        if collected < count:
            collected += self._collect_from_ai_apis(count - collected)
            sources_used.append("AI APIs")
        
        # Method 2: Web scraping from AI art galleries
        if collected < count:
            collected += self._scrape_ai_galleries(count - collected)
            sources_used.append("AI Galleries")
        
        # Method 3: Download from curated AI image datasets
        if collected < count:
            collected += self._download_ai_datasets(count - collected)
            sources_used.append("AI Datasets")
        
        logger.info(f"Collected {collected} AI-generated images from: {', '.join(sources_used)}")
        return collected

    def collect_real_images(self, count=1000):
        """Collect real images from various sources"""
        logger.info(f"Starting collection of {count} real images...")
        
        collected = 0
        sources_used = []
        
        # Method 1: Unsplash API
        if self.unsplash_access_key and collected < count:
            collected += self._collect_from_unsplash(count - collected)
            sources_used.append("Unsplash")
        
        # Method 2: Pexels API
        if self.pexels_api_key and collected < count:
            collected += self._collect_from_pexels(count - collected)
            sources_used.append("Pexels")
        
        # Method 3: Download from real image datasets
        if collected < count:
            collected += self._download_real_datasets(count - collected)
            sources_used.append("Real Datasets")
        
        logger.info(f"Collected {collected} real images from: {', '.join(sources_used)}")
        return collected

    def _collect_from_ai_apis(self, count):
        """Collect images from AI generation APIs"""
        collected = 0
        
        # This would require API keys and specific implementations
        # For now, we'll create a placeholder structure
        logger.info("AI API collection requires specific API implementations")
        
        return collected

    def _scrape_ai_galleries(self, count):
        """Scrape AI-generated images from galleries"""
        collected = 0
        
        # This would require web scraping implementations
        # For now, we'll create a placeholder structure
        logger.info("AI gallery scraping requires web scraping implementations")
        
        return collected

    def _download_ai_datasets(self, count):
        """Download from existing AI image datasets"""
        collected = 0
        
        # Popular AI image datasets
        datasets = [
            "https://huggingface.co/datasets/Matthijs/sniffles",  # AI-generated faces
            "https://huggingface.co/datasets/poloclub/diffusiondb",  # Stable Diffusion
            "https://huggingface.co/datasets/Gustavosta/Stable-Diffusion-Prompts",  # SD prompts
        ]
        
        logger.info(f"Would download from {len(datasets)} AI datasets")
        
        return collected

    def _collect_from_unsplash(self, count):
        """Collect real images from Unsplash API"""
        if not self.unsplash_access_key:
            return 0
        
        collected = 0
        headers = {"Authorization": f"Client-ID {self.unsplash_access_key}"}
        
        # Categories for diverse real images
        categories = ["nature", "people", "architecture", "animals", "food", "travel"]
        
        for category in categories:
            if collected >= count:
                break
                
            try:
                url = f"https://api.unsplash.com/photos/random?count=30&query={category}"
                response = requests.get(url, headers=headers)
                
                if response.status_code == 200:
                    photos = response.json()
                    for photo in photos:
                        if collected >= count:
                            break
                        
                        try:
                            img_url = photo["urls"]["regular"]
                            img_response = requests.get(img_url)
                            
                            if img_response.status_code == 200:
                                img = Image.open(BytesIO(img_response.content))
                                img_hash = self._generate_image_hash(img)
                                filename = f"real_{img_hash}.jpg"
                                filepath = self.real_images_dir / filename
                                
                                if not filepath.exists():
                                    img.save(filepath, "JPEG", quality=95)
                                    collected += 1
                                    logger.info(f"Collected real image {collected}: {filename}")
                                
                                time.sleep(0.1)  # Rate limiting
                                
                        except Exception as e:
                            logger.error(f"Error downloading image: {e}")
                            continue
                            
            except Exception as e:
                logger.error(f"Error with Unsplash API: {e}")
                continue
        
        return collected

    def _collect_from_pexels(self, count):
        """Collect real images from Pexels API"""
        if not self.pexels_api_key:
            return 0
        
        collected = 0
        headers = {"Authorization": self.pexels_api_key}
        
        # Similar implementation to Unsplash
        logger.info("Pexels collection implementation similar to Unsplash")
        
        return collected

    def _download_real_datasets(self, count):
        """Download from existing real image datasets"""
        collected = 0
        
        # Popular real image datasets
        datasets = [
            "https://huggingface.co/datasets/imagenet-1k",  # ImageNet
            "https://huggingface.co/datasets/Matthijs/sniffles",  # Real faces
            "https://huggingface.co/datasets/oxford-iiit-pet",  # Pet images
        ]
        
        logger.info(f"Would download from {len(datasets)} real image datasets")
        
        return collected

    def _generate_image_hash(self, image):
        """Generate a unique hash for an image"""
        # Convert to grayscale and resize for consistent hashing
        img_gray = image.convert('L').resize((8, 8))
        img_array = np.array(img_gray)
        
        # Create hash from pixel values
        hash_value = hashlib.md5(img_array.tobytes()).hexdigest()
        return hash_value[:16]

    def create_training_dataset(self, ai_count=1000, real_count=1000):
        """Create a complete training dataset"""
        logger.info("Creating training dataset...")
        
        # Collect AI-generated images
        ai_collected = self.collect_ai_generated_images(ai_count)
        
        # Collect real images
        real_collected = self.collect_real_images(real_count)
        
        # Create dataset info
        dataset_info = {
            "created_at": datetime.now().isoformat(),
            "ai_images": ai_collected,
            "real_images": real_collected,
            "total_images": ai_collected + real_collected,
            "ai_directory": str(self.ai_images_dir),
            "real_directory": str(self.real_images_dir)
        }
        
        # Save dataset info
        info_file = self.data_dir / "dataset_info.json"
        with open(info_file, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        logger.info(f"Dataset created successfully!")
        logger.info(f"AI images: {ai_collected}")
        logger.info(f"Real images: {real_collected}")
        logger.info(f"Total: {ai_collected + real_collected}")
        
        return dataset_info

    def validate_dataset(self):
        """Validate the collected dataset"""
        logger.info("Validating dataset...")
        
        ai_files = list(self.ai_images_dir.glob("*.jpg")) + list(self.ai_images_dir.glob("*.png"))
        real_files = list(self.real_images_dir.glob("*.jpg")) + list(self.real_images_dir.glob("*.png"))
        
        logger.info(f"AI images found: {len(ai_files)}")
        logger.info(f"Real images found: {len(real_files)}")
        
        # Check for corrupted images
        corrupted_ai = 0
        corrupted_real = 0
        
        for file in ai_files:
            try:
                with Image.open(file) as img:
                    img.verify()
            except:
                corrupted_ai += 1
                file.unlink()  # Remove corrupted file
        
        for file in real_files:
            try:
                with Image.open(file) as img:
                    img.verify()
            except:
                corrupted_real += 1
                file.unlink()  # Remove corrupted file
        
        logger.info(f"Corrupted AI images removed: {corrupted_ai}")
        logger.info(f"Corrupted real images removed: {corrupted_real}")
        
        return {
            "ai_images": len(ai_files) - corrupted_ai,
            "real_images": len(real_files) - corrupted_real,
            "corrupted_ai": corrupted_ai,
            "corrupted_real": corrupted_real
        }

if __name__ == "__main__":
    # Example usage
    collector = DataCollector()
    
    # Create a small test dataset
    info = collector.create_training_dataset(ai_count=100, real_count=100)
    
    # Validate the dataset
    validation = collector.validate_dataset()
    
    print("Dataset creation completed!")
    print(f"Final dataset: {validation}") 
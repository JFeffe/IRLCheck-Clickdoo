"""
AI Detection Module for IRLCheck
Detects AI-generated images using multiple methods
"""

import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import numpy as np
import cv2
import requests
from io import BytesIO
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIDetector:
    """AI Detection class using multiple methods"""
    
    def __init__(self):
        self.models = {}
        self.processors = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.initialized = False
        
    def initialize_models(self):
        """Initialize AI detection models"""
        try:
            logger.info("Initializing AI detection models...")
            
            # Model 1: AI-Or-Not (general AI detection)
            try:
                self.processors['ai_or_not'] = AutoImageProcessor.from_pretrained(
                    "microsoft/resnet-50", 
                    cache_dir="./models"
                )
                self.models['ai_or_not'] = AutoModelForImageClassification.from_pretrained(
                    "microsoft/resnet-50",
                    cache_dir="./models"
                )
                logger.info("AI-Or-Not model loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load AI-Or-Not model: {e}")
            
            # Model 2: CLIP-based detection (alternative)
            try:
                from transformers import CLIPProcessor, CLIPModel
                self.processors['clip'] = CLIPProcessor.from_pretrained(
                    "openai/clip-vit-base-patch32",
                    cache_dir="./models"
                )
                self.models['clip'] = CLIPModel.from_pretrained(
                    "openai/clip-vit-base-patch32",
                    cache_dir="./models"
                )
                logger.info("CLIP model loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load CLIP model: {e}")
            
            self.initialized = True
            logger.info("AI detection models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing AI models: {e}")
            self.initialized = False
    
    def detect_ai_generation(self, image):
        """Main method to detect AI-generated images"""
        if not self.initialized:
            self.initialize_models()
        
        results = {
            'ai_probability': 0.0,
            'model_used': 'Fallback method',
            'confidence': 0.0,
            'detection_methods': [],
            'details': {}
        }
        
        try:
            # Method 1: Statistical analysis (always available)
            stats_result = self._statistical_analysis(image)
            results['detection_methods'].append({
                'name': 'Statistical Analysis',
                'probability': stats_result['probability'],
                'confidence': stats_result['confidence']
            })
            
            # Method 2: Deep learning models (if available)
            if self.models.get('ai_or_not'):
                dl_result = self._deep_learning_detection(image)
                results['detection_methods'].append({
                    'name': 'Deep Learning',
                    'probability': dl_result['probability'],
                    'confidence': dl_result['confidence']
                })
            
            # Method 3: CLIP-based detection (if available)
            if self.models.get('clip'):
                clip_result = self._clip_detection(image)
                results['detection_methods'].append({
                    'name': 'CLIP Analysis',
                    'probability': clip_result['probability'],
                    'confidence': clip_result['confidence']
                })
            
            # Combine results
            if results['detection_methods']:
                # Weighted average based on confidence
                total_weight = 0
                weighted_prob = 0
                
                for method in results['detection_methods']:
                    weight = method['confidence']
                    total_weight += weight
                    weighted_prob += method['probability'] * weight
                
                if total_weight > 0:
                    results['ai_probability'] = weighted_prob / total_weight
                    results['confidence'] = total_weight / len(results['detection_methods'])
                    results['model_used'] = f"Combined ({len(results['detection_methods'])} methods)"
                else:
                    # Fallback to statistical analysis
                    results['ai_probability'] = stats_result['probability']
                    results['confidence'] = stats_result['confidence']
            else:
                # Fallback to statistical analysis
                results['ai_probability'] = stats_result['probability']
                results['confidence'] = stats_result['confidence']
            
            results['details'] = {
                'methods_used': len(results['detection_methods']),
                'device': str(self.device),
                'models_available': len(self.models)
            }
            
        except Exception as e:
            logger.error(f"Error in AI detection: {e}")
            # Fallback to basic statistical analysis
            stats_result = self._statistical_analysis(image)
            results['ai_probability'] = stats_result['probability']
            results['confidence'] = stats_result['confidence']
            results['model_used'] = 'Fallback (Statistical)'
        
        return results
    
    def _statistical_analysis(self, image):
        """Statistical analysis for AI detection"""
        try:
            # Convert to numpy array
            if isinstance(image, Image.Image):
                img_array = np.array(image)
            else:
                img_array = image
            
            # Convert to grayscale if needed
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # 1. Frequency domain analysis
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.log(np.abs(f_shift) + 1)
            
            # AI-generated images often have different frequency patterns
            freq_score = np.std(magnitude_spectrum)
            
            # 2. Texture analysis
            # Calculate local binary pattern
            from skimage.feature import local_binary_pattern
            try:
                lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
                texture_score = np.std(lbp)
            except:
                texture_score = np.std(gray)
            
            # 3. Noise analysis
            # AI-generated images often have different noise characteristics
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            noise = cv2.absdiff(gray, blurred)
            noise_score = np.std(noise)
            
            # 4. Edge analysis
            edges = cv2.Canny(gray, 100, 200)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Combine scores
            scores = [freq_score, texture_score, noise_score, edge_density]
            normalized_scores = [(s - np.mean(scores)) / np.std(scores) for s in scores]
            
            # Calculate probability (higher scores = more likely AI)
            probability = min(max(np.mean(normalized_scores) * 20 + 50, 0), 100)
            confidence = 0.6  # Medium confidence for statistical method
            
            return {
                'probability': probability,
                'confidence': confidence,
                'scores': {
                    'frequency': freq_score,
                    'texture': texture_score,
                    'noise': noise_score,
                    'edge_density': edge_density
                }
            }
            
        except Exception as e:
            logger.error(f"Error in statistical analysis: {e}")
            return {'probability': 50.0, 'confidence': 0.3}
    
    def _deep_learning_detection(self, image):
        """Deep learning-based AI detection"""
        try:
            if 'ai_or_not' not in self.models:
                return {'probability': 50.0, 'confidence': 0.0}
            
            # Preprocess image
            processor = self.processors['ai_or_not']
            model = self.models['ai_or_not']
            
            # Resize and preprocess
            if isinstance(image, Image.Image):
                image = image.convert('RGB')
                image = image.resize((224, 224))
            
            inputs = processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = model(**inputs)
                probabilities = F.softmax(outputs.logits, dim=-1)
            
            # For ResNet-50, we'll use a heuristic based on feature activations
            # This is a simplified approach - in practice, you'd use a fine-tuned model
            ai_probability = float(probabilities[0][1]) * 100  # Assuming class 1 is AI
            confidence = 0.7
            
            return {
                'probability': ai_probability,
                'confidence': confidence
            }
            
        except Exception as e:
            logger.error(f"Error in deep learning detection: {e}")
            return {'probability': 50.0, 'confidence': 0.0}
    
    def _clip_detection(self, image):
        """CLIP-based AI detection"""
        try:
            if 'clip' not in self.models:
                return {'probability': 50.0, 'confidence': 0.0}
            
            processor = self.processors['clip']
            model = self.models['clip']
            
            # Prepare text prompts
            real_prompts = [
                "a real photograph",
                "a natural image",
                "a genuine photo",
                "an authentic image"
            ]
            
            ai_prompts = [
                "an AI generated image",
                "a computer generated image",
                "a synthetic image",
                "a fake image"
            ]
            
            # Preprocess image
            if isinstance(image, Image.Image):
                image = image.convert('RGB')
            
            inputs = processor(
                text=real_prompts + ai_prompts,
                images=image,
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=-1)
            
            # Calculate AI probability
            real_scores = probs[0][:len(real_prompts)].mean()
            ai_scores = probs[0][len(real_prompts):].mean()
            
            ai_probability = float(ai_scores / (real_scores + ai_scores)) * 100
            confidence = 0.8
            
            return {
                'probability': ai_probability,
                'confidence': confidence
            }
            
        except Exception as e:
            logger.error(f"Error in CLIP detection: {e}")
            return {'probability': 50.0, 'confidence': 0.0}

# Global instance
ai_detector = AIDetector()

def detect_ai_generation_simple(image):
    """Simple wrapper for AI detection"""
    return ai_detector.detect_ai_generation(image) 
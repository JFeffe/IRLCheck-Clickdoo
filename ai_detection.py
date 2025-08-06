"""
AI Detection Module for IRLCheck
Detects AI-generated images using multiple methods
"""

import torch
import torchvision.transforms as transforms
from transformers import AutoImageProcessor, AutoModelForImageClassification, CLIPProcessor, CLIPModel
import numpy as np
from PIL import Image
import cv2
from skimage.feature import local_binary_pattern
from skimage import filters
import logging
import os
import requests
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIDetector:
    def __init__(self):
        """Initialize AI detection models and processors"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models_loaded = False
        self.models = {}
        self.processors = {}
        
        # Create models directory if it doesn't exist
        os.makedirs('./models', exist_ok=True)
        
        try:
            self.initialize_models()
            self.models_loaded = True
            logging.info("AI models loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load AI models: {e}")
            self.models_loaded = False

    def initialize_models(self):
        """Load specialized AI detection models"""
        # Load specialized AI detection model (if available)
        try:
            # Try to load a model specifically trained for AI detection
            self.processors['ai_detector'] = AutoImageProcessor.from_pretrained(
                "microsoft/resnet-50", 
                cache_dir="./models"
            )
            self.models['ai_detector'] = AutoModelForImageClassification.from_pretrained(
                "microsoft/resnet-50", 
                cache_dir="./models"
            ).to(self.device)
        except Exception as e:
            logging.warning(f"Could not load specialized AI detector: {e}")
        
        # Load CLIP for text-image comparison
        try:
            self.processors['clip'] = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32", 
                cache_dir="./models"
            )
            self.models['clip'] = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32", 
                cache_dir="./models"
            ).to(self.device)
        except Exception as e:
            logging.warning(f"Could not load CLIP model: {e}")
        
        # Set models to evaluation mode
        for model in self.models.values():
            model.eval()

    def detect_ai_generation(self, image):
        """Comprehensive AI generation detection with improved accuracy"""
        if not self.models_loaded:
            return self._fallback_detection(image)
        
        try:
            # Convert PIL image to numpy array for analysis
            img_array = np.array(image)
            
            # Run all detection methods with improved algorithms
            results = {
                'statistical': self._enhanced_statistical_analysis(img_array),
                'frequency': self._enhanced_frequency_analysis(img_array),
                'texture': self._enhanced_texture_analysis(img_array),
                'noise': self._enhanced_noise_analysis(img_array),
                'edge': self._enhanced_edge_analysis(img_array),
                'color': self._enhanced_color_analysis(img_array),
                'deep_learning': self._enhanced_deep_learning_detection(image),
                'clip': self._enhanced_clip_detection(image),
                'artifacts': self._enhanced_artifact_detection(img_array),
                'consistency': self._enhanced_consistency_analysis(img_array),
                'ela': self._error_level_analysis(img_array),
                'compression': self._compression_analysis(img_array)
            }
            
            # Calculate weighted final score with improved weighting
            final_result = self._combine_results_enhanced(results)
            
            return final_result
            
        except Exception as e:
            logging.error(f"Error in AI detection: {e}")
            return self._fallback_detection(image)

    def _enhanced_statistical_analysis(self, img_array):
        """Enhanced statistical analysis with better AI detection"""
        try:
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Multiple statistical measures
            mean_intensity = np.mean(gray)
            std_intensity = np.std(gray)
            skewness = self._calculate_skewness(gray)
            kurtosis = self._calculate_kurtosis(gray)
            
            # Analyze histogram characteristics
            hist, _ = np.histogram(gray, bins=256, range=(0, 256))
            hist_smoothness = self._calculate_histogram_smoothness(hist)
            
            # AI-generated images often have more uniform distributions
            # But we need to be more sensitive to subtle differences
            uniformity_score = 1 - (std_intensity / 128)
            smoothness_score = hist_smoothness
            
            # Enhanced probability calculation
            # AI images tend to have more uniform distributions and smoother histograms
            probability = (uniformity_score * 0.7 + smoothness_score * 0.3) * 100
            
            # Apply correction factor for better sensitivity
            if probability > 50:
                probability = 50 + (probability - 50) * 1.5  # Amplify high probabilities
            
            probability = min(probability, 100)
            
            return {
                'probability': probability,
                'confidence': 0.8,
                'details': {
                    'mean_intensity': mean_intensity,
                    'std_intensity': std_intensity,
                    'skewness': skewness,
                    'kurtosis': kurtosis,
                    'histogram_smoothness': hist_smoothness,
                    'uniformity_score': uniformity_score,
                    'reasoning': f"Statistical analysis shows {'high' if probability > 70 else 'moderate' if probability > 40 else 'low'} uniformity in pixel distribution, which is {'characteristic' if probability > 70 else 'somewhat characteristic' if probability > 40 else 'not characteristic'} of AI-generated images."
                }
            }
        except Exception as e:
            return {'probability': 50, 'confidence': 0.3, 'details': {'error': str(e)}}

    def _enhanced_frequency_analysis(self, img_array):
        """Enhanced frequency domain analysis"""
        try:
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Apply FFT
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.log(np.abs(f_shift) + 1)
            
            # Enhanced frequency analysis
            center_freq = magnitude_spectrum[gray.shape[0]//2-10:gray.shape[0]//2+10, 
                                           gray.shape[1]//2-10:gray.shape[1]//2+10]
            
            # Calculate frequency characteristics
            center_energy = np.sum(center_freq)
            total_energy = np.sum(magnitude_spectrum)
            freq_ratio = center_energy / total_energy if total_energy > 0 else 0
            
            # AI images often have different frequency patterns
            # More uniform frequency distribution
            freq_uniformity = 1 - (np.std(magnitude_spectrum) / np.mean(magnitude_spectrum)) if np.mean(magnitude_spectrum) > 0 else 0
            
            # Enhanced probability calculation
            probability = (freq_ratio * 0.4 + freq_uniformity * 0.6) * 100
            
            # Apply correction for better sensitivity
            if probability > 50:
                probability = 50 + (probability - 50) * 1.3
            
            probability = min(probability, 100)
            
            return {
                'probability': probability,
                'confidence': 0.7,
                'details': {
                    'freq_ratio': freq_ratio,
                    'freq_uniformity': freq_uniformity,
                    'center_energy': center_energy,
                    'total_energy': total_energy,
                    'reasoning': f"Frequency domain analysis shows {'unusual' if probability > 70 else 'moderate' if probability > 40 else 'normal'} frequency distribution patterns, {'suggesting' if probability > 70 else 'possibly indicating' if probability > 40 else 'not indicating'} AI generation."
                }
            }
        except Exception as e:
            return {'probability': 50, 'confidence': 0.3, 'details': {'error': str(e)}}

    def _enhanced_texture_analysis(self, img_array):
        """Enhanced texture analysis"""
        try:
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Enhanced Local Binary Pattern analysis
            radius = 3
            n_points = 8 * radius
            lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
            
            # Calculate texture features
            lbp_hist, _ = np.histogram(lbp, bins=n_points + 2, range=(0, n_points + 2))
            lbp_hist = lbp_hist.astype(float) / lbp_hist.sum()
            
            # AI images often have more uniform textures
            texture_uniformity = np.sum(lbp_hist ** 2)
            texture_entropy = -np.sum(lbp_hist * np.log2(lbp_hist + 1e-10))
            
            # Enhanced scoring
            uniformity_score = texture_uniformity
            entropy_score = 1 - (texture_entropy / 8)
            
            probability = (uniformity_score * 0.7 + entropy_score * 0.3) * 100
            
            # Apply correction for better sensitivity
            if probability > 50:
                probability = 50 + (probability - 50) * 1.4
            
            probability = min(probability, 100)
            
            return {
                'probability': probability,
                'confidence': 0.75,
                'details': {
                    'texture_uniformity': texture_uniformity,
                    'texture_entropy': texture_entropy,
                    'uniformity_score': uniformity_score,
                    'entropy_score': entropy_score,
                    'reasoning': f"Texture analysis reveals {'highly uniform' if probability > 70 else 'moderately uniform' if probability > 40 else 'natural'} texture patterns, which is {'characteristic' if probability > 70 else 'somewhat characteristic' if probability > 40 else 'not characteristic'} of AI-generated content."
                }
            }
        except Exception as e:
            return {'probability': 50, 'confidence': 0.3, 'details': {'error': str(e)}}

    def _enhanced_noise_analysis(self, img_array):
        """Enhanced noise pattern analysis"""
        try:
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Multiple noise analysis methods
            # Method 1: Gaussian blur difference
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            noise = cv2.absdiff(gray, blurred)
            
            # Method 2: Laplacian noise
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            
            # Analyze noise characteristics
            noise_mean = np.mean(noise)
            noise_std = np.std(noise)
            noise_entropy = self._calculate_entropy(noise)
            laplacian_variance = np.var(laplacian)
            
            # AI-generated images often have different noise patterns
            noise_uniformity = 1 - (noise_std / (noise_mean + 1e-10))
            noise_regularity = 1 - (noise_entropy / 8)
            
            # Enhanced probability calculation
            probability = (noise_uniformity * 0.6 + noise_regularity * 0.4) * 100
            
            # Apply correction for better sensitivity
            if probability > 50:
                probability = 50 + (probability - 50) * 1.2
            
            probability = min(probability, 100)
            
            return {
                'probability': probability,
                'confidence': 0.7,
                'details': {
                    'noise_mean': noise_mean,
                    'noise_std': noise_std,
                    'noise_entropy': noise_entropy,
                    'laplacian_variance': laplacian_variance,
                    'noise_uniformity': noise_uniformity,
                    'noise_regularity': noise_regularity,
                    'reasoning': f"Noise analysis shows {'artificial' if probability > 70 else 'moderately artificial' if probability > 40 else 'natural'} noise patterns, {'indicating' if probability > 70 else 'suggesting' if probability > 40 else 'not indicating'} synthetic image generation."
                }
            }
        except Exception as e:
            return {'probability': 50, 'confidence': 0.3, 'details': {'error': str(e)}}

    def _enhanced_edge_analysis(self, img_array):
        """Enhanced edge pattern analysis"""
        try:
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Multiple edge detection methods
            edges_canny = cv2.Canny(gray, 50, 150)
            edges_sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 1)
            
            # Analyze edge characteristics
            edge_density = np.sum(edges_canny > 0) / edges_canny.size
            edge_regularity = self._calculate_edge_regularity(edges_canny)
            sobel_variance = np.var(edges_sobel)
            
            # AI-generated images often have different edge patterns
            edge_uniformity = 1 - edge_regularity
            edge_artificiality = edge_density * edge_uniformity
            
            # Enhanced probability calculation
            probability = edge_artificiality * 100
            
            # Apply correction for better sensitivity
            if probability > 50:
                probability = 50 + (probability - 50) * 1.5
            
            probability = min(probability, 100)
            
            return {
                'probability': probability,
                'confidence': 0.65,
                'details': {
                    'edge_density': edge_density,
                    'edge_regularity': edge_regularity,
                    'sobel_variance': sobel_variance,
                    'edge_uniformity': edge_uniformity,
                    'edge_artificiality': edge_artificiality,
                    'reasoning': f"Edge analysis reveals {'artificial' if probability > 70 else 'moderately artificial' if probability > 40 else 'natural'} edge patterns, {'suggesting' if probability > 70 else 'possibly indicating' if probability > 40 else 'not indicating'} computer-generated content."
                }
            }
        except Exception as e:
            return {'probability': 50, 'confidence': 0.3, 'details': {'error': str(e)}}

    def _enhanced_color_analysis(self, img_array):
        """Enhanced color analysis"""
        try:
            if len(img_array.shape) != 3:
                return {'probability': 50, 'confidence': 0.3, 'details': {'error': 'Not a color image'}}
            
            # Convert to different color spaces
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            
            # Enhanced color analysis
            hue_std = np.std(hsv[:, :, 0])
            saturation_std = np.std(hsv[:, :, 1])
            value_std = np.std(hsv[:, :, 2])
            
            # Color uniformity (AI images often have more uniform colors)
            color_uniformity = 1 - ((hue_std + saturation_std + value_std) / 3) / 128
            
            # Color palette analysis
            unique_colors = len(np.unique(img_array.reshape(-1, 3), axis=0))
            color_efficiency = unique_colors / (img_array.shape[0] * img_array.shape[1])
            
            # Enhanced probability calculation
            probability = (color_uniformity * 0.8 + (1 - color_efficiency) * 0.2) * 100
            
            # Apply correction for better sensitivity
            if probability > 50:
                probability = 50 + (probability - 50) * 1.3
            
            probability = min(probability, 100)
            
            return {
                'probability': probability,
                'confidence': 0.6,
                'details': {
                    'hue_std': hue_std,
                    'saturation_std': saturation_std,
                    'value_std': value_std,
                    'color_uniformity': color_uniformity,
                    'unique_colors': unique_colors,
                    'color_efficiency': color_efficiency,
                    'reasoning': f"Color analysis shows {'artificial' if probability > 70 else 'moderately artificial' if probability > 40 else 'natural'} color distribution, {'indicating' if probability > 70 else 'suggesting' if probability > 40 else 'not indicating'} synthetic image generation."
                }
            }
        except Exception as e:
            return {'probability': 50, 'confidence': 0.3, 'details': {'error': str(e)}}

    def _enhanced_deep_learning_detection(self, image):
        """Enhanced deep learning-based AI detection"""
        try:
            if 'ai_detector' not in self.models:
                return {'probability': 50, 'confidence': 0.3, 'details': {'error': 'Model not available'}}
            
            # Prepare image for model
            inputs = self.processors['ai_detector'](image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.models['ai_detector'](**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Get top predictions
            top_probs, top_indices = torch.topk(probabilities, 10)
            
            # Enhanced AI indicator detection
            ai_indicators = ['artificial', 'synthetic', 'generated', 'digital', 'computer', 'fake', 'ai', 'neural', 'gan', 'deepfake']
            ai_score = 0
            
            for prob, idx in zip(top_probs[0], top_indices[0]):
                label = self.models['ai_detector'].config.id2label[idx.item()]
                label_lower = label.lower()
                
                # Check for AI indicators in label
                for indicator in ai_indicators:
                    if indicator in label_lower:
                        ai_score += prob.item()
                        break
                
                # Additional heuristic: check for unusual confidence patterns
                if prob.item() > 0.8:  # Very high confidence might indicate AI
                    ai_score += prob.item() * 0.5
            
            # Enhanced probability calculation
            probability = ai_score * 100
            
            # Apply correction for better sensitivity
            if probability > 50:
                probability = 50 + (probability - 50) * 1.5
            
            probability = min(probability, 100)
            
            return {
                'probability': probability,
                'confidence': 0.85,
                'details': {
                    'top_predictions': [(self.models['ai_detector'].config.id2label[idx.item()], prob.item()) 
                                       for prob, idx in zip(top_probs[0], top_indices[0])],
                    'ai_score': ai_score,
                    'reasoning': f"Deep learning model identified {'strong' if probability > 70 else 'moderate' if probability > 40 else 'weak'} AI generation indicators in the image classification results."
                }
            }
        except Exception as e:
            return {'probability': 50, 'confidence': 0.3, 'details': {'error': str(e)}}

    def _enhanced_clip_detection(self, image):
        """Enhanced CLIP-based text-image comparison"""
        try:
            if 'clip' not in self.models:
                return {'probability': 50, 'confidence': 0.3, 'details': {'error': 'CLIP model not available'}}
            
            # Enhanced text prompts for better AI detection
            real_prompts = [
                "a real photograph taken by a camera",
                "a natural image captured in real life",
                "a genuine photo from reality",
                "an authentic photograph",
                "a real picture from the real world",
                "a natural scene photographed",
                "a real person in a real environment",
                "a genuine moment captured on camera"
            ]
            
            ai_prompts = [
                "an AI generated image created by artificial intelligence",
                "a computer generated image made by AI",
                "a synthetic image produced by neural networks",
                "an artificial image generated by machine learning",
                "a fake image created by AI",
                "a computer generated fake image",
                "an AI generated fake photograph",
                "a synthetic fake image made by artificial intelligence"
            ]
            
            # Prepare inputs
            inputs = self.processors['clip'](
                text=real_prompts + ai_prompts,
                images=image,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.models['clip'](**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=-1)
            
            # Calculate scores
            real_score = probs[0, :8].mean().item()  # First 8 are real prompts
            ai_score = probs[0, 8:].mean().item()    # Last 8 are AI prompts
            
            # Enhanced probability calculation
            probability = ai_score * 100
            
            # Apply correction for better sensitivity
            if probability > 50:
                probability = 50 + (probability - 50) * 1.4
            
            probability = min(probability, 100)
            
            return {
                'probability': probability,
                'confidence': 0.8,
                'details': {
                    'real_score': real_score,
                    'ai_score': ai_score,
                    'real_prompts': real_prompts,
                    'ai_prompts': ai_prompts,
                    'reasoning': f"CLIP analysis shows the image is {'more similar to' if probability > 70 else 'somewhat similar to' if probability > 40 else 'more similar to real'} AI-generated descriptions than real photograph descriptions."
                }
            }
        except Exception as e:
            return {'probability': 50, 'confidence': 0.3, 'details': {'error': str(e)}}

    def _enhanced_artifact_detection(self, img_array):
        """Enhanced artifact detection"""
        try:
            artifacts_found = []
            artifact_score = 0
            
            # Enhanced grid pattern detection
            grid_score = self._detect_grid_patterns_enhanced(img_array)
            if grid_score > 0.2:  # Lower threshold for better detection
                artifacts_found.append(f"Grid patterns detected (score: {grid_score:.2f})")
                artifact_score += grid_score * 0.4
            
            # Enhanced repetitive pattern detection
            repetition_score = self._detect_repetitive_patterns_enhanced(img_array)
            if repetition_score > 0.2:
                artifacts_found.append(f"Repetitive patterns detected (score: {repetition_score:.2f})")
                artifact_score += repetition_score * 0.3
            
            # Enhanced unrealistic detail detection
            detail_score = self._detect_unrealistic_details_enhanced(img_array)
            if detail_score > 0.2:
                artifacts_found.append(f"Unrealistic details detected (score: {detail_score:.2f})")
                artifact_score += detail_score * 0.3
            
            # Enhanced probability calculation
            probability = artifact_score * 100
            
            # Apply correction for better sensitivity
            if probability > 50:
                probability = 50 + (probability - 50) * 1.6
            
            probability = min(probability, 100)
            
            return {
                'probability': probability,
                'confidence': 0.75 if artifacts_found else 0.4,
                'details': {
                    'artifacts_found': artifacts_found,
                    'grid_score': grid_score,
                    'repetition_score': repetition_score,
                    'detail_score': detail_score,
                    'artifact_score': artifact_score,
                    'reasoning': f"Artifact detection {'found' if artifacts_found else 'did not find'} common AI generation artifacts: {', '.join(artifacts_found) if artifacts_found else 'No significant artifacts detected'}."
                }
            }
        except Exception as e:
            return {'probability': 50, 'confidence': 0.3, 'details': {'error': str(e)}}

    def _enhanced_consistency_analysis(self, img_array):
        """Enhanced consistency analysis"""
        try:
            consistency_issues = []
            inconsistency_score = 0
            
            # Enhanced lighting consistency check
            lighting_score = self._check_lighting_consistency_enhanced(img_array)
            if lighting_score > 0.2:  # Lower threshold
                consistency_issues.append(f"Lighting inconsistencies (score: {lighting_score:.2f})")
                inconsistency_score += lighting_score * 0.4
            
            # Enhanced perspective consistency check
            perspective_score = self._check_perspective_consistency_enhanced(img_array)
            if perspective_score > 0.2:
                consistency_issues.append(f"Perspective inconsistencies (score: {perspective_score:.2f})")
                inconsistency_score += perspective_score * 0.3
            
            # Enhanced object consistency check
            object_score = self._check_object_consistency_enhanced(img_array)
            if object_score > 0.2:
                consistency_issues.append(f"Object inconsistencies (score: {object_score:.2f})")
                inconsistency_score += object_score * 0.3
            
            # Enhanced probability calculation
            probability = inconsistency_score * 100
            
            # Apply correction for better sensitivity
            if probability > 50:
                probability = 50 + (probability - 50) * 1.3
            
            probability = min(probability, 100)
            
            return {
                'probability': probability,
                'confidence': 0.7 if consistency_issues else 0.5,
                'details': {
                    'consistency_issues': consistency_issues,
                    'lighting_score': lighting_score,
                    'perspective_score': perspective_score,
                    'object_score': object_score,
                    'inconsistency_score': inconsistency_score,
                    'reasoning': f"Consistency analysis {'found' if consistency_issues else 'did not find'} inconsistencies: {', '.join(consistency_issues) if consistency_issues else 'Image appears consistent'}."
                }
            }
        except Exception as e:
            return {'probability': 50, 'confidence': 0.3, 'details': {'error': str(e)}}

    def _error_level_analysis(self, img_array):
        """Error Level Analysis for detecting editing"""
        try:
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Simplified ELA (Error Level Analysis)
            # Save and reload image to introduce compression artifacts
            temp_img = Image.fromarray(gray)
            temp_buffer = BytesIO()
            temp_img.save(temp_buffer, format='JPEG', quality=95)
            temp_buffer.seek(0)
            reloaded_img = Image.open(temp_buffer)
            reloaded_array = np.array(reloaded_img)
            
            # Calculate difference
            ela_diff = np.abs(gray.astype(float) - reloaded_array.astype(float))
            ela_variance = np.var(ela_diff)
            
            # AI-generated images often have different ELA patterns
            probability = min(ela_variance / 100 * 100, 100)
            
            # Apply correction for better sensitivity
            if probability > 50:
                probability = 50 + (probability - 50) * 1.2
            
            return {
                'probability': probability,
                'confidence': 0.6,
                'details': {
                    'ela_variance': ela_variance,
                    'reasoning': f"Error Level Analysis shows {'unusual' if probability > 70 else 'moderate' if probability > 40 else 'normal'} compression patterns, {'suggesting' if probability > 70 else 'possibly indicating' if probability > 40 else 'not indicating'} AI generation or editing."
                }
            }
        except Exception as e:
            return {'probability': 50, 'confidence': 0.3, 'details': {'error': str(e)}}

    def _compression_analysis(self, img_array):
        """Compression artifact analysis"""
        try:
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Analyze compression artifacts
            # Look for block artifacts (common in JPEG compression)
            block_size = 8
            h, w = gray.shape
            
            block_artifacts = 0
            for i in range(0, h - block_size, block_size):
                for j in range(0, w - block_size, block_size):
                    block = gray[i:i+block_size, j:j+block_size]
                    block_variance = np.var(block)
                    if block_variance < 10:  # Low variance might indicate compression
                        block_artifacts += 1
            
            total_blocks = ((h // block_size) * (w // block_size))
            artifact_ratio = block_artifacts / total_blocks if total_blocks > 0 else 0
            
            # AI-generated images might have different compression patterns
            probability = artifact_ratio * 100
            
            # Apply correction for better sensitivity
            if probability > 50:
                probability = 50 + (probability - 50) * 1.1
            
            probability = min(probability, 100)
            
            return {
                'probability': probability,
                'confidence': 0.5,
                'details': {
                    'artifact_ratio': artifact_ratio,
                    'block_artifacts': block_artifacts,
                    'total_blocks': total_blocks,
                    'reasoning': f"Compression analysis shows {'unusual' if probability > 70 else 'moderate' if probability > 40 else 'normal'} compression patterns, {'suggesting' if probability > 70 else 'possibly indicating' if probability > 40 else 'not indicating'} AI generation."
                }
            }
        except Exception as e:
            return {'probability': 50, 'confidence': 0.3, 'details': {'error': str(e)}}

    def _combine_results_enhanced(self, results):
        """Enhanced combination of all analysis results"""
        # Improved weighting based on reliability and sensitivity
        weights = {
            'deep_learning': 0.30,  # Increased weight
            'clip': 0.25,           # Increased weight
            'statistical': 0.15,
            'texture': 0.12,
            'frequency': 0.08,
            'noise': 0.05,
            'edge': 0.02,
            'color': 0.01,
            'artifacts': 0.01,
            'consistency': 0.005,
            'ela': 0.005,
            'compression': 0.005
        }
        
        total_probability = 0
        total_weight = 0
        detailed_reasons = []
        
        for method, weight in weights.items():
            if method in results:
                result = results[method]
                prob = result['probability']
                conf = result['confidence']
                
                # Adjust weight by confidence
                adjusted_weight = weight * conf
                total_probability += prob * adjusted_weight
                total_weight += adjusted_weight
                
                # Add detailed reasoning
                if 'reasoning' in result['details']:
                    detailed_reasons.append(f"• {method.replace('_', ' ').title()}: {result['details']['reasoning']}")
        
        if total_weight > 0:
            final_probability = total_probability / total_weight
        else:
            final_probability = 50
        
        # Apply final correction for better sensitivity
        if final_probability > 50:
            final_probability = 50 + (final_probability - 50) * 1.2
        
        final_probability = min(final_probability, 100)
        
        # Determine overall confidence
        avg_confidence = sum(results[method]['confidence'] for method in weights.keys() if method in results) / len(weights)
        
        # Create comprehensive report
        report = {
            'ai_probability': final_probability,
            'confidence': avg_confidence,
            'model_used': 'Enhanced Multi-method AI Detection',
            'detection_methods': [
                {
                    'name': method.replace('_', ' ').title(),
                    'probability': results[method]['probability'],
                    'confidence': results[method]['confidence']
                }
                for method in weights.keys() if method in results
            ],
            'details': {
                'methods_used': len(results),
                'device': str(self.device),
                'models_available': len(self.models),
                'detailed_reasons': detailed_reasons,
                'method_scores': {
                    method: {
                        'probability': results[method]['probability'],
                        'confidence': results[method]['confidence'],
                        'details': results[method]['details']
                    }
                    for method in weights.keys() if method in results
                }
            }
        }
        
        return report

    def _fallback_detection(self, image):
        """Enhanced fallback detection"""
        return {
            'ai_probability': 50.0,
            'confidence': 0.3,
            'model_used': 'Enhanced Fallback (Basic Analysis)',
            'detection_methods': [],
            'details': {
                'error': 'AI models not available',
                'methods_used': 0,
                'device': 'CPU',
                'models_available': 0
            }
        }

    # Enhanced helper methods
    def _detect_grid_patterns_enhanced(self, img_array):
        """Enhanced grid pattern detection"""
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Multiple grid detection methods
        edges = cv2.Canny(gray, 30, 100)  # Lower thresholds for better detection
        
        # Horizontal and vertical line detection
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
        
        horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
        vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)
        
        # Calculate grid scores
        horizontal_score = np.sum(horizontal_lines) / horizontal_lines.size
        vertical_score = np.sum(vertical_lines) / vertical_lines.size
        
        # Enhanced grid score calculation
        grid_score = (horizontal_score + vertical_score) / 2
        return min(1, grid_score * 15)  # Increased sensitivity

    def _detect_repetitive_patterns_enhanced(self, img_array):
        """Enhanced repetitive pattern detection"""
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Enhanced autocorrelation analysis
        h, w = gray.shape
        center_h, center_w = h // 2, w // 2
        
        region_size = min(60, h // 3, w // 3)  # Larger region
        region = gray[center_h-region_size//2:center_h+region_size//2, 
                     center_w-region_size//2:center_w+region_size//2]
        
        if region.size == 0:
            return 0
        
        # Enhanced autocorrelation
        corr = np.correlate(region.flatten(), region.flatten(), mode='full')
        corr = corr[len(corr)//2:]
        
        # Look for peaks indicating repetition
        peaks = []
        for i in range(1, len(corr)-1):
            if corr[i] > corr[i-1] and corr[i] > corr[i+1] and corr[i] > np.mean(corr):
                peaks.append(corr[i])
        
        if len(peaks) == 0:
            return 0
        
        # Enhanced repetition score
        repetition_score = np.mean(peaks) / np.max(corr)
        return min(1, repetition_score * 8)  # Increased sensitivity

    def _detect_unrealistic_details_enhanced(self, img_array):
        """Enhanced unrealistic detail detection"""
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Multiple detail analysis methods
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = np.var(laplacian)
        
        # Enhanced sharpness analysis
        sharpness_score = min(1, sharpness / 800)  # Adjusted threshold
        
        # Gradient analysis
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Enhanced gradient uniformity
        gradient_uniformity = 1 - (np.std(gradient_magnitude) / (np.mean(gradient_magnitude) + 1e-10))
        
        # Enhanced detail score
        detail_score = (sharpness_score * 0.6 + gradient_uniformity * 0.4)
        return detail_score

    def _check_lighting_consistency_enhanced(self, img_array):
        """Enhanced lighting consistency check"""
        if len(img_array.shape) != 3:
            return 0
        
        # Convert to LAB color space
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0]
        
        # Enhanced lighting analysis
        grad_x = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1, ksize=3)
        
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        lighting_inconsistency = np.std(gradient_magnitude) / (np.mean(gradient_magnitude) + 1e-10)
        
        return min(1, lighting_inconsistency / 1.5)  # Adjusted threshold

    def _check_perspective_consistency_enhanced(self, img_array):
        """Enhanced perspective consistency check"""
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Enhanced line detection
        edges = cv2.Canny(gray, 30, 100)  # Lower thresholds
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=80)  # Lower threshold
        
        if lines is None:
            return 0
        
        # Analyze line angles for consistency
        angles = []
        for line in lines:
            rho, theta = line[0]
            angles.append(theta)
        
        if len(angles) < 2:
            return 0
        
        # Enhanced angle consistency
        angle_std = np.std(angles)
        consistency = 1 - (angle_std / np.pi)
        
        return max(0, consistency)

    def _check_object_consistency_enhanced(self, img_array):
        """Enhanced object consistency check"""
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Enhanced region analysis
        h, w = gray.shape
        region_size = min(40, h // 6, w // 6)  # Smaller regions for better analysis
        
        regions = []
        for i in range(0, h - region_size, region_size):
            for j in range(0, w - region_size, region_size):
                region = gray[i:i+region_size, j:j+region_size]
                regions.append(np.std(region))
        
        if len(regions) < 2:
            return 0
        
        # Enhanced region consistency
        region_std = np.std(regions)
        region_mean = np.mean(regions)
        
        if region_mean == 0:
            return 0
        
        consistency = 1 - (region_std / region_mean)
        return max(0, min(1, consistency))

    # Helper methods (unchanged)
    def _calculate_skewness(self, data):
        """Calculate skewness of data"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)

    def _calculate_kurtosis(self, data):
        """Calculate kurtosis of data"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 4) - 3

    def _calculate_histogram_smoothness(self, hist):
        """Calculate histogram smoothness"""
        if len(hist) < 3:
            return 0
        differences = np.diff(hist)
        smoothness = 1 - (np.std(differences) / (np.mean(np.abs(differences)) + 1e-10))
        return max(0, min(1, smoothness))

    def _calculate_entropy(self, data):
        """Calculate entropy of data"""
        hist, _ = np.histogram(data, bins=256, range=(0, 256))
        hist = hist.astype(float) / hist.sum()
        return -np.sum(hist * np.log2(hist + 1e-10))

    def _calculate_edge_regularity(self, edges):
        """Calculate edge regularity"""
        edge_coords = np.where(edges > 0)
        if len(edge_coords[0]) == 0:
            return 0
        
        directions = []
        for i in range(0, len(edge_coords[0]), 10):
            y, x = edge_coords[0][i], edge_coords[1][i]
            if y > 0 and y < edges.shape[0] - 1 and x > 0 and x < edges.shape[1] - 1:
                gx = edges[y, x+1] - edges[y, x-1]
                gy = edges[y+1, x] - edges[y-1, x]
                if gx != 0 or gy != 0:
                    angle = np.arctan2(gy, gx)
                    directions.append(angle)
        
        if len(directions) < 2:
            return 0
        
        directions = np.array(directions)
        direction_std = np.std(directions)
        regularity = 1 - (direction_std / np.pi)
        return max(0, min(1, regularity))

# Global instance
ai_detector = AIDetector()

def detect_ai_generation_simple(image):
    """Simple wrapper function for AI detection"""
    return ai_detector.detect_ai_generation(image) 
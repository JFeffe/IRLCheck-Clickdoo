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
        """Load all AI detection models"""
        # Load ResNet-50 for general AI detection
        self.processors['resnet'] = AutoImageProcessor.from_pretrained(
            "microsoft/resnet-50", 
            cache_dir="./models"
        )
        self.models['resnet'] = AutoModelForImageClassification.from_pretrained(
            "microsoft/resnet-50", 
            cache_dir="./models"
        ).to(self.device)
        
        # Load CLIP for text-image comparison
        self.processors['clip'] = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32", 
            cache_dir="./models"
        )
        self.models['clip'] = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32", 
            cache_dir="./models"
        ).to(self.device)
        
        # Set models to evaluation mode
        for model in self.models.values():
            model.eval()

    def detect_ai_generation(self, image):
        """Comprehensive AI generation detection with detailed analysis"""
        if not self.models_loaded:
            return self._fallback_detection(image)
        
        try:
            # Convert PIL image to numpy array for analysis
            img_array = np.array(image)
            
            # Run all detection methods
            results = {
                'statistical': self._statistical_analysis(img_array),
                'frequency': self._frequency_domain_analysis(img_array),
                'texture': self._texture_analysis(img_array),
                'noise': self._noise_analysis(img_array),
                'edge': self._edge_analysis(img_array),
                'color': self._color_analysis(img_array),
                'deep_learning': self._deep_learning_detection(image),
                'clip': self._clip_detection(image),
                'artifacts': self._artifact_detection(img_array),
                'consistency': self._consistency_analysis(img_array)
            }
            
            # Calculate weighted final score
            final_result = self._combine_results(results)
            
            return final_result
            
        except Exception as e:
            logging.error(f"Error in AI detection: {e}")
            return self._fallback_detection(image)

    def _statistical_analysis(self, img_array):
        """Statistical analysis of image properties"""
        try:
            # Convert to grayscale if needed
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Calculate various statistical measures
            mean_intensity = np.mean(gray)
            std_intensity = np.std(gray)
            skewness = self._calculate_skewness(gray)
            kurtosis = self._calculate_kurtosis(gray)
            
            # Analyze histogram
            hist, _ = np.histogram(gray, bins=256, range=(0, 256))
            hist_smoothness = self._calculate_histogram_smoothness(hist)
            
            # Calculate probability based on statistical indicators
            # AI-generated images often have more uniform distributions
            uniformity_score = 1 - (std_intensity / 128)  # Normalized
            smoothness_score = hist_smoothness
            
            # Combine scores
            probability = (uniformity_score * 0.6 + smoothness_score * 0.4) * 100
            probability = min(probability, 100)
            
            return {
                'probability': probability,
                'confidence': 0.7,
                'details': {
                    'mean_intensity': mean_intensity,
                    'std_intensity': std_intensity,
                    'skewness': skewness,
                    'kurtosis': kurtosis,
                    'histogram_smoothness': hist_smoothness,
                    'uniformity_score': uniformity_score,
                    'reasoning': f"Statistical analysis shows {'high' if probability > 70 else 'moderate' if probability > 40 else 'low'} uniformity in pixel distribution, which is {'typical' if probability > 70 else 'somewhat typical' if probability > 40 else 'atypical'} of AI-generated images."
                }
            }
        except Exception as e:
            return {'probability': 50, 'confidence': 0.3, 'details': {'error': str(e)}}

    def _frequency_domain_analysis(self, img_array):
        """Analyze frequency domain characteristics"""
        try:
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Apply FFT
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.log(np.abs(f_shift) + 1)
            
            # Analyze frequency distribution
            center_freq = magnitude_spectrum[gray.shape[0]//2-10:gray.shape[0]//2+10, 
                                           gray.shape[1]//2-10:gray.shape[1]//2+10]
            edge_freq = magnitude_spectrum - center_freq
            
            # Calculate frequency ratios
            center_energy = np.sum(center_freq)
            edge_energy = np.sum(edge_freq)
            total_energy = np.sum(magnitude_spectrum)
            
            # AI-generated images often have different frequency characteristics
            freq_ratio = center_energy / total_energy if total_energy > 0 else 0
            freq_uniformity = 1 - (np.std(magnitude_spectrum) / np.mean(magnitude_spectrum)) if np.mean(magnitude_spectrum) > 0 else 0
            
            probability = (freq_ratio * 0.5 + freq_uniformity * 0.5) * 100
            probability = min(probability, 100)
            
            return {
                'probability': probability,
                'confidence': 0.6,
                'details': {
                    'freq_ratio': freq_ratio,
                    'freq_uniformity': freq_uniformity,
                    'center_energy': center_energy,
                    'edge_energy': edge_energy,
                    'reasoning': f"Frequency domain analysis shows {'unusual' if probability > 70 else 'moderate' if probability > 40 else 'normal'} frequency distribution patterns, {'suggesting' if probability > 70 else 'possibly indicating' if probability > 40 else 'not indicating'} AI generation."
                }
            }
        except Exception as e:
            return {'probability': 50, 'confidence': 0.3, 'details': {'error': str(e)}}

    def _texture_analysis(self, img_array):
        """Analyze texture patterns"""
        try:
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Local Binary Pattern analysis
            radius = 3
            n_points = 8 * radius
            lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
            
            # Calculate texture features
            lbp_hist, _ = np.histogram(lbp, bins=n_points + 2, range=(0, n_points + 2))
            lbp_hist = lbp_hist.astype(float) / lbp_hist.sum()
            
            # Texture uniformity (AI images often have more uniform textures)
            texture_uniformity = np.sum(lbp_hist ** 2)
            
            # Calculate texture entropy
            texture_entropy = -np.sum(lbp_hist * np.log2(lbp_hist + 1e-10))
            
            # Normalize scores
            uniformity_score = texture_uniformity
            entropy_score = 1 - (texture_entropy / 8)  # Normalized
            
            probability = (uniformity_score * 0.6 + entropy_score * 0.4) * 100
            probability = min(probability, 100)
            
            return {
                'probability': probability,
                'confidence': 0.65,
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

    def _noise_analysis(self, img_array):
        """Analyze noise patterns"""
        try:
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Calculate noise
            noise = cv2.absdiff(gray, blurred)
            
            # Analyze noise characteristics
            noise_mean = np.mean(noise)
            noise_std = np.std(noise)
            noise_entropy = self._calculate_entropy(noise)
            
            # AI-generated images often have different noise patterns
            noise_uniformity = 1 - (noise_std / (noise_mean + 1e-10))
            noise_regularity = 1 - (noise_entropy / 8)  # Normalized
            
            probability = (noise_uniformity * 0.5 + noise_regularity * 0.5) * 100
            probability = min(probability, 100)
            
            return {
                'probability': probability,
                'confidence': 0.6,
                'details': {
                    'noise_mean': noise_mean,
                    'noise_std': noise_std,
                    'noise_entropy': noise_entropy,
                    'noise_uniformity': noise_uniformity,
                    'noise_regularity': noise_regularity,
                    'reasoning': f"Noise analysis shows {'artificial' if probability > 70 else 'moderately artificial' if probability > 40 else 'natural'} noise patterns, {'indicating' if probability > 70 else 'suggesting' if probability > 40 else 'not indicating'} synthetic image generation."
                }
            }
        except Exception as e:
            return {'probability': 50, 'confidence': 0.3, 'details': {'error': str(e)}}

    def _edge_analysis(self, img_array):
        """Analyze edge patterns"""
        try:
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Detect edges
            edges = cv2.Canny(gray, 50, 150)
            
            # Analyze edge characteristics
            edge_density = np.sum(edges > 0) / edges.size
            edge_regularity = self._calculate_edge_regularity(edges)
            
            # AI-generated images often have different edge patterns
            edge_uniformity = 1 - edge_regularity
            edge_artificiality = edge_density * edge_uniformity
            
            probability = edge_artificiality * 100
            probability = min(probability, 100)
            
            return {
                'probability': probability,
                'confidence': 0.55,
                'details': {
                    'edge_density': edge_density,
                    'edge_regularity': edge_regularity,
                    'edge_uniformity': edge_uniformity,
                    'edge_artificiality': edge_artificiality,
                    'reasoning': f"Edge analysis reveals {'artificial' if probability > 70 else 'moderately artificial' if probability > 40 else 'natural'} edge patterns, {'suggesting' if probability > 70 else 'possibly indicating' if probability > 40 else 'not indicating'} computer-generated content."
                }
            }
        except Exception as e:
            return {'probability': 50, 'confidence': 0.3, 'details': {'error': str(e)}}

    def _color_analysis(self, img_array):
        """Analyze color characteristics"""
        try:
            if len(img_array.shape) != 3:
                return {'probability': 50, 'confidence': 0.3, 'details': {'error': 'Not a color image'}}
            
            # Convert to different color spaces
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            
            # Analyze color distribution
            hue_std = np.std(hsv[:, :, 0])
            saturation_std = np.std(hsv[:, :, 1])
            value_std = np.std(hsv[:, :, 2])
            
            # Color uniformity (AI images often have more uniform colors)
            color_uniformity = 1 - ((hue_std + saturation_std + value_std) / 3) / 128
            
            # Color palette analysis
            unique_colors = len(np.unique(img_array.reshape(-1, 3), axis=0))
            color_efficiency = unique_colors / (img_array.shape[0] * img_array.shape[1])
            
            probability = (color_uniformity * 0.7 + (1 - color_efficiency) * 0.3) * 100
            probability = min(probability, 100)
            
            return {
                'probability': probability,
                'confidence': 0.5,
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

    def _deep_learning_detection(self, image):
        """Deep learning-based AI detection"""
        try:
            # Prepare image for ResNet
            inputs = self.processors['resnet'](image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.models['resnet'](**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Get top predictions
            top_probs, top_indices = torch.topk(probabilities, 5)
            
            # Analyze predictions for AI indicators
            ai_indicators = ['artificial', 'synthetic', 'generated', 'digital', 'computer']
            ai_score = 0
            
            for prob, idx in zip(top_probs[0], top_indices[0]):
                label = self.models['resnet'].config.id2label[idx.item()]
                if any(indicator in label.lower() for indicator in ai_indicators):
                    ai_score += prob.item()
            
            probability = ai_score * 100
            confidence = 0.8
            
            return {
                'probability': probability,
                'confidence': confidence,
                'details': {
                    'top_predictions': [(self.models['resnet'].config.id2label[idx.item()], prob.item()) 
                                       for prob, idx in zip(top_probs[0], top_indices[0])],
                    'ai_score': ai_score,
                    'reasoning': f"Deep learning model identified {'strong' if probability > 70 else 'moderate' if probability > 40 else 'weak'} AI generation indicators in the image classification results."
                }
            }
        except Exception as e:
            return {'probability': 50, 'confidence': 0.3, 'details': {'error': str(e)}}

    def _clip_detection(self, image):
        """CLIP-based text-image comparison"""
        try:
            # Define text prompts for comparison
            real_prompts = [
                "a real photograph",
                "a natural image",
                "a genuine photo",
                "an authentic image",
                "a real picture"
            ]
            
            ai_prompts = [
                "an AI generated image",
                "a computer generated image",
                "a synthetic image",
                "an artificial image",
                "a generated picture"
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
            real_score = probs[0, :5].mean().item()  # First 5 are real prompts
            ai_score = probs[0, 5:].mean().item()    # Last 5 are AI prompts
            
            probability = ai_score * 100
            confidence = 0.75
            
            return {
                'probability': probability,
                'confidence': confidence,
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

    def _artifact_detection(self, img_array):
        """Detect common AI generation artifacts"""
        try:
            artifacts_found = []
            artifact_score = 0
            
            # Check for grid patterns (common in GANs)
            grid_score = self._detect_grid_patterns(img_array)
            if grid_score > 0.3:
                artifacts_found.append(f"Grid patterns detected (score: {grid_score:.2f})")
                artifact_score += grid_score * 0.3
            
            # Check for repetitive patterns
            repetition_score = self._detect_repetitive_patterns(img_array)
            if repetition_score > 0.3:
                artifacts_found.append(f"Repetitive patterns detected (score: {repetition_score:.2f})")
                artifact_score += repetition_score * 0.3
            
            # Check for unrealistic details
            detail_score = self._detect_unrealistic_details(img_array)
            if detail_score > 0.3:
                artifacts_found.append(f"Unrealistic details detected (score: {detail_score:.2f})")
                artifact_score += detail_score * 0.4
            
            probability = artifact_score * 100
            confidence = 0.7 if artifacts_found else 0.3
            
            return {
                'probability': probability,
                'confidence': confidence,
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

    def _consistency_analysis(self, img_array):
        """Analyze overall image consistency"""
        try:
            consistency_issues = []
            inconsistency_score = 0
            
            # Check lighting consistency
            lighting_score = self._check_lighting_consistency(img_array)
            if lighting_score > 0.3:
                consistency_issues.append(f"Lighting inconsistencies (score: {lighting_score:.2f})")
                inconsistency_score += lighting_score * 0.3
            
            # Check perspective consistency
            perspective_score = self._check_perspective_consistency(img_array)
            if perspective_score > 0.3:
                consistency_issues.append(f"Perspective inconsistencies (score: {perspective_score:.2f})")
                inconsistency_score += perspective_score * 0.3
            
            # Check object consistency
            object_score = self._check_object_consistency(img_array)
            if object_score > 0.3:
                consistency_issues.append(f"Object inconsistencies (score: {object_score:.2f})")
                inconsistency_score += object_score * 0.4
            
            probability = inconsistency_score * 100
            confidence = 0.6 if consistency_issues else 0.4
            
            return {
                'probability': probability,
                'confidence': confidence,
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

    def _combine_results(self, results):
        """Combine all analysis results with detailed reasoning"""
        # Weight each method based on confidence and reliability
        weights = {
            'deep_learning': 0.25,
            'clip': 0.20,
            'statistical': 0.15,
            'texture': 0.12,
            'frequency': 0.10,
            'noise': 0.08,
            'edge': 0.05,
            'color': 0.03,
            'artifacts': 0.01,
            'consistency': 0.01
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
        
        # Determine overall confidence
        avg_confidence = sum(results[method]['confidence'] for method in weights.keys() if method in results) / len(weights)
        
        # Create comprehensive report
        report = {
            'ai_probability': final_probability,
            'confidence': avg_confidence,
            'model_used': 'Multi-method AI Detection',
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
        """Fallback detection when models are not available"""
        return {
            'ai_probability': 50.0,
            'confidence': 0.3,
            'model_used': 'Fallback (Basic Analysis)',
            'detection_methods': [],
            'details': {
                'error': 'AI models not available',
                'methods_used': 0,
                'device': 'CPU',
                'models_available': 0
            }
        }

    # Helper methods for calculations
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
        # This is a simplified version - in practice, more sophisticated algorithms would be used
        edge_coords = np.where(edges > 0)
        if len(edge_coords[0]) == 0:
            return 0
        
        # Calculate edge direction consistency
        directions = []
        for i in range(0, len(edge_coords[0]), 10):  # Sample every 10th edge
            y, x = edge_coords[0][i], edge_coords[1][i]
            if y > 0 and y < edges.shape[0] - 1 and x > 0 and x < edges.shape[1] - 1:
                # Calculate gradient direction
                gx = edges[y, x+1] - edges[y, x-1]
                gy = edges[y+1, x] - edges[y-1, x]
                if gx != 0 or gy != 0:
                    angle = np.arctan2(gy, gx)
                    directions.append(angle)
        
        if len(directions) < 2:
            return 0
        
        # Calculate direction consistency
        directions = np.array(directions)
        direction_std = np.std(directions)
        regularity = 1 - (direction_std / np.pi)  # Normalize
        return max(0, min(1, regularity))

    def _detect_grid_patterns(self, img_array):
        """Detect grid-like patterns common in GANs"""
        # Simplified grid detection
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Look for horizontal and vertical lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
        
        horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
        vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)
        
        # Calculate grid score
        horizontal_score = np.sum(horizontal_lines) / horizontal_lines.size
        vertical_score = np.sum(vertical_lines) / vertical_lines.size
        
        grid_score = (horizontal_score + vertical_score) / 2
        return min(1, grid_score * 10)  # Scale up

    def _detect_repetitive_patterns(self, img_array):
        """Detect repetitive patterns"""
        # Simplified repetition detection
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Use autocorrelation to detect repetition
        h, w = gray.shape
        center_h, center_w = h // 2, w // 2
        
        # Calculate autocorrelation for a small region
        region_size = min(50, h // 4, w // 4)
        region = gray[center_h-region_size//2:center_h+region_size//2, 
                     center_w-region_size//2:center_w+region_size//2]
        
        if region.size == 0:
            return 0
        
        # Simplified autocorrelation
        corr = np.correlate(region.flatten(), region.flatten(), mode='full')
        corr = corr[len(corr)//2:]
        
        # Look for peaks indicating repetition
        peaks = []
        for i in range(1, len(corr)-1):
            if corr[i] > corr[i-1] and corr[i] > corr[i+1]:
                peaks.append(corr[i])
        
        if len(peaks) == 0:
            return 0
        
        # Calculate repetition score
        repetition_score = np.mean(peaks) / np.max(corr)
        return min(1, repetition_score * 5)  # Scale up

    def _detect_unrealistic_details(self, img_array):
        """Detect unrealistic details"""
        # Simplified unrealistic detail detection
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Look for overly sharp or blurry regions
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = np.var(laplacian)
        
        # Normalize sharpness
        sharpness_score = min(1, sharpness / 1000)
        
        # Look for unrealistic gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Calculate gradient uniformity
        gradient_uniformity = 1 - (np.std(gradient_magnitude) / np.mean(gradient_magnitude))
        
        detail_score = (sharpness_score + gradient_uniformity) / 2
        return detail_score

    def _check_lighting_consistency(self, img_array):
        """Check lighting consistency"""
        # Simplified lighting consistency check
        if len(img_array.shape) != 3:
            return 0
        
        # Convert to LAB color space
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0]
        
        # Calculate lighting gradients
        grad_x = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1, ksize=3)
        
        # Look for inconsistent lighting patterns
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        lighting_inconsistency = np.std(gradient_magnitude) / np.mean(gradient_magnitude)
        
        return min(1, lighting_inconsistency / 2)

    def _check_perspective_consistency(self, img_array):
        """Check perspective consistency"""
        # Simplified perspective consistency check
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Look for straight lines and check their consistency
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is None:
            return 0
        
        # Analyze line angles for consistency
        angles = []
        for line in lines:
            rho, theta = line[0]
            angles.append(theta)
        
        if len(angles) < 2:
            return 0
        
        # Calculate angle consistency
        angle_std = np.std(angles)
        consistency = 1 - (angle_std / np.pi)
        
        return max(0, consistency)

    def _check_object_consistency(self, img_array):
        """Check object consistency"""
        # Simplified object consistency check
        # This would typically involve object detection and analysis
        # For now, we'll use a simple texture-based approach
        
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Divide image into regions and check consistency
        h, w = gray.shape
        region_size = min(32, h // 8, w // 8)
        
        regions = []
        for i in range(0, h - region_size, region_size):
            for j in range(0, w - region_size, region_size):
                region = gray[i:i+region_size, j:j+region_size]
                regions.append(np.std(region))
        
        if len(regions) < 2:
            return 0
        
        # Calculate region consistency
        region_std = np.std(regions)
        region_mean = np.mean(regions)
        
        if region_mean == 0:
            return 0
        
        consistency = 1 - (region_std / region_mean)
        return max(0, min(1, consistency))

# Global instance
ai_detector = AIDetector()

def detect_ai_generation_simple(image):
    """Simple wrapper function for AI detection"""
    return ai_detector.detect_ai_generation(image) 
"""
Custom AI Detection Integration
Integrates trained custom model into the main application
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
from PIL import Image
import logging
from pathlib import Path
import json
from typing import Dict, Any, Tuple

# Import our custom model
from custom_trainer import CustomAIDetector

logger = logging.getLogger(__name__)

class CustomAIDetectionEngine:
    """Custom AI detection engine using trained model"""
    
    def __init__(self, model_path="./custom_models/best_model.pth"):
        self.model_path = Path(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.transform = None
        self.is_loaded = False
        
        # Initialize transform (must match training transform)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load model if available
        self.load_model()
    
    def load_model(self) -> bool:
        """Load the trained custom model"""
        if not self.model_path.exists():
            logger.warning(f"Custom model not found at {self.model_path}")
            return False
        
        try:
            # Create model
            self.model = CustomAIDetector(num_classes=2, pretrained=False)
            
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Move to device and set to eval mode
            self.model = self.model.to(self.device)
            self.model.eval()
            
            self.is_loaded = True
            logger.info(f"Custom AI detection model loaded successfully from {self.model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load custom model: {e}")
            self.is_loaded = False
            return False
    
    def detect_ai_generation(self, image: Image.Image) -> Dict[str, Any]:
        """Detect AI generation using custom trained model"""
        if not self.is_loaded:
            return self._fallback_detection()
        
        try:
            # Preprocess image
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                outputs, attention = self.model(img_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                
                # Get AI probability (class 1)
                ai_probability = probabilities[0][1].item()
                real_probability = probabilities[0][0].item()
                
                # Get predicted class
                predicted_class = torch.argmax(outputs, dim=1).item()
                
                # Calculate confidence
                confidence = max(ai_probability, real_probability)
                
                # Generate detailed reasoning
                reasoning = self._generate_reasoning(ai_probability, attention)
                
                return {
                    'probability': ai_probability,
                    'confidence': confidence,
                    'predicted_class': predicted_class,
                    'real_probability': real_probability,
                    'model_used': 'Custom Trained AI Detector',
                    'detailed_reasons': reasoning,
                    'attention_map': attention.cpu().numpy() if attention is not None else None
                }
                
        except Exception as e:
            logger.error(f"Error in custom AI detection: {e}")
            return self._fallback_detection()
    
    def _generate_reasoning(self, ai_probability: float, attention_map=None) -> list:
        """Generate detailed reasoning for the prediction"""
        reasons = []
        
        # Base reasoning based on probability
        if ai_probability > 0.8:
            reasons.append("**HIGH CONFIDENCE AI DETECTION**: The custom model is very confident this image was AI-generated")
        elif ai_probability > 0.6:
            reasons.append("**MODERATE AI DETECTION**: The custom model detected significant AI generation patterns")
        elif ai_probability > 0.4:
            reasons.append("**SUSPICIOUS PATTERNS**: The custom model detected some AI-like characteristics")
        else:
            reasons.append("**LIKELY REAL**: The custom model did not detect significant AI generation patterns")
        
        # Attention-based reasoning
        if attention_map is not None:
            attention_std = np.std(attention_map)
            if attention_std > 0.1:
                reasons.append("**ATTENTION ANALYSIS**: The model focused on specific regions, suggesting artificial patterns")
            else:
                reasons.append("**ATTENTION ANALYSIS**: The model's attention was evenly distributed, typical of real images")
        
        # Confidence-based reasoning
        if ai_probability > 0.9:
            reasons.append("**VERY HIGH CONFIDENCE**: Model confidence > 90% - strong AI generation indicators")
        elif ai_probability > 0.7:
            reasons.append("**HIGH CONFIDENCE**: Model confidence > 70% - clear AI generation patterns")
        elif ai_probability > 0.5:
            reasons.append("**MODERATE CONFIDENCE**: Model confidence > 50% - some AI generation indicators")
        else:
            reasons.append("**LOW CONFIDENCE**: Model confidence < 50% - minimal AI generation indicators")
        
        # Training-based reasoning
        reasons.append("**CUSTOM TRAINING**: This model was specifically trained on AI vs real image datasets")
        reasons.append("**SPECIALIZED FEATURES**: Uses attention mechanisms and feature fusion for better detection")
        
        return reasons
    
    def _fallback_detection(self) -> Dict[str, Any]:
        """Fallback detection when custom model is not available"""
        return {
            'probability': 0.5,
            'confidence': 0.5,
            'predicted_class': 0,
            'real_probability': 0.5,
            'model_used': 'Custom Model Not Available (Fallback)',
            'detailed_reasons': [
                "**FALLBACK MODE**: Custom trained model not available",
                "**DEFAULT PROBABILITY**: Using 50% as default probability",
                "**RECOMMENDATION**: Train and load custom model for better accuracy"
            ],
            'attention_map': None
        }

class HybridAIDetector:
    """Hybrid detector combining custom model with existing methods"""
    
    def __init__(self, custom_model_path="./custom_models/best_model.pth"):
        self.custom_engine = CustomAIDetectionEngine(custom_model_path)
        self.use_custom_model = self.custom_engine.is_loaded
        
        # Weights for hybrid approach
        self.custom_weight = 0.7 if self.use_custom_model else 0.0
        self.existing_weight = 0.3 if self.use_custom_model else 1.0
        
        logger.info(f"Hybrid AI Detector initialized - Custom model: {self.use_custom_model}")
    
    def detect_ai_generation(self, image: Image.Image, existing_results: Dict[str, Any] = None) -> Dict[str, Any]:
        """Hybrid AI detection combining custom model with existing methods"""
        
        # Get custom model prediction
        custom_result = self.custom_engine.detect_ai_generation(image)
        
        # If no existing results provided, use custom model only
        if existing_results is None:
            return custom_result
        
        # Combine results
        if self.use_custom_model:
            # Weighted combination
            combined_probability = (
                self.custom_weight * custom_result['probability'] +
                self.existing_weight * existing_results.get('probability', 0.5)
            )
            
            # Combine reasoning
            combined_reasons = custom_result['detailed_reasons'] + [
                "**HYBRID APPROACH**: Combined custom trained model with existing methods",
                f"**CUSTOM WEIGHT**: {self.custom_weight:.1%} (trained model)",
                f"**EXISTING WEIGHT**: {self.existing_weight:.1%} (rule-based methods)"
            ]
            
            # Add existing method details
            if 'detailed_reasons' in existing_results:
                combined_reasons.extend([
                    "**EXISTING METHODS**:",
                    *existing_results['detailed_reasons'][:3]  # First 3 reasons from existing
                ])
            
            return {
                'probability': combined_probability,
                'confidence': max(custom_result['confidence'], existing_results.get('confidence', 0.5)),
                'predicted_class': 1 if combined_probability > 0.5 else 0,
                'real_probability': 1 - combined_probability,
                'model_used': f'Hybrid: Custom Trained ({self.custom_weight:.1%}) + Existing ({self.existing_weight:.1%})',
                'detailed_reasons': combined_reasons,
                'custom_probability': custom_result['probability'],
                'existing_probability': existing_results.get('probability', 0.5),
                'attention_map': custom_result.get('attention_map')
            }
        else:
            # Use existing results only
            return existing_results

def integrate_custom_model_into_app():
    """Integration function to use custom model in main app"""
    
    # Create hybrid detector
    hybrid_detector = HybridAIDetector()
    
    if hybrid_detector.use_custom_model:
        logger.info("✅ Custom trained model successfully integrated!")
        return hybrid_detector
    else:
        logger.warning("⚠️ Custom model not found - using existing methods only")
        return None

# Global instance for app integration
custom_detector = None

def initialize_custom_detector():
    """Initialize custom detector for app use"""
    global custom_detector
    custom_detector = integrate_custom_model_into_app()
    return custom_detector is not None

def detect_ai_with_custom_model(image: Image.Image, existing_results: Dict[str, Any] = None) -> Dict[str, Any]:
    """Main function for app integration"""
    global custom_detector
    
    if custom_detector is None:
        # Fallback to existing results
        return existing_results if existing_results else {
            'probability': 0.5,
            'confidence': 0.5,
            'model_used': 'Custom Model Not Available',
            'detailed_reasons': ['Custom model not loaded']
        }
    
    return custom_detector.detect_ai_generation(image, existing_results)

if __name__ == "__main__":
    # Test the custom detector
    print("Testing Custom AI Detection Engine...")
    
    # Initialize
    success = initialize_custom_detector()
    print(f"Custom detector initialized: {success}")
    
    if success:
        # Create a test image (you would load a real image here)
        test_image = Image.new('RGB', (224, 224), color='red')
        
        # Test detection
        result = detect_ai_with_custom_model(test_image)
        print(f"Test result: {result}")
    else:
        print("Custom model not available - check model path and training") 
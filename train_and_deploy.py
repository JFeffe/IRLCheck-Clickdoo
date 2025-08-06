"""
Train and Deploy Custom AI Detection Model
Automated script to collect data, train model, and integrate into app
"""

import os
import sys
import logging
import argparse
from pathlib import Path
import json
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Main training and deployment script"""
    parser = argparse.ArgumentParser(description='Train and deploy custom AI detection model')
    parser.add_argument('--mode', choices=['collect', 'train', 'deploy', 'full'], 
                       default='full', help='Operation mode')
    parser.add_argument('--ai-count', type=int, default=1000, 
                       help='Number of AI images to collect')
    parser.add_argument('--real-count', type=int, default=1000, 
                       help='Number of real images to collect')
    parser.add_argument('--epochs', type=int, default=30, 
                       help='Number of training epochs')
    parser.add_argument('--data-dir', default='./training_data', 
                       help='Training data directory')
    parser.add_argument('--model-dir', default='./custom_models', 
                       help='Model output directory')
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting IRLCheck Custom AI Detection Training Pipeline")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"AI Images: {args.ai_count}")
    logger.info(f"Real Images: {args.real_count}")
    logger.info(f"Epochs: {args.epochs}")
    
    # Create directories
    Path(args.data_dir).mkdir(parents=True, exist_ok=True)
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    
    if args.mode in ['collect', 'full']:
        collect_data(args)
    
    if args.mode in ['train', 'full']:
        train_model(args)
    
    if args.mode in ['deploy', 'full']:
        deploy_model(args)
    
    logger.info("✅ Training and deployment pipeline completed!")

def collect_data(args):
    """Collect training data"""
    logger.info("📥 Starting data collection...")
    
    try:
        from data_collector import DataCollector
        
        # Initialize collector
        collector = DataCollector(data_dir=args.data_dir)
        
        # Collect AI-generated images
        logger.info("🤖 Collecting AI-generated images...")
        ai_collected = collector.collect_ai_generated_images(args.ai_count)
        
        # Collect real images
        logger.info("📸 Collecting real images...")
        real_collected = collector.collect_real_images(args.real_count)
        
        # Create dataset
        logger.info("📊 Creating training dataset...")
        dataset_info = collector.create_training_dataset(
            ai_count=ai_collected, 
            real_count=real_collected
        )
        
        # Validate dataset
        logger.info("🔍 Validating dataset...")
        validation = collector.validate_dataset()
        
        logger.info(f"✅ Data collection completed!")
        logger.info(f"AI images: {validation['ai_images']}")
        logger.info(f"Real images: {validation['real_images']}")
        logger.info(f"Total: {validation['ai_images'] + validation['real_images']}")
        
        # Save collection report
        report = {
            'collection_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'dataset_info': dataset_info,
            'validation': validation,
            'settings': vars(args)
        }
        
        with open(Path(args.data_dir) / 'collection_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        return validation
        
    except Exception as e:
        logger.error(f"❌ Data collection failed: {e}")
        raise

def train_model(args):
    """Train custom model"""
    logger.info("🧠 Starting model training...")
    
    try:
        from custom_trainer import train_custom_model
        
        # Check if we have enough data
        data_dir = Path(args.data_dir)
        ai_files = list((data_dir / "ai_generated").glob("*.jpg")) + \
                  list((data_dir / "ai_generated").glob("*.png"))
        real_files = list((data_dir / "real").glob("*.jpg")) + \
                    list((data_dir / "real").glob("*.png"))
        
        total_images = len(ai_files) + len(real_files)
        
        if total_images < 100:
            logger.warning(f"⚠️ Only {total_images} images found. Consider collecting more data.")
            logger.info("Continuing with available data...")
        
        # Train model
        trainer, results = train_custom_model(
            data_dir=args.data_dir,
            epochs=args.epochs
        )
        
        logger.info(f"✅ Model training completed!")
        logger.info(f"Final test accuracy: {results['accuracy']:.4f}")
        logger.info(f"Precision: {results['precision']:.4f}")
        logger.info(f"Recall: {results['recall']:.4f}")
        logger.info(f"F1-Score: {results['f1_score']:.4f}")
        
        # Save training report
        training_report = {
            'training_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'results': results,
            'settings': vars(args),
            'model_path': str(Path(args.model_dir) / 'best_model.pth')
        }
        
        with open(Path(args.model_dir) / 'training_report.json', 'w') as f:
            json.dump(training_report, f, indent=2)
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Model training failed: {e}")
        raise

def deploy_model(args):
    """Deploy model to main application"""
    logger.info("🚀 Deploying model to main application...")
    
    try:
        # Check if model exists
        model_path = Path(args.model_dir) / 'best_model.pth'
        if not model_path.exists():
            logger.error(f"❌ Model not found at {model_path}")
            logger.info("Please train the model first using --mode train")
            return False
        
        # Test custom detector
        from custom_ai_detection import initialize_custom_detector
        
        success = initialize_custom_detector()
        
        if success:
            logger.info("✅ Custom model successfully integrated into application!")
            
            # Create deployment report
            deployment_report = {
                'deployment_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'model_path': str(model_path),
                'integration_success': True,
                'next_steps': [
                    "1. Restart your Streamlit application",
                    "2. Test with known AI and real images",
                    "3. Monitor performance and accuracy",
                    "4. Consider retraining with more data if needed"
                ]
            }
        else:
            logger.warning("⚠️ Custom model integration failed - using existing methods")
            deployment_report = {
                'deployment_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'model_path': str(model_path),
                'integration_success': False,
                'error': 'Model integration failed',
                'fallback': 'Using existing AI detection methods'
            }
        
        # Save deployment report
        with open(Path(args.model_dir) / 'deployment_report.json', 'w') as f:
            json.dump(deployment_report, f, indent=2)
        
        return success
        
    except Exception as e:
        logger.error(f"❌ Model deployment failed: {e}")
        raise

def check_requirements():
    """Check if all required packages are installed"""
    logger.info("🔍 Checking requirements...")
    
    required_packages = [
        'torch', 'torchvision', 'transformers', 'scikit-learn',
        'matplotlib', 'seaborn', 'pillow', 'numpy', 'requests'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            logger.error(f"❌ {package} - MISSING")
    
    if missing_packages:
        logger.error(f"❌ Missing packages: {', '.join(missing_packages)}")
        logger.info("Please install missing packages:")
        logger.info(f"pip install {' '.join(missing_packages)}")
        return False
    
    logger.info("✅ All requirements satisfied!")
    return True

def show_usage_instructions():
    """Show usage instructions"""
    logger.info("\n📖 USAGE INSTRUCTIONS:")
    logger.info("=" * 50)
    logger.info("1. COLLECT DATA:")
    logger.info("   python train_and_deploy.py --mode collect --ai-count 1000 --real-count 1000")
    logger.info("")
    logger.info("2. TRAIN MODEL:")
    logger.info("   python train_and_deploy.py --mode train --epochs 30")
    logger.info("")
    logger.info("3. DEPLOY MODEL:")
    logger.info("   python train_and_deploy.py --mode deploy")
    logger.info("")
    logger.info("4. FULL PIPELINE:")
    logger.info("   python train_and_deploy.py --mode full")
    logger.info("")
    logger.info("5. CUSTOM SETTINGS:")
    logger.info("   python train_and_deploy.py --mode full --ai-count 2000 --real-count 2000 --epochs 50")
    logger.info("=" * 50)

if __name__ == "__main__":
    try:
        # Check requirements first
        if not check_requirements():
            logger.error("❌ Requirements not met. Please install missing packages.")
            sys.exit(1)
        
        # Show instructions
        show_usage_instructions()
        
        # Run main pipeline
        main()
        
    except KeyboardInterrupt:
        logger.info("⏹️ Training interrupted by user")
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        sys.exit(1) 
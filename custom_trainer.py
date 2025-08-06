"""
Custom AI Detection Model Trainer
Trains a specialized model for AI image detection
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import numpy as np
from PIL import Image
import os
import json
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIImageDataset(Dataset):
    """Custom dataset for AI vs Real image classification"""
    
    def __init__(self, data_dir, transform=None, split='train'):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.split = split
        
        # Load dataset info
        info_file = self.data_dir / "dataset_info.json"
        if info_file.exists():
            with open(info_file, 'r') as f:
                self.dataset_info = json.load(f)
        else:
            self.dataset_info = {}
        
        # Get file paths
        self.ai_files = list((self.data_dir / "ai_generated").glob("*.jpg")) + \
                       list((self.data_dir / "ai_generated").glob("*.png"))
        self.real_files = list((self.data_dir / "real").glob("*.jpg")) + \
                         list((self.data_dir / "real").glob("*.png"))
        
        # Create labels (0 = real, 1 = AI-generated)
        self.images = []
        self.labels = []
        
        # Add AI images (label 1)
        for file in self.ai_files:
            self.images.append(str(file))
            self.labels.append(1)
        
        # Add real images (label 0)
        for file in self.real_files:
            self.images.append(str(file))
            self.labels.append(0)
        
        # Split data
        if split in ['train', 'val']:
            train_imgs, val_imgs, train_labels, val_labels = train_test_split(
                self.images, self.labels, test_size=0.2, random_state=42, stratify=self.labels
            )
            
            if split == 'train':
                self.images = train_imgs
                self.labels = train_labels
            else:  # val
                self.images = val_imgs
                self.labels = val_labels
        
        logger.info(f"{split} dataset: {len(self.images)} images ({sum(self.labels)} AI, {len(self.labels)-sum(self.labels)} real)")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
            
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            # Return a placeholder image
            placeholder = torch.zeros(3, 224, 224)
            return placeholder, label

class CustomAIDetector(nn.Module):
    """Custom neural network for AI image detection"""
    
    def __init__(self, num_classes=2, pretrained=True):
        super(CustomAIDetector, self).__init__()
        
        # Use a pre-trained backbone (ResNet50)
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Freeze early layers for transfer learning
        for param in list(self.backbone.parameters())[:-20]:
            param.requires_grad = False
        
        # Replace the final layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
        # Add attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(2048, 512, 1),
            nn.ReLU(),
            nn.Conv2d(512, 1, 1),
            nn.Sigmoid()
        )
        
        # Feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Linear(num_features + 49, 512),  # 49 = 7x7 attention map
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # Extract features from backbone
        features = self.backbone.conv1(x)
        features = self.backbone.bn1(features)
        features = self.backbone.relu(features)
        features = self.backbone.maxpool(features)
        
        features = self.backbone.layer1(features)
        features = self.backbone.layer2(features)
        features = self.backbone.layer3(features)
        features = self.backbone.layer4(features)
        
        # Apply attention
        attention_weights = self.attention(features)
        attended_features = features * attention_weights
        
        # Global average pooling
        pooled_features = torch.mean(attended_features, dim=[2, 3])
        
        # Flatten attention weights
        attention_flat = torch.mean(attention_weights, dim=1).view(x.size(0), -1)
        
        # Concatenate features
        combined_features = torch.cat([pooled_features, attention_flat], dim=1)
        
        # Final classification
        output = self.feature_fusion(combined_features)
        
        return output, attention_weights

class CustomTrainer:
    """Trainer for custom AI detection model"""
    
    def __init__(self, data_dir="./training_data", model_dir="./custom_models"):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Data transforms
        self.train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Model and training parameters
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

    def prepare_data(self):
        """Prepare training and validation datasets"""
        logger.info("Preparing datasets...")
        
        # Create datasets
        self.train_dataset = AIImageDataset(
            self.data_dir, 
            transform=self.train_transform, 
            split='train'
        )
        
        self.val_dataset = AIImageDataset(
            self.data_dir, 
            transform=self.val_transform, 
            split='val'
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=32, 
            shuffle=True, 
            num_workers=4,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset, 
            batch_size=32, 
            shuffle=False, 
            num_workers=4,
            pin_memory=True
        )
        
        logger.info(f"Train loader: {len(self.train_loader)} batches")
        logger.info(f"Val loader: {len(self.val_loader)} batches")

    def setup_model(self, learning_rate=0.001):
        """Setup model, loss function, and optimizer"""
        logger.info("Setting up model...")
        
        # Create model
        self.model = CustomAIDetector(num_classes=2, pretrained=True)
        self.model = self.model.to(self.device)
        
        # Loss function (weighted for imbalanced datasets)
        # Calculate class weights
        ai_count = sum(self.train_dataset.labels)
        real_count = len(self.train_dataset.labels) - ai_count
        total_count = len(self.train_dataset.labels)
        
        class_weights = torch.tensor([
            ai_count / total_count,  # Weight for real class (0)
            real_count / total_count  # Weight for AI class (1)
        ]).to(self.device)
        
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=3, 
            verbose=True
        )
        
        logger.info(f"Model setup complete. Class weights: {class_weights}")

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs, attention = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if batch_idx % 10 == 0:
                logger.info(f"Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}")
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc

    def validate_epoch(self):
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs, attention = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100 * correct / total
        
        # Calculate additional metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        return epoch_loss, epoch_acc, precision, recall, f1

    def train(self, epochs=50, early_stopping_patience=10):
        """Train the model"""
        logger.info(f"Starting training for {epochs} epochs...")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            logger.info(f"\nEpoch {epoch+1}/{epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc, precision, recall, f1 = self.validate_epoch()
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Store history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Log results
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            logger.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_model("best_model.pth")
                logger.info("New best model saved!")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping after {epoch+1} epochs")
                break
        
        # Save final model
        self.save_model("final_model.pth")
        logger.info("Training completed!")

    def save_model(self, filename):
        """Save model to file"""
        model_path = self.model_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'model_config': {
                'num_classes': 2,
                'pretrained': True
            }
        }, model_path)
        logger.info(f"Model saved to {model_path}")

    def load_model(self, filename):
        """Load model from file"""
        model_path = self.model_dir / filename
        if model_path.exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Setup model first
            self.setup_model()
            
            # Load state dict
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load history
            self.train_losses = checkpoint.get('train_losses', [])
            self.val_losses = checkpoint.get('val_losses', [])
            self.train_accuracies = checkpoint.get('train_accuracies', [])
            self.val_accuracies = checkpoint.get('val_accuracies', [])
            
            logger.info(f"Model loaded from {model_path}")
            return True
        else:
            logger.error(f"Model file not found: {model_path}")
            return False

    def plot_training_history(self):
        """Plot training history"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Val Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy
        ax2.plot(self.train_accuracies, label='Train Accuracy')
        ax2.plot(self.val_accuracies, label='Val Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        # Learning rate
        if hasattr(self.scheduler, 'get_last_lr'):
            lrs = [self.scheduler.get_last_lr()[0]] * len(self.train_losses)
            ax3.plot(lrs)
            ax3.set_title('Learning Rate')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Learning Rate')
            ax3.grid(True)
        
        # Final metrics
        if self.val_accuracies:
            final_acc = self.val_accuracies[-1]
            ax4.bar(['Final Validation Accuracy'], [final_acc])
            ax4.set_title(f'Final Validation Accuracy: {final_acc:.2f}%')
            ax4.set_ylabel('Accuracy (%)')
            ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.model_dir / 'training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

    def evaluate_model(self, test_data_dir=None):
        """Evaluate model on test data"""
        if test_data_dir is None:
            test_data_dir = self.data_dir
        
        # Create test dataset
        test_dataset = AIImageDataset(
            test_data_dir, 
            transform=self.val_transform, 
            split='val'  # Use validation split for testing
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=32, 
            shuffle=False, 
            num_workers=4
        )
        
        # Evaluate
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs, attention = self.model(images)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        logger.info(f"Test Results:")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1-Score: {f1:.4f}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probabilities
        }

def train_custom_model(data_dir="./training_data", epochs=50):
    """Main function to train custom AI detection model"""
    logger.info("Starting custom AI detection model training...")
    
    # Initialize trainer
    trainer = CustomTrainer(data_dir=data_dir)
    
    # Prepare data
    trainer.prepare_data()
    
    # Setup model
    trainer.setup_model()
    
    # Train model
    trainer.train(epochs=epochs)
    
    # Plot training history
    trainer.plot_training_history()
    
    # Evaluate model
    results = trainer.evaluate_model()
    
    logger.info("Custom model training completed!")
    return trainer, results

if __name__ == "__main__":
    # Example usage
    trainer, results = train_custom_model(epochs=30)
    print(f"Final test accuracy: {results['accuracy']:.4f}") 
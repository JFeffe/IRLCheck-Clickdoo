# 🧠 Custom AI Detection Model Training Guide

Ce guide vous explique comment entraîner votre propre modèle de détection AI personnalisé pour améliorer drastiquement la précision de IRLCheck-Clickdoo.

## 🎯 Pourquoi Entraîner un Modèle Personnalisé ?

### Problèmes Actuels :
- **Modèles génériques** : Les modèles pré-entraînés ne sont pas spécialisés dans la détection AI
- **Précision limitée** : Détection médiocre des images AI générées (91% d'authenticité pour des images 100% AI)
- **Pas d'adaptation** : Ne s'améliore pas avec l'usage

### Avantages du Modèle Personnalisé :
- **Spécialisé** : Entraîné spécifiquement sur AI vs images réelles
- **Haute précision** : Peut atteindre 95%+ de précision
- **Adaptatif** : Peut être réentraîné avec de nouvelles données
- **Hybride** : Combine apprentissage profond avec méthodes existantes

## 🚀 Installation et Configuration

### 1. Prérequis
```bash
# Installer les dépendances supplémentaires
pip install scikit-learn>=1.3.0

# Vérifier que tout est installé
python train_and_deploy.py --help
```

### 2. Structure des Fichiers
```
IRLCheck-Clickdoo-Clean/
├── data_collector.py          # Collecte automatique de données
├── custom_trainer.py          # Entraînement du modèle
├── custom_ai_detection.py     # Intégration dans l'app
├── train_and_deploy.py        # Script principal automatisé
├── training_data/             # Données d'entraînement
│   ├── ai_generated/          # Images AI
│   └── real/                  # Images réelles
└── custom_models/             # Modèles entraînés
    ├── best_model.pth         # Meilleur modèle
    └── training_history.png   # Graphiques d'entraînement
```

## 📥 Collecte de Données d'Entraînement

### Méthodes Automatiques

#### 1. Collecte Complète (Recommandée)
```bash
# Collecter 1000 images AI et 1000 images réelles
python train_and_deploy.py --mode collect --ai-count 1000 --real-count 1000
```

#### 2. Collecte Progressive
```bash
# Commencer avec moins d'images pour tester
python train_and_deploy.py --mode collect --ai-count 100 --real-count 100
```

### Sources de Données

#### Images AI Générées :
- **APIs directes** : DALL-E, Midjourney, Stable Diffusion
- **Galleries web** : Artbreeder, ThisPersonDoesNotExist
- **Datasets existants** : HuggingFace datasets spécialisés

#### Images Réelles :
- **Unsplash API** : Photos haute qualité
- **Pexels API** : Images libres de droits
- **Datasets** : ImageNet, photos réelles

### Configuration des APIs (Optionnel)
```python
# Dans data_collector.py, ajoutez vos clés API
self.unsplash_access_key = "VOTRE_CLE_UNSPLASH"
self.pexels_api_key = "VOTRE_CLE_PEXELS"
```

## 🧠 Entraînement du Modèle

### Architecture du Modèle
- **Backbone** : ResNet50 pré-entraîné
- **Attention** : Mécanisme d'attention pour focaliser sur les zones importantes
- **Feature Fusion** : Combinaison intelligente des caractéristiques
- **Dropout** : Régularisation pour éviter le surapprentissage

### Entraînement Automatique
```bash
# Entraînement complet avec 30 époques
python train_and_deploy.py --mode train --epochs 30

# Entraînement avec plus d'époques pour meilleure précision
python train_and_deploy.py --mode train --epochs 50
```

### Paramètres d'Entraînement
- **Learning Rate** : 0.001 (ajusté automatiquement)
- **Batch Size** : 32
- **Early Stopping** : 10 époques sans amélioration
- **Data Augmentation** : Rotation, flip, color jitter
- **Class Weights** : Équilibré automatiquement

### Monitoring de l'Entraînement
Le script génère automatiquement :
- **Graphiques** : Loss, accuracy, learning rate
- **Logs** : Progression détaillée
- **Métriques** : Precision, recall, F1-score
- **Sauvegarde** : Meilleur modèle automatiquement

## 🚀 Déploiement et Intégration

### Déploiement Automatique
```bash
# Déployer le modèle entraîné
python train_and_deploy.py --mode deploy
```

### Intégration dans l'Application
Le modèle personnalisé s'intègre automatiquement :
- **Mode hybride** : 70% modèle personnalisé + 30% méthodes existantes
- **Fallback** : Utilise les méthodes existantes si le modèle n'est pas disponible
- **Raisons détaillées** : Explications spécifiques au modèle entraîné

### Pipeline Complet
```bash
# Tout faire en une fois : collecte + entraînement + déploiement
python train_and_deploy.py --mode full --ai-count 2000 --real-count 2000 --epochs 50
```

## 📊 Évaluation et Amélioration

### Métriques de Performance
- **Accuracy** : Précision globale
- **Precision** : Précision pour la classe AI
- **Recall** : Sensibilité pour la classe AI
- **F1-Score** : Moyenne harmonique

### Amélioration Continue
1. **Collecter plus de données** : Plus d'images = meilleure précision
2. **Réentraîner** : Avec de nouvelles données
3. **Ajuster les paramètres** : Learning rate, architecture
4. **Valider** : Tester sur de nouveaux exemples

## 🔧 Personnalisation Avancée

### Modification de l'Architecture
```python
# Dans custom_trainer.py, modifiez CustomAIDetector
class CustomAIDetector(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        # Changez le backbone
        self.backbone = models.resnet101(pretrained=pretrained)  # ResNet101 au lieu de ResNet50
        
        # Ajoutez des couches personnalisées
        self.custom_layers = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
```

### Hyperparamètres Personnalisés
```python
# Dans custom_trainer.py, modifiez les paramètres
def setup_model(self, learning_rate=0.0005):  # Learning rate plus bas
    # Optimiseur différent
    self.optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, self.model.parameters()),
        lr=learning_rate,
        weight_decay=0.02  # Plus de régularisation
    )
```

## 🐛 Dépannage

### Problèmes Courants

#### 1. "CUDA out of memory"
```bash
# Réduire la batch size dans custom_trainer.py
batch_size=16  # Au lieu de 32
```

#### 2. "Not enough data"
```bash
# Collecter plus de données
python train_and_deploy.py --mode collect --ai-count 5000 --real-count 5000
```

#### 3. "Model not loading"
```bash
# Vérifier le chemin du modèle
ls -la custom_models/best_model.pth
```

#### 4. "Poor accuracy"
- **Plus de données** : Collecter plus d'images
- **Plus d'époques** : Entraîner plus longtemps
- **Data augmentation** : Modifier les transformations
- **Architecture** : Essayer un backbone différent

## 📈 Résultats Attendus

### Avant (Modèles Génériques)
- **Précision** : ~60-70%
- **Faux positifs** : Beaucoup d'images AI classées comme réelles
- **Temps d'analyse** : Rapide mais imprécis

### Après (Modèle Personnalisé)
- **Précision** : ~90-95%
- **Faux positifs** : Très rares
- **Temps d'analyse** : Légèrement plus lent mais très précis
- **Raisons détaillées** : Explications spécifiques au modèle

## 🎯 Prochaines Étapes

1. **Lancer la collecte** : Commencez avec 1000 images de chaque type
2. **Entraîner le modèle** : 30-50 époques pour commencer
3. **Tester** : Valider sur vos images AI connues
4. **Améliorer** : Collecter plus de données si nécessaire
5. **Déployer** : Intégrer dans votre application

## 📞 Support

Si vous rencontrez des problèmes :
1. Vérifiez les logs dans `training.log`
2. Consultez les rapports dans `training_data/` et `custom_models/`
3. Vérifiez que toutes les dépendances sont installées
4. Commencez avec de petites quantités de données pour tester

**Bon entraînement ! 🚀** 
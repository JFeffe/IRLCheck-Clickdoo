# 🚀 IRLCheck-Clickdoo Deployment Guide

## 📋 Prérequis

- **GitHub Account**: Pour héberger le code
- **Streamlit Cloud Account**: Pour le déploiement (gratuit)
- **Git**: Installé localement

## 🌐 Déploiement sur Streamlit Cloud (Recommandé)

### Étape 1: Préparer le Repository GitHub

1. **Créer un nouveau repository sur GitHub**
   - Allez sur [github.com](https://github.com)
   - Cliquez sur "New repository"
   - Nommez-le `IRLCheck-Clickdoo`
   - Choisissez "Public" ou "Private"

2. **Pousser le code vers GitHub**
   ```bash
   # Dans le dossier IRLCheck-Clickdoo_v1.0
   git init
   git add .
   git commit -m "Initial commit: IRLCheck-Clickdoo v1.0"
   git branch -M main
   git remote add origin https://github.com/VOTRE_USERNAME/IRLCheck-Clickdoo.git
   git push -u origin main
   ```

### Étape 2: Déployer sur Streamlit Cloud

1. **Aller sur Streamlit Cloud**
   - Visitez [share.streamlit.io](https://share.streamlit.io)
   - Connectez-vous avec votre compte GitHub

2. **Créer une nouvelle application**
   - Cliquez sur "New app"
   - Sélectionnez votre repository: `VOTRE_USERNAME/IRLCheck-Clickdoo`
   - Définissez le chemin du fichier principal: `streamlit_app.py`
   - Cliquez sur "Deploy!"

3. **Attendre le déploiement**
   - Le premier déploiement peut prendre 5-10 minutes
   - Les modèles IA seront téléchargés automatiquement

### Étape 3: Accéder à l'application

- **URL**: `https://votre-app-name.streamlit.app`
- **Statut**: Vérifiez que l'application fonctionne correctement

## 🐳 Déploiement avec Docker (Optionnel)

### Déploiement Local avec Docker

```bash
# Construire et lancer avec docker-compose
docker-compose up --build

# Ou avec Docker directement
docker build -t irlcheck .
docker run -p 8501:8501 irlcheck
```

### Déploiement sur Serveur

```bash
# Cloner le repository
git clone https://github.com/VOTRE_USERNAME/IRLCheck-Clickdoo.git
cd IRLCheck-Clickdoo

# Lancer avec docker-compose
docker-compose up -d

# Vérifier les logs
docker-compose logs -f
```

## 🔧 Configuration Avancée

### Variables d'Environnement

Créez un fichier `.env` pour les variables d'environnement :

```env
# AI Model Configuration
AI_CACHE_DIR=./models
AI_USE_GPU=false
AI_MAX_MEMORY=4GB

# Application Settings
MAX_FILE_SIZE=100
DEBUG_MODE=false

# Performance
ENABLE_CACHING=true
CACHE_TTL=3600
```

### Configuration Streamlit

Le fichier `.streamlit/config.toml` contient :

```toml
[global]
developmentMode = false

[server]
headless = true
port = 8501
enableCORS = true
enableXsrfProtection = true

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
```

## 📊 Monitoring et Maintenance

### Vérifier les Logs

```bash
# Streamlit Cloud
# Les logs sont disponibles dans l'interface Streamlit Cloud

# Docker
docker-compose logs -f irlcheck
```

### Mettre à Jour l'Application

```bash
# Pull les dernières modifications
git pull origin main

# Redéployer sur Streamlit Cloud
# Les mises à jour sont automatiques

# Ou avec Docker
docker-compose down
docker-compose up --build -d
```

## 🚨 Dépannage

### Problèmes Courants

1. **Erreur de Port**
   - Vérifiez que le port 8501 est disponible
   - Changez le port dans `config.toml` si nécessaire

2. **Modèles IA non chargés**
   - Vérifiez la connexion internet
   - Les modèles se téléchargent automatiquement au premier lancement

3. **Erreur de Mémoire**
   - Réduisez `AI_MAX_MEMORY` dans les variables d'environnement
   - Utilisez des modèles plus légers

4. **Problèmes de Dépendances**
   - Vérifiez que `requirements.txt` est à jour
   - Reconstruisez l'image Docker si nécessaire

### Support

- **Documentation**: Consultez le `README.md`
- **Issues**: Créez une issue sur GitHub
- **Logs**: Vérifiez les logs de déploiement

## 🎯 Prochaines Étapes

Après le déploiement réussi :

1. **Tester l'application** avec différentes images
2. **Configurer un domaine personnalisé** (optionnel)
3. **Mettre en place le monitoring** (optionnel)
4. **Optimiser les performances** selon l'usage

## 📞 Support

Pour toute question ou problème :
- Créez une issue sur GitHub
- Consultez la documentation
- Contactez l'équipe de développement

---

**🎉 Félicitations ! Votre application IRLCheck-Clickdoo est maintenant déployée et accessible en ligne !** 
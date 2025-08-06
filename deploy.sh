#!/bin/bash

# IRLCheck-Clickdoo Deployment Script
# This script helps deploy the application to Streamlit Cloud

echo "🚀 IRLCheck-Clickdoo Deployment Script"
echo "======================================"

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "📁 Initializing Git repository..."
    git init
    git add .
    git commit -m "Initial commit: IRLCheck-Clickdoo v1.0"
    echo "✅ Git repository initialized"
else
    echo "✅ Git repository already exists"
fi

# Check if remote origin exists
if ! git remote get-url origin > /dev/null 2>&1; then
    echo "⚠️  No remote origin found!"
    echo "Please add your GitHub repository as origin:"
    echo "git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git"
    echo ""
    echo "Then run: git push -u origin main"
else
    echo "✅ Remote origin configured"
    echo "📤 Pushing to GitHub..."
    git add .
    git commit -m "Update: Prepare for Streamlit Cloud deployment"
    git push origin main
    echo "✅ Code pushed to GitHub"
fi

echo ""
echo "🌐 Next Steps for Streamlit Cloud Deployment:"
echo "============================================="
echo "1. Go to https://share.streamlit.io"
echo "2. Sign in with your GitHub account"
echo "3. Click 'New app'"
echo "4. Select your repository: YOUR_USERNAME/YOUR_REPO"
echo "5. Set main file path: streamlit_app.py"
echo "6. Click 'Deploy!'"
echo ""
echo "📋 Deployment Checklist:"
echo "✅ All files committed to Git"
echo "✅ streamlit_app.py created"
echo "✅ requirements.txt updated"
echo "✅ .streamlit/config.toml configured"
echo "✅ packages.txt created"
echo "✅ README.md updated"
echo ""
echo "🎉 Your app will be available at: https://your-app-name.streamlit.app"
echo ""
echo "📞 Need help? Check the README.md for detailed instructions!" 
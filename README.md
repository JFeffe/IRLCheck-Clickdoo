# IRLCheck-Clickdoo 🔍

**Advanced Image Authenticity Detection Tool with AI**

IRLCheck-Clickdoo is a comprehensive web application that analyzes images for authenticity using multiple detection methods including metadata analysis, editing detection, and AI generation detection.

## ✨ Features

### 🔍 **Core Analysis**
- **Metadata Extraction**: EXIF data, GPS coordinates, camera info, software used
- **Editing Detection**: Photoshop traces, recompression artifacts, Error Level Analysis (ELA)
- **AI Detection**: Advanced AI generation detection using multiple models

### 🤖 **AI Detection Methods**
- **Statistical Analysis**: Frequency domain, texture, noise, and edge analysis
- **Deep Learning**: ResNet-50 model for AI detection
- **CLIP Analysis**: Text-image comparison for authenticity assessment
- **Combined Results**: Weighted average of multiple detection methods

### 📊 **Results & Reporting**
- **Real-time Analysis**: Instant results with progress tracking
- **Detailed Reports**: Comprehensive analysis with confidence scores
- **PDF Export**: Generate detailed reports for documentation
- **Visual Analysis**: Multiple analysis views and visualizations

### 🎨 **User Interface**
- **Drag & Drop**: Simple file upload interface
- **Modern Design**: Beautiful, responsive UI with custom styling
- **Multi-tab Results**: Organized results in separate tabs
- **Progress Tracking**: Real-time analysis progress

## 🚀 Quick Start

### Local Development

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd IRLCheck-Clickdoo_v1.0
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Open in browser**
   - Local: http://localhost:8501
   - Network: http://your-ip:8501

### Streamlit Cloud Deployment

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Initial deployment"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub repository
   - Set main file path: `streamlit_app.py`
   - Deploy!

## 📁 Project Structure

```
IRLCheck-Clickdoo_v1.0/
├── app.py                 # Main application file
├── ai_detection.py        # AI detection module
├── streamlit_app.py       # Streamlit Cloud entry point
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── .streamlit/           # Streamlit configuration
│   └── config.toml
├── .gitignore           # Git ignore rules
└── models/              # AI model cache (auto-created)
```

## 🔧 Technical Details

### **Supported File Formats**
- **Images**: PNG, JPG, JPEG, WEBP
- **Max Size**: 100MB per file
- **Analysis**: Real-time processing

### **AI Models Used**
- **ResNet-50**: Microsoft's pre-trained model for image classification
- **CLIP**: OpenAI's vision-language model for text-image comparison
- **Statistical Analysis**: Custom algorithms for frequency and texture analysis

### **Detection Methods**
1. **Metadata Analysis**: EXIF data extraction and validation
2. **Editing Detection**: Noise analysis and compression artifacts
3. **AI Generation Detection**: Multi-model approach with confidence scoring

## 📊 Analysis Results

### **Summary Tab**
- Overall authenticity score
- AI generation risk percentage
- Editing detection probability
- Confidence levels for each method

### **Metadata Tab**
- Complete EXIF data display
- GPS coordinates (if available)
- Camera and software information
- File properties

### **Visual Analysis Tab**
- Detailed analysis results
- Technical indicators
- Visual representations
- Statistical data

### **AI Analysis Tab**
- AI detection probability
- Model confidence scores
- Detection methods used
- Technical details and explanations

## 🌐 Deployment

### **Streamlit Cloud**
- **URL**: [Your Streamlit Cloud URL]
- **Status**: Ready for deployment
- **Configuration**: Optimized for cloud deployment

### **Local Deployment**
- **Port**: 8501 (configurable)
- **Requirements**: Python 3.8+, 4GB+ RAM
- **Dependencies**: All included in requirements.txt

## 🔮 Future Enhancements

### **Planned Features**
- **Batch Processing**: Analyze multiple images simultaneously
- **API Integration**: REST API for external applications
- **Advanced Models**: More sophisticated AI detection models
- **Performance Optimization**: Faster processing and caching
- **User Authentication**: Multi-user support with history

### **Technical Improvements**
- **GPU Acceleration**: CUDA support for faster AI processing
- **Model Optimization**: Quantized models for better performance
- **Caching System**: Intelligent caching for repeated analyses
- **Error Handling**: Enhanced error recovery and user feedback

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Streamlit**: For the amazing web framework
- **Hugging Face**: For pre-trained AI models
- **OpenCV**: For image processing capabilities
- **Pillow**: For image manipulation
- **PyTorch**: For deep learning capabilities

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Documentation**: [Project Wiki](https://github.com/your-repo/wiki)
- **Contact**: [Your Email]

---

**IRLCheck-Clickdoo** - Making image authenticity detection accessible to everyone! 🔍✨ 
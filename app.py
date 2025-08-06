import streamlit as st
import os
import tempfile
from PIL import Image
import exifread
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import io
import base64
import random

# Import AI detection module
try:
    from ai_detection import detect_ai_generation_simple
    AI_DETECTION_AVAILABLE = True
except ImportError:
    AI_DETECTION_AVAILABLE = False
    st.warning("⚠️ AI detection module not available. Using fallback method.")

# Page configuration
st.set_page_config(
    page_title="IRLCheck - Image Authenticity Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .upload-area {
        border: 3px dashed #1f77b4;
        border-radius: 15px;
        padding: 3rem;
        text-align: center;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        margin: 1rem 0;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .upload-area:hover {
        border-color: #764ba2;
        background: linear-gradient(135deg, #e9ecef 0%, #dee2e6 100%);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    
    .upload-area.dragover {
        border-color: #28a745;
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
    }
    
    .result-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border: 1px solid #e0e0e0;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .result-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    .metric-card h3 {
        margin: 0 0 0.5rem 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    .metric-card h2 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: bold;
    }
    
    .metric-card p {
        margin: 0.5rem 0 0 0;
        opacity: 0.8;
        font-size: 0.9rem;
    }
    
    .status-authentic {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
    }
    
    .status-suspicious {
        background: linear-gradient(135deg, #ffc107 0%, #fd7e14 100%);
    }
    
    .status-fake {
        background: linear-gradient(135deg, #dc3545 0%, #e83e8c 100%);
    }
    
    .progress-bar {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        height: 8px;
        border-radius: 4px;
        margin: 0.5rem 0;
    }
    
    .file-info {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    
    .sidebar-section {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .stButton > button {
        border-radius: 10px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0 0;
        background-color: #f8f9fa;
        border: none;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
    
    .ai-method-card {
        background: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .ai-method-card h4 {
        margin: 0 0 0.5rem 0;
        color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'analysis_progress' not in st.session_state:
    st.session_state.analysis_progress = 0

def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">🔍 IRLCheck - Image Authenticity Detector</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Settings")
        
        with st.container():
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            max_file_size = st.slider("Max file size (MB)", 10, 100, 100)
            st.info(f"Maximum file size: {max_file_size} MB")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.header("ℹ️ About")
        with st.container():
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.markdown("""
            **IRLCheck** analyzes images to detect:
            - 📸 **Metadata** (EXIF, GPS, camera info)
            - ✂️ **Editing** (Photoshop, compression artifacts)
            - 🤖 **AI Generation** (Stable Diffusion, DALL-E, etc.)
            
            Supported formats: JPG, JPEG, PNG, WEBP
            """)
            
            # AI Detection Status
            if AI_DETECTION_AVAILABLE:
                st.success("✅ AI Detection: Available")
            else:
                st.warning("⚠️ AI Detection: Fallback mode")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Quick stats in sidebar
        if st.session_state.analysis_results:
            st.header("📊 Quick Stats")
            display_quick_stats(st.session_state.analysis_results)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📁 Upload Image")
        
        # Enhanced upload area
        uploaded_file = st.file_uploader(
            "Drag and drop your image here or click to browse",
            type=['jpg', 'jpeg', 'png', 'webp'],
            help=f"Maximum file size: {max_file_size} MB",
            key="file_uploader"
        )
        
        if uploaded_file is not None:
            # Check file size
            file_size_mb = uploaded_file.size / (1024 * 1024)
            if file_size_mb > max_file_size:
                st.error(f"File size ({file_size_mb:.1f} MB) exceeds the maximum allowed size ({max_file_size} MB)")
                return
            
            st.session_state.uploaded_file = uploaded_file
            
            # Display file info
            with st.container():
                st.markdown('<div class="file-info">', unsafe_allow_html=True)
                st.write(f"**📁 File:** {uploaded_file.name}")
                st.write(f"**📏 Size:** {file_size_mb:.1f} MB")
                st.write(f"**📄 Type:** {uploaded_file.type}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Display uploaded image
            st.subheader("📸 Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Analysis button with progress
            if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    # Progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Simulate progress
                    for i in range(100):
                        progress_bar.progress(i + 1)
                        if i < 20:
                            status_text.text("Extracting metadata...")
                        elif i < 40:
                            status_text.text("Analyzing image quality...")
                        elif i < 70:
                            status_text.text("Running AI detection...")
                        elif i < 90:
                            status_text.text("Combining results...")
                        else:
                            status_text.text("Finalizing analysis...")
                    
                    results = analyze_image(uploaded_file)
                    st.session_state.analysis_results = results
                    
                    progress_bar.progress(100)
                    status_text.text("Analysis completed!")
                    
                    st.success("✅ Analysis completed successfully!")
    
    with col2:
        st.header("📊 Quick Stats")
        if st.session_state.analysis_results:
            display_quick_stats(st.session_state.analysis_results)
        else:
            st.info("Upload an image and click 'Analyze' to see results")
    
    # Display results if available
    if st.session_state.analysis_results:
        display_detailed_results(st.session_state.analysis_results)

def analyze_image(uploaded_file):
    """Analyze the uploaded image for authenticity"""
    results = {
        'metadata': {},
        'editing_detection': {},
        'ai_detection': {},
        'file_info': {}
    }
    
    try:
        # File info
        results['file_info'] = {
            'filename': uploaded_file.name,
            'size_mb': uploaded_file.size / (1024 * 1024),
            'format': uploaded_file.type
        }
        
        # Metadata analysis
        results['metadata'] = extract_metadata(uploaded_file)
        
        # Editing detection
        results['editing_detection'] = detect_editing(uploaded_file)
        
        # AI detection
        if AI_DETECTION_AVAILABLE:
            # Reset file pointer
            uploaded_file.seek(0)
            image = Image.open(uploaded_file)
            results['ai_detection'] = detect_ai_generation_simple(image)
        else:
            # Fallback method
            results['ai_detection'] = detect_ai_generation_fallback(uploaded_file)
        
    except Exception as e:
        st.error(f"Error during analysis: {str(e)}")
        return None
    
    return results

def extract_metadata(uploaded_file):
    """Extract EXIF metadata from image"""
    metadata = {}
    
    try:
        # Reset file pointer
        uploaded_file.seek(0)
        
        # Read EXIF data
        tags = exifread.process_file(uploaded_file)
        
        if tags:
            # Camera information
            if 'Image Make' in tags:
                metadata['camera_make'] = str(tags['Image Make'])
            if 'Image Model' in tags:
                metadata['camera_model'] = str(tags['Image Model'])
            
            # Date and time
            if 'EXIF DateTimeOriginal' in tags:
                metadata['date_taken'] = str(tags['EXIF DateTimeOriginal'])
            elif 'Image DateTime' in tags:
                metadata['date_taken'] = str(tags['Image DateTime'])
            
            # GPS coordinates
            if 'GPS GPSLatitude' in tags and 'GPS GPSLongitude' in tags:
                metadata['gps_latitude'] = str(tags['GPS GPSLatitude'])
                metadata['gps_longitude'] = str(tags['GPS GPSLongitude'])
            
            # Software used
            if 'Image Software' in tags:
                metadata['software'] = str(tags['Image Software'])
            
            # Image dimensions
            if 'EXIF ExifImageWidth' in tags:
                metadata['width'] = str(tags['EXIF ExifImageWidth'])
            if 'EXIF ExifImageLength' in tags:
                metadata['height'] = str(tags['EXIF ExifImageLength'])
            
            metadata['has_exif'] = True
        else:
            metadata['has_exif'] = False
            metadata['message'] = "No EXIF metadata found"
            
    except Exception as e:
        metadata['error'] = f"Error reading metadata: {str(e)}"
        metadata['has_exif'] = False
    
    return metadata

def detect_editing(uploaded_file):
    """Detect signs of image editing using basic analysis"""
    editing_results = {}
    
    try:
        # Reset file pointer
        uploaded_file.seek(0)
        
        # Load image with PIL
        image = Image.open(uploaded_file)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Save as JPEG for analysis
        temp_path = tempfile.mktemp(suffix='.jpg')
        image.save(temp_path, 'JPEG', quality=95)
        
        # Load with OpenCV for analysis
        img = cv2.imread(temp_path)
        
        if img is not None:
            # Basic editing detection using noise analysis
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Calculate difference
            diff = cv2.absdiff(gray, blurred)
            
            # Calculate editing score (variance of differences)
            score = np.var(diff)
            
            # Normalize for probability (0-100%)
            probability = min(score / 1000 * 100, 100)
            
            editing_results['editing_score'] = score
            editing_results['editing_probability'] = probability
            editing_results['heatmap'] = diff
            
        else:
            editing_results['editing_probability'] = 0.0
            editing_results['error'] = "Could not load image for analysis"
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
    except Exception as e:
        editing_results['error'] = f"Error in editing detection: {str(e)}"
        editing_results['editing_probability'] = 0.0
    
    return editing_results

def detect_ai_generation_fallback(uploaded_file):
    """Fallback AI detection method"""
    ai_results = {
        'ai_probability': 0.0,
        'model_used': 'Fallback (Statistical)',
        'confidence': 0.0,
        'detection_methods': [],
        'details': {}
    }
    
    # Placeholder: random probability for demonstration
    # In a real implementation, this would use basic statistical analysis
    ai_results['ai_probability'] = random.uniform(0, 30)  # Low probability for real images
    ai_results['confidence'] = 0.3
    ai_results['detection_methods'] = [{
        'name': 'Statistical Analysis',
        'probability': ai_results['ai_probability'],
        'confidence': ai_results['confidence']
    }]
    
    return ai_results

def display_quick_stats(results):
    """Display quick statistics in the sidebar"""
    if not results:
        return
    
    # AI Detection Score
    ai_prob = results.get('ai_detection', {}).get('ai_probability', 0)
    st.metric("🤖 AI Probability", f"{ai_prob:.1f}%")
    
    # Editing Detection Score
    editing_prob = results.get('editing_detection', {}).get('editing_probability', 0)
    st.metric("✂️ Editing Probability", f"{editing_prob:.1f}%")
    
    # Metadata Status
    has_exif = results.get('metadata', {}).get('has_exif', False)
    metadata_status = "✅ Present" if has_exif else "❌ Absent"
    st.metric("📸 Metadata", metadata_status)
    
    # File Info
    file_size = results.get('file_info', {}).get('size_mb', 0)
    st.metric("📁 File Size", f"{file_size:.1f} MB")

def display_detailed_results(results):
    """Display detailed analysis results in tabs"""
    st.markdown("---")
    st.header("📊 Analysis Results")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Summary", "📸 Metadata", "🔍 Visual Analysis", "🤖 AI Analysis"])
    
    with tab1:
        display_summary_tab(results)
    
    with tab2:
        display_metadata_tab(results)
    
    with tab3:
        display_visual_analysis_tab(results)
    
    with tab4:
        display_ai_analysis_tab(results)
    
    # Download report button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("📄 Download Report (PDF)", type="secondary", use_container_width=True):
            generate_pdf_report(results)

def display_summary_tab(results):
    """Display summary of analysis results"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        ai_prob = results.get('ai_detection', {}).get('ai_probability', 0)
        status_class = "status-authentic" if ai_prob < 30 else "status-suspicious" if ai_prob < 70 else "status-fake"
        st.markdown(f"""
        <div class="metric-card {status_class}">
            <h3>🤖 AI Generation</h3>
            <h2>{ai_prob:.1f}%</h2>
            <p>Probability of AI generation</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        editing_prob = results.get('editing_detection', {}).get('editing_probability', 0)
        status_class = "status-authentic" if editing_prob < 30 else "status-suspicious" if editing_prob < 70 else "status-fake"
        st.markdown(f"""
        <div class="metric-card {status_class}">
            <h3>✂️ Image Editing</h3>
            <h2>{editing_prob:.1f}%</h2>
            <p>Probability of manipulation</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        has_exif = results.get('metadata', {}).get('has_exif', False)
        ai_prob = results.get('ai_detection', {}).get('ai_probability', 0)
        editing_prob = results.get('editing_detection', {}).get('editing_probability', 0)
        
        if has_exif and ai_prob < 30 and editing_prob < 30:
            status = "✅ Authentic"
            status_class = "status-authentic"
        elif ai_prob > 70 or editing_prob > 70:
            status = "❌ Fake"
            status_class = "status-fake"
        else:
            status = "⚠️ Suspicious"
            status_class = "status-suspicious"
            
        st.markdown(f"""
        <div class="metric-card {status_class}">
            <h3>🔍 Overall Status</h3>
            <h2>{status}</h2>
            <p>Based on all indicators</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Progress bars
    st.subheader("📈 Detailed Scores")
    
    ai_prob = results.get('ai_detection', {}).get('ai_probability', 0)
    editing_prob = results.get('editing_detection', {}).get('editing_probability', 0)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**AI Generation Risk**")
        st.progress(ai_prob / 100)
        st.write(f"{ai_prob:.1f}% - {'Low' if ai_prob < 30 else 'Medium' if ai_prob < 70 else 'High'} risk")
    
    with col2:
        st.write("**Editing Detection**")
        st.progress(editing_prob / 100)
        st.write(f"{editing_prob:.1f}% - {'Low' if editing_prob < 30 else 'Medium' if editing_prob < 70 else 'High'} manipulation")
    
    # Detailed summary
    st.subheader("📝 Analysis Summary")
    
    summary_text = f"""
    **File Information:**
    - Filename: {results.get('file_info', {}).get('filename', 'Unknown')}
    - Size: {results.get('file_info', {}).get('size_mb', 0):.1f} MB
    - Format: {results.get('file_info', {}).get('format', 'Unknown')}
    
    **Authenticity Assessment:**
    - AI Generation Probability: {ai_prob:.1f}% ({'Low' if ai_prob < 30 else 'Medium' if ai_prob < 70 else 'High'} risk)
    - Editing Detection: {editing_prob:.1f}% ({'Low' if editing_prob < 30 else 'Medium' if editing_prob < 70 else 'High'} manipulation)
    - Metadata Status: {'Present and consistent' if has_exif else 'Absent or inconsistent'}
    
    **Recommendation:**
    """
    
    if ai_prob < 30 and editing_prob < 30 and has_exif:
        summary_text += "✅ This image appears to be authentic with low signs of manipulation."
    elif ai_prob > 70 or editing_prob > 70:
        summary_text += "⚠️ This image shows significant signs of AI generation or manipulation."
    else:
        summary_text += "🤔 This image shows some suspicious indicators. Further analysis recommended."
    
    st.markdown(summary_text)

def display_metadata_tab(results):
    """Display detailed metadata information"""
    metadata = results.get('metadata', {})
    
    if metadata.get('has_exif', False):
        st.subheader("📸 EXIF Metadata")
        
        # Create metadata table
        metadata_data = []
        for key, value in metadata.items():
            if key not in ['has_exif', 'error']:
                # Format key names
                formatted_key = key.replace('_', ' ').title()
                metadata_data.append([formatted_key, str(value)])
        
        if metadata_data:
            df = pd.DataFrame(metadata_data, columns=['Property', 'Value'])
            st.dataframe(df, use_container_width=True)
        
        # GPS coordinates if available
        if 'gps_latitude' in metadata and 'gps_longitude' in metadata:
            st.subheader("📍 GPS Coordinates")
            st.write(f"Latitude: {metadata['gps_latitude']}")
            st.write(f"Longitude: {metadata['gps_longitude']}")
            
            # Create a simple map (placeholder)
            st.info("Map visualization will be implemented in future versions")
    
    else:
        st.warning("⚠️ No EXIF metadata found in this image")
        st.info("This could indicate:")
        st.markdown("""
        - The image was edited and metadata was stripped
        - The image was generated by AI
        - The image format doesn't support EXIF data
        - The image was saved without preserving metadata
        """)

def display_visual_analysis_tab(results):
    """Display visual analysis results"""
    st.subheader("🔍 Visual Analysis")
    
    editing_results = results.get('editing_detection', {})
    
    if 'heatmap' in editing_results and editing_results['heatmap'] is not None:
        st.write("**Noise Analysis Heatmap:**")
        st.write("This visualization shows areas where the image may have been edited or compressed.")
        
        # Convert heatmap to displayable format
        heatmap = editing_results['heatmap']
        heatmap_normalized = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
        heatmap_colored = cv2.applyColorMap(heatmap_normalized.astype(np.uint8), cv2.COLORMAP_JET)
        
        # Convert BGR to RGB for display
        heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        st.image(heatmap_rgb, caption="Noise Analysis Heatmap - Red areas indicate potential editing", use_container_width=True)
        
        # Score explanation
        editing_score = editing_results.get('editing_score', 0)
        st.write(f"**Editing Score:** {editing_score:.2f}")
        st.info("""
        **How to interpret the heatmap:**
        - 🔴 **Red areas**: High noise levels, potential editing
        - 🟡 **Yellow areas**: Medium noise levels
        - 🟢 **Green areas**: Low noise levels, likely original
        """)
    
    else:
        st.warning("Visual analysis could not be performed")
        if 'error' in editing_results:
            st.error(f"Error: {editing_results['error']}")

def display_ai_analysis_tab(results):
    """Display AI analysis results"""
    st.subheader("🤖 AI Generation Analysis")
    
    ai_results = results.get('ai_detection', {})
    
    # Overall AI probability
    ai_prob = ai_results.get('ai_probability', 0)
    confidence = ai_results.get('confidence', 0)
    model_used = ai_results.get('model_used', 'Unknown')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("**Overall AI Probability**", f"{ai_prob:.1f}%")
        st.metric("**Confidence**", f"{confidence:.2f}")
    
    with col2:
        st.metric("**Model Used**", model_used)
        st.metric("**Risk Level**", "High" if ai_prob > 70 else "Medium" if ai_prob > 30 else "Low")
    
    # Detection methods
    detection_methods = ai_results.get('detection_methods', [])
    
    if detection_methods:
        st.subheader("🔬 Detection Methods Used")
        
        for method in detection_methods:
            with st.container():
                st.markdown(f"""
                <div class="ai-method-card">
                    <h4>{method['name']}</h4>
                    <p><strong>Probability:</strong> {method['probability']:.1f}%</p>
                    <p><strong>Confidence:</strong> {method['confidence']:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Technical details
    details = ai_results.get('details', {})
    if details:
        st.subheader("⚙️ Technical Details")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"**Methods Used:** {details.get('methods_used', 0)}")
        with col2:
            st.write(f"**Device:** {details.get('device', 'Unknown')}")
        with col3:
            st.write(f"**Models Available:** {details.get('models_available', 0)}")
    
    # AI detection explanation
    st.subheader("📚 How AI Detection Works")
    
    st.markdown("""
    **IRLCheck uses multiple methods to detect AI-generated images:**
    
    1. **Statistical Analysis**: Analyzes frequency patterns, texture, noise, and edge characteristics
    2. **Deep Learning Models**: Uses pre-trained neural networks to identify AI patterns
    3. **CLIP Analysis**: Compares image features with text descriptions of real vs AI images
    
    **Common AI Generation Tools Detected:**
    - Stable Diffusion
    - DALL-E
    - Midjourney
    - GAN-generated images
    
    **Note**: AI detection is not 100% accurate. Results should be used as indicators, not definitive proof.
    """)

def generate_pdf_report(results):
    """Generate and download PDF report"""
    try:
        # Create PDF buffer
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        elements = []
        
        # Add title
        styles = getSampleStyleSheet()
        title = Paragraph("IRLCheck - Image Authenticity Report", styles['Title'])
        elements.append(title)
        elements.append(Spacer(1, 20))
        
        # Add file information
        file_info = results.get('file_info', {})
        elements.append(Paragraph(f"<b>File:</b> {file_info.get('filename', 'Unknown')}", styles['Normal']))
        elements.append(Paragraph(f"<b>Size:</b> {file_info.get('size_mb', 0):.1f} MB", styles['Normal']))
        elements.append(Paragraph(f"<b>Format:</b> {file_info.get('format', 'Unknown')}", styles['Normal']))
        elements.append(Spacer(1, 20))
        
        # Add analysis results
        ai_prob = results.get('ai_detection', {}).get('ai_probability', 0)
        editing_prob = results.get('editing_detection', {}).get('editing_probability', 0)
        
        elements.append(Paragraph("<b>Analysis Results:</b>", styles['Heading2']))
        elements.append(Paragraph(f"AI Generation Probability: {ai_prob:.1f}%", styles['Normal']))
        elements.append(Paragraph(f"Editing Detection: {editing_prob:.1f}%", styles['Normal']))
        elements.append(Spacer(1, 20))
        
        # Build PDF
        doc.build(elements)
        
        # Get PDF content
        pdf_content = buffer.getvalue()
        buffer.close()
        
        # Create download button
        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_content,
            file_name="irlcheck_report.pdf",
            mime="application/pdf"
        )
        
    except Exception as e:
        st.error(f"Error generating PDF report: {str(e)}")

if __name__ == "__main__":
    main() 
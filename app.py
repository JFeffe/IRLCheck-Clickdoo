import streamlit as st
import cv2
import numpy as np
from PIL import Image
import exifread
import io
import base64
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import time
import os
from datetime import datetime

# Import AI detection with caching
try:
    from ai_detection import detect_ai_generation_simple
    AI_DETECTION_AVAILABLE = True
except ImportError:
    AI_DETECTION_AVAILABLE = False
    def detect_ai_generation_simple(image):
        return {
            'ai_probability': 50.0,
            'confidence': 0.5,
            'model_used': 'Fallback method',
            'detection_methods': [],
            'details': {}
        }

# Configure page
st.set_page_config(
    page_title="IRLCheck-Clickdoo - Image Authenticity Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern design
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .upload-area {
        border: 3px dashed #FF6B6B;
        border-radius: 15px;
        padding: 3rem;
        text-align: center;
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        transition: all 0.3s ease;
        margin-bottom: 2rem;
    }
    
    .upload-area:hover {
        border-color: #4ECDC4;
        background: linear-gradient(135deg, #e9ecef, #f8f9fa);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .result-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        border-left: 5px solid #FF6B6B;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .status-authentic {
        border-left-color: #28a745 !important;
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
    }
    
    .status-suspicious {
        border-left-color: #ffc107 !important;
        background: linear-gradient(135deg, #fff3cd, #ffeaa7);
    }
    
    .status-fake {
        border-left-color: #dc3545 !important;
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
    }
    
    .ai-method-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .sidebar-section {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }
    
    .file-info {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state with caching
@st.cache_resource
def initialize_session_state():
    """Initialize session state with caching for better performance"""
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False

# Cache AI detection results
@st.cache_data(ttl=3600)  # Cache for 1 hour
def cached_ai_detection(image_bytes):
    """Cache AI detection results to avoid recomputation"""
    # Convert bytes back to PIL Image for AI detection
    image = Image.open(io.BytesIO(image_bytes))
    return detect_ai_generation_simple(image)

# Cache metadata extraction
@st.cache_data(ttl=3600)
def cached_metadata_extraction(image_bytes):
    """Cache metadata extraction results"""
    return extract_metadata(image_bytes)

def main():
    initialize_session_state()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 style="color: white; text-align: center; margin: 0;">🔍 IRLCheck-Clickdoo</h1>
        <p style="color: white; text-align: center; margin: 0.5rem 0 0 0; font-size: 1.2rem;">
            Advanced Image Authenticity Detection Tool with AI
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.header("⚙️ Settings")
        
        # AI Detection Status
        ai_status = "🟢 Available" if AI_DETECTION_AVAILABLE else "🟡 Fallback Mode"
        st.info(f"**AI Detection:** {ai_status}")
        
        # File size limit
        max_file_size = st.slider("Max file size (MB)", 10, 200, 100)
        
        # Analysis options
        st.subheader("Analysis Options")
        enable_metadata = st.checkbox("Metadata Analysis", value=True)
        enable_editing = st.checkbox("Editing Detection", value=True)
        enable_ai = st.checkbox("AI Detection", value=True, disabled=not AI_DETECTION_AVAILABLE)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # About section
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.header("ℹ️ About")
        st.markdown("""
        **IRLCheck-Clickdoo** is an advanced image authenticity detection tool that combines:
        
        • **Metadata Analysis** - EXIF data extraction
        • **Editing Detection** - Photoshop traces & artifacts
        • **AI Detection** - AI-generated image identification
        
        **Supported formats:** PNG, JPG, JPEG, WEBP
        **Max size:** {}MB
        """.format(max_file_size))
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # File upload area
        st.markdown('<div class="upload-area">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "📁 Drag and drop your image here or click to browse",
            type=['png', 'jpg', 'jpeg', 'webp'],
            help="Upload an image to analyze its authenticity"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Display file information
        if uploaded_file is not None:
            file_size = len(uploaded_file.getvalue()) / (1024 * 1024)  # MB
            st.markdown(f"""
            <div class="file-info">
                <strong>📄 File Information:</strong><br>
                • Name: {uploaded_file.name}<br>
                • Size: {file_size:.2f} MB<br>
                • Type: {uploaded_file.type}
            </div>
            """, unsafe_allow_html=True)
            
            # Check file size
            if file_size > max_file_size:
                st.error(f"❌ File too large! Maximum size is {max_file_size}MB")
                return
            
            # Display image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Analysis button
            if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                with st.spinner("🔄 Analyzing image..."):
                    # Progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Step 1: Extract metadata
                    status_text.text("📊 Extracting metadata...")
                    progress_bar.progress(20)
                    metadata = cached_metadata_extraction(uploaded_file.getvalue()) if enable_metadata else {}
                    
                    # Step 2: Detect editing
                    status_text.text("✂️ Detecting editing traces...")
                    progress_bar.progress(40)
                    editing_result = detect_editing(image) if enable_editing else {'probability': 0, 'confidence': 0}
                    
                    # Step 3: AI detection
                    status_text.text("🤖 Running AI detection...")
                    progress_bar.progress(60)
                    ai_result = cached_ai_detection(uploaded_file.getvalue()) if enable_ai else {
                        'ai_probability': 0,
                        'confidence': 0,
                        'model_used': 'Disabled',
                        'detection_methods': [],
                        'details': {}
                    }
                    
                    # Step 4: Combine results
                    status_text.text("🔄 Combining results...")
                    progress_bar.progress(80)
                    
                    # Store results in session state
                    st.session_state.analysis_results = {
                        'metadata': metadata,
                        'editing': editing_result,
                        'ai': ai_result,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'filename': uploaded_file.name
                    }
                    st.session_state.uploaded_file = uploaded_file
                    st.session_state.analysis_complete = True
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Analysis complete!")
                    time.sleep(1)
                    progress_bar.empty()
                    status_text.empty()
    
    with col2:
        # Quick stats
        if st.session_state.analysis_complete and st.session_state.analysis_results:
            display_quick_stats(st.session_state.analysis_results)
    
    # Display detailed results
    if st.session_state.analysis_complete and st.session_state.analysis_results:
        display_detailed_results(st.session_state.analysis_results)

def display_quick_stats(results):
    """Display quick statistics in the sidebar"""
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.subheader("📊 Quick Stats")
    
    # Overall authenticity score
    ai_prob = results['ai']['ai_probability']
    editing_prob = results['editing']['probability']
    
    # Calculate overall score (weighted average)
    overall_score = 100 - ((ai_prob * 0.6) + (editing_prob * 0.4))
    
    # Determine status
    if overall_score >= 80:
        status = "🟢 Authentic"
        status_class = "status-authentic"
    elif overall_score >= 50:
        status = "🟡 Suspicious"
        status_class = "status-suspicious"
    else:
        status = "🔴 Likely Fake"
        status_class = "status-fake"
    
    st.markdown(f"""
    <div class="metric-card {status_class}">
        <h3>Overall Score</h3>
        <h2>{overall_score:.1f}%</h2>
        <p>{status}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # AI Generation Risk
    st.markdown(f"""
    <div class="metric-card">
        <h4>AI Generation Risk</h4>
        <h3>{ai_prob:.1f}%</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Editing Detection
    st.markdown(f"""
    <div class="metric-card">
        <h4>Editing Detection</h4>
        <h3>{editing_prob:.1f}%</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_detailed_results(results):
    """Display detailed analysis results in tabs"""
    st.markdown("## 📋 Detailed Analysis Results")
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Summary", "📄 Metadata", "🔍 Visual Analysis", "🤖 AI Analysis"])
    
    with tab1:
        display_summary_tab(results)
    
    with tab2:
        display_metadata_tab(results['metadata'])
    
    with tab3:
        display_visual_analysis_tab(results)
    
    with tab4:
        display_ai_analysis_tab(results['ai'])
    
    # Download report button
    if st.button("📄 Download PDF Report", type="secondary"):
        pdf_bytes = generate_pdf_report(results)
        st.download_button(
            label="💾 Download Report",
            data=pdf_bytes,
            file_name=f"IRLCheck_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf"
        )

def display_summary_tab(results):
    """Display summary tab with overall results"""
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    
    # Overall assessment
    ai_prob = results['ai']['ai_probability']
    editing_prob = results['editing']['probability']
    overall_score = 100 - ((ai_prob * 0.6) + (editing_prob * 0.4))
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Overall Authenticity", f"{overall_score:.1f}%")
    
    with col2:
        st.metric("AI Generation Risk", f"{ai_prob:.1f}%")
    
    with col3:
        st.metric("Editing Detection", f"{editing_prob:.1f}%")
    
    # Progress bars
    st.subheader("Risk Assessment")
    
    # AI Generation Risk
    st.write("🤖 AI Generation Risk")
    st.progress(ai_prob / 100)
    st.caption(f"Confidence: {results['ai']['confidence']:.1%}")
    
    # Editing Detection
    st.write("✂️ Editing Detection")
    st.progress(editing_prob / 100)
    st.caption(f"Confidence: {results['editing']['confidence']:.1%}")
    
    # Overall status
    if overall_score >= 80:
        st.success("✅ **VERDICT: Likely Authentic** - This image appears to be genuine with minimal signs of manipulation.")
    elif overall_score >= 50:
        st.warning("⚠️ **VERDICT: Suspicious** - This image shows some signs of potential manipulation. Further investigation recommended.")
    else:
        st.error("❌ **VERDICT: Likely Fake** - This image shows strong signs of AI generation or heavy editing.")
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_metadata_tab(metadata):
    """Display metadata analysis results"""
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    
    if not metadata:
        st.info("No metadata found or metadata analysis disabled.")
        return
    
    st.subheader("📄 EXIF Metadata Analysis")
    
    # Basic file info
    if 'File Information' in metadata:
        st.write("**File Information:**")
        for key, value in metadata['File Information'].items():
            st.write(f"• {key}: {value}")
    
    # Camera info
    if 'Camera Information' in metadata:
        st.write("**Camera Information:**")
        for key, value in metadata['Camera Information'].items():
            st.write(f"• {key}: {value}")
    
    # GPS info
    if 'GPS Information' in metadata:
        st.write("**GPS Information:**")
        for key, value in metadata['GPS Information'].items():
            st.write(f"• {key}: {value}")
    
    # Software info
    if 'Software Information' in metadata:
        st.write("**Software Information:**")
        for key, value in metadata['Software Information'].items():
            st.write(f"• {key}: {value}")
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_visual_analysis_tab(results):
    """Display visual analysis results"""
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    
    st.subheader("🔍 Visual Analysis Results")
    
    # Editing detection details
    editing_result = results['editing']
    st.write("**Editing Detection Analysis:**")
    st.write(f"• Probability of editing: {editing_result['probability']:.1f}%")
    st.write(f"• Confidence: {editing_result['confidence']:.1%}")
    
    if 'details' in editing_result:
        st.write("**Technical Details:**")
        for key, value in editing_result['details'].items():
            st.write(f"• {key}: {value}")
    
    # Create visualization
    fig = go.Figure()
    
    # Add bars for different metrics
    fig.add_trace(go.Bar(
        x=['AI Generation Risk', 'Editing Detection'],
        y=[results['ai']['ai_probability'], editing_result['probability']],
        marker_color=['#FF6B6B', '#4ECDC4'],
        text=[f"{results['ai']['ai_probability']:.1f}%", f"{editing_result['probability']:.1f}%"],
        textposition='auto'
    ))
    
    fig.update_layout(
        title="Risk Assessment Visualization",
        xaxis_title="Analysis Type",
        yaxis_title="Risk Percentage (%)",
        yaxis_range=[0, 100],
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_ai_analysis_tab(ai_results):
    """Display AI analysis results"""
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    
    st.subheader("🤖 AI Generation Analysis")
    
    # Overall AI probability
    st.metric("AI Generation Probability", f"{ai_results['ai_probability']:.1f}%")
    st.metric("Confidence", f"{ai_results['confidence']:.1%}")
    st.metric("Model Used", ai_results['model_used'])
    
    # Risk level
    ai_prob = ai_results['ai_probability']
    if ai_prob < 30:
        risk_level = "🟢 Low Risk"
        risk_color = "#28a745"
    elif ai_prob < 70:
        risk_level = "🟡 Medium Risk"
        risk_color = "#ffc107"
    else:
        risk_level = "🔴 High Risk"
        risk_color = "#dc3545"
    
    st.markdown(f"""
    <div style="background-color: {risk_color}; color: white; padding: 1rem; border-radius: 10px; text-align: center;">
        <h4>Risk Level: {risk_level}</h4>
    </div>
    """, unsafe_allow_html=True)
    
    # Detection methods
    if ai_results['detection_methods']:
        st.subheader("Detection Methods Used:")
        for method in ai_results['detection_methods']:
            st.markdown(f"""
            <div class="ai-method-card">
                <strong>{method['name']}</strong><br>
                Probability: {method['probability']:.1f}%<br>
                Confidence: {method['confidence']:.1%}
            </div>
            """, unsafe_allow_html=True)
    
    # Technical details
    if 'details' in ai_results:
        st.subheader("Technical Details:")
        details = ai_results['details']
        st.write(f"• Methods used: {details.get('methods_used', 'N/A')}")
        st.write(f"• Device: {details.get('device', 'N/A')}")
        st.write(f"• Models available: {details.get('models_available', 'N/A')}")
    
    # Explanation
    st.subheader("How AI Detection Works:")
    st.markdown("""
    Our AI detection system uses multiple methods:
    
    1. **Statistical Analysis** - Analyzes frequency patterns, texture, and noise characteristics
    2. **Deep Learning** - Uses pre-trained neural networks to identify AI-generated patterns
    3. **CLIP Analysis** - Compares image features with text descriptions of real vs AI images
    
    The final probability is a weighted combination of these methods.
    """)
    
    st.markdown('</div>', unsafe_allow_html=True)

def extract_metadata(image_bytes):
    """Extract metadata from image with caching"""
    try:
        # Convert bytes to file-like object
        image_file = io.BytesIO(image_bytes)
        
        # Extract EXIF data
        tags = exifread.process_file(image_file)
        
        if not tags:
            return {"File Information": {"Status": "No EXIF data found"}}
        
        # Organize metadata
        metadata = {
            "File Information": {},
            "Camera Information": {},
            "GPS Information": {},
            "Software Information": {}
        }
        
        for tag, value in tags.items():
            if tag.startswith('Image'):
                metadata["File Information"][tag] = str(value)
            elif tag.startswith('EXIF'):
                metadata["Camera Information"][tag] = str(value)
            elif tag.startswith('GPS'):
                metadata["GPS Information"][tag] = str(value)
            elif tag.startswith('Software') or tag.startswith('Processing'):
                metadata["Software Information"][tag] = str(value)
        
        return metadata
        
    except Exception as e:
        return {"Error": f"Failed to extract metadata: {str(e)}"}

def detect_editing(image):
    """Detect editing traces in image"""
    try:
        # Convert PIL image to numpy array
        img_array = np.array(image)
        
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Basic noise analysis (simplified version)
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Calculate difference
        diff = cv2.absdiff(gray, blurred)
        
        # Calculate variance of the difference
        variance = np.var(diff)
        
        # Normalize variance to get a probability
        # Higher variance might indicate editing
        probability = min(variance / 1000 * 100, 100)
        confidence = 0.6  # Medium confidence for this method
        
        return {
            'probability': probability,
            'confidence': confidence,
            'details': {
                'noise_variance': variance,
                'method': 'Noise Analysis'
            }
        }
        
    except Exception as e:
        return {
            'probability': 50.0,
            'confidence': 0.3,
            'details': {'error': str(e)}
        }

def generate_pdf_report(results):
    """Generate a comprehensive PDF report"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=1
    )
    story.append(Paragraph("IRLCheck-Clickdoo Analysis Report", title_style))
    story.append(Spacer(1, 20))
    
    # Summary
    story.append(Paragraph("Executive Summary", styles['Heading2']))
    ai_prob = results['ai']['ai_probability']
    editing_prob = results['editing']['probability']
    overall_score = 100 - ((ai_prob * 0.6) + (editing_prob * 0.4))
    
    summary_text = f"""
    Overall Authenticity Score: {overall_score:.1f}%
    AI Generation Risk: {ai_prob:.1f}%
    Editing Detection: {editing_prob:.1f}%
    Analysis Date: {results['timestamp']}
    Filename: {results['filename']}
    """
    story.append(Paragraph(summary_text, styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Detailed results
    story.append(Paragraph("Detailed Analysis", styles['Heading2']))
    
    # AI Analysis
    story.append(Paragraph("AI Generation Analysis", styles['Heading3']))
    ai_text = f"""
    Probability: {ai_prob:.1f}%
    Confidence: {results['ai']['confidence']:.1%}
    Model Used: {results['ai']['model_used']}
    """
    story.append(Paragraph(ai_text, styles['Normal']))
    story.append(Spacer(1, 12))
    
    # Editing Analysis
    story.append(Paragraph("Editing Detection", styles['Heading3']))
    editing_text = f"""
    Probability: {editing_prob:.1f}%
    Confidence: {results['editing']['confidence']:.1%}
    """
    story.append(Paragraph(editing_text, styles['Normal']))
    story.append(Spacer(1, 12))
    
    # Metadata
    if results['metadata']:
        story.append(Paragraph("Metadata Analysis", styles['Heading3']))
        for category, data in results['metadata'].items():
            if isinstance(data, dict):
                story.append(Paragraph(f"{category}:", styles['Heading4']))
                for key, value in data.items():
                    story.append(Paragraph(f"• {key}: {value}", styles['Normal']))
                story.append(Spacer(1, 6))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

if __name__ == "__main__":
    main() 
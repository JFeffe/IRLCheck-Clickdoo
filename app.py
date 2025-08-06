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
import json
from datetime import datetime, timedelta
import hashlib

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
    
    .history-item {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #FF6B6B;
        transition: all 0.3s ease;
    }
    
    .history-item:hover {
        background: #e9ecef;
        transform: translateX(5px);
    }
    
    .batch-progress {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
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
    if 'analysis_progress' not in st.session_state:
        st.session_state.analysis_progress = 0
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    if 'batch_results' not in st.session_state:
        st.session_state.batch_results = []

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

def generate_file_hash(image_bytes):
    """Generate a unique hash for the image file"""
    return hashlib.md5(image_bytes).hexdigest()

def save_to_history(results):
    """Save analysis results to history"""
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    
    # Add timestamp and file hash
    results['id'] = generate_file_hash(results.get('image_bytes', b''))
    results['timestamp'] = datetime.now().isoformat()
    
    # Keep only last 50 analyses
    st.session_state.analysis_history = st.session_state.analysis_history[-49:] + [results]

def load_from_history(history_id):
    """Load analysis results from history"""
    for item in st.session_state.analysis_history:
        if item.get('id') == history_id:
            return item
    return None

def batch_analyze_images(uploaded_files, max_file_size, enable_metadata, enable_editing, enable_ai):
    """Analyze multiple images in batch"""
    batch_results = []
    total_files = len(uploaded_files)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, uploaded_file in enumerate(uploaded_files):
        try:
            status_text.text(f"Analyzing {uploaded_file.name} ({i+1}/{total_files})")
            
            # Check file size
            file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
            if file_size > max_file_size:
                batch_results.append({
                    'filename': uploaded_file.name,
                    'status': 'error',
                    'error': f'File too large ({file_size:.1f}MB > {max_file_size}MB)'
                })
                continue
            
            # Analyze image
            image = Image.open(uploaded_file)
            
            # Extract metadata
            metadata = cached_metadata_extraction(uploaded_file.getvalue()) if enable_metadata else {}
            
            # Detect editing
            editing_result = detect_editing(image) if enable_editing else {'probability': 0, 'confidence': 0}
            
            # AI detection
            ai_result = cached_ai_detection(uploaded_file.getvalue()) if enable_ai else {
                'ai_probability': 0,
                'confidence': 0,
                'model_used': 'Disabled',
                'detection_methods': [],
                'details': {}
            }
            
            # Calculate overall score
            ai_prob = ai_result['ai_probability']
            editing_prob = editing_result['probability']
            overall_score = 100 - ((ai_prob * 0.6) + (editing_prob * 0.4))
            
            batch_results.append({
                'filename': uploaded_file.name,
                'status': 'success',
                'overall_score': overall_score,
                'ai_probability': ai_prob,
                'editing_probability': editing_prob,
                'metadata': metadata,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
        except Exception as e:
            batch_results.append({
                'filename': uploaded_file.name,
                'status': 'error',
                'error': str(e)
            })
        
        # Update progress
        progress_bar.progress((i + 1) / total_files)
    
    progress_bar.empty()
    status_text.empty()
    
    return batch_results

def main():
    # Initialize session state
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'analysis_progress' not in st.session_state:
        st.session_state.analysis_progress = 0
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    if 'batch_results' not in st.session_state:
        st.session_state.batch_results = []
    
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
        
        # History section
        if st.session_state.analysis_history:
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.header("📚 Analysis History")
            
            # Show recent analyses
            recent_history = st.session_state.analysis_history[-5:]  # Last 5
            for item in reversed(recent_history):
                filename = item.get('filename', 'Unknown')
                timestamp = item.get('timestamp', '')
                overall_score = item.get('overall_score', 0)
                
                if overall_score >= 80:
                    status_emoji = "🟢"
                elif overall_score >= 50:
                    status_emoji = "🟡"
                else:
                    status_emoji = "🔴"
                
                st.markdown(f"""
                <div class="history-item">
                    <strong>{status_emoji} {filename}</strong><br>
                    <small>Score: {overall_score:.1f}% | {timestamp}</small>
                </div>
                """, unsafe_allow_html=True)
            
            if st.button("🗑️ Clear History"):
                st.session_state.analysis_history = []
                st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["🔍 Single Analysis", "📦 Batch Analysis", "📊 Analytics"])
    
    with tab1:
        display_single_analysis(max_file_size, enable_metadata, enable_editing, enable_ai)
    
    with tab2:
        display_batch_analysis(max_file_size, enable_metadata, enable_editing, enable_ai)
    
    with tab3:
        display_analytics()

def display_single_analysis(max_file_size, enable_metadata, enable_editing, enable_ai):
    """Display single image analysis interface"""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # File upload area
        st.markdown('<div class="upload-area">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "📁 Drag and drop your image here or click to browse",
            type=['png', 'jpg', 'jpeg', 'webp'],
            help="Upload an image to analyze its authenticity",
            key="single_uploader"
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
                    results = {
                        'metadata': metadata,
                        'editing': editing_result,
                        'ai': ai_result,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'filename': uploaded_file.name,
                        'image_bytes': uploaded_file.getvalue()
                    }
                    
                    # Calculate overall score
                    ai_prob = ai_result['ai_probability']
                    editing_prob = editing_result['probability']
                    results['overall_score'] = 100 - ((ai_prob * 0.6) + (editing_prob * 0.4))
                    
                    st.session_state.analysis_results = results
                    st.session_state.uploaded_file = uploaded_file
                    st.session_state.analysis_complete = True
                    
                    # Save to history
                    save_to_history(results)
                    
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

def display_batch_analysis(max_file_size, enable_metadata, enable_editing, enable_ai):
    """Display batch analysis interface"""
    st.header("📦 Batch Analysis")
    st.markdown("Upload multiple images to analyze them all at once.")
    
    # Batch file uploader
    uploaded_files = st.file_uploader(
        "📁 Drag and drop multiple images here",
        type=['png', 'jpg', 'jpeg', 'webp'],
        accept_multiple_files=True,
        help="Upload multiple images for batch analysis",
        key="batch_uploader"
    )
    
    if uploaded_files:
        st.info(f"📁 {len(uploaded_files)} files selected for batch analysis")
        
        # Show file list
        with st.expander("📋 File List", expanded=True):
            for i, file in enumerate(uploaded_files):
                file_size = len(file.getvalue()) / (1024 * 1024)
                st.write(f"{i+1}. {file.name} ({file_size:.2f} MB)")
        
        # Batch analysis button
        if st.button("🚀 Start Batch Analysis", type="primary", use_container_width=True):
            st.session_state.batch_results = batch_analyze_images(
                uploaded_files, max_file_size, enable_metadata, enable_editing, enable_ai
            )
    
    # Display batch results
    if st.session_state.batch_results:
        st.header("📊 Batch Analysis Results")
        
        # Summary statistics
        successful = [r for r in st.session_state.batch_results if r['status'] == 'success']
        errors = [r for r in st.session_state.batch_results if r['status'] == 'error']
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Files", len(st.session_state.batch_results))
        with col2:
            st.metric("Successful", len(successful))
        with col3:
            st.metric("Errors", len(errors))
        with col4:
            if successful:
                avg_score = sum(r['overall_score'] for r in successful) / len(successful)
                st.metric("Avg Score", f"{avg_score:.1f}%")
        
        # Results table
        if successful:
            st.subheader("✅ Successful Analyses")
            results_data = []
            for result in successful:
                results_data.append({
                    'Filename': result['filename'],
                    'Overall Score': f"{result['overall_score']:.1f}%",
                    'AI Risk': f"{result['ai_probability']:.1f}%",
                    'Editing': f"{result['editing_probability']:.1f}%",
                    'Status': "🟢 Authentic" if result['overall_score'] >= 80 else "🟡 Suspicious" if result['overall_score'] >= 50 else "🔴 Fake"
                })
            
            df = pd.DataFrame(results_data)
            st.dataframe(df, use_container_width=True)
        
        # Error list
        if errors:
            st.subheader("❌ Errors")
            for error in errors:
                st.error(f"**{error['filename']}**: {error['error']}")
        
        # Download batch report
        if st.button("📄 Download Batch Report", type="secondary"):
            batch_report = generate_batch_report(st.session_state.batch_results)
            st.download_button(
                label="💾 Download Batch Report",
                data=batch_report,
                file_name=f"IRLCheck_Batch_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf"
            )

def display_analytics():
    """Display analytics and statistics"""
    st.header("📊 Analytics Dashboard")
    
    if not st.session_state.analysis_history:
        st.info("No analysis history available. Run some analyses to see statistics.")
        return
    
    # Convert history to DataFrame
    history_data = []
    for item in st.session_state.analysis_history:
        if 'overall_score' in item:
            history_data.append({
                'filename': item.get('filename', 'Unknown'),
                'overall_score': item.get('overall_score', 0),
                'ai_probability': item.get('ai', {}).get('ai_probability', 0),
                'editing_probability': item.get('editing', {}).get('probability', 0),
                'timestamp': item.get('timestamp', ''),
                'date': datetime.fromisoformat(item.get('timestamp', '')).date() if item.get('timestamp') else None
            })
    
    if not history_data:
        st.info("No valid analysis data found in history.")
        return
    
    df = pd.DataFrame(history_data)
    
    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Analyses", len(df))
    with col2:
        st.metric("Average Score", f"{df['overall_score'].mean():.1f}%")
    with col3:
        st.metric("Authentic Images", len(df[df['overall_score'] >= 80]))
    with col4:
        st.metric("Suspicious/Fake", len(df[df['overall_score'] < 80]))
    
    # Score distribution chart
    st.subheader("📈 Score Distribution")
    fig = px.histogram(df, x='overall_score', nbins=20, 
                      title="Distribution of Authenticity Scores",
                      labels={'overall_score': 'Authenticity Score (%)', 'count': 'Number of Images'})
    fig.add_vline(x=80, line_dash="dash", line_color="green", annotation_text="Authentic Threshold")
    fig.add_vline(x=50, line_dash="dash", line_color="orange", annotation_text="Suspicious Threshold")
    st.plotly_chart(fig, use_container_width=True)
    
    # Time series analysis
    if 'date' in df.columns and df['date'].notna().any():
        st.subheader("📅 Analysis Over Time")
        daily_stats = df.groupby('date').agg({
            'overall_score': ['mean', 'count'],
            'ai_probability': 'mean',
            'editing_probability': 'mean'
        }).reset_index()
        
        daily_stats.columns = ['date', 'avg_score', 'count', 'avg_ai', 'avg_editing']
        
        fig = px.line(daily_stats, x='date', y='avg_score', 
                     title="Average Authenticity Score Over Time",
                     labels={'avg_score': 'Average Score (%)', 'date': 'Date'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Risk analysis
    st.subheader("🎯 Risk Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.scatter(df, x='ai_probability', y='editing_probability', 
                        color='overall_score', size='overall_score',
                        title="AI vs Editing Risk",
                        labels={'ai_probability': 'AI Generation Risk (%)', 
                               'editing_probability': 'Editing Detection (%)',
                               'overall_score': 'Overall Score'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Top 10 most suspicious images
        suspicious_df = df.nsmallest(10, 'overall_score')
        fig = px.bar(suspicious_df, x='filename', y='overall_score',
                    title="Top 10 Most Suspicious Images",
                    labels={'overall_score': 'Authenticity Score (%)', 'filename': 'Filename'})
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

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
    """Display AI analysis results with detailed reasoning"""
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    
    st.subheader("🤖 AI Generation Analysis")
    
    # Overall AI probability
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("AI Generation Probability", f"{ai_results['ai_probability']:.1f}%")
    with col2:
        st.metric("Confidence", f"{ai_results['confidence']:.1%}")
    with col3:
        st.metric("Model Used", ai_results['model_used'])
    
    # Risk level
    ai_prob = ai_results['ai_probability']
    if ai_prob < 30:
        risk_level = "🟢 Low Risk"
        risk_color = "#28a745"
        risk_description = "This image shows minimal signs of AI generation"
    elif ai_prob < 70:
        risk_level = "🟡 Medium Risk"
        risk_color = "#ffc107"
        risk_description = "This image shows some signs of potential AI generation"
    else:
        risk_level = "🔴 High Risk"
        risk_color = "#dc3545"
        risk_description = "This image shows strong signs of AI generation"
    
    st.markdown(f"""
    <div style="background-color: {risk_color}; color: white; padding: 1rem; border-radius: 10px; text-align: center; margin: 1rem 0;">
        <h4>Risk Level: {risk_level}</h4>
        <p>{risk_description}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Detailed reasoning section
    if 'details' in ai_results and 'detailed_reasons' in ai_results['details']:
        st.subheader("📋 Detailed Analysis Report")
        st.markdown("**Why this image was classified as AI-generated or not:**")
        
        reasons = ai_results['details']['detailed_reasons']
        if reasons:
            for reason in reasons:
                st.markdown(f"<div style='background: #f8f9fa; padding: 0.5rem; margin: 0.5rem 0; border-radius: 5px; border-left: 3px solid #007bff;'>{reason}</div>", unsafe_allow_html=True)
        else:
            st.info("No detailed reasoning available for this analysis.")
    
    # Detection methods with individual scores
    if ai_results['detection_methods']:
        st.subheader("🔬 Individual Method Results:")
        
        # Create a DataFrame for better display
        methods_data = []
        for method in ai_results['detection_methods']:
            methods_data.append({
                'Method': method['name'],
                'Probability': f"{method['probability']:.1f}%",
                'Confidence': f"{method['confidence']:.1%}",
                'Status': "🟢 Low" if method['probability'] < 30 else "🟡 Medium" if method['probability'] < 70 else "🔴 High"
            })
        
        df = pd.DataFrame(methods_data)
        st.dataframe(df, use_container_width=True)
        
        # Method details
        if 'details' in ai_results and 'method_scores' in ai_results['details']:
            with st.expander("🔍 Detailed Method Analysis", expanded=False):
                method_scores = ai_results['details']['method_scores']
                for method_name, method_data in method_scores.items():
                    st.write(f"**{method_name.replace('_', ' ').title()}**")
                    if 'details' in method_data:
                        details = method_data['details']
                        if 'reasoning' in details:
                            st.write(f"*{details['reasoning']}*")
                        
                        # Show technical metrics
                        tech_metrics = {k: v for k, v in details.items() if k not in ['reasoning', 'error']}
                        if tech_metrics:
                            for metric, value in tech_metrics.items():
                                if isinstance(value, (int, float)):
                                    st.write(f"• {metric.replace('_', ' ').title()}: {value:.3f}")
                                else:
                                    st.write(f"• {metric.replace('_', ' ').title()}: {value}")
                    st.divider()
    
    # Technical details
    if 'details' in ai_results:
        st.subheader("⚙️ Technical Information:")
        details = ai_results['details']
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"• **Methods Used:** {details.get('methods_used', 'N/A')}")
            st.write(f"• **Device:** {details.get('device', 'N/A')}")
        with col2:
            st.write(f"• **Models Available:** {details.get('models_available', 'N/A')}")
            st.write(f"• **Analysis Time:** {datetime.now().strftime('%H:%M:%S')}")
    
    # How it works explanation
    st.subheader("📚 How AI Detection Works:")
    st.markdown("""
    Our advanced AI detection system uses **10 different analysis methods**:
    
    1. **Deep Learning** (25% weight) - Neural networks trained to identify AI patterns
    2. **CLIP Analysis** (20% weight) - Compares image with text descriptions
    3. **Statistical Analysis** (15% weight) - Analyzes pixel distribution patterns
    4. **Texture Analysis** (12% weight) - Examines surface texture characteristics
    5. **Frequency Domain** (10% weight) - Analyzes frequency patterns in the image
    6. **Noise Analysis** (8% weight) - Studies noise patterns and characteristics
    7. **Edge Analysis** (5% weight) - Examines edge patterns and consistency
    8. **Color Analysis** (3% weight) - Analyzes color distribution and uniformity
    9. **Artifact Detection** (1% weight) - Looks for AI generation artifacts
    10. **Consistency Analysis** (1% weight) - Checks overall image consistency
    
    **Each method provides specific reasoning** for why it thinks the image is AI-generated or not.
    The final probability is a **weighted combination** of all methods.
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

def generate_batch_report(batch_results):
    """Generate a comprehensive batch analysis PDF report"""
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
    story.append(Paragraph("IRLCheck-Clickdoo Batch Analysis Report", title_style))
    story.append(Spacer(1, 20))
    
    # Summary
    successful = [r for r in batch_results if r['status'] == 'success']
    errors = [r for r in batch_results if r['status'] == 'error']
    
    summary_text = f"""
    Batch Analysis Summary:
    Total Files: {len(batch_results)}
    Successful Analyses: {len(successful)}
    Errors: {len(errors)}
    Analysis Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    """
    story.append(Paragraph("Executive Summary", styles['Heading2']))
    story.append(Paragraph(summary_text, styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Successful results
    if successful:
        story.append(Paragraph("Successful Analyses", styles['Heading2']))
        
        # Create table
        table_data = [['Filename', 'Overall Score', 'AI Risk', 'Editing', 'Status']]
        for result in successful:
            status = "Authentic" if result['overall_score'] >= 80 else "Suspicious" if result['overall_score'] >= 50 else "Fake"
            table_data.append([
                result['filename'],
                f"{result['overall_score']:.1f}%",
                f"{result['ai_probability']:.1f}%",
                f"{result['editing_probability']:.1f}%",
                status
            ])
        
        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(table)
        story.append(Spacer(1, 20))
    
    # Errors
    if errors:
        story.append(Paragraph("Errors", styles['Heading2']))
        for error in errors:
            story.append(Paragraph(f"• {error['filename']}: {error['error']}", styles['Normal']))
        story.append(Spacer(1, 20))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

if __name__ == "__main__":
    main() 
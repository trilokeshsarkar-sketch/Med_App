import streamlit as st
from medical_processor import MedicalOCRProcessor
from datetime import datetime
import json
import time
import os
import shutil
import subprocess
import sys

# Configure Tesseract path with automatic detection
import pytesseract

def setup_tesseract():
    """Setup Tesseract path with multiple fallback options"""
    tesseract_found = False
    
    # Try to find tesseract in PATH
    tesseract_path = shutil.which("tesseract")
    if tesseract_path:
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        st.success(f"✅ Tesseract found in PATH: {tesseract_path}")
        tesseract_found = True
        return tesseract_found
    
    # Common installation paths for different OS
    common_paths = []
    
    if sys.platform == "win32":
        # Windows paths
        common_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            r"C:\Users\{}\AppData\Local\Tesseract-OCR\tesseract.exe".format(os.getlogin()),
        ]
    elif sys.platform == "darwin":
        # macOS paths
        common_paths = [
            "/usr/local/bin/tesseract",
            "/opt/homebrew/bin/tesseract",
            "/usr/bin/tesseract",
        ]
    else:
        # Linux paths
        common_paths = [
            "/usr/bin/tesseract",
            "/usr/local/bin/tesseract",
            "/snap/bin/tesseract",
        ]
    
    # Check common paths
    for path in common_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            st.success(f"✅ Tesseract found at: {path}")
            tesseract_found = True
            return tesseract_found
    
    # Try to install tesseract using package managers (for cloud environments)
    if not tesseract_found:
        st.warning("⚠️ Tesseract not found. Attempting to install...")
        
        try:
            if sys.platform == "linux" or sys.platform == "linux2":
                # Try to install on Linux
                result = subprocess.run(['apt-get', 'update'], capture_output=True, text=True)
                result = subprocess.run(['apt-get', 'install', '-y', 'tesseract-ocr'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    tesseract_path = shutil.which("tesseract")
                    if tesseract_path:
                        pytesseract.pytesseract.tesseract_cmd = tesseract_path
                        st.success(f"✅ Tesseract installed and found at: {tesseract_path}")
                        tesseract_found = True
                        return tesseract_found
            
            elif sys.platform == "darwin":
                # Try Homebrew on macOS
                result = subprocess.run(['brew', 'install', 'tesseract'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    tesseract_path = shutil.which("tesseract")
                    if tesseract_path:
                        pytesseract.pytesseract.tesseract_cmd = tesseract_path
                        st.success(f"✅ Tesseract installed and found at: {tesseract_path}")
                        tesseract_found = True
                        return tesseract_found
                        
        except Exception as e:
            st.warning(f"Could not auto-install Tesseract: {e}")
    
    return tesseract_found

# Setup Tesseract
tesseract_available = setup_tesseract()

if not tesseract_available:
    st.error("""
    ❌ Tesseract OCR is not installed!
    
    **Please install Tesseract OCR:**
    
    **Windows:**
    - Download from: https://github.com/UB-Mannheim/tesseract/wiki
    - Add Tesseract to your PATH environment variable
    
    **macOS:**
    ```bash
    brew install tesseract
    ```
    
    **Linux (Ubuntu/Debian):**
    ```bash
    sudo apt-get update
    sudo apt-get install tesseract-ocr
    ```
    
    **For Streamlit Cloud:**
    Add this to your requirements.txt:
    ```
    pytesseract
    ```
    And ensure Tesseract is installed in your environment.
    """)

# Download NLTK data for TextBlob if needed
try:
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('brown', quiet=True)
except:
    pass

def load_default_api_key():
    """Load API key from key.txt if available"""
    if os.path.exists("key.txt"):
        with open("key.txt", "r", encoding="utf-8") as f:
            return f.read().strip()
    return ""

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Medical OCR Analyzer",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header { font-size: 2.5rem; color: #1f77b4; text-align: center; margin-bottom: 1.5rem; }
    .sub-header { font-size: 1.4rem; color: #2ca02c; margin-top: 1.2rem; margin-bottom: 0.8rem; }
    .success-box { background-color: #d4edda; color: #155724; padding: 12px; border-radius: 6px; margin: 8px 0; }
    .error-box { background-color: #f8d7da; color: #721c24; padding: 12px; border-radius: 6px; margin: 8px 0; }
    .warning-box { background-color: #fff3cd; color: #856404; padding: 12px; border-radius: 6px; margin: 8px 0; }
    .text-display, .analysis-display {
        max-height: 600px; overflow-y: auto; border: 1px solid #ddd; padding: 15px;
        border-radius: 8px; background-color: #ffffff; white-space: pre-wrap;
        color: #000000; line-height: 1.4; font-family: monospace;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🏥 Medical OCR Analyzer</h1>', unsafe_allow_html=True)
    
    # Check if Tesseract is properly configured
    try:
        pytesseract.get_tesseract_version()
        tesseract_available = True
        st.success("✅ Tesseract is properly configured!")
    except Exception as e:
        tesseract_available = False
        st.error(f"❌ Tesseract configuration error: {e}")
    
    # Initialize processor only if Tesseract is available
    if tesseract_available:
        if 'processor' not in st.session_state:
            st.session_state.processor = MedicalOCRProcessor()
        
        processor = st.session_state.processor
        
        # Sidebar
        with st.sidebar:
            st.header("🔑 API Configuration")
            default_api_key = load_default_api_key()
            api_key = st.text_input(
                "OpenRouter API Key",
                type="password",
                value=default_api_key,
                help="Get your API key from https://openrouter.ai"
            )
            
            st.header("📋 Instructions")
            st.markdown("""
            1. Enter your OpenRouter API key
            2. Upload medical documents
            3. Click 'Process Documents'
            """)
            
            # Debug info
            with st.expander("🔧 Debug Info"):
                try:
                    st.write(f"Python: {sys.version}")
                    st.write(f"Platform: {sys.platform}")
                    st.write(f"Tesseract path: {pytesseract.pytesseract.tesseract_cmd}")
                    st.write(f"Tesseract version: {pytesseract.get_tesseract_version()}")
                except:
                    st.write("Tesseract not available")
        
        # File upload
        st.markdown('<div class="sub-header">📁 Upload Medical Documents</div>', unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "Choose medical documents to analyze",
            type=['png', 'jpg', 'jpeg', 'tiff', 'bmp', 'pdf'],
            accept_multiple_files=True,
            help="Upload images or PDFs of medical documents"
        )
        
        if uploaded_files:
            if st.button("🚀 Process Documents", use_container_width=True, type="primary"):
                # Initialize progress trackers
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_file_progress(completed, total):
                    progress_bar.progress(completed / total)
                    status_text.text(f"Processing {completed}/{total} files...")
                
                # Process files
                with st.spinner("Processing documents..."):
                    try:
                        data, all_ocr_text, combined_text_path, processed_files = processor.process_uploaded_files(
                            uploaded_files, progress_callback=update_file_progress
                        )
                    except Exception as e:
                        st.error(f"Error processing files: {e}")
                        progress_bar.empty()
                        status_text.empty()
                        return
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                # Display results
                if processed_files:
                    st.markdown('<div class="sub-header">📊 Processing Results</div>', unsafe_allow_html=True)
                    
                    for file_info in processed_files:
                        status_class = "success-box" if file_info['status'] == 'Success' else "error-box"
                        st.markdown(f"""
                        <div class="{status_class}">
                            <b>{file_info['filename']}</b> ({file_info['type']}) - {file_info['status']}<br>
                            {f"Extracted {file_info['text_length']} characters" if 'text_length' in file_info else ''}
                            {f"- {file_info['page_count']} pages" if 'page_count' in file_info else ''}
                        </div>
                        """, unsafe_allow_html=True)
                
                # Show extracted text
                if all_ocr_text:
                    st.markdown('<div class="sub-header">📝 Full Extracted Text</div>', unsafe_allow_html=True)
                    
                    with st.expander("View Full Extracted Text", expanded=False):
                        st.markdown(f'<div class="text-display">{all_ocr_text}</div>', unsafe_allow_html=True)
                        
                        st.download_button(
                            label="📥 Download Full Text",
                            data=all_ocr_text,
                            file_name="full_extracted_text.txt",
                            mime="text/plain"
                        )
                    
                    # Analyze text if API key is provided
                    if api_key:
                        st.markdown('<div class="sub-header">🔍 Medical Analysis</div>', unsafe_allow_html=True)
                        
                        text_for_analysis = all_ocr_text
                        
                        # Setup analysis progress
                        analysis_progress = st.progress(0)
                        analysis_status = st.empty()
                        
                        def update_analysis_progress(chunk_index, total_chunks, attempt):
                            analysis_progress.progress(chunk_index / total_chunks)
                            analysis_status.text(f"Analyzing chunk {chunk_index}/{total_chunks} (Attempt {attempt})...")
                        
                        analysis_text = processor.analyze_medical_text(
                            text_for_analysis, api_key, progress_callback=update_analysis_progress
                        )
                        
                        # Clear analysis progress
                        analysis_progress.empty()
                        analysis_status.empty()
                        
                        if analysis_text:
                            if analysis_text.startswith("❌"):
                                st.error(analysis_text)
                            else:
                                st.markdown(f'<div class="analysis-display">{analysis_text}</div>', unsafe_allow_html=True)
                                
                                # Save and provide download options
                                json_file, text_file = processor.save_results(analysis_text, combined_text_path)
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.download_button(
                                        label="📥 Download Analysis (TXT)",
                                        data=analysis_text,
                                        file_name="medical_analysis.txt",
                                        mime="text/plain",
                                        use_container_width=True
                                    )
                                with col2:
                                    analysis_data = {
                                        "timestamp": str(datetime.now()),
                                        "analysis": analysis_text,
                                        "processed_files": [f["filename"] for f in processed_files if f["status"] == "Success"],
                                        "text_length_original": len(all_ocr_text),
                                        "text_length_analyzed": len(text_for_analysis),
                                        "truncated": False
                                    }
                                    st.download_button(
                                        label="📥 Download Analysis (JSON)",
                                        data=json.dumps(analysis_data, indent=2),
                                        file_name="medical_analysis.json",
                                        mime="application/json",
                                        use_container_width=True
                                    )
                    else:
                        st.warning("Please enter your OpenRouter API key to enable medical analysis.")
        
        elif not uploaded_files:
            st.info("📋 Please upload medical documents to get started.")
    else:
        st.info("Please install Tesseract OCR to use this application.")

if __name__ == "__main__":
    main()

import streamlit as st
from medical_processor import MedicalOCRProcessor
from datetime import datetime
import json
import time
import os

# Configure Tesseract path with automatic detection
import pytesseract

# Search for tesseract binary in path
@st.cache_resource
def find_tesseract_binary() -> str:
    """Find Tesseract binary in system PATH"""
    return shutil.which("tesseract")

# Set tesseract binary path
tesseract_path = find_tesseract_binary()
if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
    st.success(f"✅ Tesseract found at: {tesseract_path}")
else:
    # Try common installation paths as fallback
    common_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        "/usr/bin/tesseract",
        "/usr/local/bin/tesseract",
    ]
    
    for path in common_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            st.success(f"✅ Tesseract found at: {path}")
            break
    else:
        st.error("❌ Tesseract not found. Please install Tesseract OCR")
        st.info("""
        **Installation Instructions:**
        - **Windows**: Download from https://github.com/UB-Mannheim/tesseract/wiki
        - **macOS**: `brew install tesseract`
        - **Linux**: `sudo apt-get install tesseract-ocr`
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
    
    # Custom CSS - Improved text visibility
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
    .stProgress > div > div > div > div { background-color: #1f77b4; }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🏥 Medical OCR Analyzer</h1>', unsafe_allow_html=True)
    
    # Initialize processor
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
        1. Enter your OpenRouter API key (or auto-loaded from `key.txt`)
        2. Upload medical documents
        3. Click 'Process Documents'
        """)
    
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
                data, all_ocr_text, combined_text_path, processed_files = processor.process_uploaded_files(
                    uploaded_files, progress_callback=update_file_progress
                )
            
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
                    
                    text_for_analysis = all_ocr_text  # No truncation
                    
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

if __name__ == "__main__":
    main()





import streamlit as st
from medical_processor import MedicalOCRProcessor
from datetime import datetime
import json
import os

def load_default_api_key():
    """Load API key from key.txt if available"""
    if os.path.exists("key.txt"):
        with open("key.txt", "r", encoding="utf-8") as f:
            return f.read().strip()
    return ""

def main():
    st.set_page_config(
        page_title="Medical OCR Analyzer (LLM-powered)",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown('<h1 style="text-align:center;color:#1f77b4;">🏥 Medical OCR Analyzer (via OpenRouter)</h1>', unsafe_allow_html=True)

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
            help="Auto-loads from key.txt if available"
        )

        if api_key:
            st.success("✅ API key loaded")
        else:
            st.warning("⚠️ No API key found. Please add key.txt or enter manually.")

    # File upload
    st.subheader("📁 Upload Medical Documents")
    uploaded_files = st.file_uploader(
        "Upload images or PDFs",
        type=['png', 'jpg', 'jpeg', 'tiff', 'bmp', 'pdf'],
        accept_multiple_files=True
    )

    if uploaded_files and api_key:
        if st.button("🚀 Process & Analyze Documents", use_container_width=True, type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_file_progress(completed, total):
                progress_bar.progress(completed / total)
                status_text.text(f"Processing {completed}/{total} files...")

            with st.spinner("Extracting documents..."):
                _, all_ocr_text, combined_text_path, processed_files = processor.process_uploaded_files(
                    uploaded_files, progress_callback=update_file_progress
                )

            progress_bar.empty()
            status_text.empty()

            if processed_files:
                st.subheader("📊 Processing Results")
                for f in processed_files:
                    st.success(f"{f['filename']} ({f['type']}) - {f['status']}")

            st.subheader("🔍 OCR + Medical Analysis")
            with st.spinner("Analyzing via OpenRouter..."):
                analysis_text = processor.analyze_medical_text(all_ocr_text, api_key)

            if analysis_text.startswith("❌"):
                st.error(analysis_text)
            else:
                st.markdown(f"<div style='white-space:pre-wrap;'>{analysis_text}</div>", unsafe_allow_html=True)

                json_file, text_file = processor.save_results(analysis_text, combined_text_path)

                col1, col2 = st.columns(2)
                with col1:
                    st.download_button("📥 Download TXT", analysis_text, "medical_analysis.txt", "text/plain")
                with col2:
                    st.download_button("📥 Download JSON", json.dumps({"analysis": analysis_text}, indent=2),
                                       "medical_analysis.json", "application/json")

    elif uploaded_files and not api_key:
        st.warning("⚠️ Please enter your OpenRouter API key.")

    else:
        st.info("📋 Upload medical documents to begin.")

if __name__ == "__main__":
    main()

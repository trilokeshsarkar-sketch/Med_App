import streamlit as st
import fitz  # PyMuPDF
from PIL import Image
import io, os, base64, requests, json
from datetime import datetime


# ========== Utility Functions ==========

def load_default_api_key():
    """Load API key from key.txt if available"""
    if os.path.exists("key.txt"):
        with open("key.txt", "r", encoding="utf-8") as f:
            return f.read().strip()
    return ""


def pdf_to_images(pdf_file):
    """Convert PDF pages to list of PIL images"""
    images = []
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=200)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        images.append(img)
    return images


def image_to_base64(image):
    """Convert PIL image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def ocr_via_openrouter(images, api_key, progress_callback=None):
    """Send images to OpenRouter model for OCR"""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "Medical OCR Analyzer"
    }

    all_text = []
    total = len(images)

    for idx, img in enumerate(images, start=1):
        if progress_callback:
            progress_callback(idx, total)

        img_b64 = image_to_base64(img)
        payload = {
            "model": "gpt-4.1-mini",  # ✅ free-tier OCR-capable model
            "messages": [
                {"role": "system", "content": "You are an OCR assistant. Extract all text exactly as seen in the image."},
                {"role": "user", "content": [
                    {"type": "text", "text": f"OCR page {idx}:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                ]}
            ]
        }

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            if resp.status_code == 200:
                text = resp.json()["choices"][0]["message"]["content"]
                all_text.append(text)
            else:
                all_text.append(f"❌ OCR failed for page {idx}: {resp.text}")
        except Exception as e:
            all_text.append(f"⚠️ OCR error on page {idx}: {str(e)}")

    return "\n\n".join(all_text)


# ========== Streamlit App ==========

def main():
    st.set_page_config(
        page_title="Medical OCR Analyzer",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown('<h1 style="text-align:center;color:#1f77b4;">🏥 Medical OCR Analyzer</h1>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("🔑 API Configuration")
        default_api_key = load_default_api_key()
        api_key = st.text_input(
            "OpenRouter API Key",
            type="password",
            value=default_api_key,
            help="Stored in key.txt or paste here"
        )

        st.header("📋 Instructions")
        st.markdown("""
        1. Enter your OpenRouter API key  
        2. Upload PDFs or images  
        3. Click 'Process Documents'
        """)

    # File uploader
    uploaded_files = st.file_uploader(
        "📁 Upload Medical Documents",
        type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp"],
        accept_multiple_files=True
    )

    if uploaded_files and api_key:
        if st.button("🚀 Process Documents", use_container_width=True, type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(done, total):
                progress_bar.progress(done / total)
                status_text.text(f"Processing page {done}/{total}...")

            full_text = ""
            file_results = []

            for file in uploaded_files:
                try:
                    if file.type == "application/pdf":
                        images = pdf_to_images(file)
                    else:
                        img = Image.open(file)
                        images = [img]

                    text = ocr_via_openrouter(images, api_key, progress_callback=update_progress)

                    full_text += f"\n\n===== {file.name} =====\n\n{text}\n"
                    file_results.append({"filename": file.name, "status": "Success", "pages": len(images)})
                except Exception as e:
                    file_results.append({"filename": file.name, "status": f"Failed - {str(e)}"})

            progress_bar.empty()
            status_text.empty()

            # Results summary
            st.subheader("📊 Processing Results")
            for res in file_results:
                if res["status"] == "Success":
                    st.success(f"{res['filename']} - ✅ Success ({res['pages']} pages)")
                else:
                    st.error(f"{res['filename']} - ❌ {res['status']}")

            # Full text
            if full_text.strip():
                st.subheader("📝 Extracted Text")
                with st.expander("View Full OCR Text", expanded=False):
                    st.text_area("OCR Result", full_text, height=400)

                    st.download_button(
                        label="📥 Download OCR Result (TXT)",
                        data=full_text,
                        file_name=f"ocr_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )

                # Save JSON too
                result_json = {
                    "timestamp": str(datetime.now()),
                    "results": file_results,
                    "text": full_text
                }
                st.download_button(
                    label="📥 Download OCR Result (JSON)",
                    data=json.dumps(result_json, indent=2),
                    file_name=f"ocr_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

    elif uploaded_files and not api_key:
        st.warning("⚠️ Please provide your OpenRouter API key.")


if __name__ == "__main__":
    main()

import fitz  # PyMuPDF
from PIL import Image
import io, base64, requests

class MedicalOCRProcessor:
    def __init__(self):
        pass

    def pdf_to_images(self, pdf_file):
        """Convert PDF pages to list of PIL images"""
        images = []
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(dpi=200)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            images.append(img)
        return images

    def image_to_base64(self, image):
        """Convert PIL image to base64 string"""
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def ocr_via_openrouter(self, images, api_key):
        """Send images to OpenRouter model for OCR"""
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "http://localhost:8501",
            "X-Title": "Medical OCR Analyzer"
        }

        all_text = []
        for idx, img in enumerate(images, start=1):
            img_b64 = self.image_to_base64(img)
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

            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            if resp.status_code == 200:
                text = resp.json()["choices"][0]["message"]["content"]
                all_text.append(text)
            else:
                all_text.append(f"❌ OCR failed for page {idx}: {resp.text}")

        return "\n\n".join(all_text)

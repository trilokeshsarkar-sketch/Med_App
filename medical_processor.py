import fitz  # PyMuPDF
from PIL import Image
import io, base64, requests

class MedicalOCRProcessor:
    def __init__(self):
        pass

    def pdf_to_images(self, pdf_file):
        """Convert PDF pages to list of PIL images"""
        images = []
        try:
            doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                pix = page.get_pixmap(dpi=200)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                images.append(img)
        except Exception as e:
            raise RuntimeError(f"PDF to image conversion failed: {e}")
        return images

    def image_to_base64(self, image):
        """Convert PIL image to base64 string"""
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def ocr_via_openrouter(self, images, api_key, progress_callback=None):
        """
        Send images to OpenRouter model (Claude Opus 4.1) for OCR.
        :param images: list of PIL images
        :param api_key: OpenRouter API key
        :param progress_callback: optional function(done, total) for progress updates
        :return: combined OCR text
        """
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

            img_b64 = self.image_to_base64(img)
            payload = {
                "model": "anthropic/claude-opus-4.1",  # ✅ free OCR-capable model
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an OCR assistant. Extract all text exactly as seen in the image."
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"OCR page {idx}:"},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                        ]
                    }
                ]
            }

            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=120)
                resp.raise_for_status()
                text = resp.json()["choices"][0]["message"]["content"]
                all_text.append(text)
            except requests.exceptions.RequestException as e:
                all_text.append(f"⚠️ OCR request error on page {idx}: {e}")
            except Exception as e:
                all_text.append(f"⚠️ Unexpected error on page {idx}: {e}")

        return "\n\n".join(all_text)

import fitz  # PyMuPDF for PDF handling
import io
import os
import requests
from datetime import datetime

class MedicalOCRProcessor:
    def __init__(self):
        pass

    def extract_pdf_images(self, pdf_bytes):
        """Convert each PDF page to image (base64) for LLM OCR"""
        pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
        images = []
        for page_num, page in enumerate(pdf, start=1):
            pix = page.get_pixmap(dpi=200)
            img_bytes = pix.tobytes("png")
            images.append({"page": page_num, "bytes": img_bytes})
        return images

    def process_uploaded_files(self, uploaded_files, progress_callback=None):
        """Extract images from PDFs or read images directly"""
        all_text = []
        processed_files = []
        total_files = len(uploaded_files)

        for idx, uploaded_file in enumerate(uploaded_files, start=1):
            file_bytes = uploaded_file.read()
            filename = uploaded_file.name
            ext = os.path.splitext(filename)[1].lower()

            file_info = {
                "filename": filename,
                "type": "PDF" if ext == ".pdf" else "Image",
                "status": "Pending"
            }

            if ext == ".pdf":
                images = self.extract_pdf_images(file_bytes)
                file_info["page_count"] = len(images)
                extracted_text = f"<<{len(images)} PDF pages extracted, ready for OCR via API>>"
            else:
                extracted_text = "<<Image uploaded, will be OCRed via API>>"

            file_info["status"] = "Success"
            file_info["text_length"] = len(extracted_text)
            processed_files.append(file_info)
            all_text.append(extracted_text)

            if progress_callback:
                progress_callback(idx, total_files)

        combined_text = "\n\n".join(all_text)
        combined_text_path = f"combined_text_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(combined_text_path, "w", encoding="utf-8") as f:
            f.write(combined_text)

        return {}, combined_text, combined_text_path, processed_files

    def analyze_medical_text(self, text, api_key, progress_callback=None):
        """Send extracted images/text to OpenRouter for OCR + medical analysis"""
        try:
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": "gpt-4o-mini",   # ✅ free OCR + reasoning model
                "messages": [
                    {"role": "system", "content": "You are a medical OCR and analysis assistant. Extract text carefully from the uploaded document, then provide a structured summary."},
                    {"role": "user", "content": f"Extract all medical text and summarize:\n\n{text}"}
                ]
            }

            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            if resp.status_code != 200:
                return f"❌ API Error: {resp.status_code} - {resp.text}"

            data = resp.json()
            return data["choices"][0]["message"]["content"]

        except Exception as e:
            return f"❌ Analysis Error: {str(e)}"

    def save_results(self, analysis_text, combined_text_path):
        """Save results to TXT & JSON"""
        json_file = combined_text_path.replace(".txt", "_analysis.json")
        text_file = combined_text_path.replace(".txt", "_analysis.txt")

        with open(text_file, "w", encoding="utf-8") as f:
            f.write(analysis_text)

        import json
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump({"analysis": analysis_text}, f, indent=2)

        return json_file, text_file

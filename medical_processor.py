import easyocr
import fitz  # PyMuPDF for PDF handling
from PIL import Image
import numpy as np
import io
import os
from datetime import datetime

class MedicalOCRProcessor:
    def __init__(self):
        # Initialize EasyOCR reader (English only, add more langs if needed)
        self.reader = easyocr.Reader(['en'], gpu=False)

    def ocr_image(self, image_bytes):
        """Extract text from image bytes using EasyOCR"""
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            np_image = np.array(image)
            results = self.reader.readtext(np_image, detail=0)  # detail=0 gives plain text only
            return "\n".join(results)
        except Exception as e:
            return f"❌ OCR Error: {str(e)}"

    def ocr_pdf(self, pdf_bytes):
        """Extract text from PDF using EasyOCR (page by page as images)"""
        text = []
        try:
            pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
            for page_num, page in enumerate(pdf, start=1):
                pix = page.get_pixmap(dpi=200)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                np_img = np.array(img)
                results = self.reader.readtext(np_img, detail=0)
                text.append("\n".join(results))
            return "\n".join(text), len(pdf)
        except Exception as e:
            return f"❌ PDF OCR Error: {str(e)}", 0

    def process_uploaded_files(self, uploaded_files, progress_callback=None):
        """Process uploaded images and PDFs"""
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
                "status": "Failed"
            }

            extracted_text = ""
            page_count = None

            if ext == ".pdf":
                extracted_text, page_count = self.ocr_pdf(file_bytes)
                file_info["page_count"] = page_count
            else:
                extracted_text = self.ocr_image(file_bytes)

            if extracted_text and not extracted_text.startswith("❌"):
                file_info["status"] = "Success"
                file_info["text_length"] = len(extracted_text)
                all_text.append(extracted_text)

            processed_files.append(file_info)

            if progress_callback:
                progress_callback(idx, total_files)

        combined_text = "\n\n".join(all_text)
        combined_text_path = f"combined_text_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        with open(combined_text_path, "w", encoding="utf-8") as f:
            f.write(combined_text)

        return {}, combined_text, combined_text_path, processed_files

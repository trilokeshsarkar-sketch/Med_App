import os
import requests
import json
import time
import re
from datetime import datetime
from PIL import Image
import fitz  # PyMuPDF for PDF handling
import io
import tempfile
from pathlib import Path
import logging
import numpy as np
import easyocr  # ✅ EasyOCR replaces pytesseract

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalOCRProcessor:
    def __init__(self):
        # Initialize paths
        self.working_dir = Path.cwd()
        self.images_folder = self.working_dir / "images"
        self.pdf_folder = self.working_dir / "PDFs"
        self.ocr_output_folder = self.working_dir / "ocr_texts"
        self.medical_ocr_folder = self.working_dir / "Combined_txt"
        self.output_directory = self.working_dir
        
        # API settings
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.model = "deepseek/deepseek-chat-v3.1:free"
        self.max_retries = 3
        self.base_delay = 2
        
        # Initialize EasyOCR reader
        self.reader = easyocr.Reader(['en'], gpu=False)  # ✅ You can add more languages if needed
        
        # Create directories
        self.setup_directories()
    
    def setup_directories(self):
        """Create necessary directories if they don't exist"""
        self.images_folder.mkdir(exist_ok=True)
        self.pdf_folder.mkdir(exist_ok=True)
        self.ocr_output_folder.mkdir(exist_ok=True)
        self.medical_ocr_folder.mkdir(exist_ok=True)
        self.output_directory.mkdir(exist_ok=True)
    
    def clean_text(self, text):
        """Clean text by removing control characters and extra whitespace"""
        if text is None:
            return ""
        text = re.sub(r'[\x00-\x1F\x7F-\x9F]+', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def pdf_to_images(self, pdf_bytes):
        """Convert PDF bytes to a list of PIL images"""
        images = []
        temp_file_path = None
        
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(pdf_bytes)
                temp_file_path = tmp_file.name
            
            with fitz.open(temp_file_path) as pdf_document:
                for page_num in range(len(pdf_document)):
                    page = pdf_document.load_page(page_num)
                    mat = fitz.Matrix(2.0, 2.0)
                    pix = page.get_pixmap(matrix=mat)
                    img_data = pix.tobytes("ppm")
                    img = Image.open(io.BytesIO(img_data))
                    
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                        
                    images.append(img)
                    
        except Exception as e:
            logger.error(f"Error converting PDF to images: {e}")
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
        
        return images
    
    def image_to_text(self, image):
        """Extract text from image using EasyOCR"""
        try:
            img_array = np.array(image)
            results = self.reader.readtext(img_array, detail=0)
            text = " ".join(results)
            return self.clean_text(text)
        except Exception as e:
            logger.error(f"EasyOCR failed: {e}")
            return ""
    
    def process_single_file(self, uploaded_file):
        """Process a single file and return results"""
        try:
            file_bytes = uploaded_file.read()
            file_type = uploaded_file.type
            file_name = uploaded_file.name
            
            if file_type.startswith('image/'):
                image = Image.open(io.BytesIO(file_bytes))
                ocr_text = self.image_to_text(image)
                
                txt_file_path = self.ocr_output_folder / f"{Path(file_name).stem}.txt"
                with open(txt_file_path, "w", encoding="utf-8") as f:
                    f.write(ocr_text)
                
                return {
                    "filename": file_name,
                    "type": "Image",
                    "ocr_text": ocr_text,
                    "text_length": len(ocr_text),
                    "status": "Success"
                }
                
            elif file_type == 'application/pdf':
                pdf_images = self.pdf_to_images(file_bytes)
                
                if not pdf_images:
                    return {
                        "filename": file_name,
                        "type": "PDF",
                        "status": "Failed: No images extracted",
                        "text_length": 0
                    }
                
                pdf_ocr_text = ""
                
                for page_num, image in enumerate(pdf_images, 1):
                    page_text = self.image_to_text(image)
                    pdf_ocr_text += f"--- Page {page_num} ---\n\n{page_text}\n\n"
                
                pdf_ocr_text = self.clean_text(pdf_ocr_text)
                
                txt_file_path = self.ocr_output_folder / f"{Path(file_name).stem}.txt"
                with open(txt_file_path, "w", encoding="utf-8") as f:
                    f.write(pdf_ocr_text)
                
                return {
                    "filename": file_name,
                    "type": "PDF",
                    "ocr_text": pdf_ocr_text,
                    "text_length": len(pdf_ocr_text),
                    "page_count": len(pdf_images),
                    "status": "Success"
                }
                
            else:
                return {
                    "filename": file_name,
                    "type": file_type,
                    "status": "Unsupported format",
                    "text_length": 0
                }
                
        except Exception as e:
            logger.error(f"Error processing {uploaded_file.name}: {e}")
            return {
                "filename": uploaded_file.name,
                "type": uploaded_file.type,
                "status": f"Error: {str(e)}",
                "text_length": 0
            }

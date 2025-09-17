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
import torch
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalOCRProcessor:
    def __init__(self, output_base_dir=None):
        # Initialize paths
        self.working_dir = Path.cwd()
        
        # Use custom output directory if provided, otherwise use working directory
        if output_base_dir:
            self.output_base_dir = Path(output_base_dir)
        else:
            self.output_base_dir = self.working_dir
        
        # Create specific folders within the output base directory
        self.images_folder = self.output_base_dir / "images"
        self.pdf_folder = self.output_base_dir / "PDFs"
        self.ocr_output_folder = self.output_base_dir / "ocr_texts"
        self.medical_ocr_folder = self.output_base_dir / "Combined_txt"
        self.analysis_folder = self.output_base_dir / "analysis_results"
        
        # API settings
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.ocr_model = "nvidia/nemotron-nano-9b-v2:free"
        self.analysis_model = "deepseek/deepseek-chat-v3.1:free"
        self.max_retries = 3
        self.base_delay = 2
        
        # OCR settings
        self.use_nemotron = True  # Use NVIDIA Nemotron by default
        self.openrouter_api_key = None
        
        # Create directories
        self.setup_directories()
    
    def setup_directories(self):
        """Create necessary directories if they don't exist"""
        self.images_folder.mkdir(exist_ok=True, parents=True)
        self.pdf_folder.mkdir(exist_ok=True, parents=True)
        self.ocr_output_folder.mkdir(exist_ok=True, parents=True)
        self.medical_ocr_folder.mkdir(exist_ok=True, parents=True)
        self.analysis_folder.mkdir(exist_ok=True, parents=True)
    
    def clean_text(self, text):
        """Clean text by removing control characters and extra whitespace"""
        if text is None:
            return ""
        text = re.sub(r'[\x00-\x1F\x7F-\x9F]+', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def image_to_base64(self, image):
        """Convert PIL image to base64 string"""
        try:
            # Resize large images to avoid API limits
            max_size = (1024, 1024)
            if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
                image.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            buffered = io.BytesIO()
            image.save(buffered, format="PNG", optimize=True)
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
        except Exception as e:
            logger.error(f"Error converting image to base64: {e}")
            return None
    
    def extract_text_with_nemotron(self, image, api_key):
        """Extract text from image using NVIDIA Nemotron via OpenRouter"""
        if not api_key:
            return None, "No API key"
        
        for attempt in range(self.max_retries):
            try:
                # Convert image to base64
                image_base64 = self.image_to_base64(image)
                if not image_base64:
                    return None, "Image conversion failed"
                
                # Prepare the prompt with image
                prompt = {
                    "model": self.ocr_model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are an expert OCR system specialized in medical documents. Extract all text from the provided image exactly as it appears. Preserve formatting, spacing, and special characters. Return only the extracted text without any additional commentary. Be very accurate with medical terminology, numbers, and measurements."
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Extract all text from this medical document image with high precision. Pay special attention to medical terms, numbers, dates, and measurements."
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{image_base64}"
                                    }
                                }
                            ]
                        }
                    ],
                    "max_tokens": 4000,
                    "temperature": 0.1
                }
                
                # Send request to OpenRouter
                response = requests.post(
                    self.api_url,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=prompt,
                    timeout=90  # Increased timeout for image processing
                )
                
                if response.status_code == 429:
                    wait_time = self.exponential_backoff(attempt)
                    time.sleep(wait_time)
                    continue
                
                response.raise_for_status()
                result = response.json()
                
                extracted_text = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                return self.clean_text(extracted_text), "NVIDIA Nemotron"
                
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    wait_time = self.exponential_backoff(attempt)
                    time.sleep(wait_time)
                    continue
                logger.error(f"NVIDIA Nemotron OCR API error: {e}")
                return None, f"NVIDIA Nemotron failed: {str(e)}"
            except Exception as e:
                logger.error(f"NVIDIA Nemotron OCR error: {e}")
                return None, f"NVIDIA Nemotron failed: {str(e)}"
        
        return None, "NVIDIA Nemotron failed after multiple retries"
    
    def extract_text_with_tesseract(self, image):
        """Extract text from image using Tesseract (fallback)"""
        try:
            import pytesseract
            text = pytesseract.image_to_string(image)
            return self.clean_text(text), "Tesseract"
        except ImportError:
            logger.error("Tesseract not available")
            return None, "Tesseract not installed"
        except Exception as e:
            logger.error(f"Tesseract error: {e}")
            return None, f"Tesseract failed: {str(e)}"
    
    def extract_text_from_image(self, image, api_key=None):
        """Extract text from image using available OCR method"""
        if self.use_nemotron and api_key:
            text, method = self.extract_text_with_nemotron(image, api_key)
            if text and len(text.strip()) > 10:  # Minimum text length check
                return text, method
        
        # Fallback to Tesseract
        text, method = self.extract_text_with_tesseract(image)
        if text:
            return text, method
        
        return "OCR failed - no text extracted", "Failed"
    
    def pdf_to_images(self, pdf_bytes, dpi=200):
        """Convert PDF bytes to a list of high-quality PIL images"""
        images = []
        temp_file_path = None
        
        try:
            # Save PDF to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(pdf_bytes)
                temp_file_path = tmp_file.name
            
            # Open PDF and convert each page to image
            with fitz.open(temp_file_path) as pdf_document:
                for page_num in range(len(pdf_document)):
                    page = pdf_document.load_page(page_num)
                    
                    # Create high-resolution matrix for better OCR
                    zoom = dpi / 72  # 72 is the default DPI in PDF
                    mat = fitz.Matrix(zoom, zoom)
                    
                    # Render page to pixmap
                    pix = page.get_pixmap(matrix=mat)
                    
                    # Convert to PIL Image
                    img_data = pix.tobytes("ppm")
                    img = Image.open(io.BytesIO(img_data))
                    
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Enhance image for better OCR
                    img = self.enhance_image_for_ocr(img)
                    
                    images.append(img)
                    
        except Exception as e:
            logger.error(f"Error converting PDF to images: {e}")
        finally:
            # Clean up temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
        
        return images
    
    def enhance_image_for_ocr(self, image):
        """Enhance image for better OCR results"""
        try:
            # Convert to grayscale for better contrast
            if image.mode != 'L':
                image = image.convert('L')
            
            # Enhance contrast
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.5)  # Increase contrast
            
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.2)  # Increase sharpness
            
        except Exception as e:
            logger.warning(f"Image enhancement failed: {e}")
        
        return image
    
    def process_single_file(self, uploaded_file, api_key=None):
        """Process a single file and return results"""
        try:
            file_bytes = uploaded_file.read()
            file_type = uploaded_file.type
            file_name = uploaded_file.name
            
            if file_type.startswith('image/'):
                # Process image file
                image = Image.open(io.BytesIO(file_bytes))
                # Enhance image for better OCR
                image = self.enhance_image_for_ocr(image)
                ocr_text, method = self.extract_text_from_image(image, api_key)
                
                txt_file_path = self.ocr_output_folder / f"{Path(file_name).stem}.txt"
                with open(txt_file_path, "w", encoding="utf-8") as f:
                    f.write(ocr_text)
                
                return {
                    "filename": file_name,
                    "type": "Image",
                    "ocr_text": ocr_text,
                    "text_length": len(ocr_text),
                    "status": "Success",
                    "method": method
                }
                
            elif file_type == 'application/pdf':
                # Process PDF file - convert to images first
                pdf_images = self.pdf_to_images(file_bytes)
                
                if not pdf_images:
                    return {
                        "filename": file_name,
                        "type": "PDF",
                        "status": "Failed: No images extracted",
                        "text_length": 0
                    }
                
                pdf_ocr_text = ""
                method_used = "Tesseract"  # Default
                successful_pages = 0
                
                for page_num, image in enumerate(pdf_images, 1):
                    page_text, method = self.extract_text_from_image(image, api_key)
                    method_used = method  # Track which method was used
                    
                    if page_text and len(page_text.strip()) > 10:  # Valid text check
                        pdf_ocr_text += f"--- Page {page_num} ---\n\n{page_text}\n\n"
                        successful_pages += 1
                    else:
                        pdf_ocr_text += f"--- Page {page_num} ---\n\n[OCR failed or no text detected]\n\n"
                
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
                    "successful_pages": successful_pages,
                    "status": "Success",
                    "method": method_used
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
    
    def process_uploaded_files(self, uploaded_files, progress_callback=None):
        """Process all uploaded files sequentially"""
        all_ocr_text = ""
        processed_files_info = []
        file_data = []
        
        for i, uploaded_file in enumerate(uploaded_files):
            if progress_callback:
                progress_callback(i + 1, len(uploaded_files))
            
            result = self.process_single_file(uploaded_file, self.openrouter_api_key)
            processed_files_info.append(result)
            
            if result.get('status') == 'Success' and 'ocr_text' in result:
                all_ocr_text += f"--- {result['filename']} ---\n\n{result['ocr_text']}\n\n"
                file_data.append(result)
        
        if all_ocr_text:
            combined_file_path = self.medical_ocr_folder / "all_ocr_text.txt"
            with open(combined_file_path, "w", encoding="utf-8") as f:
                f.write(all_ocr_text)
            
            return file_data, all_ocr_text, str(combined_file_path), processed_files_info
        
        return None, None, None, processed_files_info
    
    def exponential_backoff(self, attempt, retry_after=None):
        """Calculate wait time with exponential backoff"""
        if retry_after:
            return retry_after
        return min(self.base_delay * (2 ** attempt), 60)
    
    def send_to_api(self, prompt, api_key, progress_callback=None):
        """Send individual request to API with robust error handling"""
        for attempt in range(self.max_retries):
            try:
                if progress_callback:
                    progress_callback(1, 1, attempt + 1)
                
                response = requests.post(
                    url=self.api_url,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.analysis_model,
                        "messages": [
                            {
                                "role": "system", 
                                "content": """You are an experienced medical doctor and clinical analyst. Provide comprehensive, detailed medical analysis of the provided medical document.

CRITICAL INSTRUCTIONS:
1. Analyze the ENTIRE document thoroughly
2. Provide structured, organized analysis with clear sections
3. Focus on medical accuracy and clinical relevance
4. Include specific findings, interpretations, and recommendations
5. Use professional medical terminology
6. Be comprehensive but avoid unnecessary repetition"""
                            },
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.1,
                        "max_tokens": 4000,
                    },
                    timeout=120
                )
                
                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 30))
                    wait_time = self.exponential_backoff(attempt, retry_after)
                    
                    if attempt < self.max_retries - 1:
                        time.sleep(wait_time)
                        continue
                    else:
                        response.raise_for_status()
                
                response.raise_for_status()
                result = response.json()
                
                analysis_text = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                
                if not analysis_text:
                    return f"Empty response from API"
                
                return analysis_text
                    
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    wait_time = self.exponential_backoff(attempt)
                    time.sleep(wait_time)
                    continue
                return f"Request error: {e}"
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = self.exponential_backoff(attempt)
                    time.sleep(wait_time)
                    continue
                return f"Unexpected error: {e}"
        
        return "Failed after multiple retries"
    
    def analyze_medical_text(self, combined_text, api_key, progress_callback=None):
        """Send full text to API for comprehensive medical analysis"""
        if not api_key:
            return "❌ API key is required for analysis"
        
        if not combined_text or len(combined_text.strip()) < 100:
            return "❌ No meaningful text extracted for analysis"
        
        logger.info(f"Processing full text for analysis (length: {len(combined_text)})")
        
        prompt = f"""COMPREHENSIVE MEDICAL DOCUMENT ANALYSIS REQUEST

Please provide a detailed, structured analysis of the following medical document(s). The text may contain OCR errors, so use clinical judgment to interpret ambiguous information.

MEDICAL TEXT TO ANALYZE:
{combined_text}

REQUIRED ANALYSIS FORMAT:

1. DOCUMENT OVERVIEW
   - Type of document(s) (e.g., lab report, clinical notes, imaging report)
   - Overall clinical context and purpose
   - Date relevance and temporal considerations

2. PATIENT DEMOGRAPHICS & HISTORY (if available)
   - Age, gender, relevant background
   - Medical history findings
   - Current symptoms or complaints

3. CLINICAL FINDINGS - DETAILED BREAKDOWN
   For each major section (lab results, imaging, clinical notes):
   - Normal/abnormal parameters with specific values
   - Clinical significance of each finding
   - Patterns and trends across multiple tests/measures
   - Critical values requiring immediate attention

4. DIFFERENTIAL DIAGNOSIS
   - Potential conditions based on findings
   - Most likely diagnoses with supporting evidence
   - Ruled-out conditions with reasoning

5. RISK ASSESSMENT
   - Immediate health risks (urgent/emergent)
   - Medium-term clinical concerns
   - Long-term health implications

6. RECOMMENDATIONS & NEXT STEPS
   - Immediate actions required (if any)
   - Specialist referrals needed
   - Follow-up testing and timing
   - Patient monitoring requirements
   - Treatment considerations

7. CLINICAL IMPRESSION & SUMMARY
   - Overall assessment of patient status
   - Key takeaways for healthcare providers
   - Documentation quality assessment

Please provide this analysis in a clear, structured format using medical terminology appropriate for healthcare professionals. Focus on actionable insights and clinical relevance."""

        result = self.send_to_api(prompt, api_key, progress_callback)
        
        if result.startswith(("Error:", "Request error:", "Failed")):
            return f"❌ Analysis failed: {result}"
        
        return result
    
    def save_results(self, analysis_text, combined_text_path):
        """Save analysis results to files with error handling"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        json_output_file = self.analysis_folder / f"medical_analysis_{timestamp}.json"
        result_data = {
            "timestamp": str(datetime.now()),
            "combined_text_path": combined_text_path,
            "analysis": analysis_text,
        }
        
        try:
            with open(json_output_file, "w", encoding="utf-8") as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            return None, f"❌ Error saving JSON: {e}"
        
        text_output_file = self.analysis_folder / f"medical_analysis_{timestamp}.txt"
        try:
            with open(text_output_file, "w", encoding="utf-8") as f:
                f.write(analysis_text)
            return str(json_output_file), str(text_output_file)
        except Exception as e:
            return str(json_output_file), f"❌ Error saving text file: {e}"

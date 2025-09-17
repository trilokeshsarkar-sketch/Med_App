import os
import requests
import json
import time
import re
from datetime import datetime
from PIL import Image
import pytesseract
import fitz  # PyMuPDF for PDF handling
import io
import tempfile
from pathlib import Path
import logging

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
    
    def process_single_file(self, uploaded_file):
        """Process a single file and return results"""
        try:
            file_bytes = uploaded_file.read()
            file_type = uploaded_file.type
            file_name = uploaded_file.name
            
            if file_type.startswith('image/'):
                image = Image.open(io.BytesIO(file_bytes))
                ocr_text = pytesseract.image_to_string(image)
                ocr_text = self.clean_text(ocr_text)
                
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
                    page_text = pytesseract.image_to_string(image)
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
    
    def process_uploaded_files(self, uploaded_files, progress_callback=None):
        """Process all uploaded files sequentially"""
        all_ocr_text = ""
        processed_files_info = []
        file_data = []
        
        for i, uploaded_file in enumerate(uploaded_files):
            if progress_callback:
                progress_callback(i + 1, len(uploaded_files))
            
            result = self.process_single_file(uploaded_file)
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
                    progress_callback(1, 1, attempt + 1)  # Single request now
                
                response = requests.post(
                    url=self.api_url,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
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
                        "temperature": 0.1,  # Lower temperature for more factual responses
                        "max_tokens": 4000,  # Increased tokens for detailed analysis
                    },
                    timeout=120  # Increased timeout for larger responses
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
        
        json_output_file = self.output_directory / f"medical_analysis_{timestamp}.json"
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
        
        text_output_file = self.output_directory / f"medical_analysis_{timestamp}.txt"
        try:
            with open(text_output_file, "w", encoding="utf-8") as f:
                f.write(analysis_text)
            return str(json_output_file), str(text_output_file)
        except Exception as e:
            return str(json_output_file), f"❌ Error saving text file: {e}"

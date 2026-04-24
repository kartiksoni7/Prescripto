"""
Prescription Text Extractor using Google Gemini API
Extracts text from prescription images/PDFs using Gemini's vision capabilities
"""

import re
import base64
from PIL import Image
from typing import Dict, List, Optional
import json
from pathlib import Path
import google.generativeai as genai
import os
from io import BytesIO


class PrescriptionExtractor:
    """Extract text from prescription images and PDFs using Gemini API"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the extractor with Gemini API
        
        Args:
            api_key: Google Gemini API key (if not provided, reads from GEMINI_API_KEY env variable)
        """
        # Get API key from parameter or environment variable
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        
        if not self.api_key:
            raise ValueError(
                "Gemini API key not provided. Either pass it to the constructor or set GEMINI_API_KEY environment variable"
            )
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        
        # Use Gemini 1.5 Flash for fast and free processing
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Common medical abbreviations to preserve during text cleaning
        self.medical_abbr = {
            'od': 'once daily',
            'bd': 'twice daily', 
            'td': 'thrice daily',
            'qid': 'four times daily',
            'prn': 'as needed',
            'sos': 'if necessary',
            'stat': 'immediately',
            'ac': 'before meals',
            'pc': 'after meals',
            'hs': 'at bedtime',
            'po': 'by mouth',
            'im': 'intramuscular',
            'iv': 'intravenous',
            'sc': 'subcutaneous',
            'tab': 'tablet',
            'cap': 'capsule',
            'syp': 'syrup',
            'inj': 'injection',
            'susp': 'suspension',
            'ml': 'milliliter',
            'mg': 'milligram',
            'mcg': 'microgram',
            'gm': 'gram',
        }
    
    def load_image(self, image_path: str) -> Image.Image:
        """
        Load image from file path
        
        Args:
            image_path: Path to image file
            
        Returns:
            PIL Image object
        """
        try:
            img = Image.open(image_path)
            # Convert to RGB if needed (handles RGBA, grayscale, etc.)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            return img
        except Exception as e:
            raise ValueError(f"Failed to load image: {e}")
    
    def convert_pdf_to_images(self, pdf_path: str) -> List[Image.Image]:
        """
        Convert PDF pages to images
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of PIL Image objects (one per page)
        """
        try:
            import fitz  # PyMuPDF
            
            doc = fitz.open(pdf_path)
            images = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                # Render page to image at 300 DPI
                pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
                
                # Convert to PIL Image
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                images.append(img)
            
            doc.close()
            return images
            
        except ImportError:
            raise ImportError("PyMuPDF (fitz) is required for PDF processing. Install with: pip install PyMuPDF")
        except Exception as e:
            raise ValueError(f"Failed to process PDF: {e}")
    
    def extract_text_with_gemini(self, image: Image.Image) -> str:
        """
        Extract text from image using Gemini Vision API
        
        Args:
            image: PIL Image object
            
        Returns:
            Extracted text string
        """
        try:
            # Create prompt for prescription extraction
            prompt = """
            You are a medical document OCR expert. Extract ALL text from this prescription image.
            
            Please provide:
            1. Complete text extraction of the entire prescription
            2. Maintain the original structure and formatting as much as possible
            3. Include all medicine names, dosages, frequencies, and instructions
            4. Include patient information, doctor information, and dates if visible
            5. Be accurate with medical terminology and drug names
            
            Extract the text exactly as it appears in the image, line by line.
            """
            
            # Generate content using Gemini
            response = self.model.generate_content([prompt, image])
            
            return response.text
            
        except Exception as e:
            print(f"Error in Gemini API call: {e}")
            return ""
    
    def extract_structured_info_with_gemini(self, image: Image.Image) -> Dict:
        """
        Extract structured information from prescription using Gemini
        
        Args:
            image: PIL Image object
            
        Returns:
            Dictionary with structured prescription information
        """
        try:
            prompt = """
            You are a medical document analysis expert. Analyze this prescription image and extract structured information.
            
            Please extract and provide the following information in JSON format:
            {
                "patient_name": "patient's name or null",
                "doctor_name": "doctor's name or null",
                "date": "prescription date or null",
                "diagnosis": "diagnosis/condition or null",
                "medicines_section": "list of all medicines with dosage and frequency, each on a new line",
                "instructions": "any additional instructions or null"
            }
            
            For medicines_section, format each medicine as:
            Medicine Name [dosage] [frequency/timing]
            
            Examples:
            Paracetamol 500mg twice daily after meals
            Amoxicillin 250mg thrice daily before meals
            
            If any field is not visible or unclear in the image, use null.
            Return ONLY the JSON object, no additional text.
            """
            
            response = self.model.generate_content([prompt, image])
            
            # Parse JSON response
            response_text = response.text.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.startswith('```'):
                response_text = response_text[3:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            
            response_text = response_text.strip()
            
            structured_info = json.loads(response_text)
            return structured_info
            
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON from Gemini response: {e}")
            # Fallback to basic extraction
            return self.extract_structured_info_fallback(response.text if 'response' in locals() else "")
        except Exception as e:
            print(f"Error in Gemini structured extraction: {e}")
            return {}
    
    def extract_structured_info_fallback(self, text: str) -> Dict:
        """
        Fallback method to extract structured info from text
        """
        structured = {
            'patient_name': None,
            'doctor_name': None,
            'date': None,
            'medicines_section': None,
            'diagnosis': None,
            'instructions': None
        }
        
        if not text:
            return structured
        
        lines = text.split('\n')
        
        # Extract patient name
        for i, line in enumerate(lines[:10]):
            if re.search(r'patient|name|mr\.|mrs\.|ms\.', line, re.IGNORECASE):
                if i + 1 < len(lines):
                    structured['patient_name'] = lines[i + 1].strip()
                break
        
        # Extract doctor name
        for line in lines:
            if re.search(r'dr\.|doctor|physician', line, re.IGNORECASE):
                structured['doctor_name'] = line.strip()
                break
        
        # Extract date
        date_pattern = r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}'
        for line in lines:
            match = re.search(date_pattern, line)
            if match:
                structured['date'] = match.group(0)
                break
        
        # Extract diagnosis
        for i, line in enumerate(lines):
            if re.search(r'diagnosis|condition|complaint', line, re.IGNORECASE):
                if i + 1 < len(lines):
                    structured['diagnosis'] = lines[i + 1].strip()
                break
        
        # Extract medicine section
        medicine_start = -1
        medicine_end = -1
        
        for i, line in enumerate(lines):
            if re.search(r'rx|prescription|medicines?|drugs?', line, re.IGNORECASE):
                medicine_start = i + 1
            if re.search(r'advice|instructions|follow.?up|next.?visit', line, re.IGNORECASE) and medicine_start != -1:
                medicine_end = i
                break
        
        if medicine_start != -1:
            if medicine_end == -1:
                medicine_end = len(lines)
            structured['medicines_section'] = '\n'.join(lines[medicine_start:medicine_end])
        
        return structured
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep medical notation
        text = re.sub(r'[^\w\s\.\,\-\:\(\)\/\+]', '', text)
        
        # Normalize line breaks
        text = text.replace('\n\n', '\n').replace('\r', '\n')
        
        return text.strip()
    
    def process_prescription(self, file_path: str, use_structured_extraction: bool = True) -> Dict:
        """
        Main processing function for prescription files
        
        Args:
            file_path: Path to prescription image or PDF
            use_structured_extraction: Whether to use Gemini for structured extraction (recommended)
            
        Returns:
            Dictionary with extracted and structured data
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            return {"error": "File not found"}
        
        try:
            # Load image(s)
            if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                images = [self.load_image(str(file_path))]
            elif file_path.suffix.lower() == '.pdf':
                images = self.convert_pdf_to_images(str(file_path))
            else:
                return {"error": "Unsupported file format"}
            
            # Process all images (for multi-page PDFs)
            all_text = []
            structured_info = {}
            
            for i, image in enumerate(images):
                # Extract text
                text = self.extract_text_with_gemini(image)
                all_text.append(text)
                
                # Extract structured info from first page
                if i == 0 and use_structured_extraction:
                    structured_info = self.extract_structured_info_with_gemini(image)
            
            # Combine text from all pages
            raw_text = "\n\n--- Page Break ---\n\n".join(all_text)
            
            if not raw_text.strip():
                return {"error": "No text extracted from file"}
            
            # Clean text
            cleaned_text = self.clean_text(raw_text)
            
            # If structured extraction wasn't used or failed, use fallback
            if not structured_info:
                structured_info = self.extract_structured_info_fallback(cleaned_text)
            
            return {
                'raw_text': raw_text,
                'cleaned_text': cleaned_text,
                'structured_info': structured_info,
                'ready_for_ner': True
            }
            
        except Exception as e:
            return {"error": f"Processing failed: {str(e)}"}
    
    def save_to_json(self, data: Dict, output_path: str):
        """Save extracted data to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def save_text_for_ner(self, data: Dict, output_path: str):
        """
        Save cleaned text in format ready for NER processing
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(data['cleaned_text'])
    
    def batch_process(self, file_paths: List[str], output_dir: str) -> List[Dict]:
        """
        Process multiple prescription files
        
        Args:
            file_paths: List of file paths to process
            output_dir: Directory to save outputs
            
        Returns:
            List of extraction results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        results = []
        
        for file_path in file_paths:
            print(f"Processing: {file_path}")
            
            result = self.process_prescription(file_path)
            results.append(result)
            
            if 'error' not in result:
                # Save individual results
                file_name = Path(file_path).stem
                json_path = output_dir / f"{file_name}_extracted.json"
                txt_path = output_dir / f"{file_name}_for_ner.txt"
                
                self.save_to_json(result, str(json_path))
                self.save_text_for_ner(result, str(txt_path))
                
                print(f"✓ Saved: {json_path}")
            else:
                print(f"✗ Error: {result['error']}")
        
        return results


def main():
    """Main function for command-line usage"""
    import sys
    
    # Check if API key is set
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("=" * 60)
        print("ERROR: GEMINI_API_KEY environment variable not set")
        print("=" * 60)
        print("\nPlease set your Gemini API key:")
        print("  Windows: set GEMINI_API_KEY=your_api_key_here")
        print("  Linux/Mac: export GEMINI_API_KEY=your_api_key_here")
        print("\nGet your free API key at: https://makersuite.google.com/app/apikey")
        return
    
    extractor = PrescriptionExtractor()
    
    print("=" * 60)
    print("PRESCRIPTION TEXT EXTRACTOR (Powered by Google Gemini)")
    print("=" * 60)
    
    # Get file path
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = input("\nEnter path to prescription (image or PDF): ").strip().strip('"')
    
    if not file_path:
        print("No file path provided. Exiting.")
        return
    
    print(f"\nProcessing prescription with Gemini AI...")
    print("-" * 60)
    
    # Process prescription
    result = extractor.process_prescription(file_path)
    
    # Display results
    if 'error' in result:
        print(f"\n❌ Error: {result['error']}")
        return
    
    print("\n✅ EXTRACTION SUCCESSFUL")
    print("=" * 60)
    
    print("\n📄 RAW TEXT (first 500 chars):")
    print("-" * 60)
    print(result['raw_text'][:500])
    
    print("\n\n🧹 CLEANED TEXT (first 500 chars):")
    print("-" * 60)
    print(result['cleaned_text'][:500])
    
    print("\n\n📋 STRUCTURED INFORMATION:")
    print("-" * 60)
    structured = result['structured_info']
    for key, value in structured.items():
        if value:
            display_value = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
            print(f"{key.replace('_', ' ').title()}: {display_value}")
    
    # Save options
    save = input("\n\nSave results? (y/n): ").strip().lower()
    if save == 'y':
        base_name = Path(file_path).stem
        
        # Save JSON
        json_path = f"{base_name}_extracted.json"
        extractor.save_to_json(result, json_path)
        print(f"✓ Saved JSON: {json_path}")
        
        # Save text for NER
        txt_path = f"{base_name}_for_ner.txt"
        extractor.save_text_for_ner(result, txt_path)
        print(f"✓ Saved NER-ready text: {txt_path}")
    
    print("\n" + "=" * 60)
    print("Text extraction complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
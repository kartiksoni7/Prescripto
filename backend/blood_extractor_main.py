import re
import cv2
import numpy as np
from PIL import Image
import pytesseract
import pandas as pd
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path
import fitz  # PyMuPDF
import subprocess
import shutil
import sys



class BloodReportExtractor:
    def __init__(self):
        # (ALL your existing regex dictionaries and patterns — unchanged)
        self.blood_parameters: Dict[str, List[str]] = {
            # CBC
            "Hemoglobin": [r"\bhemoglobin[:\s\-]*([\d\.]+)", r"\bhaemoglobin[:\s\-]*([\d\.]+)", r"\bhb[:\s\-]*([\d\.]+)"],
            "RBC Count": [r"\brbc[:\s\-]*([\d\.]+)", r"rbc.*?count.*?([\d\.]+)", r"red\s*blood\s*cell.*?([\d\.]+)"],
            "WBC Count": [
                r"\bwbc[:\s\-]*([\d\.]+)",
                r"wbc.*?count.*?([\d\.]+)",
                r"white\s*blood\s*cell.*?([\d\.]+)",
                r"\btlc[:\s\-]*([\d\.]+)",
            ],
            "Platelet Count": [
                r"platelet.*?count.*?([\d\.]+)",
                r"\bplt[:\s\-]*([\d\.]+)",
                r"platelet.*?(?:x10[\^]?\d*\/?\s*ul)?.*?([\d\.]+)",
                r"thrombocyte.*?([\d\.]+)",
            ],
            "PCV": [r"\bpcv[:\s\-]*([\d\.]+)", r"packed\s*cell\s*volume.*?([\d\.]+)", r"\bhct[:\s\-]*([\d\.]+)"],
            "MCV": [r"\bmcv[:\s\-]*([\d\.]+)"],
            "MCH": [r"\bmch(?!c)[:\s\-]*([\d\.]+)"],
            "MCHC": [r"\bmchc[:\s\-]*([\d\.]+)"],
            "RDW": [r"\brdw[:\s\-]*([\d\.]+)"],

            # Differential Count
            "Neutrophils": [r"\bneutrophils[:\s\-]*([\d\.]+)", r"\bpolymorphs[:\s\-]*([\d\.]+)", r"\bpmn[:\s\-]*([\d\.]+)"],
            "Lymphocytes": [r"\blymphocytes[:\s\-]*([\d\.]+)"],
            "Eosinophils": [r"\beosinophils[:\s\-]*([\d\.]+)"],
            "Monocytes": [r"\bmonocytes[:\s\-]*([\d\.]+)"],
            "Basophils": [r"\bbasophils[:\s\-]*([\d\.]+)"],

            # Absolute Counts
            "Absolute Neutrophil Count": [r"absolute\s*neutrophil\s*count\s*\(anc\).*?([\d\.]+)", r"\banc[:\s\-]*([\d\.]+)"],
            "Absolute Lymphocyte Count": [r"absolute\s*lymphocyte\s*count\s*\(alc\).*?([\d\.]+)", r"\balc[:\s\-]*([\d\.]+)"],
            "Absolute Monocyte Count": [r"absolute\s*monocyte\s*count\s*\(amc\).*?([\d\.]+)", r"\bamc[:\s\-]*([\d\.]+)"],
            "Absolute Eosinophil Count": [r"absolute\s*eosinophil\s*count\s*\(aec\).*?([\d\.]+)", r"\baec[:\s\-]*([\d\.]+)"],
            "Absolute Basophil Count": [r"absolute\s*basophil\s*count\s*\(abc\).*?([\d\.]+)", r"\babc[:\s\-]*([\d\.]+)"],
            "Absolute Polymorphs Count": [r"absolute\s*polymorphs\s*count\s*\(apc\).*?([\d\.]+)", r"absolute\s*polymorphs\s*count.*?([\d\.]+)"],

            # Inflammation
            "ESR": [r"\besr[:\s\-]*([\d\.]+)", r"erythrocyte.*?sedimentation.*?rate.*?([\d\.]+)"],
            "CRP": [r"\bcrp[:\s\-]*([\d\.]+)", r"c[\s\-]?reactive\s*protein.*?([\d\.]+)"],
            "Procalcitonin": [r"procalcitonin[:\s\-]*([\d\.]+)"],
            "D-Dimer": [r"d[\s\-]?dimer[:\s\-]*([\d\.]+)"],
            "LDH": [r"\bldh[:\s\-]*([\d\.]+)"],

            # Diabetes
            "Glucose": [r"\bglucose[:\s\-]*([\d\.]+)", r"blood\s*sugar[:\s\-]*([\d\.]+)", r"\bfbs[:\s\-]*([\d\.]+)", r"\brbs[:\s\-]*([\d\.]+)"],
            "HbA1c": [r"hba1c[:\s\-]*([\d\.]+)", r"glycated\s*haemoglobin.*?([\d\.]+)"],
            "Average Blood Glucose": [r"average\s*glucose[:\s\-]*([\d\.]+)"],

            # Renal
            "Creatinine": [r"creatinine[:\s\-]*([\d\.]+)"],
            "Urea": [r"\burea[:\s\-]*([\d\.]+)"],
            "BUN": [r"\bbun[:\s\-]*([\d\.]+)"],
            "Uric Acid": [r"uric\s*acid[:\s\-]*([\d\.]+)"],
            "eGFR": [r"egfr[:\s\-]*([\d\.]+)"],

            # Lipid Profile
            "Total Cholesterol": [r"total\s*cholesterol[:\s\-]*([\d\.]+)"],
            "HDL": [r"\bhdl[:\s\-]*([\d\.]+)"],
            "LDL": [r"\bldl[:\s\-]*([\d\.]+)"],
            "Triglycerides": [r"triglycerides[:\s\-]*([\d\.]+)", r"\btg[:\s\-]*([\d\.]+)"],
            "VLDL": [r"\bvldl[:\s\-]*([\d\.]+)"],

            # Liver Function Tests
            "SGPT/ALT": [r"\b(?:sgpt|alt)[:\s\-]*([\d\.]+)"],
            "SGOT/AST": [r"\b(?:sgot|ast)[:\s\-]*([\d\.]+)"],
            "ALP": [r"(?:alkaline\s*phosphatase|alp)[:\s\-]*([\d\.]+)"],
            "GGT": [r"\bggt[:\s\-]*([\d\.]+)"],
            "Total Protein": [r"total\s*protein[:\s\-]*([\d\.]+)"],
            "Albumin": [r"\balbumin[:\s\-]*([\d\.]+)"],
            "Globulin": [r"\bglobulin[:\s\-]*([\d\.]+)"],
            "A/G Ratio": [r"a[\s\/\-]?g\s*ratio[:\s\-]*([\d\.]+)"],
            "Total Bilirubin": [r"total\s*bilirubin[:\s\-]*([\d\.]+)"],
            "Direct Bilirubin": [r"direct\s*bilirubin[:\s\-]*([\d\.]+)"],
            "Indirect Bilirubin": [r"indirect\s*bilirubin[:\s\-]*([\d\.]+)"],

            # Thyroid
            "TSH": [r"\btsh[:\s\-]*([\d\.]+)"],
            "Free T3": [r"(?:free\s*t3|ft3)[:\s\-]*([\d\.]+)"],
            "Free T4": [r"(?:free\s*t4|ft4)[:\s\-]*([\d\.]+)"],
            "Total T3": [r"(?:total\s*t3)[:\s\-]*([\d\.]+)"],
            "Total T4": [r"(?:total\s*t4)[:\s\-]*([\d\.]+)"],

            # Electrolytes
            "Sodium": [r"(?:sodium|na[\+\-]?)[:\s\-]*([\d\.]+)"],
            "Potassium": [r"(?:potassium|k[\+\-]?)[:\s\-]*([\d\.]+)"],
            "Chloride": [r"(?:chloride|cl[\+\-]?)[:\s\-]*([\d\.]+)"],
            "Calcium": [r"(?:calcium|ca)[:\s\-]*([\d\.]+)"],
            "Phosphorus": [r"(?:phosphorus|phosphate)[:\s\-]*([\d\.]+)"],
            "Magnesium": [r"(?:magnesium|mg)[:\s\-]*([\d\.]+)"],

            # Iron Studies
            "Serum Iron": [r"(?:serum\s*iron)[:\s\-]*([\d\.]+)"],
            "Ferritin": [r"(?:ferritin)[:\s\-]*([\d\.]+)"],
            "TIBC": [r"(?:tibc)[:\s\-]*([\d\.]+)"],
            "Transferrin Saturation": [r"(?:transferrin\s*saturation)[:\s\-]*([\d\.]+)"],

            # Vitamins
            "Vitamin D": [r"(?:vitamin\s*d)[:\s\-]*([\d\.]+)"],
            "Vitamin B12": [r"(?:vitamin\s*b12)[:\s\-]*([\d\.]+)"],
            "Folate": [r"(?:folate|folic\s*acid)[:\s\-]*([\d\.]+)"],

            # Coagulation
            "PT": [r"(?:pt|prothrombin\s*time)[:\s\-]*([\d\.]+)"],
            "INR": [r"\binr[:\s\-]*([\d\.]+)"],
            "APTT": [r"\baptt[:\s\-]*([\d\.]+)"],

            # Tumor Markers
            "PSA": [r"\bpsac[:\s\-]*([\d\.]+)", r"\bpsa[:\s\-]*([\d\.]+)"],
            "CEA": [r"\bcea[:\s\-]*([\d\.]+)"],
            "CA-125": [r"ca[\s\-]?125[:\s\-]*([\d\.]+)"],
            "CA-19-9": [r"ca[\s\-]?19[\s\-]?9[:\s\-]*([\d\.]+)"],
            "AFP": [r"\bafp[:\s\-]*([\d\.]+)"],

            # Cardiac
            "Troponin": [r"troponin[:\s\-]*([\d\.]+)"],
            "CK-MB": [r"ck[\s\-]?mb[:\s\-]*([\d\.]+)"],
            "BNP": [r"\bbnp[:\s\-]*([\d\.]+)"],
        }

        # Range patterns (expanded for many text formats)
        self.range_patterns = [
            r"(\d+\.?\d*)\s*[-–—]\s*(\d+\.?\d*)",               # 4.0–10.0
            r"(\d+\.?\d*)\s*to\s*(\d+\.?\d*)",                   # 4.0 to 10.0
            r"reference\s*(?:range|interval)[:\s]*(\d+\.?\d*)\s*[-–—]\s*(\d+\.?\d*)",
            r"ref\.?\s*(?:range|interval)[:\s]*(\d+\.?\d*)\s*[-–—]\s*(\d+\.?\d*)",
            r"normal\s*(?:range|value|values)?[:\s]*(\d+\.?\d*)\s*[-–—]\s*(\d+\.?\d*)",
            r"([\d\.]+)\s*-\s*([\d\.]+)",
        ]

        # Status indicators
        self.status_patterns = {
            "High": [r"\bhigh\b", r"\belevated\b", r"\bincreased\b", r"\babove\s*normal\b", r"\b↑+", r"\bhigher\b"],
            "Low": [r"\blow\b", r"\bdecreased\b", r"\breduced\b", r"\bbelow\s*normal\b", r"\b↓+", r"\blower\b"],
            "Normal": [r"\bnormal\b", r"\bwithin\s+normal\b", r"\bwithin\s+range\b", r"\bwnl\b"],
            "Borderline": [r"\bborderline\b", r"\bmarginal\b", r"\bslightly\s*(?:high|low)\b", r"\bnear\s*normal\b"],
        }

    # ---------------- Image / PDF preprocessing and OCR ----------------
    def preprocess_image(self, image_path_or_array) -> np.ndarray:
        """Preprocess image for better OCR results. Accepts file path or numpy array."""
        if isinstance(image_path_or_array, str):
            image = cv2.imread(image_path_or_array)
            if image is None:
                raise FileNotFoundError(f"Image not found: {image_path_or_array}")
        else:
            image = image_path_or_array

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Use a slightly larger kernel for morphological ops if needed
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        return cleaned

    def extract_text_from_image(self, image_path: str) -> str:
        """Extract text from an image using Tesseract OCR."""
        try:
            processed_image = self.preprocess_image(image_path)
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,()-/:% '
            text = pytesseract.image_to_string(processed_image, config=custom_config)
            return text.lower()
        except Exception as e:
            print(f"Error in OCR: {e}")
            return ""

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF using PyMuPDF (fitz)."""
        try:
            doc = fitz.open(pdf_path)
            text_parts = []
            for page in doc:
                text_parts.append(page.get_text("text"))
            doc.close()
            return "\n".join(text_parts).lower()
        except Exception as e:
            print(f"Error extracting PDF text: {e}")
            return ""

    
    def extract_patient_info(self, text: str) -> Dict[str, str]:
        """Extract patient name, age, sex from text."""
        patient_info: Dict[str, str] = {}

        name_patterns = [
            r"patient.*?name.*?:?\s*([a-z\.\s]+)",
            r"^name[:\s]*([a-z\.\s]+)",
            r"\bmr\.?\s*([a-z\.\s]+)",
            r"\bmrs\.?\s*([a-z\.\s]+)",
            r"^([a-z\.\s]+)\n\s*age[:\s]*\d+",  
        ]

        age_patterns = [r"age[:\s]*([0-9]{1,3})", r"(\b[0-9]{1,3})\s*years?"]

        sex_patterns = [r"sex[:\s]*(male|female|m|f)", r"gender[:\s]*(male|female|m|f)"]

        # Name
        for pat in name_patterns:
            m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
            if m:
                name_str = m.group(1).strip()
                # basic filters
                if "pathology" not in name_str and len(name_str) < 60:
                    patient_info["name"] = name_str.title()
                    break

        # Age (if not included already)
        if "age" not in patient_info:
            for pat in age_patterns:
                m = re.search(pat, text, re.IGNORECASE)
                if m:
                    patient_info["age"] = m.group(1)
                    break

        # Sex
        for pat in sex_patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                sex = m.group(1).lower()
                patient_info["sex"] = "Male" if sex in ("male", "m") else "Female"
                break

        return patient_info

    # ---------------- Helper utilities ----------------
    @staticmethod
    def _numeric_from_match(match: re.Match) -> Optional[str]:
        """
        Given a regex match object, pick the last group that looks numeric.
        Returns string or None.
        """
        groups = match.groups()
        # iterate from last to first to pick the last numeric group
        for g in reversed(groups):
            if g is None:
                continue
            g_str = str(g).strip()
            if re.match(r"^-?\d+(\.\d+)?$", g_str):
                return g_str
            # sometimes captured text like '4.5 g/dl' — extract numeric substring
            num_search = re.search(r"-?\d+(\.\d+)?", g_str)
            if num_search:
                return num_search.group(0)
        return None

    def find_reference_range(self, text: str, parameter_name: str, value_position: int) -> Optional[Tuple[str, str]]:
        """Try to find a reference range near the value position in the text."""
        # First look after the value (common in table rows)
        start = value_position
        search_window = text[start:start + 200]
        for pat in self.range_patterns:
            m = re.search(pat, search_window, re.IGNORECASE)
            if m:
                # returning first two groups as min/max
                try:
                    return m.group(1), m.group(2)
                except IndexError:
                    continue

        # Wider search around the value
        start = max(0, value_position - 200)
        search_window = text[start:value_position + 200]
        for pat in self.range_patterns:
            m = re.search(pat, search_window, re.IGNORECASE)
            if m:
                try:
                    return m.group(1), m.group(2)
                except IndexError:
                    continue
        return None

    def determine_status(self, text: str, parameter_name: str, value_position: int, patient_value: float,
                         ref_range: Optional[Tuple[str, str]]) -> str:
        """Determine High/Low/Normal/Borderline/Unknown for a numeric value."""
        # Look for nearby explicit words
        start = max(0, value_position - 100)
        window = text[start:value_position + 100]

        for status, patterns in self.status_patterns.items():
            for pat in patterns:
                if re.search(pat, window, re.IGNORECASE):
                    return status

        # If no explicit status, use reference range if available
        if ref_range:
            try:
                min_val, max_val = float(ref_range[0]), float(ref_range[1])
                if patient_value < min_val:
                    return "Low"
                elif patient_value > max_val:
                    return "High"
                elif patient_value == min_val or patient_value == max_val:
                    return "Borderline"
                else:
                    return "Normal"
            except (ValueError, TypeError):
                pass

        return "Unknown"

    # ---------------- Parameter extraction ----------------
    def extract_blood_parameters(self, text: str) -> Dict[str, Dict]:
        """Extract blood parameters, reference ranges and status from text."""
        results: Dict[str, Dict] = {}

        for param_name, patterns in self.blood_parameters.items():
            found = False
            for pat in patterns:
                for match in re.finditer(pat, text, re.IGNORECASE):
                    numeric_str = self._numeric_from_match(match)
                    if numeric_str is None:
                        continue
                    try:
                        value = float(numeric_str)
                    except ValueError:
                        continue

                    # position of the matched numeric group relative to entire text
                    # match.start() gives the char position of the whole match;
                    # we attempt to find numeric position within the match string
                    match_text = match.group(0)
                    # find index of the numeric_str within the matched substring
                    idx_in_match = match_text.lower().find(numeric_str.lower())
                    if idx_in_match >= 0:
                        value_pos = match.start() + idx_in_match
                    else:
                        value_pos = match.start()

                    ref_range = self.find_reference_range(text, param_name, value_pos)
                    status = self.determine_status(text, param_name, value_pos, value, ref_range)

                    results[param_name] = {
                        "value": value,
                        "reference_range": f"{ref_range[0]}-{ref_range[1]}" if ref_range else "Not found",
                        "status": status,
                    }
                    found = True
                    break
                if found:
                    break

        return results

    # ---------------- High-level processing ----------------
    def process_report(self, file_path: str) -> Dict:
        """Process a report file (image or pdf) and return extracted info."""
        file_path = Path(file_path)
        if not file_path.exists():
            return {"error": "File not found"}

        if file_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
            text = self.extract_text_from_image(str(file_path))
        elif file_path.suffix.lower() == ".pdf":
            text = self.extract_text_from_pdf(str(file_path))
        else:
            return {"error": "Unsupported file format"}

        if not text.strip():
            return {"error": "No text extracted from file"}

        patient_info = self.extract_patient_info(text)
        blood_parameters = self.extract_blood_parameters(text)

        return {
            "patient_info": patient_info,
            "blood_parameters": blood_parameters,
            "extracted_text": text[:500] + "..." if len(text) > 500 else text,
        }

    # ---------------- Output helpers ----------------
    def create_summary_report(self, extracted_data: Dict) -> str:
        """Create a readable summary report string for console output."""
        if "error" in extracted_data:
            return f"Error: {extracted_data['error']}"

        lines = []
        lines.append("=" * 60)
        lines.append("BLOOD TEST REPORT SUMMARY")
        lines.append("=" * 60)
        lines.append("\nPATIENT INFORMATION:")
        lines.append("-" * 20)
        patient_info = extracted_data.get("patient_info", {})
        if not patient_info:
            lines.append("No patient information found.")
        else:
            for k, v in patient_info.items():
                lines.append(f"{k.title()}: {v}")
        lines.append("\nBLOOD TEST RESULTS:")
        lines.append("-" * 20)

        blood_params = extracted_data.get("blood_parameters", {})
        if not blood_params:
            lines.append("No blood parameters found in the report.")
        else:
            header = f"{'Parameter':<30} {'Value':<10} {'Reference Range':<20} {'Status':<12}"
            lines.append(header)
            lines.append("-" * 80)
            for param, data in blood_params.items():
                value = data.get("value", "N/A")
                ref_range = data.get("reference_range", "N/A")
                status = data.get("status", "Unknown")
                status_indicator = ""
                if status == "High":
                    status_indicator = " ↑"
                elif status == "Low":
                    status_indicator = " ↓"
                elif status == "Borderline":
                    status_indicator = " ~"
                lines.append(f"{param:<30} {str(value):<10} {ref_range:<20} {status + status_indicator:<12}")

        lines.append("\nLegend: ↑ High, ↓ Low, ~ Borderline")
        lines.append("=" * 60)
        return "\n".join(lines)

    def save_results_to_json(self, extracted_data: Dict, output_path: str):
        """Save extracted data to a JSON file."""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(extracted_data, f, indent=2, ensure_ascii=False)

    def save_results_to_csv(self, extracted_data: Dict, output_path: str):
        """Save blood parameters to CSV."""
        blood_params = extracted_data.get("blood_parameters", {})
        if blood_params:
            df_data = []
            for param, data in blood_params.items():
                df_data.append(
                    {
                        "Parameter": param,
                        "Value": data.get("value", "N/A"),
                        "Reference_Range": data.get("reference_range", "N/A"),
                        "Status": data.get("status", "Unknown"),
                    }
                )
            df = pd.DataFrame(df_data)
            df.to_csv(output_path, index=False)


# --------------------------- STEP 1: Local summariser ---------------------------

def is_ollama_available() -> bool:
    """Check if ollama binary is available in PATH."""
    return shutil.which("ollama") is not None


def summarise_with_local_model(extracted_params: Dict[str, Dict]) -> str:
    """
    Try to call a local summarisation model using Ollama.
    If Ollama/model is not available, fall back to rule_based_summariser() below.
    """
    try:
        if is_ollama_available():
            prompt = f"""
Summarize these blood test results in simple human language.
Do NOT give medical advice. Just describe which parameters are high, low or normal in 4-6 short sentences.

Lab Results:
{json.dumps(extracted_params, indent=2)}
"""
            # Use 'llama3.2' or any model you have pulled to Ollama. Change the model name if needed.
            # This runs: ollama run <model> and sends prompt on stdin.
            proc = subprocess.run(
                ["ollama", "run", "llama3.2"],
                input=prompt.encode("utf-8"),
                capture_output=True,
                timeout=30
            )
            out = proc.stdout.decode("utf-8").strip()
            if out:
                return out
            # if empty output, fall back
        # Ollama not available or failed -> fallback
    except Exception as e:
        # Don't crash — fallback
        print(f"[summariser] Ollama call failed: {e}", file=sys.stderr)

    # fallback
    return rule_based_summariser(extracted_params)


def rule_based_summariser(extracted_params: Dict[str, Dict]) -> str:
    """
    Simple deterministic summariser to use when no local LLM is available.
    Produces a short human-friendly summary from the structured extraction.
    """
    lines = []
    abnormal_count = 0
    for param, info in extracted_params.items():
        status = info.get("status", "Unknown")
        val = info.get("value", None)
        if status in ("High", "Low", "Borderline"):
            abnormal_count += 1
            if status == "High":
                lines.append(f"{param} is higher than the expected range (value: {val}).")
            elif status == "Low":
                lines.append(f"{param} is lower than the expected range (value: {val}).")
            else:
                lines.append(f"{param} is borderline (value: {val}).")

    # Summarise normal / nothing abnormal
    if abnormal_count == 0:
        # show 2-3 normal examples for positive feedback
        example_normals = []
        for param, info in extracted_params.items():
            if info.get("status", "") == "Normal":
                example_normals.append(f"{param} ({info.get('value')})")
            if len(example_normals) >= 3:
                break
        base = "No clearly abnormal values found." if not example_normals else f"No clearly abnormal values found. Examples: {', '.join(example_normals)}."
        return base

    # join and keep it short
    summary = " ".join(lines)
    # keep maximum ~3 sentences for brevity
    sentences = re.split(r'(?<=[.!?])\s+', summary)
    short = " ".join(sentences[:4])
    return short


def generate_abnormal_suggestions(extracted_params: Dict[str, Dict]) -> List[str]:
    """
    Create a list of suggestion strings ONLY for abnormal parameters in the exact format you asked.
    E.g. "- Hemoglobin is high. Consider discussing with a doctor for possible causes and follow-up testing."
    """
    suggestions = []
    for param, info in extracted_params.items():
        status = info.get("status", "Unknown")
        if status == "High":
            suggestions.append(f"- {param} is high. Consider discussing with a doctor for possible causes and follow-up testing.")
        elif status == "Low":
            suggestions.append(f"- {param} is low. It may require nutritional, lifestyle, or medical evaluation.")
        elif status == "Borderline":
            suggestions.append(f"- {param} is borderline. Monitoring or retesting may be helpful.")
    if not suggestions:
        suggestions.append("All parameters appear within the normal range.")
    return suggestions


def full_rule_based_analysis(extracted_params: Dict[str, Dict]) -> str:
    """
    A slightly longer rule-based 'full analysis' that groups by categories.
    This is not a medical diagnosis — it's a human-friendly explanation and next-step suggestions.
    """
    categories = {
        "CBC": ["Hemoglobin", "RBC Count", "WBC Count", "Platelet Count", "PCV", "MCV", "MCH", "MCHC", "RDW"],
        "Renal": ["Creatinine", "Urea", "BUN", "Uric Acid", "eGFR"],
        "Liver": ["SGPT/ALT", "SGOT/AST", "ALP", "GGT", "Total Bilirubin", "Direct Bilirubin"],
        "Lipid": ["Total Cholesterol", "HDL", "LDL", "Triglycerides", "VLDL"],
        "Thyroid": ["TSH", "Free T3", "Free T4"],
        "Vitamins": ["Vitamin D", "Vitamin B12", "Folate"]
    }

    analysis_lines = []
    for cat, params in categories.items():
        cat_abnormal = []
        for p in params:
            if p in extracted_params and extracted_params[p].get("status") in ("High", "Low", "Borderline"):
                val = extracted_params[p].get("value")
                st = extracted_params[p].get("status")
                cat_abnormal.append(f"{p} ({val}) is {st.lower()}")
        if cat_abnormal:
            analysis_lines.append(f"{cat}: " + "; ".join(cat_abnormal) + ".")

    # General guidance lines
    guidance = [
        "This analysis is informational and non-diagnostic.",
        "For abnormal values consider following up with your primary care physician or specialist.",
        "Bring this report to your next consult; your doctor may order repeat tests or additional investigations."
    ]
    if not analysis_lines:
        return "No grouped abnormalities detected. " + " ".join(guidance)
    return "\n".join(analysis_lines + [""] + guidance)


# --------------------------- END STEP 1 ---------------------------


def main():
    extractor = BloodReportExtractor()

    file_path = input("Enter path to blood report (image or PDF): ").strip().strip('"')
    if not file_path:
        print("No file path provided. Exiting.")
        return

    print("Processing report...")
    results = extractor.process_report(file_path)

    # If error, show and exit
    if "error" in results:
        print("Error:", results["error"])
        return

    # Print the tabular-like summary you already had
    summary = extractor.create_summary_report(results)
    print(summary)

    # ----------------- STEP 1 usage: call local summariser (or fallback) -----------------
    params = results.get("blood_parameters", {})
    print("\n--- HUMAN SUMMARY (light local summariser) ---")
    human_summary = summarise_with_local_model(params)
    print(human_summary)

    # ----------------- Abnormal-only suggestions (exact requested format) -----------------
    print("\n--- ABNORMAL PARAMETERS / SUGGESTIONS ---")
    suggestions = generate_abnormal_suggestions(params)
    for s in suggestions:
        print(s)

    # ----------------- Full rule-based analysis (longer) -----------------
    print("\n--- FULL ANALYSIS (rule-based) ---")
    analysis = full_rule_based_analysis(params)
    print(analysis)

    # Save options
    save_option = input("\nSave results? (y/n): ").strip().lower()
    if save_option == "y":
        base_name = Path(file_path).stem
        json_path = f"{base_name}_results.json"
        extractor.save_results_to_json(results, json_path)
        print(f"Results saved to {json_path}")

        csv_path = f"{base_name}_results.csv"
        extractor.save_results_to_csv(results, csv_path)
        print(f"Results saved to {csv_path}")


if __name__ == "__main__":
    main()

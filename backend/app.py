from dotenv import load_dotenv
import os

load_dotenv()

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import tempfile
from pathlib import Path
import json

# Import your BloodReportExtractor class
from blood_extractor_main import (
    BloodReportExtractor,
    summarise_with_local_model,
    generate_abnormal_suggestions
)

# Import PrescriptionExtractor class
from prescription_extractor import PrescriptionExtractor

app = Flask(__name__)
CORS(app)  # Enable CORS for Next.js frontend

# Configuration
UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'tiff', 'bmp'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Get Gemini API key from environment variable
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/analyze-blood-report', methods=['POST'])
def analyze_blood_report():
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload PDF, JPEG, PNG, or TIFF'}), 400
        
        # Save file temporarily
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(temp_path)
        
        try:
            # Initialize extractor and process
            extractor = BloodReportExtractor()
            results = extractor.process_report(temp_path)
            
            # Check for extraction errors
            if "error" in results:
                return jsonify({'error': results['error']}), 400
            
            # Generate human summary using local model or fallback
            blood_parameters = results.get("blood_parameters", {})
            human_summary = summarise_with_local_model(blood_parameters)
            
            # Generate suggestions for abnormal values
            suggestions = generate_abnormal_suggestions(blood_parameters)
            
            # Format response
            response = {
                'patient_info': results.get('patient_info', {}),
                'blood_parameters': blood_parameters,
                'human_summary': human_summary,
                'suggestions': suggestions,
                'extracted_text': results.get('extracted_text', '')
            }
            
            return jsonify(response), 200
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    except Exception as e:
        print(f"Error processing report: {str(e)}")
        return jsonify({'error': f'Failed to process report: {str(e)}'}), 500

@app.route('/api/extract-prescription', methods=['POST'])
def extract_prescription():
    try:
        # Check if Gemini API key is configured
        if not GEMINI_API_KEY:
            return jsonify({
                'error': 'Gemini API key not configured. Please set GEMINI_API_KEY environment variable.'
            }), 500
        
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload PDF, JPEG, PNG, BMP, or TIFF'}), 400
        
        # Save file temporarily
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(temp_path)
        
        try:
            # Initialize prescription extractor with Gemini API
            extractor = PrescriptionExtractor(api_key=GEMINI_API_KEY)
            
            # Process prescription
            results = extractor.process_prescription(temp_path)
            
            # Check for extraction errors
            if "error" in results:
                return jsonify({'error': results['error']}), 400
            
            # Format response
            response = {
                'raw_text': results.get('raw_text', ''),
                'cleaned_text': results.get('cleaned_text', ''),
                'structured_info': results.get('structured_info', {}),
                'ready_for_ner': results.get('ready_for_ner', False)
            }
            
            return jsonify(response), 200
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    except Exception as e:
        print(f"Error processing prescription: {str(e)}")
        return jsonify({'error': f'Failed to process prescription: {str(e)}'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    gemini_configured = GEMINI_API_KEY is not None
    
    return jsonify({
        'status': 'ok', 
        'message': 'Medical Document API is running',
        'gemini_api_configured': gemini_configured,
        'endpoints': {
            'blood_report': '/api/analyze-blood-report',
            'prescription': '/api/extract-prescription',
            'health': '/api/health'
        }
    }), 200

if __name__ == '__main__':
    # Check if Gemini API key is set
    if not GEMINI_API_KEY:
        print("\n" + "="*60)
        print("WARNING: GEMINI_API_KEY environment variable not set!")
        print("="*60)
        print("Prescription extraction will not work without it.")
        print("Get your free API key at: https://makersuite.google.com/app/apikey")
        print("\nSet it with:")
        print("  Windows: set GEMINI_API_KEY=your_api_key_here")
        print("  Linux/Mac: export GEMINI_API_KEY=your_api_key_here")
        print("="*60 + "\n")
    
    app.run(debug=True, port=5000)
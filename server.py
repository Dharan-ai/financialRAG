"""
Flask Web Application for Financial RAG System
"""

from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import json
from app import rag_system
import pandas as pd
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create upload folder
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload CSV or Excel file'}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'filepath': filepath
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Analyze uploaded bank statement"""
    try:
        data = request.json
        filepath = data.get('filepath')
        
        if not filepath or not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
        
        # Process the bank statement
        result = rag_system.process_bank_statement(filepath)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/query', methods=['POST'])
def query_rag():
    """Query the RAG system with custom questions"""
    try:
        data = request.json
        question = data.get('question', '')
        
        if not question:
            return jsonify({'error': 'No question provided'}), 400
        
        # Use vector store to find relevant transactions
        if not hasattr(rag_system, 'vector_store'):
            return jsonify({'error': 'No analysis has been performed yet. Please upload and analyze a file first.'}), 400
        
        relevant_docs = rag_system.vector_store.similarity_search(question, k=5)
        context = "\n".join([doc.page_content for doc in relevant_docs])
        
        # Generate answer using LLM
        prompt = f"""Based on the following transaction data, answer the question:

QUESTION: {question}

RELEVANT TRANSACTIONS:
{context}

Provide a clear, concise answer based on the data.
"""
        
        response = rag_system.llm.invoke(prompt)
        answer = response.content
        
        return jsonify({
            'question': question,
            'answer': answer,
            'relevant_transactions': len(relevant_docs)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/export', methods=['POST'])
def export_reports():
    """Export financial reports as Excel"""
    try:
        data = request.json
        profit_loss = data.get('profit_loss', {})
        balance_sheet = data.get('balance_sheet', {})
        cash_flow = data.get('cash_flow', {})
        
        # Create Excel file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"financial_reports_{timestamp}.xlsx"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Profit & Loss
            pl_df = pd.DataFrame([profit_loss]).T
            pl_df.columns = ['Amount (₹)']
            pl_df.to_excel(writer, sheet_name='Profit & Loss')
            
            # Balance Sheet
            bs_data = []
            if 'Assets' in balance_sheet:
                for key, value in balance_sheet['Assets'].items():
                    bs_data.append({'Category': 'Assets', 'Item': key, 'Amount (₹)': value})
            if 'Liabilities' in balance_sheet:
                for key, value in balance_sheet['Liabilities'].items():
                    bs_data.append({'Category': 'Liabilities', 'Item': key, 'Amount (₹)': value})
            if 'Equity' in balance_sheet:
                for key, value in balance_sheet['Equity'].items():
                    bs_data.append({'Category': 'Equity', 'Item': key, 'Amount (₹)': value})
            
            bs_df = pd.DataFrame(bs_data)
            bs_df.to_excel(writer, sheet_name='Balance Sheet', index=False)
            
            # Cash Flow
            cf_df = pd.DataFrame([cash_flow]).T
            cf_df.columns = ['Amount (₹)']
            cf_df.to_excel(writer, sheet_name='Cash Flow')
        
        return send_file(filepath, as_attachment=True, download_name=filename)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    # For production, use gunicorn or similar WSGI server
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

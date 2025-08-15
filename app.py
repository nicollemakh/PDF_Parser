from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify
import os
from utils import extract_text_from_pdf, summarize_text, generate_quiz, text_to_speech, highlight_pdf


UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_pdf():
    if 'pdf' not in request.files:
        return redirect(url_for('index'))
    file = request.files['pdf']
    if file and allowed_file(file.filename):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        return jsonify({'success': True, 'filename': file.filename})
    return jsonify({'success': False})

@app.route('/search', methods=['POST'])
def search_text():
    data = request.json
    filename = data['filename']
    query = data['query']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # Highlighted output path
    highlighted_path = os.path.join(app.config['UPLOAD_FOLDER'], f"highlighted_{filename}")
    result_path = highlight_pdf(filepath, query, highlighted_path)

    if result_path:
        return send_file(result_path, as_attachment=True)
    else:
        return jsonify({'error': 'Highlighting failed'}), 500


@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.json
    filename = data['filename']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    text = extract_text_from_pdf(filepath)
    summary = summarize_text(text)
    return jsonify({'summary': summary})

@app.route('/quiz', methods=['POST'])
def quiz():
    data = request.json
    filename = data['filename']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    text = extract_text_from_pdf(filepath)
    quiz = generate_quiz(text)
    return jsonify({'quiz': quiz})

@app.route('/tts', methods=['POST'])
def tts():
    data = request.json
    filename = data['filename']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    text = extract_text_from_pdf(filepath)
    summary = summarize_text(text)
    audio_path = text_to_speech(summary)
    return send_file(audio_path, as_attachment=True)

# Instructions:
# 1. Install Python 3.x
# 2. pip install -r requirements.txt
# 3. python app.py
# 4. Open http://127.0.0.1:5000 in browser

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
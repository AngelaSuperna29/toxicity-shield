from flask import Flask, render_template, request, jsonify
from models import analyze_toxicity, rewrite_text
from utils import detect_language, translate_to_english, translate_back

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    # accept JSON or form
    if request.is_json:
        data = request.get_json() or {}
        text = data.get('text', '') or ''
        out_lang = (data.get('lang') or '').strip()
    else:
        text = request.form.get('text', '') or ''
        out_lang = (request.form.get('lang') or '').strip()

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    # detect language if not provided
    if not out_lang:
        out_lang = detect_language(text)

    # translate to english for processing
    text_en = translate_to_english(text)

    # analyze toxicity
    tox = analyze_toxicity(text_en)

    # generate polite rewrite in English
    rewritten_en = rewrite_text(text_en)

    # translate back if requested (keep same language by default)
    rewritten_final = translate_back(rewritten_en, out_lang)

    return jsonify({
        'original': text,
        'toxicity': tox.get('label'),
        'score': round(tox.get('score', 0), 3),
        'rewritten': rewritten_final
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

# Toxicity Shield — AI-Powered Toxic Comment Filter & Rewriter

This is a ready-to-run Flask project that detects toxic comments and rewrites them politely.

## Features
- Toxicity detection using `unitary/toxic-bert` (Hugging Face)
- Polite rewriter using `google/flan-t5-small` (T5 family)
- Language detection and optional translation using `deep-translator`
- Simple Bootstrap UI

## Quick start
1. Create & activate virtualenv
   ```bash
   python -m venv .venv
   .venv\\Scripts\\activate   # windows
   source .venv/bin/activate  # mac/linux
   ```
2. Install requirements
   ```bash
   pip install -r requirements.txt
   ```
3. Run
   ```bash
   python app.py
   ```
4. Open http://127.0.0.1:5000

## Notes
- Models will download the first time and may take a few minutes.
- If you want to avoid downloading large models, consider swapping the rewriter for a rule-based approach in `models.py`.

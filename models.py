import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Device
device = 0 if torch.cuda.is_available() else -1

# Toxicity detector (sequence classification pipeline)
tox_model_name = 'unitary/toxic-bert'
tox_pipeline = pipeline('text-classification', model=tox_model_name, return_all_scores=True, device=device)

# Polite rewriter (T5-family)
rewriter_model_name = 'google/flan-t5-small'
rewriter_tokenizer = AutoTokenizer.from_pretrained(rewriter_model_name)
rewriter_model = AutoModelForSeq2SeqLM.from_pretrained(rewriter_model_name)
if device >= 0:
    rewriter_model = rewriter_model.to(device)

def analyze_toxicity(text, threshold=0.5):
    try:
        scores = tox_pipeline(text[:1000])
        items = scores[0] if isinstance(scores, list) and len(scores)>0 else scores
        best = max(items, key=lambda x: x.get('score',0))
        label = best.get('label','').lower()
        score = best.get('score',0.0)
        is_toxic = any(k in label for k in ['toxic','insult','abuse','hate','offensive']) and score>=threshold
        return {'label': 'toxic' if is_toxic else 'non-toxic', 'score': score}
    except Exception as e:
        return {'label': 'error', 'score': 0.0, 'error': str(e)}

def rewrite_text(text, max_length=128):
    prompt = f"Rewrite the following sentence to be polite, respectful, and constructive in English: {text}"
    inputs = rewriter_tokenizer.encode(prompt, return_tensors='pt', truncation=True, max_length=512)
    if device>=0:
        inputs = inputs.to(device)
    outputs = rewriter_model.generate(inputs, max_length=max_length, num_beams=5, early_stopping=True, no_repeat_ngram_size=3)
    dec = rewriter_tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    out = dec.strip()
    if not out or out.lower() == text.lower():
        out = text
        out = out.replace('stupid','not very thoughtful')
        out = out.replace('idiot','person')
    return out

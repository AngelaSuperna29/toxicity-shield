from deep_translator import GoogleTranslator
from langdetect import detect

def detect_language(text):
    try:
        return detect(text)
    except Exception:
        return 'en'

def translate_to_english(text):
    lang = detect_language(text)
    if lang != 'en':
        try:
            return GoogleTranslator(source=lang, target='en').translate(text)
        except Exception:
            return text
    return text

def translate_back(text, target_lang):
    if not target_lang or target_lang == 'en':
        return text
    try:
        return GoogleTranslator(source='en', target=target_lang).translate(text)
    except Exception:
        return text

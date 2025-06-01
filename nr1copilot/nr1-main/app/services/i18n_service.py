
from typing import Dict
from ..schemas import SetLanguageIn, SetLanguageOut, TranslationsOut

class I18nError(Exception):
    pass

# Sample translations (you can expand this)
TRANSLATIONS = {
    "en": {
        "welcome": "Welcome",
        "process_video": "Process Video",
        "download": "Download",
        "upload": "Upload",
        "error": "Error",
        "success": "Success"
    },
    "es": {
        "welcome": "Bienvenido",
        "process_video": "Procesar Video",
        "download": "Descargar",
        "upload": "Subir",
        "error": "Error",
        "success": "Éxito"
    }
}

# Global language state (in production, use Redis or database)
current_language = "en"

def get_translations_service() -> TranslationsOut:
    """Get translations for current language"""
    global current_language
    
    if current_language not in TRANSLATIONS:
        raise I18nError(f"Language {current_language} not supported")
    
    return TranslationsOut(
        translations=TRANSLATIONS[current_language],
        language=current_language
    )

def set_language_service(data: SetLanguageIn) -> SetLanguageOut:
    """Set the active language"""
    global current_language
    
    if data.language not in TRANSLATIONS:
        raise I18nError(f"Language {data.language} not supported")
    
    current_language = data.language
    
    return SetLanguageOut(language=current_language)

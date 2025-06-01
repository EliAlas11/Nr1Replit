"""
I18n Service Layer

- Handles internationalization (i18n) logic: language setting, translation retrieval, etc.
- All business logic, validation, and error handling for i18n is centralized here.
- Designed for auditability, security, and testability (Stripe/Netflix standards).
- All functions are stateless and side-effect free except for language/translation updates.
- TODO: Replace in-memory store with persistent database for production.
"""

from typing import Dict

from app.schemas import SetLanguageIn, SetLanguageOut, TranslationsOut

class I18nError(Exception):
    """Custom exception for i18n service errors."""
    pass

# In-memory language/translation store for demo
_fake_language = "en"
_fake_translations = {"en": {"hello": "Hello", "bye": "Goodbye"}, "es": {"hello": "Hola", "bye": "Adiós"}}

def get_translations_service() -> TranslationsOut:
    """
    Retrieve current translations for the active language.

    Returns:
        TranslationsOut: The translations for the current language.

    Raises:
        I18nError: If translations are not found.
    """
    translations = _fake_translations.get(_fake_language)
    if not translations:
        raise I18nError("No translations found for current language.")
    return TranslationsOut(translations=translations)

def set_language_service(data: SetLanguageIn) -> SetLanguageOut:
    """
    Set the active language for translations.

    Args:
        data (SetLanguageIn): The language to set.

    Returns:
        SetLanguageOut: Confirmation message.

    Raises:
        I18nError: If language is not supported.
    """
    global _fake_language
    if data.language not in _fake_translations:
        raise I18nError(f"Language '{data.language}' not supported.")
    _fake_language = data.language
    return SetLanguageOut(message=f"Language set to {data.language}")

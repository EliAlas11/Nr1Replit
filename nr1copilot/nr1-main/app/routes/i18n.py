
<old_str></old_str>
<new_str>"""
Internationalization Routes
"""

from fastapi import APIRouter, HTTPException
import logging
from ..schemas import TranslationOut, LanguageOut, SuccessResponse

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/i18n/languages", response_model=SuccessResponse)
async def get_languages():
    """Get available languages"""
    languages = [
        LanguageOut(code="en", name="English", native_name="English", is_active=True),
        LanguageOut(code="es", name="Spanish", native_name="Español", is_active=True),
        LanguageOut(code="fr", name="French", native_name="Français", is_active=True)
    ]
    return SuccessResponse(message="Languages retrieved", data=languages)

@router.get("/i18n/translations/{language}", response_model=SuccessResponse)
async def get_translations(language: str):
    """Get translations for a language"""
    translations = {
        "welcome": "Welcome",
        "login": "Login",
        "signup": "Sign Up",
        "process_video": "Process Video"
    }
    return SuccessResponse(message="Translations retrieved", data=translations)</new_str>

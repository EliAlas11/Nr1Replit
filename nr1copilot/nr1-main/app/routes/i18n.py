from fastapi import APIRouter
from app.schemas import SetLanguageIn, SetLanguageOut, TranslationsOut
from ..controllers.i18n_controller import get_translations, set_language

router = APIRouter(prefix="/api/v1/i18n", tags=["I18N"])

@router.get("/translations", response_model=TranslationsOut)
def translations():
    return get_translations()

@router.post("/set-language", response_model=SetLanguageOut)
def set_lang(data: SetLanguageIn):
    return set_language(data)

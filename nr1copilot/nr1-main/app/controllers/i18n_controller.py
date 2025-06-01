from fastapi import HTTPException
from app.schemas import SetLanguageIn, SetLanguageOut, TranslationsOut
from ..services.i18n_service import (
    get_translations_service,
    set_language_service,
    I18nError,
)

def get_translations() -> TranslationsOut:
    """
    Retrieve current translations for the active language. Raises HTTPException on error.
    """
    try:
        return get_translations_service()
    except I18nError as e:
        raise HTTPException(status_code=400, detail=str(e))

def set_language(data: SetLanguageIn) -> SetLanguageOut:
    """
    Set the active language for translations. Raises HTTPException on error.
    """
    try:
        return set_language_service(data)
    except I18nError as e:
        raise HTTPException(status_code=400, detail=str(e))

# Minimal test/assertion to confirm endpoint signature compiles
if __name__ == "__main__":
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)
    # Sanity-check GET /api/v1/i18n returns 200 and expected shape
    resp = client.get("/api/v1/i18n")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, dict) and "translations" in data

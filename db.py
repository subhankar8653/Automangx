import os
import logging
from motor.motor_asyncio import AsyncIOMotorClient

logger = logging.getLogger(__name__)

MONGO_URI = os.environ.get("MONGO_URI", "YOUR_MONGODB_URI_HERE")
DB_NAME = os.environ.get("MONGO_DB_NAME", "manga_bot")

_client = AsyncIOMotorClient(MONGO_URI)
_db = _client[DB_NAME]
_settings_collection = _db["user_settings"]

# ─────────────────────────────────────────
# Default settings — naye user ke liye
# ─────────────────────────────────────────
DEFAULT_SETTINGS = {
    "quality": "720p",      # 360p, 480p, 720p, 1080p, 4K
    "voice": "hi-female",   # hi-female, hi-male, hi-female-alt
    "bgm_enabled": True,
    "bgm_volume": 30,       # 0-100 (%)
    "text_removal": False,  # True = panel ka text/speech-bubble hata do, False = waisa hi rakho
}

QUALITY_OPTIONS = {
    "360p": 360,
    "480p": 480,
    "720p": 720,
    "1080p": 1080,
    "4K": 2160,
}

# ─────────────────────────────────────────
# Voice options — edge-tts (Microsoft Neural) primary
# gTTS automatically fallback hoga agar edge-tts fail ho
# ─────────────────────────────────────────
VOICE_OPTIONS = {
    "hi-female": {
        "label": "Hindi Female — Swara 🙆‍♀️",
        "engine": "edge",
        "edge_voice": "hi-IN-SwaraNeural",
        "gtts_tld": "co.in",
    },
    "hi-male": {
        "label": "Hindi Male — Madhur 🙆‍♂️",
        "engine": "edge",
        "edge_voice": "hi-IN-MadhurNeural",
        "gtts_tld": "com",
    },
    "hi-female-alt": {
        "label": "Hindi Female Alt — Ananya 🌸",
        "engine": "edge",
        "edge_voice": "hi-IN-AnanyaNeural",
        "gtts_tld": "co.in",
    },
}


async def get_user_settings(user_id: int) -> dict:
    """User ki settings MongoDB se laata hai. Nahi milti to default deta hai."""
    try:
        doc = await _settings_collection.find_one({"user_id": user_id})
        if doc:
            settings = DEFAULT_SETTINGS.copy()
            settings.update({k: v for k, v in doc.items() if k in DEFAULT_SETTINGS})
            return settings
        return DEFAULT_SETTINGS.copy()
    except Exception as e:
        logger.error(f"DB read error: {e}")
        return DEFAULT_SETTINGS.copy()


async def update_user_setting(user_id: int, key: str, value) -> bool:
    """Ek setting update karta hai (upsert)."""
    try:
        await _settings_collection.update_one(
            {"user_id": user_id},
            {"$set": {key: value}},
            upsert=True
        )
        return True
    except Exception as e:
        logger.error(f"DB write error: {e}")
        return False


async def reset_user_settings(user_id: int) -> bool:
    """Default settings pe wapas le jaata hai."""
    try:
        await _settings_collection.update_one(
            {"user_id": user_id},
            {"$set": DEFAULT_SETTINGS.copy()},
            upsert=True
        )
        return True
    except Exception as e:
        logger.error(f"DB reset error: {e}")
        return False

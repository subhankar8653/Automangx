import os
import re
import gc
import cv2
import zipfile
import numpy as np
import asyncio
import tempfile
import logging
import json
import time
from pathlib import Path
from PIL import Image
from google import genai
from google.genai import types
from gtts import gTTS
from moviepy.editor import (
    ImageClip, AudioClip, AudioFileClip, concatenate_videoclips, concatenate_audioclips,
    CompositeAudioClip, CompositeVideoClip, VideoClip, VideoFileClip, afx
)
from pdf2image import convert_from_path

logger = logging.getLogger(__name__)

# ── Multi-key rotation ──
def _load_gemini_keys() -> list:
    keys = []
    for i in range(1, 11):
        k = os.environ.get(f"GEMINI_API_KEY_{i}", "").strip()
        if k and k != "YOUR_GEMINI_KEY_HERE":
            keys.append(k)
    k = os.environ.get("GEMINI_API_KEY", "").strip()
    if k and k != "YOUR_GEMINI_KEY_HERE" and k not in keys:
        keys.append(k)
    if not keys:
        keys = ["YOUR_GEMINI_KEY_HERE"]
    return keys

GEMINI_API_KEYS = _load_gemini_keys()
_current_key_idx = 0

def _get_genai_client(key_idx: int = None):
    idx = key_idx if key_idx is not None else _current_key_idx
    idx = idx % len(GEMINI_API_KEYS)
    return genai.Client(api_key=GEMINI_API_KEYS[idx])

genai_client = _get_genai_client(0)

DEFAULT_BGM_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "assets", "default_bgm.mp3"
)

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')

CANVAS_W, CANVAS_H = 1280, 720  # 16:9 landscape


class MangaProcessor:

    BEAT_PAUSE = 0.15  # beats ke beech silence gap — thoda breathing room

    MIN_GEMINI_GAP = 6.5  # seconds per key (free tier: 10 RPM)

    AUDIO_SPEED = 1.25  # 1.4 pe robotic lagti thi, 1.25 natural + fast enough

    # FIX #4: Blank panel threshold 0.90 → 0.96 — sirf EXTREME white panels skip
    # Previous 0.90 threshold ne bahut saare valid light-bg panels skip kar diye the
    BLANK_PANEL_WHITE_RATIO = 0.96

    def __init__(self):
        self.model_name = 'gemini-2.5-flash'
        self.temp_files = []
        self._key_last_call = [0.0] * len(GEMINI_API_KEYS)

    # ─────────────────────────────────────────
    # 1. PDF → Images
    # ─────────────────────────────────────────
    def pdf_to_images(self, pdf_path: str) -> list:
        try:
            from pdf2image import pdfinfo_from_path
            info = pdfinfo_from_path(pdf_path)
            total_pages = info.get("Pages", 0)
            logger.info(f"PDF mein total {total_pages} pages hain")
        except Exception as e:
            logger.warning(f"pdfinfo error: {e}")
            total_pages = 0

        image_paths = []

        if total_pages > 0:
            for page_num in range(1, total_pages + 1):
                try:
                    pages = convert_from_path(
                        pdf_path, dpi=200,
                        first_page=page_num, last_page=page_num
                    )
                    if not pages:
                        continue
                    tmp = tempfile.NamedTemporaryFile(
                        suffix=f'_page{page_num}.jpg', delete=False)
                    pages[0].save(tmp.name, 'JPEG', quality=92)
                    image_paths.append(tmp.name)
                    self.temp_files.append(tmp.name)
                except Exception as e:
                    logger.error(f"Page {page_num} error: {e}")
                    continue
        else:
            try:
                pages = convert_from_path(pdf_path, dpi=200)
                for i, page in enumerate(pages):
                    tmp = tempfile.NamedTemporaryFile(suffix=f'_page{i}.jpg', delete=False)
                    page.save(tmp.name, 'JPEG', quality=92)
                    image_paths.append(tmp.name)
                    self.temp_files.append(tmp.name)
            except Exception as e:
                logger.error(f"Bulk PDF conversion error: {e}")
                raise ValueError(f"PDF process nahi ho payi: {e}")

        if not image_paths:
            raise ValueError("PDF se ek bhi page extract nahi hui!")

        logger.info(f"PDF se {len(image_paths)} pages nikale")
        return image_paths

    # ─────────────────────────────────────────
    # 1b. ZIP → Images
    # ─────────────────────────────────────────
    def zip_to_images(self, zip_path: str) -> list:
        image_paths = []
        extract_dir = tempfile.mkdtemp(prefix='manga_zip_')
        self.temp_files.append(extract_dir)

        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(extract_dir)
        except zipfile.BadZipFile:
            raise ValueError("ZIP file corrupt hai ya invalid hai!")

        all_files = []
        for root, _, files in os.walk(extract_dir):
            for fname in files:
                if fname.lower().endswith(IMAGE_EXTENSIONS):
                    all_files.append(os.path.join(root, fname))

        def natural_key(path):
            name = os.path.basename(path)
            return [int(t) if t.isdigit() else t.lower()
                    for t in re.split(r'(\d+)', name)]

        all_files.sort(key=natural_key)

        if not all_files:
            raise ValueError("ZIP ke andar koi image nahi mili!")

        for f in all_files:
            image_paths.append(f)
            self.temp_files.append(f)

        logger.info(f"ZIP se {len(image_paths)} images nikali")
        return image_paths

    # ─────────────────────────────────────────
    # 1c. Blank panel filter
    # FIX #4: Threshold 0.90 → 0.96 (sirf extreme white pages skip)
    # ─────────────────────────────────────────
    def is_blank_panel(self, img_path: str) -> bool:
        try:
            img = cv2.imread(img_path)
            if img is None:
                return False
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            white_pixels = np.sum(gray > 240)
            total_pixels = gray.size
            white_ratio = white_pixels / total_pixels if total_pixels else 0
            return white_ratio >= self.BLANK_PANEL_WHITE_RATIO
        except Exception as e:
            logger.warning(f"Blank-panel check error: {e}")
            return False

    def filter_blank_panels(self, image_paths: list, *parallel_lists) -> tuple:
        keep_indices = []
        for i, p in enumerate(image_paths):
            if self.is_blank_panel(p):
                logger.info(f"Blank panel skip: {os.path.basename(p)}")
            else:
                keep_indices.append(i)

        filtered_main = [image_paths[i] for i in keep_indices]
        filtered_others = tuple(
            [lst[i] for i in keep_indices] for lst in parallel_lists
        )
        return (filtered_main,) + filtered_others

    # ─────────────────────────────────────────
    # 2. Text Removal (OpenCV) — improved, less destructive
    # FIX #5: Better bubble detection — art ko preserve karo
    # ─────────────────────────────────────────
    def remove_text_from_images(self, image_paths: list) -> list:
        cleaned_paths = []
        for img_path in image_paths:
            try:
                img = cv2.imread(img_path)
                if img is None:
                    cleaned_paths.append(img_path)
                    continue

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # FIX: Sirf clearly-white speech bubbles target karo
                # Previous code zyada aggressive tha — dark panels mein bhi art erase hota tha
                # New approach: white-filled closed regions (speech bubbles) only
                _, white_mask = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY)

                # Speech bubble contours — RETR_EXTERNAL sirf outer boundary
                contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                bubble_mask = np.zeros_like(gray)
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    # Size range: bahut chhote dots skip, bahut bade (poora panel) skip
                    if 800 < area < 40000:
                        # Convexity check: speech bubbles mostly convex hote hain
                        hull = cv2.convexHull(cnt)
                        hull_area = cv2.contourArea(hull)
                        if hull_area > 0 and (area / hull_area) > 0.6:
                            cv2.drawContours(bubble_mask, [cnt], -1, 255, -1)

                # Text pixels inside bubbles = edge pixels within white regions
                edges = cv2.Canny(gray, 80, 160)
                text_mask = cv2.bitwise_and(edges, bubble_mask)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                text_mask = cv2.dilate(text_mask, kernel, iterations=3)

                # Inpaint only if meaningful mask exists
                if np.sum(text_mask > 0) > 100:
                    result = cv2.inpaint(img, text_mask, inpaintRadius=5,
                                         flags=cv2.INPAINT_TELEA)
                else:
                    result = img.copy()

                tmp = tempfile.NamedTemporaryFile(suffix='_clean.jpg', delete=False)
                cv2.imwrite(tmp.name, result, [cv2.IMWRITE_JPEG_QUALITY, 88])
                cleaned_paths.append(tmp.name)
                self.temp_files.append(tmp.name)
            except Exception as e:
                logger.warning(f"Clean error ({img_path}): {e} — original use")
                cleaned_paths.append(img_path)

        logger.info(f"{len(cleaned_paths)} images process ho gayi")
        return cleaned_paths

    # ─────────────────────────────────────────
    # 2b. Best available Gemini key
    # ─────────────────────────────────────────
    def _pick_best_key(self, exclude_idx: int = None) -> int:
        now = time.time()
        best_idx, best_rest = -1, -1.0
        for i, last in enumerate(self._key_last_call):
            if i == exclude_idx:
                continue
            rest = now - last
            if rest > best_rest:
                best_rest = rest
                best_idx = i
        if best_idx == -1:
            best_idx = 0
        return best_idx

    # ─────────────────────────────────────────
    # 3. Gemini se per-panel beats — IMPROVED PROMPT
    # FIX #6: Quality vs concise conflict resolve kiya
    # ─────────────────────────────────────────
    def _make_fallback_beats(self) -> list:
        return [
            {"text": "Yeh panel abhi load nahi ho paya, aage badhte hain.", "position": 50},
        ]

    async def generate_panel_script(self, image_path: str, story_context: str = "") -> tuple:
        """
        Returns: (beats_list, updated_story_context)
        """
        try:
            with open(image_path, 'rb') as f:
                img_bytes = f.read()
        except Exception as e:
            logger.warning(f"Image read error ({image_path}): {e}")
            return self._make_fallback_beats(), story_context

        context_block = (
            f"📖 AB TAK KI KAHANI (pichle panels se):\n{story_context}\n\n"
            if story_context.strip() else
            "📖 Yeh PEHLA panel hai is comic ka — koi pichla context nahi hai.\n\n"
        )

        # Prompt: short beats, strict sync, smart pan direction
        prompt = (
            "Tu ek YouTube manga Hindi narrator hai. Seedha aur fast bata — "
            "jaise koi dost 30 second mein poori scene explain kare.\n\n"
            + context_block +
            "Is panel ko TOP se BOTTOM tak dekh aur beats mein bata:\n\n"
            "STRICT RULES:\n"
            "1. Har beat = sirf EK moment — dialogue ya action, dono ek saath.\n"
            "   SAHI: 'Rote hue usne kaha — main tumhe bacha nahi paya...'\n"
            "   GALAT: 'Usne kaha kuch. [alag beat] Woh ro raha tha.'\n"
            "2. Har beat SIRF 1 sentence — max 15 words. Zyada nahi.\n"
            "3. Speech bubbles ko Hindi mein naturally retell karo — translate mat karo.\n"
            "4. Character naam use karo agar panel mein dikh raha ho.\n"
            "5. FORBIDDEN (yeh likhoge toh output reject hoga):\n"
            "   - Apni reaction: 'waah', 'kya scene hai', 'dil bhar aaya' etc.\n"
            "   - Dialogue ke baad explanation: 'matlab woh dukhi tha', 'iska matlab...'\n"
            "   - Filler: 'is panel mein', 'yahan', 'dekho', 'aur phir'\n"
            "   - Koi bhi cheez jo panel mein clearly nahi dikh rahi\n"
            "6. Jo bol rahe ho wahi screen pe dikh raha hona chahiye — sync zaroori hai.\n"
            "7. Beats ki sankhya: jo scene mein distinct moments hain utne — faaltu beat mat banao.\n\n"
            "8. Har beat ke liye \"pan\" field bhi do:\n"
            "   - \"top_to_bottom\": is beat mein camera upar se niche move kare (e.g. pehle face dikha, phir body)\n"
            "   - \"bottom_to_top\": camera niche se upar move kare (e.g. pehle feet, phir face)\n"
            "   - \"static\": camera ek jagah ruke (jab poora scene ek hi area mein ho)\n"
            "   position = is beat ka center point (0=top, 100=bottom). "
            "pan direction = crop window us center ke around kaise move kare.\n\n"
            "SIRF JSON return karo:\n"
            '{"beats": [{"position": 10, "pan": "top_to_bottom", "text": "..."}, '
            '{"position": 70, "pan": "static", "text": "..."}], '
            '"updated_context": "2-3 sentence summary of story so far"}'
        )

        content_parts = [
            types.Part.from_text(text=prompt),
            types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
        ]

        gen_config = types.GenerateContentConfig(
            safety_settings=[
                types.SafetySetting(category='HARM_CATEGORY_HARASSMENT', threshold='BLOCK_NONE'),
                types.SafetySetting(category='HARM_CATEGORY_HATE_SPEECH', threshold='BLOCK_NONE'),
                types.SafetySetting(category='HARM_CATEGORY_SEXUALLY_EXPLICIT', threshold='BLOCK_NONE'),
                types.SafetySetting(category='HARM_CATEGORY_DANGEROUS_CONTENT', threshold='BLOCK_NONE'),
            ],
        )

        max_attempts = min(3 * len(GEMINI_API_KEYS), max(3, len(GEMINI_API_KEYS) * 2))
        last_429_key = None

        for attempt in range(1, max_attempts + 1):
            key_idx = self._pick_best_key(exclude_idx=last_429_key)
            client = _get_genai_client(key_idx)

            elapsed = time.time() - self._key_last_call[key_idx]
            if elapsed < self.MIN_GEMINI_GAP:
                wait_needed = self.MIN_GEMINI_GAP - elapsed
                logger.info(f"Key {key_idx+1} throttle — {wait_needed:.1f}s wait...")
                await asyncio.sleep(wait_needed)

            try:
                logger.info(f"Gemini call attempt {attempt} (key {key_idx+1}/{len(GEMINI_API_KEYS)})")
                loop = asyncio.get_event_loop()
                self._key_last_call[key_idx] = time.time()
                response = await loop.run_in_executor(
                    None, lambda c=client: c.models.generate_content(
                        model=self.model_name,
                        contents=content_parts,
                        config=gen_config,
                    )
                )

                if not response.candidates:
                    raise ValueError("Gemini ne koi candidate nahi diya")
                candidate = response.candidates[0]
                finish_reason = getattr(candidate, 'finish_reason', None)
                finish_str = str(getattr(finish_reason, 'name', finish_reason) or "")
                if finish_str and 'STOP' not in finish_str.upper():
                    raise ValueError(f"Gemini finish_reason: {finish_str}")

                text = response.text.strip()
                if '```json' in text:
                    text = text.split('```json')[1].split('```')[0].strip()
                elif '```' in text:
                    text = text.split('```')[1].split('```')[0].strip()

                parsed = json.loads(text)
                if not isinstance(parsed, dict):
                    raise ValueError("Response dict nahi hai")

                beats_data = parsed.get("beats", [])
                new_context = (parsed.get("updated_context") or "").strip()

                if not isinstance(beats_data, list) or len(beats_data) == 0:
                    raise ValueError("Empty beats from Gemini")

                beats = []
                for item in beats_data:
                    txt = (item.get("text") or "").strip()
                    if not txt:
                        continue
                    pos = item.get("position", 50)
                    try:
                        pos = max(0, min(100, float(pos)))
                    except (TypeError, ValueError):
                        pos = 50
                    pan = (item.get("pan") or "static").strip().lower()
                    if pan not in ("top_to_bottom", "bottom_to_top", "static"):
                        pan = "static"
                    beats.append({"text": txt, "position": pos, "pan": pan})

                if not beats:
                    raise ValueError("Saare beats empty nikle")

                beats.sort(key=lambda b: b["position"])

                if not new_context:
                    new_context = story_context

                logger.info(f"Panel se {len(beats)} beats mile")
                return beats, new_context

            except Exception as e:
                err_str = str(e)
                logger.error(f"Gemini error (attempt {attempt}) key {key_idx+1}: {err_str[:300]}")

                is_rate_limit = ('429' in err_str or
                                 'quota' in err_str.lower() or
                                 'RESOURCE_EXHAUSTED' in err_str)
                is_overloaded = ('503' in err_str or
                                  'unavailable' in err_str.lower() or
                                  'overloaded' in err_str.lower())

                if is_rate_limit:
                    last_429_key = key_idx
                    if len(GEMINI_API_KEYS) > 1:
                        logger.info(f"Key {key_idx+1} rate-limited — switch...")
                        await asyncio.sleep(1)
                    else:
                        wait_time = 65
                        m = re.search(r'seconds:\s*(\d+)', err_str)
                        if m:
                            wait_time = int(m.group(1)) + 10
                        logger.info(f"1 key — {wait_time}s wait...")
                        await asyncio.sleep(wait_time)
                elif is_overloaded:
                    wait_time = 5 * attempt
                    logger.info(f"Gemini overloaded — {wait_time}s wait...")
                    await asyncio.sleep(wait_time)
                else:
                    break

        logger.info("Fallback beats use kar raha hoon")
        return self._make_fallback_beats(), story_context

    # ─────────────────────────────────────────
    # 3b. Batch: 2 panels ek hi Gemini call mein
    # ─────────────────────────────────────────
    async def generate_panel_scripts_batch(
        self, image_paths: list, story_context: str = ""
    ) -> tuple:
        if len(image_paths) == 1:
            beats, ctx = await self.generate_panel_script(image_paths[0], story_context)
            return [beats], ctx

        img_parts = []
        for i, path in enumerate(image_paths):
            try:
                with open(path, 'rb') as f:
                    img_bytes = f.read()
                img_parts.append((i + 1, img_bytes))
            except Exception as e:
                logger.warning(f"Batch image read error ({path}): {e}")
                img_parts.append((i + 1, None))

        context_block = (
            f"📖 AB TAK KI KAHANI:\n{story_context}\n\n"
            if story_context.strip() else
            "📖 Yeh pehle panels hain — koi pichla context nahi.\n\n"
        )

        batch_prompt = (
            "Tu ek YouTube manga Hindi narrator hai. Tujhe 2 manga panels diye ja rahe "
            "hain (Panel 1 aur Panel 2). Dono ko top-to-bottom order mein seedha suna.\n\n"
            + context_block +
            "STRICT RULES (dono panels ke liye):\n"
            "1. Har beat = sirf EK moment — dialogue ya action, dono ek saath.\n"
            "   SAHI: 'Rote hue usne kaha — main tumhe bacha nahi paya...'\n"
            "   GALAT: 'Usne kaha kuch.' [alag beat] 'Woh ro raha tha.'\n"
            "2. Har beat SIRF 1 sentence — max 15 words. Zyada nahi.\n"
            "3. Speech bubbles ko Hindi mein naturally retell karo — translate mat karo.\n"
            "4. Character naam use karo agar panel mein dikh raha ho.\n"
            "5. FORBIDDEN:\n"
            "   - Apni reaction: 'waah', 'kya scene hai', 'dil bhar aaya' etc.\n"
            "   - Dialogue ke baad explanation: 'matlab woh dukhi tha', 'iska matlab...'\n"
            "   - Filler: 'is panel mein', 'yahan', 'dekho', 'aur phir'\n"
            "   - Jo panel mein clearly nahi dikh raha\n"
            "6. Jo bol rahe ho wahi screen pe dikh raha hona chahiye — sync zaroori hai.\n"
            "7. Beats ki sankhya: sirf actual distinct moments — faaltu beat mat banao.\n"
            "8. Har beat ke liye \"pan\" field bhi do:\n"
            "   - \"top_to_bottom\": camera upar se niche (e.g. face → body)\n"
            "   - \"bottom_to_top\": camera niche se upar (e.g. feet → face)\n"
            "   - \"static\": ek jagah ruko\n\n"
            "SIRF JSON return karo:\n"
            '{"panel_1": {"beats": [{"position": 10, "pan": "top_to_bottom", "text": "..."}, ...], '
            '"updated_context": "..."}, '
            '"panel_2": {"beats": [{"position": 20, "pan": "static", "text": "..."}, ...], '
            '"updated_context": "..."}}'
        )

        content_parts = [types.Part.from_text(text=batch_prompt)]
        for panel_num, img_bytes in img_parts:
            if img_bytes is not None:
                content_parts.append(
                    types.Part.from_text(text=f"[Panel {panel_num}]")
                )
                content_parts.append(
                    types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg")
                )

        gen_config = types.GenerateContentConfig(
            safety_settings=[
                types.SafetySetting(category='HARM_CATEGORY_HARASSMENT', threshold='BLOCK_NONE'),
                types.SafetySetting(category='HARM_CATEGORY_HATE_SPEECH', threshold='BLOCK_NONE'),
                types.SafetySetting(category='HARM_CATEGORY_SEXUALLY_EXPLICIT', threshold='BLOCK_NONE'),
                types.SafetySetting(category='HARM_CATEGORY_DANGEROUS_CONTENT', threshold='BLOCK_NONE'),
            ],
        )

        max_attempts = min(3 * len(GEMINI_API_KEYS), max(3, len(GEMINI_API_KEYS) * 2))
        last_429_key = None
        ctx = story_context

        for attempt in range(1, max_attempts + 1):
            key_idx = self._pick_best_key(exclude_idx=last_429_key)
            client = _get_genai_client(key_idx)

            elapsed = time.time() - self._key_last_call[key_idx]
            if elapsed < self.MIN_GEMINI_GAP:
                wait_needed = self.MIN_GEMINI_GAP - elapsed
                logger.info(f"Batch key {key_idx+1} throttle — {wait_needed:.1f}s wait...")
                await asyncio.sleep(wait_needed)

            try:
                logger.info(f"Gemini BATCH call attempt {attempt} (key {key_idx+1})")
                loop = asyncio.get_event_loop()
                self._key_last_call[key_idx] = time.time()
                response = await loop.run_in_executor(
                    None, lambda c=client: c.models.generate_content(
                        model=self.model_name,
                        contents=content_parts,
                        config=gen_config,
                    )
                )

                if not response.candidates:
                    raise ValueError("No candidates")
                candidate = response.candidates[0]
                finish_str = str(getattr(getattr(candidate, 'finish_reason', None), 'name', '') or "")
                if finish_str and 'STOP' not in finish_str.upper():
                    raise ValueError(f"finish_reason: {finish_str}")

                text = response.text.strip()
                if '```json' in text:
                    text = text.split('```json')[1].split('```')[0].strip()
                elif '```' in text:
                    text = text.split('```')[1].split('```')[0].strip()

                parsed = json.loads(text)
                if not isinstance(parsed, dict):
                    raise ValueError("Response dict nahi hai")

                all_beats = []
                for pi in range(len(image_paths)):
                    key = f"panel_{pi + 1}"
                    pdata = parsed.get(key, {})
                    raw_beats = pdata.get("beats", [])
                    beats = []
                    for item in raw_beats:
                        txt = (item.get("text") or "").strip()
                        if not txt:
                            continue
                        pos = item.get("position", 50)
                        try:
                            pos = max(0, min(100, float(pos)))
                        except (TypeError, ValueError):
                            pos = 50
                        pan = (item.get("pan") or "static").strip().lower()
                        if pan not in ("top_to_bottom", "bottom_to_top", "static"):
                            pan = "static"
                        beats.append({"text": txt, "position": pos, "pan": pan})
                    if not beats:
                        beats = self._make_fallback_beats()
                    beats.sort(key=lambda b: b["position"])
                    all_beats.append(beats)
                    new_ctx = (pdata.get("updated_context") or "").strip()
                    if new_ctx:
                        ctx = new_ctx

                logger.info(f"Batch: {len(image_paths)} panels done")
                return all_beats, ctx

            except Exception as e:
                err_str = str(e)
                logger.error(f"Batch error (attempt {attempt}) key {key_idx+1}: {err_str[:300]}")

                is_rate_limit = ('429' in err_str or 'RESOURCE_EXHAUSTED' in err_str or 'quota' in err_str.lower())
                is_overloaded = ('503' in err_str or 'unavailable' in err_str.lower())

                if is_rate_limit:
                    last_429_key = key_idx
                    if len(GEMINI_API_KEYS) > 1:
                        logger.info(f"Batch key {key_idx+1} rate-limited — switch...")
                        await asyncio.sleep(1)
                    else:
                        wait_time = 65
                        m = re.search(r'seconds:\s*(\d+)', err_str)
                        if m:
                            wait_time = int(m.group(1)) + 10
                        await asyncio.sleep(wait_time)
                elif is_overloaded:
                    await asyncio.sleep(5 * attempt)
                else:
                    break

        # Batch fail — individually fallback
        logger.warning("Batch fail — individually fallback...")
        all_beats = []
        for path in image_paths:
            beats, ctx = await self.generate_panel_script(path, ctx)
            all_beats.append(beats)
        return all_beats, ctx

    # gTTS voice map
    VOICE_TLD_MAP = {
        "hi-female": "co.in",
        "hi-male": "com",
    }

    async def text_to_speech(self, text: str, output_path: str, voice: str = "hi-female") -> float:
        tld = self.VOICE_TLD_MAP.get(voice, "co.in")
        loop = asyncio.get_event_loop()
        result = {"duration": 0.0}

        def _gen():
            gTTS(text=text, lang='hi', tld=tld, slow=False).save(output_path)

            from pydub import AudioSegment
            sound = AudioSegment.from_file(output_path)

            if self.AUDIO_SPEED and self.AUDIO_SPEED != 1.0:
                sped_path = output_path + '.sped.mp3'
                try:
                    new_rate = int(sound.frame_rate * self.AUDIO_SPEED)
                    fast_sound = sound._spawn(
                        sound.raw_data,
                        overrides={"frame_rate": new_rate}
                    ).set_frame_rate(sound.frame_rate)
                    fast_sound.export(sped_path, format="mp3")
                    os.replace(sped_path, output_path)
                    result["duration"] = len(fast_sound) / 1000.0
                except Exception as e:
                    logger.warning(f"Audio speed-up fail ({e})")
                    if os.path.exists(sped_path):
                        try:
                            os.remove(sped_path)
                        except Exception:
                            pass
                    result["duration"] = len(sound) / 1000.0
            else:
                result["duration"] = len(sound) / 1000.0

        await loop.run_in_executor(None, _gen)
        return result["duration"]

    # ─────────────────────────────────────────
    # 5. Panel ko CANVAS mein scale karo
    # ─────────────────────────────────────────
    def _load_scaled_panel(self, img_path: str):
        """
        Option A — zoom-crop approach:
        - Horizontal/square panels: scale to fit canvas width, center vertically (unchanged)
        - Tall vertical panels: scale so WIDTH = CANVAS_W (full bleed)
          At render time, make_frame crops a CANVAS_H window at the beat's position
          → viewer sees a zoomed-in, fully readable section, no thin strip
        Returns: (panel_rgb, is_tall, scaled_h)
          panel_rgb  — full scaled panel (may be taller than CANVAS_H for tall panels)
          is_tall    — True if panel height > CANVAS_H after scaling
          scaled_h   — actual pixel height of panel_rgb
        """
        img = cv2.imread(img_path)
        if img is None:
            blank = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
            return blank, False, CANVAS_H

        h, w = img.shape[:2]
        aspect = w / h  # <1 for tall panels

        if aspect >= 1.0:
            # Horizontal/square — fit to canvas width, may have letterbox
            scale = CANVAS_W / w
            scaled_w = CANVAS_W
            scaled_h = max(1, int(h * scale))
            is_tall = False
        else:
            # Tall vertical — scale width to CANVAS_W so it fills full bleed
            # Height will be >> CANVAS_H; we'll crop per-beat in make_frame
            scale = CANVAS_W / w
            scaled_w = CANVAS_W
            scaled_h = max(1, int(h * scale))
            is_tall = scaled_h > CANVAS_H

        # Safety cap to avoid OOM on insanely tall panels
        MAX_PANEL_H = 12000
        if scaled_h > MAX_PANEL_H:
            scale_down = MAX_PANEL_H / scaled_h
            scaled_h = MAX_PANEL_H
            scaled_w = max(1, int(scaled_w * scale_down))

        panel_resized = cv2.resize(img, (scaled_w, scaled_h))
        panel_rgb = cv2.cvtColor(panel_resized, cv2.COLOR_BGR2RGB)

        return panel_rgb, is_tall, scaled_h

    # ─────────────────────────────────────────
    # 6. Panel clip with scroll + audio sync
    # ─────────────────────────────────────────
    async def create_panel_clip(self, img_path: str, beats: list, voice: str = "hi-female"):
        panel_rgb, is_tall, panel_h = self._load_scaled_panel(img_path)
        scroll_range = max(0, panel_h - CANVAS_H)

        beat_audio_paths = []
        beat_durations = []
        for i, beat in enumerate(beats):
            audio_tmp = tempfile.NamedTemporaryFile(suffix=f'_beat{i}.mp3', delete=False)
            audio_path = audio_tmp.name
            audio_tmp.close()
            self.temp_files.append(audio_path)

            try:
                actual_dur = await self.text_to_speech(beat["text"], audio_path, voice=voice)
                if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
                    raise ValueError("Audio empty")
                if actual_dur <= 0:
                    from pydub import AudioSegment
                    actual_dur = len(AudioSegment.from_file(audio_path)) / 1000.0
                dur = actual_dur + self.BEAT_PAUSE
            except Exception as e:
                logger.warning(f"Beat {i} TTS error: {e} — silence fallback")
                dur = 1.5

            beat_audio_paths.append(audio_path)
            beat_durations.append(dur)

        # SYNC FIX: Pehle audio combine karo, real durations measure karo,
        # phir timeline banao — estimated durations pe nahi, actual pe
        combined_audio_path = None
        audio_clip = None
        actual_duration = sum(beat_durations)
        if actual_duration <= 0:
            actual_duration = 1.5

        # Real measured durations for timeline (pydub se accurate ms)
        real_durations = []

        try:
            from pydub import AudioSegment
            silence_ms = int(self.BEAT_PAUSE * 1000)
            silence_seg = AudioSegment.silent(duration=silence_ms)

            combined = AudioSegment.empty()
            valid_count = 0
            for idx, p in enumerate(beat_audio_paths):
                if os.path.exists(p) and os.path.getsize(p) > 0:
                    seg = AudioSegment.from_file(p)
                    seg_dur_s = len(seg) / 1000.0
                    combined += seg
                    combined += silence_seg
                    real_durations.append(seg_dur_s + self.BEAT_PAUSE)
                    valid_count += 1
                else:
                    fallback_dur_ms = int((beat_durations[idx] - self.BEAT_PAUSE) * 1000)
                    fallback_seg = AudioSegment.silent(duration=max(500, fallback_dur_ms))
                    combined += fallback_seg
                    combined += silence_seg
                    real_durations.append((max(500, fallback_dur_ms) / 1000.0) + self.BEAT_PAUSE)

            if valid_count > 0 and len(combined) > 0:
                combined_tmp = tempfile.NamedTemporaryFile(
                    suffix='_combined_audio.mp3', delete=False)
                combined_audio_path = combined_tmp.name
                combined_tmp.close()
                self.temp_files.append(combined_audio_path)
                combined.export(combined_audio_path, format="mp3", bitrate="128k")
                actual_duration = len(combined) / 1000.0
                audio_clip = AudioFileClip(combined_audio_path)
                logger.info(f"Audio: {actual_duration:.2f}s, {valid_count} beats")
            else:
                logger.warning("Koi valid beat audio nahi")
                real_durations = beat_durations

        except Exception as e:
            logger.warning(f"pydub error: {e} — moviepy fallback")
            real_durations = beat_durations
            try:
                valid_audio_clips = []
                for idx, p in enumerate(beat_audio_paths):
                    if os.path.exists(p) and os.path.getsize(p) > 0:
                        valid_audio_clips.append(AudioFileClip(p))
                    if idx < len(beat_audio_paths) - 1:
                        silence = AudioClip(lambda t: 0, duration=self.BEAT_PAUSE, fps=44100)
                        valid_audio_clips.append(silence)
                if valid_audio_clips:
                    audio_clip = concatenate_audioclips(valid_audio_clips)
                    actual_duration = audio_clip.duration
            except Exception as e2:
                logger.warning(f"Moviepy fallback bhi fail: {e2}")
                audio_clip = None

        # Timeline: real measured durations pe banao
        if not real_durations:
            real_durations = beat_durations

        timeline = []
        t_cursor = 0.0
        for beat, dur in zip(beats, real_durations):
            y_target = int((beat["position"] / 100.0) * scroll_range)
            pan = beat.get("pan", "static")
            timeline.append({
                "start": t_cursor,
                "end": t_cursor + dur,
                "y": y_target,
                "pan": pan,
            })
            t_cursor += dur

        # Last beat actual audio end tak extend karo
        if timeline:
            timeline[-1]["end"] = actual_duration

        def get_y_at_time(t):
            if scroll_range <= 0 or not timeline:
                return 0

            # Find active beat
            active_idx = len(timeline) - 1
            for i, seg in enumerate(timeline):
                if t < seg["end"]:
                    active_idx = i
                    break

            seg = timeline[active_idx]
            seg_start = seg["start"]
            seg_end = seg["end"]
            seg_dur = max(0.01, seg_end - seg_start)
            center_y = seg["y"]
            pan = seg.get("pan", "static")

            # Pan range: sirf 12% of canvas — gentle reveal, scenes miss nahi honge
            # 30% bahut zyada tha — screen jaldi jump karta tha
            PAN_RANGE = int(CANVAS_H * 0.12)

            # Transition: beats ke beech smooth ease (first 25% of beat duration)
            # Zyada transition time = smoother feel between beats
            TRANSITION_FRAC = 0.25
            transition_time = min(0.4, seg_dur * TRANSITION_FRAC)
            time_into_seg = t - seg_start

            # Base y: smooth ease from prev beat center to this beat center
            if active_idx > 0 and transition_time > 0 and time_into_seg < transition_time:
                prev_y = timeline[active_idx - 1]["y"]
                progress = time_into_seg / transition_time
                # Ease-in-out: starts slow, fast in middle, slow at end
                progress = progress * progress * (3 - 2 * progress)
                base_y = prev_y + (center_y - prev_y) * progress
            else:
                base_y = float(center_y)

            # Pan offset within beat — gentle reveal motion
            # Only kicks in after transition is done
            remaining_frac = max(0.0, time_into_seg - transition_time)
            remaining_dur = max(0.01, seg_dur - transition_time)
            pan_progress = min(1.0, remaining_frac / remaining_dur)
            # Ease-out: fast start, gentle finish
            pan_progress = 1 - (1 - pan_progress) ** 2

            if pan == "top_to_bottom":
                pan_offset = -PAN_RANGE + (2 * PAN_RANGE * pan_progress)
            elif pan == "bottom_to_top":
                pan_offset = PAN_RANGE - (2 * PAN_RANGE * pan_progress)
            else:
                pan_offset = 0

            final_y = int(base_y + pan_offset)
            return max(0, min(scroll_range, final_y))

        def _ken_burns(source_frame, t, dur, zoom_max=0.03):
            if dur <= 0:
                return source_frame
            zoom = 1.0 + zoom_max * (t / dur)
            if abs(zoom - 1.0) < 0.001:
                return source_frame
            sh, sw = source_frame.shape[:2]
            zh = int(sh * zoom)
            zw = int(sw * zoom)
            zoomed = cv2.resize(source_frame, (zw, zh))
            cy = (zh - sh) // 2
            cx = (zw - sw) // 2
            return zoomed[cy:cy + sh, cx:cx + sw]

        if not is_tall or scroll_range <= 0:
            # Horizontal / short panel — blurred bg + centered panel (screenshot 2/3 style)
            # Original image se blurred bg banana — black bars nahi
            orig_img = cv2.imread(img_path)
            if orig_img is not None:
                oh, ow = orig_img.shape[:2]
                bg_scale = max(CANVAS_W / ow, CANVAS_H / oh)
                bg_w = max(1, int(ow * bg_scale))
                bg_h = max(1, int(oh * bg_scale))
                bg_res = cv2.resize(orig_img, (bg_w, bg_h))
                bx = (bg_w - CANVAS_W) // 2
                by = (bg_h - CANVAS_H) // 2
                bg_crop = bg_res[by:by + CANVAS_H, bx:bx + CANVAS_W]
                blur_k = 71
                bg_blurred = cv2.GaussianBlur(bg_crop, (blur_k, blur_k), 0)
                bg_blurred = (bg_blurred * 0.45).clip(0, 255).astype(np.uint8)
                canvas_base = cv2.cvtColor(bg_blurred, cv2.COLOR_BGR2RGB)
            else:
                canvas_base = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)

            # Panel ko center mein overlay karo
            pw = panel_rgb.shape[1]
            ph = min(panel_h, CANVAS_H)
            x_off = max(0, (CANVAS_W - pw) // 2)
            y_off = max(0, (CANVAS_H - ph) // 2)
            x_end = min(CANVAS_W, x_off + pw)
            y_end = min(CANVAS_H, y_off + ph)
            canvas_base[y_off:y_end, x_off:x_end] = panel_rgb[:y_end - y_off, :x_end - x_off]

            def make_frame(t):
                return _ken_burns(canvas_base, t, actual_duration, zoom_max=0.04)
        else:
            # Tall vertical panel — 10% margin dono side, blurred bg
            # Panel width = 80% of CANVAS_W (10% left + 10% right padding)
            MARGIN = int(CANVAS_W * 0.10)
            panel_display_w = CANVAS_W - 2 * MARGIN  # 1024px

            # Scale panel to fit in panel_display_w width
            orig_tall = cv2.imread(img_path)
            if orig_tall is not None:
                th, tw = orig_tall.shape[:2]
                # Scale so width = panel_display_w
                t_scale = panel_display_w / tw
                t_scaled_w = panel_display_w
                t_scaled_h = max(1, int(th * t_scale))
                if t_scaled_h > 12000:
                    t_scale2 = 12000 / t_scaled_h
                    t_scaled_h = 12000
                    t_scaled_w = max(1, int(t_scaled_w * t_scale2))
                panel_display = cv2.resize(orig_tall, (t_scaled_w, t_scaled_h))
                panel_display_rgb = cv2.cvtColor(panel_display, cv2.COLOR_BGR2RGB)
                scroll_range_display = max(0, t_scaled_h - CANVAS_H)

                # Blurred bg from original
                bg_scale2 = max(CANVAS_W / tw, CANVAS_H / th)
                bg2_w = max(1, int(tw * bg_scale2))
                bg2_h = max(1, int(th * bg_scale2))
                bg2_res = cv2.resize(orig_tall, (bg2_w, bg2_h))
                bx2 = (bg2_w - CANVAS_W) // 2
                by2 = (bg2_h - CANVAS_H) // 2
                bg2_crop = bg2_res[by2:by2 + CANVAS_H, bx2:bx2 + CANVAS_W]
                bg2_blur = cv2.GaussianBlur(bg2_crop, (71, 71), 0)
                bg2_dark = (bg2_blur * 0.40).clip(0, 255).astype(np.uint8)
                bg2_rgb = cv2.cvtColor(bg2_dark, cv2.COLOR_BGR2RGB)
            else:
                panel_display_rgb = panel_rgb
                scroll_range_display = scroll_range
                bg2_rgb = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)

            # Actual panel width after scaling (may be less than panel_display_w if height capped)
            actual_panel_w = panel_display_rgb.shape[1]
            # Center the panel horizontally
            x_start = (CANVAS_W - actual_panel_w) // 2
            x_end_paste = x_start + actual_panel_w

            # Override scroll_range for tall panel (get_y_at_time uses outer scroll_range)
            # We re-clamp here to scroll_range_display
            def make_frame(t):
                y2 = get_y_at_time(t)
                y2 = max(0, min(scroll_range_display, y2))
                ph_display = panel_display_rgb.shape[0]
                slice_h = min(CANVAS_H, ph_display - y2)
                panel_slice = panel_display_rgb[y2:y2 + slice_h, 0:actual_panel_w]

                # Start from blurred bg
                frame = bg2_rgb.copy()
                # Paste panel centered — actual width, no hardcoded assumptions
                paste_h = panel_slice.shape[0]
                paste_w = panel_slice.shape[1]
                frame[0:paste_h, x_start:x_start + paste_w] = panel_slice

                return _ken_burns(frame, t, actual_duration, zoom_max=0.02)

        video_clip = VideoClip(make_frame, duration=actual_duration)
        if audio_clip:
            video_clip = video_clip.set_audio(audio_clip)

        return video_clip

    # ─────────────────────────────────────────
    # 6b. Intro title screen clip
    # FIX #1: AudioArrayClip ab properly import ho raha hai upar se
    # ─────────────────────────────────────────
    def _make_intro_clip(self, title: str, duration: float = 3.0):
        from PIL import Image as PILImage, ImageDraw, ImageFont
        import textwrap

        frame = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
        for y in range(CANVAS_H):
            val = int(15 + 20 * (y / CANVAS_H))
            frame[y, :] = [val, val, val]

        pil_img = PILImage.fromarray(frame)
        draw = ImageDraw.Draw(pil_img)

        try:
            font_big = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 52)
            font_sub = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28)
        except Exception:
            font_big = ImageFont.load_default()
            font_sub = font_big

        wrapped = textwrap.wrap(title, width=18) or [title]
        line_h = 65
        total_h = len(wrapped) * line_h
        y_start = (CANVAS_H - total_h) // 2 - 40

        accent_y = y_start - 30
        draw.rectangle([(CANVAS_W // 2 - 60, accent_y), (CANVAS_W // 2 + 60, accent_y + 4)],
                        fill=(220, 30, 30))

        for i, line in enumerate(wrapped):
            bbox = draw.textbbox((0, 0), line, font=font_big)
            tw = bbox[2] - bbox[0]
            x = (CANVAS_W - tw) // 2
            y = y_start + i * line_h
            draw.text((x + 3, y + 3), line, font=font_big, fill=(0, 0, 0, 180))
            draw.text((x, y), line, font=font_big, fill=(255, 255, 255))

        sub_text = "📖 Manga Story"
        bbox2 = draw.textbbox((0, 0), sub_text, font=font_sub)
        sw = bbox2[2] - bbox2[0]
        draw.text(((CANVAS_W - sw) // 2, y_start + total_h + 20), sub_text,
                    font=font_sub, fill=(180, 180, 180))

        intro_arr = np.array(pil_img)

        def make_intro_frame(t):
            alpha = 1.0
            if t < 0.5:
                alpha = t / 0.5
            elif t > duration - 0.5:
                alpha = (duration - t) / 0.5
            alpha = max(0.0, min(1.0, alpha))
            return (intro_arr * alpha).astype(np.uint8)

        video = VideoClip(make_intro_frame, duration=duration)

        # moviepy 1.0.3 mein AudioArrayClip nahi hai — AudioClip lambda use karo
        silent_audio = AudioClip(
            lambda t: np.zeros((1, 2)) if np.isscalar(t) else np.zeros((len(t), 2)),
            duration=duration,
            fps=44100
        )
        return video.set_audio(silent_audio)

    # ─────────────────────────────────────────
    # 7. Full video pipeline
    # FIX #2: story_title bot.py se ab pass hoga
    # ─────────────────────────────────────────
    async def create_video_from_panels(self, image_paths: list,
                                        panel_beats: list,
                                        quality_height: int = 720,
                                        voice: str = "hi-female",
                                        bgm_enabled: bool = True,
                                        bgm_volume: int = 30,
                                        story_title: str = "Manga Story",
                                        progress_callback=None) -> str:
        if not image_paths or not panel_beats:
            raise ValueError("Images ya beats empty hain!")

        loop = asyncio.get_event_loop()
        part_paths = []

        # Intro screen
        intro_part_path = None
        try:
            intro_clip = self._make_intro_clip(story_title, duration=3.0)
            intro_tmp = tempfile.NamedTemporaryFile(suffix='_intro.mp4', delete=False)
            intro_part_path = intro_tmp.name
            intro_tmp.close()
            self.temp_files.append(intro_part_path)
            intro_audio_tmp = os.path.join(tempfile.gettempdir(), f'manga_intro_audio_{os.getpid()}.m4a')
            self.temp_files.append(intro_audio_tmp)
            await loop.run_in_executor(
                None,
                lambda c=intro_clip, p=intro_part_path, pa=intro_audio_tmp: c.write_videofile(
                    p, fps=24, codec='libx264', audio_codec='aac',
                    temp_audiofile=pa, remove_temp=True,
                    threads=2, preset='ultrafast', verbose=False, logger=None,
                )
            )
            logger.info("Intro render ho gaya")
        except Exception as e:
            logger.warning(f"Intro clip error: {e} — skip")
            intro_part_path = None

        # Panel-by-panel render
        for idx, (img_path, beats) in enumerate(zip(image_paths, panel_beats)):
            clip = None
            try:
                clip = await self.create_panel_clip(img_path, beats, voice=voice)

                part_tmp = tempfile.NamedTemporaryFile(
                    suffix=f'_part{idx}.mp4', delete=False)
                part_path = part_tmp.name
                part_tmp.close()
                self.temp_files.append(part_path)

                part_audio_tmp = os.path.join(
                    tempfile.gettempdir(),
                    f'manga_part_audio_{os.getpid()}_{idx}.m4a')
                self.temp_files.append(part_audio_tmp)

                await loop.run_in_executor(
                    None,
                    lambda c=clip, p=part_path, pa=part_audio_tmp: c.write_videofile(
                        p, fps=24, codec='libx264',
                        audio_codec='aac' if c.audio is not None else None,
                        temp_audiofile=pa if c.audio is not None else None,
                        remove_temp=True, threads=2, preset='ultrafast',
                        verbose=False, logger=None,
                    )
                )
                part_paths.append(part_path)
                logger.info(f"Panel {idx + 1}/{len(image_paths)} render done")

            except Exception as e:
                logger.warning(f"Panel clip error ({img_path}): {e} — skip")
            finally:
                if clip is not None:
                    try:
                        clip.close()
                    except Exception:
                        pass
                    del clip
                gc.collect()
                if progress_callback:
                    try:
                        await progress_callback(idx + 1, len(image_paths))
                    except Exception as e:
                        logger.warning(f"Progress callback error: {e}")

        if not part_paths:
            raise ValueError("Koi panel clip nahi bani!")

        # ffmpeg concat + BGM
        output_path = None

        try:
            out_tmp = tempfile.NamedTemporaryFile(suffix='_manga_video.mp4', delete=False)
            output_path = out_tmp.name
            out_tmp.close()
            self.temp_files.append(output_path)

            concat_list_f = tempfile.NamedTemporaryFile(
                mode='w', suffix='_concat.txt', delete=False)
            concat_list_path = concat_list_f.name
            self.temp_files.append(concat_list_path)
            if intro_part_path and os.path.exists(intro_part_path):
                concat_list_f.write(f"file '{intro_part_path}'\n")
            for p in part_paths:
                concat_list_f.write(f"file '{p}'\n")
            concat_list_f.close()

            vf_filter = ""
            if quality_height and quality_height != CANVAS_H:
                aspect = CANVAS_W / CANVAS_H
                new_w = int(quality_height * aspect)
                new_w = new_w if new_w % 2 == 0 else new_w + 1
                new_h = quality_height if quality_height % 2 == 0 else quality_height + 1
                vf_filter = f"scale={new_w}:{new_h}"

            has_bgm = bgm_enabled and os.path.exists(DEFAULT_BGM_PATH)

            def _ffmpeg_merge():
                import subprocess

                if has_bgm:
                    merged_tmp = tempfile.NamedTemporaryFile(suffix='_merged_nobgm.mp4', delete=False)
                    merged_path = merged_tmp.name
                    merged_tmp.close()
                    self.temp_files.append(merged_path)

                    cmd_concat = [
                        'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
                        '-i', concat_list_path,
                        '-c:v', 'copy',
                        '-c:a', 'aac', '-b:a', '128k',
                        merged_path
                    ]
                    subprocess.run(cmd_concat, check=True, capture_output=True)

                    vol = max(0, min(100, bgm_volume)) / 100.0
                    audio_filter = (
                        f"[0:a]aformat=fltp,volume=1.0[main];"
                        f"[1:a]aformat=fltp,volume={vol:.2f},aloop=loop=-1:size=2e+09[bgm];"
                        f"[main][bgm]amix=inputs=2:duration=first:dropout_transition=2[aout]"
                    )
                    cmd_parts = [
                        'ffmpeg', '-y',
                        '-i', merged_path,
                        '-stream_loop', '-1', '-i', DEFAULT_BGM_PATH,
                    ]
                    if vf_filter:
                        cmd_parts += ['-vf', vf_filter]
                    cmd_parts += [
                        '-filter_complex', audio_filter,
                        '-map', '0:v', '-map', '[aout]',
                        '-c:v', 'libx264', '-preset', 'ultrafast',
                        '-c:a', 'aac', '-shortest',
                        output_path
                    ]
                    subprocess.run(cmd_parts, check=True, capture_output=True)
                    logger.info("BGM mix ho gaya")

                else:
                    cmd = [
                        'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
                        '-i', concat_list_path,
                    ]
                    if vf_filter:
                        cmd += ['-vf', vf_filter]
                    cmd += [
                        '-c:v', 'libx264', '-preset', 'ultrafast',
                        '-c:a', 'aac', '-b:a', '128k',
                    ]
                    cmd.append(output_path)
                    subprocess.run(cmd, check=True, capture_output=True)

                logger.info(f"Video ready: {output_path}")

            if progress_callback:
                try:
                    await progress_callback(-1, len(image_paths))
                except Exception:
                    pass

            await loop.run_in_executor(None, _ffmpeg_merge)
            return output_path

        finally:
            gc.collect()

    # ─────────────────────────────────────────
    # 8. Cleanup
    # ─────────────────────────────────────────
    def cleanup(self, *file_lists):
        import shutil
        all_files = list(self.temp_files)
        for file_list in file_lists:
            if not file_list:
                continue
            if isinstance(file_list, str):
                all_files.append(file_list)
            elif isinstance(file_list, list):
                all_files.extend(file_list)
        for f in all_files:
            try:
                if f and os.path.isdir(f):
                    shutil.rmtree(f, ignore_errors=True)
                elif f and os.path.exists(f):
                    os.remove(f)
            except Exception:
                pass
        self.temp_files = []

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

# ── Multi-key rotation — Railway mein GEMINI_API_KEY_1 se _10 tak set karo ──
# Ek hi key se 10 RPM free-tier limit bohot jaldi hit hoti hai. Multiple
# alag Google accounts se keys banao (ai.google.dev) aur .env mein dalo.
# Agar sirf purana GEMINI_API_KEY set hai (bina number ke), to woh bhi
# fallback ke taur par use hoga.
def _load_gemini_keys() -> list:
    keys = []
    # numbered keys: GEMINI_API_KEY_1 ... GEMINI_API_KEY_10
    for i in range(1, 11):
        k = os.environ.get(f"GEMINI_API_KEY_{i}", "").strip()
        if k and k != "YOUR_GEMINI_KEY_HERE":
            keys.append(k)
    # unnumbered fallback
    k = os.environ.get("GEMINI_API_KEY", "").strip()
    if k and k != "YOUR_GEMINI_KEY_HERE" and k not in keys:
        keys.append(k)
    if not keys:
        keys = ["YOUR_GEMINI_KEY_HERE"]   # dummy so app at least starts
    return keys

GEMINI_API_KEYS = _load_gemini_keys()
# _current_key_idx is module-level so all MangaProcessor instances share it
_current_key_idx = 0

def _get_genai_client(key_idx: int = None):
    """key_idx doge to us specific key ka client milega, else current one."""
    idx = key_idx if key_idx is not None else _current_key_idx
    idx = idx % len(GEMINI_API_KEYS)
    return genai.Client(api_key=GEMINI_API_KEYS[idx])

genai_client = _get_genai_client(0)   # backward-compat (batch path ke liye)

# Default BGM track — isi folder mein assets/default_bgm.mp3 rakhna hai
DEFAULT_BGM_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "assets", "default_bgm.mp3"
)

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')

CANVAS_W, CANVAS_H = 1280, 720  # 16:9 landscape canvas (YouTube style)


class MangaProcessor:

    BEAT_PAUSE = 0.35  # beats ke beech chhota silence gap — yeh value
                        # timeline aur actual audio dono mein EXACT same
                        # honi chahiye, warna scroll/audio drift ho jaata hai

    # gemini-2.0-flash free-tier — v1beta mein supported, fast, stable.
    MIN_GEMINI_GAP = 4.0  # seconds per key

    AUDIO_SPEED = 1.5  # voice playback speed multiplier — TTS audio
                        # generate hone ke turant baad isi speed se
                        # permanently speed-up kar dete hain, taaki
                        # duration/timeline calculation (jo audio ki
                        # actual file-duration use karte hain) automatically
                        # sahi (chhoti) duration dekhein — kahin aur extra
                        # speed-adjustment ki zaroorat nahi padti

    def __init__(self):
        self.model_name = 'gemini-2.0-flash'
        self.temp_files = []
        # Per-key last-call timestamp — index matches GEMINI_API_KEYS list
        self._key_last_call = [0.0] * len(GEMINI_API_KEYS)

    # ─────────────────────────────────────────
    # 1. PDF → Images
    # ─────────────────────────────────────────
    def pdf_to_images(self, pdf_path: str) -> list:
        # DPI 200 — taaki speech-bubble ka chhota text bhi Gemini ko clearly
        # dikhe (150dpi par fine text blur ho jaata hai). 220 se thoda kam
        # rakha (200) taaki bade PDFs par memory/time issue na ho Railway
        # jaise limited-resource environment mein.
        try:
            from pdf2image import pdfinfo_from_path
            info = pdfinfo_from_path(pdf_path)
            total_pages = info.get("Pages", 0)
            logger.info(f"PDF mein total {total_pages} pages hain")
        except Exception as e:
            logger.warning(f"pdfinfo error: {e} — page count nahi mila, continue kar rahe")
            total_pages = 0

        image_paths = []

        # Bahut bade PDFs (50+ pages) ko ek saath load karna risky hai
        # (memory + time) — isliye chunks mein process karte hain, ek-ek
        # page convert karke turant save karte hain.
        if total_pages > 0:
            for page_num in range(1, total_pages + 1):
                try:
                    pages = convert_from_path(
                        pdf_path, dpi=200,
                        first_page=page_num, last_page=page_num
                    )
                    if not pages:
                        logger.warning(f"Page {page_num} convert nahi hui — skip")
                        continue
                    tmp = tempfile.NamedTemporaryFile(
                        suffix=f'_page{page_num}.jpg', delete=False)
                    pages[0].save(tmp.name, 'JPEG', quality=92)
                    image_paths.append(tmp.name)
                    self.temp_files.append(tmp.name)
                except Exception as e:
                    logger.error(f"Page {page_num} conversion error: {e} — skip kar raha hoon")
                    continue
        else:
            # Fallback: pdfinfo fail hua to purana bulk-convert tareeka
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

        logger.info(f"PDF se {len(image_paths)} pages nikale (200dpi, q92)")
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
    # 1c. Blank / text-only panels filter out karo
    # ─────────────────────────────────────────
    # Kuch PDFs mein occasionally ek poora page sirf EK speech-bubble ya
    # text-recap ke liye hota hai — koi actual manga artwork nahi, sirf
    # plain white background + ek bubble (jaise "Dannazione..." jaisa
    # transition/recap panel). Aise panels video mein dikhana waste hai —
    # viewer ko sirf ek khaali white frame dikhta hai jab voice bol rahi
    # hoti hai.
    #
    # Detection heuristic: agar image ka 97%+ area near-white hai (sirf
    # thoda sa black text/lines), to woh likely ek blank/text-only panel
    # hai, real artwork nahi. Threshold deliberately HIGH (97%) rakha hai
    # taaki genuine minimalist-style manga panels (jo legitimately safed
    # background use karte hain par real character/art bhi hota hai)
    # galti se skip na ho jaayein — sirf EXTREME cases catch karte hain.
    # 0.90 rakha hai — text-only panels (white bg + bold text) ~92-95% white
    # hote hain, genuine art panels mein usually <85% white hota hai.
    BLANK_PANEL_WHITE_RATIO = 0.90

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
            logger.warning(f"Blank-panel check error ({img_path}): {e} — safe side, blank NAHI maante")
            return False

    def filter_blank_panels(self, image_paths: list, *parallel_lists) -> tuple:
        """
        image_paths ke saath-saath jitni bhi parallel lists diye jaayein
        (jaise cleaned_images), sabko SAME index par filter karta hai —
        taaki saari lists sync rahein.

        Returns: (filtered_image_paths, filtered_parallel_list1, ...)
        """
        keep_indices = []
        for i, p in enumerate(image_paths):
            if self.is_blank_panel(p):
                logger.info(f"Blank/text-only panel skip kiya: {os.path.basename(p)}")
            else:
                keep_indices.append(i)

        filtered_main = [image_paths[i] for i in keep_indices]
        filtered_others = tuple(
            [lst[i] for i in keep_indices] for lst in parallel_lists
        )
        return (filtered_main,) + filtered_others

    # ─────────────────────────────────────────
    # 2. Text Removal (OpenCV) — optional, settings se control hota hai
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
                _, white_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                edges = cv2.Canny(gray, 50, 150)
                text_in_bubbles = cv2.bitwise_and(edges, white_mask)
                dilated = cv2.dilate(text_in_bubbles, kernel, iterations=4)
                contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                bubble_mask = np.zeros_like(gray)
                for cnt in contours:
                    if 500 < cv2.contourArea(cnt) < 50000:
                        cv2.drawContours(bubble_mask, [cnt], -1, 255, -1)
                final_mask = cv2.bitwise_and(dilated, bubble_mask)
                final_mask = cv2.dilate(final_mask, kernel, iterations=2)
                result = cv2.inpaint(img, final_mask, inpaintRadius=7,
                                     flags=cv2.INPAINT_TELEA)
                tmp = tempfile.NamedTemporaryFile(suffix='_clean.jpg', delete=False)
                cv2.imwrite(tmp.name, result, [cv2.IMWRITE_JPEG_QUALITY, 85])
                cleaned_paths.append(tmp.name)
                self.temp_files.append(tmp.name)
            except Exception as e:
                logger.warning(f"Clean error ({img_path}): {e} — original use kar raha hoon")
                cleaned_paths.append(img_path)
        logger.info(f"{len(cleaned_paths)} images clean ho gayi")
        return cleaned_paths

    # ─────────────────────────────────────────
    # 2b. Best available Gemini key choose karo
    # ─────────────────────────────────────────
    def _pick_best_key(self, exclude_idx: int = None) -> int:
        """
        Jo key sabse zyada rest kar chuki ho (last call sabse pehle tha)
        use return karo. Agar koi key 'exclude_idx' pe hai (abhi 429 diya),
        usse skip karo — rotation ka core logic yahi hai.
        """
        now = time.time()
        best_idx, best_rest = -1, -1.0
        for i, last in enumerate(self._key_last_call):
            if i == exclude_idx:
                continue
            rest = now - last
            if rest > best_rest:
                best_rest = rest
                best_idx = i
        # Agar sab exclude hain (sirf 1 key hai), wahi return karo
        if best_idx == -1:
            best_idx = 0
        return best_idx

    # ─────────────────────────────────────────
    # 3. Gemini se per-panel "beats" — explainer-style script
    # ─────────────────────────────────────────
    def _make_fallback_beats(self) -> list:
        # NOTE: Yeh tab use hota hai jab Gemini call 3 baar fail ho jaaye.
        return [
            {"text": "Yeh panel abhi load nahi ho paya, aage badhte hain.", "position": 50},
        ]

    async def generate_panel_script(self, image_path: str, story_context: str = "") -> tuple:
        """
        story_context: ab tak ki kahani ka short summary — pichle panels se
        carry hota hai taaki Gemini continuity rakh sake.

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

        prompt = (
            "Tu ek POPULAR YouTube manga/comic EXPLAINER hai — sochkar bol "
            "jaise tu seedha camera ke saamne baithkar apne viewers ko REAL "
            "MEIN yeh kahani sunna raha hai, jaise tune khud yeh panel "
            "dekha hai aur ab excited ho ke bata raha hai. Yeh kisi "
            "encyclopedia ya subtitle jaisa translation NAHI hai — yeh ek "
            "INSAAN ki awaaz honi chahiye jo kahani mein invested hai.\n\n"
            + context_block +
            "Ab is NAYE panel ko dhyaan se dekh aur kahani aage badhao:\n\n"
            "RULES:\n"
            "1. Speech bubbles/captions/sound-effects ka text padh — yeh "
            "KISI BHI language mein ho sakta hai (English, Italian, "
            "Japanese, Korean, etc.) — uska meaning samajh aur Hindi/"
            "Hinglish mein apne natural words mein bata. Kabhi 'samajh nahi "
            "aaya' mat bol, best-effort translate kar.\n"
            "2. Character ka NAAM pata chal jaaye (text se ya pichle context "
            "se) to naam se refer kar ('Nancy ne kaha', 'Vampire boy ne "
            "muskura ke jawab diya') — generic 'ladki' ya 'ladka' avoid kar "
            "jab tak naam na pata ho.\n"
            "3. Sirf dialogue translate mat kar — characters ki facial "
            "expression, body language, scene ka mood, background bhi "
            "dekh aur natural storytelling mein pirona ('Nancy darr ke "
            "peeche dekhti hai aur kehti hai - ...').\n"
            "4. EK REAL NARRATOR ki tarah bol, copy-paste explainer ki "
            "tarah nahi:\n"
            "   - Kabhi-kabhi viewer se seedha baat kar ('Dekho yahan kya "
            "hota hai', 'Ab yahan twist aata hai bhai')\n"
            "   - Apni khud ki reaction daal jab scene shocking/funny/sad "
            "ho ('Yeh dekh ke toh maza aa gaya', 'Arre yaar yeh toh "
            "heartbreaking hai')\n"
            "   - Rhetorical sawaal pooch jab suspense ho ('Ab yeh kya "
            "karega?', 'Kya yeh sach mein possible hai?')\n"
            "   - Panel-to-panel ek flowing kahani lage, har panel ek "
            "ALAG generic intro/outro line se shuru/khatam mat kar — seedha "
            "kahani mein dive kar jaise pichla panel abhi khatam hua ho\n"
            "   - Natural Hindi filler/emphasis use kar jahan organic lage "
            "('toh phir', 'lekin yaar', 'ekdum se') — lekin overdo mat kar\n"
            "5. KABHI bhi generic filler lines mat de jaise 'is panel mein "
            "scene shuru hota hai', 'kahani yahan aage badhti hai', 'yeh "
            "panel yahan khatam hota hai' — yeh khaali placeholder lagti "
            "hain, ek real narrator yeh kabhi nahi bolega. Har beat mein "
            "ACTUAL content hona chahiye — dialogue, action, ya emotion.\n"
            "6. Agar panel mein bilkul koi text/bubble nahi hai (sirf art "
            "hai), to scene ko apne style mein dramatically describe kar — "
            "kya ho raha hai, characters kaise feel kar rahe hain, kya "
            "tension build ho raha hai — generic line KABHI mat de.\n"
            "7. CONCISE rakh — personality aur energy ke naam par lambi "
            "lines mat bana. Episode already kaafi lamba ho jaata hai "
            "agar har beat 2-3 sentence ka ho jaaye. Jo bhi reaction/"
            "rhetorical-sawaal/filler daal rahe ho (rule 4), use EK CHHOTI "
            "phrase mein fit kar do — pura alag sentence mat bana usi ke "
            "liye. Har beat ideally EK hi sentence mein poora ho (zaroorat "
            "pade tabhi 2 chhote sentences), aur sirf woh information jo "
            "is exact panel-portion ke liye zaroori hai — repeat ya "
            "already-bola-hua context dobara mat bata.\n\n"
            "Panel ko TOP se BOTTOM tak 'beats' mein todo — har beat ek "
            "chhota narration-chunk hai jo panel ke specific vertical hisse "
            "se related hai. 2 bubbles (upar-niche) hain to kam se kam 2 "
            "beats banao.\n\n"
            "Har beat ke liye 'position' do (0=top, 100=bottom panel mein).\n\n"
            "Sirf JSON object return karo, kuch aur nahi, is exact format mein:\n"
            '{"beats": [{"position": 10, "text": "..."}, {"position": 70, "text": "..."}], '
            '"updated_context": "ek chhota (2-3 sentence) summary jo ab tak '
            'ki kahani capture kare — character names, important events — '
            'jo NEXT panel ke liye yaad rakhna zaroori hai"}'
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

        # Total attempts = min(3, number_of_keys * 2) — taaki multiple keys
        # ke saath zyada chances milein bina infinite loop ke
        max_attempts = min(3 * len(GEMINI_API_KEYS), max(3, len(GEMINI_API_KEYS) * 2))
        last_429_key = None   # 429 dene wali key ko next attempt mein avoid karo

        for attempt in range(1, max_attempts + 1):
            # Best key choose karo — jo sabse zyada rest kar chuki ho
            key_idx = self._pick_best_key(exclude_idx=last_429_key)
            client = _get_genai_client(key_idx)

            # Proactive throttle sirf us key ke liye
            elapsed = time.time() - self._key_last_call[key_idx]
            if elapsed < self.MIN_GEMINI_GAP:
                wait_needed = self.MIN_GEMINI_GAP - elapsed
                logger.info(f"Key {key_idx+1} throttle — {wait_needed:.1f}s wait...")
                await asyncio.sleep(wait_needed)

            try:
                logger.info(f"Gemini panel-script call attempt {attempt} (key {key_idx+1}/{len(GEMINI_API_KEYS)})")
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
                    raise ValueError("Gemini ne koi candidate nahi diya (shayad safety block)")
                candidate = response.candidates[0]
                finish_reason = getattr(candidate, 'finish_reason', None)
                finish_str = str(getattr(finish_reason, 'name', finish_reason) or "")
                if finish_str and 'STOP' not in finish_str.upper():
                    raise ValueError(f"Gemini finish_reason: {finish_str} (safety/length block ho sakta hai)")

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
                    raise ValueError("Empty/invalid beats from Gemini")

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
                    beats.append({"text": txt, "position": pos})

                if not beats:
                    raise ValueError("Saare beats empty nikle")

                beats.sort(key=lambda b: b["position"])

                if not new_context:
                    new_context = story_context

                logger.info(f"Panel se {len(beats)} beats mile, context updated")
                return beats, new_context

            except Exception as e:
                err_str = str(e)
                logger.error(f"Gemini error (attempt {attempt}) key {key_idx+1} for {os.path.basename(image_path)}: {err_str[:300]}")

                is_rate_limit = ('429' in err_str or
                                 'quota' in err_str.lower() or
                                 'RESOURCE_EXHAUSTED' in err_str)
                is_overloaded = ('503' in err_str or
                                  'unavailable' in err_str.lower() or
                                  'overloaded' in err_str.lower())

                if is_rate_limit:
                    last_429_key = key_idx  # next attempt mein yeh key avoid karo
                    if len(GEMINI_API_KEYS) > 1:
                        # Doosri key available hai — turant switch, no 65s wait
                        logger.info(f"Key {key_idx+1} rate-limited — dusri key pe switch kar raha hoon...")
                        await asyncio.sleep(1)   # tiny buffer only
                    else:
                        # Sirf 1 key hai — purana 65s wait fallback
                        wait_time = 65
                        m = re.search(r'seconds:\s*(\d+)', err_str)
                        if m:
                            wait_time = int(m.group(1)) + 10
                        logger.info(f"Sirf 1 key hai — {wait_time}s wait karke retry...")
                        await asyncio.sleep(wait_time)
                elif is_overloaded:
                    wait_time = 5 * attempt
                    logger.info(f"Gemini overloaded (503) — {wait_time}s wait karke retry...")
                    await asyncio.sleep(wait_time)
                else:
                    break   # non-retryable error

        logger.info("Fallback beats use kar raha hoon is panel ke liye")
        return self._make_fallback_beats(), story_context

    # ─────────────────────────────────────────
    # 3b. Batch: 2 panels ek hi Gemini call mein process karo
    # ─────────────────────────────────────────
    # Isse Gemini calls aadhi ho jaati hain — 8 pages ke liye 8 calls ki
    # jagah sirf 4 calls. Free tier (10 RPM) par yeh bahut fark dalta hai.
    async def generate_panel_scripts_batch(
        self, image_paths: list, story_context: str = ""
    ) -> tuple:
        """
        image_paths: 2 panel images ki list (ek ya do)
        Returns: (list_of_beats_per_panel, updated_story_context)
        Agar batch call fail ho to har panel pe individually fallback karta hai.
        """
        if len(image_paths) == 1:
            # Sirf ek panel — single call hi use karo
            beats, ctx = await self.generate_panel_script(image_paths[0], story_context)
            return [beats], ctx

        # 2 images ek sath bhejna
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
            "Tu ek POPULAR YouTube manga/comic EXPLAINER hai. Tujhe EKTATH "
            "2 manga panels diye ja rahe hain (Panel 1 aur Panel 2, is "
            "sequence mein). Dono ko top-to-bottom order mein process kar.\n\n"
            + context_block +
            "RULES (dono panels ke liye):\n"
            "1. Speech bubbles ka text KISI BHI language mein ho — Hindi/Hinglish mein natural translate karo.\n"
            "2. Character naam pata ho to naam use karo.\n"
            "3. Dialogue ke saath expression, mood, body language bhi bata.\n"
            "4. Real narrator ki tarah bol — viewer ko engage rakho.\n"
            "5. CONCISE rakh — har beat ideally ek sentence.\n"
            "6. Panel ke top se bottom tak 'beats' mein todo.\n\n"
            "Sirf JSON return karo is EXACT format mein (kuch aur nahi):\n"
            '{"panel_1": {"beats": [{"position": 10, "text": "..."}, ...], '
            '"updated_context": "..."}, '
            '"panel_2": {"beats": [{"position": 20, "text": "..."}, ...], '
            '"updated_context": "..."}}'
        )

        content_parts = [types.Part.from_text(text=batch_prompt)]
        valid_indices = []
        for panel_num, img_bytes in img_parts:
            if img_bytes is not None:
                content_parts.append(
                    types.Part.from_text(text=f"[Panel {panel_num}]")
                )
                content_parts.append(
                    types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg")
                )
                valid_indices.append(panel_num - 1)

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
                logger.info(f"Gemini BATCH call attempt {attempt} (key {key_idx+1}/{len(GEMINI_API_KEYS)}) for {len(image_paths)} panels")
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
                    raise ValueError("No candidates (safety block)")
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
                        beats.append({"text": txt, "position": pos})
                    if not beats:
                        beats = self._make_fallback_beats()
                    beats.sort(key=lambda b: b["position"])
                    all_beats.append(beats)
                    # last panel ka context carry karo
                    new_ctx = (pdata.get("updated_context") or "").strip()
                    if new_ctx:
                        ctx = new_ctx

                logger.info(f"Batch: {len(image_paths)} panels process hue")
                return all_beats, ctx

            except Exception as e:
                err_str = str(e)
                logger.error(f"Batch Gemini error (attempt {attempt}) key {key_idx+1}: {err_str[:300]}")

                is_rate_limit = ('429' in err_str or 'RESOURCE_EXHAUSTED' in err_str or 'quota' in err_str.lower())
                is_overloaded = ('503' in err_str or 'unavailable' in err_str.lower())

                if is_rate_limit:
                    last_429_key = key_idx
                    if len(GEMINI_API_KEYS) > 1:
                        logger.info(f"Batch key {key_idx+1} rate-limited — switch kar raha hoon...")
                        await asyncio.sleep(1)
                    else:
                        wait_time = 65
                        m = re.search(r'seconds:\s*(\d+)', err_str)
                        if m:
                            wait_time = int(m.group(1)) + 10
                        logger.info(f"Batch 1-key fallback — {wait_time}s wait...")
                        await asyncio.sleep(wait_time)
                elif is_overloaded:
                    await asyncio.sleep(5 * attempt)
                else:
                    break

        # Batch fail — individually fallback
        logger.warning("Batch call fail — har panel individually try kar raha hoon...")
        all_beats = []
        for path in image_paths:
            beats, ctx = await self.generate_panel_script(path, ctx)
            all_beats.append(beats)
        return all_beats, ctx
    # NOTE: gTTS sirf ek hi Hindi voice deta hai — true male/female switch
    # iske paas nahi hai. "voice" setting yahan sirf tld (accent/tone) badalti
    # hai — halka sa style difference, na ki alag gender wali awaaz.
    VOICE_TLD_MAP = {
        "hi-female": "co.in",
        "hi-male": "com",
    }

    async def text_to_speech(self, text: str, output_path: str, voice: str = "hi-female") -> float:
        """
        Returns: actual audio duration in seconds (after speed-up).
        Caller should use this value — NOT re-read from AudioFileClip,
        because MP3 header-declared duration can differ from actual playback.
        """
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
                    # pydub se exact duration lo — headers pe bharosa mat karo
                    result["duration"] = len(fast_sound) / 1000.0
                except Exception as e:
                    logger.warning(f"Audio speed-up fail ({e}) — normal speed use kar raha hoon")
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
    # 5. Panel ko CANVAS_W tak scale karna (no distortion, no stretch)
    # ─────────────────────────────────────────
    def _load_scaled_panel(self, img_path: str):
        """
        16:9 YouTube landscape rendering:
        - Vertical manga panel ko landscape canvas (1280x720) mein dikhana hai
        - Strategy: panel ki WIDTH ko canvas ke ek portion mein fit karo (centered),
          baaki sides mein blurred background fill karo
        - Panel ki scaled height CANVAS_H se zyada hogi (tall panels) —
          yahi scroll_range deta hai (top se bottom tak scroll)
        - Returns: (fg_rgb, bg_rgb, fg_h)
          fg_rgb = foreground panel (panel_display_w wide, proportional height)
          bg_rgb = blurred background (1280x720 full canvas)
          fg_h   = foreground panel height (scroll range ke liye)
          panel_display_w = foreground panel ki actual width (canvas center mein)

        Memory cap: extra-tall panels ko 8000px pe clamp karo.

        Landscape mein manga panel ka optimum width:
        - Square panels (aspect ~1:1): CANVAS_H tak height fit karo — no scroll needed
        - Tall panels (h > w): panel ko CANVAS_H ki ~95% height se fit karo as starting
          point, phir scroll se baaki dikhao. Width = CANVAS_H * (w/h), centered.
        - Agar panel width > CANVAS_W: width = CANVAS_W, height proportional (tall scroll)
        """
        img = cv2.imread(img_path)
        if img is None:
            blank = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
            return blank, blank, CANVAS_H

        h, w = img.shape[:2]
        aspect = w / h  # < 1 for tall vertical panels, > 1 for wide panels

        # ── Foreground panel sizing ──
        # Tall manga panels (aspect < 1): width ko canvas width ke max 60% tak rakho
        # taaki sides mein blurred bg visible rahe aur cinematic lage.
        # Agar panel bahut wide hai (aspect >= 1): CANVAS_W mein fit karo.
        if aspect >= 1.0:
            # Wide/square panel — width = canvas width
            fg_w = CANVAS_W
            fg_h = max(1, int(CANVAS_W / aspect))
        else:
            # Tall vertical manga panel — isko landscape canvas ke center mein
            # display karo. Max width = 55% of CANVAS_W taaki dono sides pe
            # blurred background ka "letterbox" effect mile.
            # Height: panel ko CANVAS_H se scale karo — agar scaled height >
            # CANVAS_H tab scroll hoga (jo scroll effect deta hai)
            max_fg_w = int(CANVAS_W * 0.55)
            # Height-based scale: agar pure CANVAS_H fit karen to fg_w kitna hoga?
            h_based_w = max(1, int(CANVAS_H * aspect))
            if h_based_w <= max_fg_w:
                # Panel CANVAS_H mein fit ho jaata hai — use karo (no scroll ya minimal)
                fg_w = h_based_w
                fg_h = CANVAS_H
            else:
                # Panel bahut tall hai — width = max_fg_w, height proportional
                # (CANVAS_H se zyada hogi, yahi scroll_range dega)
                fg_w = max_fg_w
                fg_h = max(1, int(fg_w / aspect))

        MAX_PANEL_H = 8000
        if fg_h > MAX_PANEL_H:
            scale_down = MAX_PANEL_H / fg_h
            fg_h = MAX_PANEL_H
            fg_w = max(1, int(fg_w * scale_down))
            logger.warning(f"Panel capped: original {h}px -> {fg_h}px")

        fg_resized = cv2.resize(img, (fg_w, fg_h))
        fg_rgb = cv2.cvtColor(fg_resized, cv2.COLOR_BGR2RGB)

        # ── Background: cover-fill 1280x720, then heavy blur + darken ──
        bg_scale = max(CANVAS_W / w, CANVAS_H / h)
        bg_w = max(1, int(w * bg_scale))
        bg_h_raw = max(1, int(h * bg_scale))
        bg_resized = cv2.resize(img, (bg_w, bg_h_raw))
        cx = (bg_w - CANVAS_W) // 2
        cy = (bg_h_raw - CANVAS_H) // 2
        bg_crop = bg_resized[cy:cy + CANVAS_H, cx:cx + CANVAS_W]
        blur_k = 71  # must be odd
        bg_blurred = cv2.GaussianBlur(bg_crop, (blur_k, blur_k), 0)
        bg_blurred = (bg_blurred * 0.40).clip(0, 255).astype(np.uint8)
        bg_rgb = cv2.cvtColor(bg_blurred, cv2.COLOR_BGR2RGB)

        return fg_rgb, bg_rgb, fg_h

    # ─────────────────────────────────────────
    # 6. Ek panel ke liye scroll-synced video clip banao
    # ─────────────────────────────────────────
    # Yeh function:
    #   1. Panel ko CANVAS_W tak scale karta hai (full height milti hai)
    #   2. Har beat ka TTS audio banata hai
    #   3. Har beat ki audio-duration ke hisaab se ek scroll-timeline banata
    #      hai — beat N bolte waqt scroll uske 'position' tak smoothly move
    #      ho jaata hai aur wahin hold karta hai jab tak audio chal rahi hai
    #   4. Agar panel ki scaled height <= CANVAS_H (chhota panel), to scroll
    #      ki zaroorat nahi — halka cinematic zoom-in de dete hain
    async def create_panel_clip(self, img_path: str, beats: list, voice: str = "hi-female"):
        fg_rgb, bg_rgb, fg_h = self._load_scaled_panel(img_path)
        scroll_range = max(0, fg_h - CANVAS_H)

        # ── Step 1: Har beat ka audio banao, duration nikaalo ──
        # CRITICAL: duration sirf TTS return value se lo — AudioFileClip se
        # nahi. VBR MP3 headers mein declared duration actual playback se
        # differ kar sakti hai (50-200ms off). Agar yahan galat duration
        # aaye to timeline aur audio drift ho jaata hai.
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
                    # Fallback: pydub se dobara read karo
                    from pydub import AudioSegment
                    actual_dur = len(AudioSegment.from_file(audio_path)) / 1000.0
                dur = actual_dur + self.BEAT_PAUSE
            except Exception as e:
                logger.warning(f"Beat {i} TTS error: {e} — 1.5s silence fallback")
                dur = 1.5

            beat_audio_paths.append(audio_path)
            beat_durations.append(dur)

        total_duration = sum(beat_durations)
        if total_duration <= 0:
            total_duration = 1.5

        # ── Step 2: Beat-timeline banao — har beat ka (start_time, end_time, position_px) ──
        # NOTE: Pehle timeline banate hain, fir actual_duration milne ke baad
        # last segment ka end clamp karte hain — taaki video duration aur
        # last beat ka end_time exactly match karein (no hanging frames).
        timeline = []
        t_cursor = 0.0
        for beat, dur in zip(beats, beat_durations):
            y_target = int((beat["position"] / 100.0) * scroll_range)
            timeline.append({
                "start": t_cursor,
                "end": t_cursor + dur,
                "y": y_target,
            })
            t_cursor += dur

        # ── Step 3: pydub se saare beats ek file mein concat karo ──
        # Pydub sample-accurate hai — koi floating point clock drift nahi,
        # koi VBR header ambiguity nahi. Silence bhi exact milliseconds mein.
        # Final combined file ka duration hi video duration hoga — yeh
        # timeline ke sum se EXACTLY match karega kyunki hum wahi beats +
        # wahi silence gaps dono jagah use kar rahe hain.
        combined_audio_path = None
        audio_clip = None
        actual_duration = total_duration  # fallback

        try:
            from pydub import AudioSegment
            silence_ms = int(self.BEAT_PAUSE * 1000)
            silence_seg = AudioSegment.silent(duration=silence_ms)

            combined = AudioSegment.empty()
            valid_count = 0
            for idx, p in enumerate(beat_audio_paths):
                if os.path.exists(p) and os.path.getsize(p) > 0:
                    seg = AudioSegment.from_file(p)
                    combined += seg
                    valid_count += 1
                else:
                    # Missing beat audio — silence se fill karo
                    fallback_dur_ms = int((beat_durations[idx] - self.BEAT_PAUSE) * 1000)
                    combined += AudioSegment.silent(duration=max(500, fallback_dur_ms))
                # Har beat ke baad (last ko chhodke) exact silence gap daalo
                if idx < len(beat_audio_paths) - 1:
                    combined += silence_seg

            if valid_count > 0 and len(combined) > 0:
                combined_tmp = tempfile.NamedTemporaryFile(
                    suffix='_combined_audio.mp3', delete=False)
                combined_audio_path = combined_tmp.name
                combined_tmp.close()
                self.temp_files.append(combined_audio_path)
                combined.export(combined_audio_path, format="mp3", bitrate="128k")

                # pydub se exact duration — timeline ke saath guaranteed match
                actual_duration = len(combined) / 1000.0
                audio_clip = AudioFileClip(combined_audio_path)
                logger.info(f"Audio concat done: {actual_duration:.2f}s, {valid_count} beats")
            else:
                logger.warning("Koi valid beat audio nahi mili — video bina audio ke banegi")

        except Exception as e:
            logger.warning(f"pydub audio concat error: {e} — moviepy fallback")
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

        # ── Step 3b: Timeline last segment clamp — video end pe audio cut na ho ──
        # actual_duration ab pydub se aaya hai (exact). Last beat ka end_time
        # timeline mein actual_duration se match karna chahiye, warna video
        # actual_duration pe khatam hoga lekin last beat ka scroll position
        # hold nahi hoga, ya ek extra frame dikhega.
        if timeline:
            timeline[-1]["end"] = actual_duration

        # ── Step 4: Scroll-position-at-time function ──
        def get_y_at_time(t):
            if scroll_range <= 0 or not timeline:
                return 0
            active_idx = len(timeline) - 1
            for i, seg in enumerate(timeline):
                if t <= seg["end"]:
                    active_idx = i
                    break
            seg = timeline[active_idx]
            seg_dur = seg["end"] - seg["start"]
            transition_time = min(0.4, seg_dur * 0.3) if seg_dur > 0 else 0
            if active_idx > 0 and transition_time > 0 and t < seg["start"] + transition_time:
                prev_y = timeline[active_idx - 1]["y"]
                progress = (t - seg["start"]) / transition_time
                progress = max(0.0, min(1.0, progress))
                progress = 1 - (1 - progress) ** 3
                return int(prev_y + (seg["y"] - prev_y) * progress)
            return seg["y"]

        # ── Step 5: Frame-generator — 9:16 blurred bg + Ken Burns zoom ──
        # Dono cases (scroll + no-scroll) mein:
        #   - bg_rgb (blurred, full 720x1280) pehle daal do
        #   - fg panel uske upar center mein composite karo
        #   - Ken Burns: gentle 5% zoom-in over clip duration

        def _composite_frame(fg_crop, bg_frame):
            """
            Landscape canvas mein fg panel ko center mein rakho:
            - Horizontally: center (left-right blurred bg visible)
            - Vertically: center (agar panel CANVAS_H se chhota hai)
            """
            frame = bg_frame.copy()
            fh, fw = fg_crop.shape[:2]

            # Horizontal center
            x_off = max(0, (CANVAS_W - fw) // 2)
            x_end = min(CANVAS_W, x_off + fw)

            # Vertical center (for short panels; scroll panels fill full height)
            y_off = max(0, (CANVAS_H - fh) // 2)
            y_end = min(CANVAS_H, y_off + fh)

            fg_h_use = y_end - y_off
            fg_w_use = x_end - x_off

            if fg_h_use > 0 and fg_w_use > 0:
                frame[y_off:y_end, x_off:x_end] = fg_crop[:fg_h_use, :fg_w_use]
            return frame

        def _ken_burns(source_frame, t, dur, zoom_max=0.05):
            """source_frame ko gentle zoom-in karo (Ken Burns effect)."""
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

        if scroll_range <= 0:
            # Chhota/square panel — CANVAS_H mein fit hai already, center mein rakho
            # Landscape mein: fg panel center mein, sides pe blurred bg
            def make_frame(t):
                composite = _composite_frame(fg_rgb, bg_rgb)
                return _ken_burns(composite, t, actual_duration, zoom_max=0.04)
        else:
            # Tall panel — upar se neeche scroll karo (landscape canvas ke center mein)
            def make_frame(t):
                y = get_y_at_time(t)
                y = max(0, min(scroll_range, y))
                # fg_rgb ka vertical slice — CANVAS_H pixels (jo landscape frame ki height hai)
                fg_slice_h = min(CANVAS_H, fg_rgb.shape[0] - y)
                fg_slice = fg_rgb[y:y + fg_slice_h, 0:fg_rgb.shape[1]]
                # Composite: fg ko bg ke upar center mein daalo
                composite = _composite_frame(fg_slice, bg_rgb)
                return _ken_burns(composite, t, actual_duration, zoom_max=0.02)
        video_clip = VideoClip(make_frame, duration=actual_duration)
        if audio_clip:
            video_clip = video_clip.set_audio(audio_clip)

        return video_clip

    # ─────────────────────────────────────────
    # 6b. Intro title screen clip
    # ─────────────────────────────────────────
    def _make_intro_clip(self, title: str, duration: float = 3.0):
        """
        1280x720 landscape intro screen jisme:
        - Title text centered
        - Subtle gradient overlay
        Returns: moviepy VideoClip (no audio)
        """
        from PIL import Image as PILImage, ImageDraw, ImageFont
        import textwrap

        frame = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)

        # Gradient: top dark, middle slightly lighter
        for y in range(CANVAS_H):
            val = int(15 + 20 * (y / CANVAS_H))
            frame[y, :] = [val, val, val]

        # PIL mein convert karo text ke liye
        pil_img = PILImage.fromarray(frame)
        draw = ImageDraw.Draw(pil_img)

        # Font — system default use karo (PIL ka default)
        try:
            font_big = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 52)
            font_sub = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28)
        except Exception:
            font_big = ImageFont.load_default()
            font_sub = font_big

        # Title wrap
        wrapped = textwrap.wrap(title, width=18) or [title]
        line_h = 65
        total_h = len(wrapped) * line_h
        y_start = (CANVAS_H - total_h) // 2 - 40

        # Red accent line above title
        accent_y = y_start - 30
        draw.rectangle([(CANVAS_W // 2 - 60, accent_y), (CANVAS_W // 2 + 60, accent_y + 4)],
                        fill=(220, 30, 30))

        for i, line in enumerate(wrapped):
            bbox = draw.textbbox((0, 0), line, font=font_big)
            tw = bbox[2] - bbox[0]
            x = (CANVAS_W - tw) // 2
            y = y_start + i * line_h
            # Shadow
            draw.text((x + 3, y + 3), line, font=font_big, fill=(0, 0, 0, 180))
            draw.text((x, y), line, font=font_big, fill=(255, 255, 255))

        # Subtitle
        sub_text = "📖 Manga Story"
        bbox2 = draw.textbbox((0, 0), sub_text, font=font_sub)
        sw = bbox2[2] - bbox2[0]
        draw.text(((CANVAS_W - sw) // 2, y_start + total_h + 20), sub_text,
                    font=font_sub, fill=(180, 180, 180))

        intro_arr = np.array(pil_img)

        def make_intro_frame(t):
            # Fade-in first 0.5s, fade-out last 0.5s
            alpha = 1.0
            if t < 0.5:
                alpha = t / 0.5
            elif t > duration - 0.5:
                alpha = (duration - t) / 0.5
            alpha = max(0.0, min(1.0, alpha))
            return (intro_arr * alpha).astype(np.uint8)

        video = VideoClip(make_intro_frame, duration=duration)

        # Silent audio track attach karo — bina iske ffmpeg concat mein
        # audio stream mismatch hoti hai aur final video mein voice nahi aata
        silent_arr = np.zeros((int(44100 * duration), 2), dtype=np.float32)
        silent_audio = AudioArrayClip(silent_arr, fps=44100).set_duration(duration)
        return video.set_audio(silent_audio)


    # ─────────────────────────────────────────
    # 7. Pura video banao — har panel ko ALAG se render karke disk pe
    #    save karte hain, fir saari chhoti files ko merge karte hain
    # ─────────────────────────────────────────
    # IMPORTANT — yeh function pehle saare 8 panels ke VideoClip objects
    # (jo apna poora uncompressed RGB numpy array RAM mein hold karte
    # hain) ek list mein jod ke rakhta tha, fir EK saath concatenate +
    # render karta tha. Matlab jab tak video poori nahi ban jaati, RAM
    # mein saare panels ka combined data ek saath baitha rehta tha.
    #
    # Ab naya approach (chunk-render-then-merge):
    #   1. Har panel ko ALAG se render karo -> chhoti _part_N.mp4 file
    #      disk par save ho jaati hai
    #   2. Us panel ka clip/numpy array IMMEDIATELY close + gc karo —
    #      RAM se hat jaata hai, sirf disk par chhoti mp4 file reh
    #      jaati hai
    #   3. Saare panels render hone ke baad, un chhoti mp4 files ko
    #      VideoFileClip se (disk-backed, RAM-light) reload karke
    #      concatenate karo aur final video banao
    #
    # Isse peak RAM ~1 panel ke barabar rehta hai, saare panels ka sum
    # nahi — jitne bhi panels hon (8 ho ya 20), memory same rahegi.
    async def create_video_from_panels(self, image_paths: list,
                                        panel_beats: list,
                                        quality_height: int = 720,
                                        voice: str = "hi-female",
                                        bgm_enabled: bool = True,
                                        bgm_volume: int = 30,
                                        story_title: str = "Manga Story",
                                        progress_callback=None) -> str:
        """
        image_paths: har panel ki image path
        panel_beats: same length list, har entry list-of-beats hai
                     (generate_panel_script se aaya)
        progress_callback: optional async function(done, total) — har
                     panel render hone ke baad call hota hai, taaki bot.py
                     Telegram status message update kar sake ("Panel 3/8
                     ban gaya")
        """
        if not image_paths or not panel_beats:
            raise ValueError("Images ya beats empty hain!")

        loop = asyncio.get_event_loop()
        part_paths = []

        # ── Step 0: Intro title screen (optional) ──
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
            logger.info("Intro screen render ho gaya")
        except Exception as e:
            logger.warning(f"Intro clip error: {e} — skip kar raha hoon")
            intro_part_path = None

        # ── Step 1: Har panel ko ALAG render karo, disk pe save karo, RAM free karo ──
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
                        p,
                        fps=24,
                        codec='libx264',
                        audio_codec='aac' if c.audio is not None else None,
                        temp_audiofile=pa if c.audio is not None else None,
                        remove_temp=True,
                        threads=2,
                        preset='ultrafast',
                        verbose=False,
                        logger=None,
                    )
                )
                part_paths.append(part_path)
                logger.info(f"Panel {idx + 1}/{len(image_paths)} render ho gaya -> {os.path.basename(part_path)}")

            except Exception as e:
                logger.warning(f"Panel clip error ({img_path}): {e} — skip")
            finally:
                # Yeh panel ka numpy array/audio RAM se IMMEDIATELY hatao,
                # agle panel pe jaane se pehle — yahi is fix ka core hai
                if clip is not None:
                    try:
                        clip.close()
                    except Exception:
                        pass
                    del clip
                gc.collect()
                try:
                    import resource
                    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
                    logger.info(f"Panel {idx + 1}/{len(image_paths)} done — peak RAM so far: {peak_mb:.0f} MB")
                except Exception:
                    pass
                if progress_callback:
                    try:
                        await progress_callback(idx + 1, len(image_paths))
                    except Exception as e:
                        logger.warning(f"Progress callback error: {e}")

        if not part_paths:
            raise ValueError("Koi panel clip nahi bani!")

        # ── Step 2: ffmpeg se zero-RAM concat ──
        # moviepy concatenate_videoclips + write_videofile saari part clips
        # RAM mein reload karta hai — 8 panels * ~60MB = ~480MB+ peak RAM,
        # jo Railway free plan pe OOM-kill + container restart karta hai.
        # Iske bajaye ffmpeg concat demuxer use karo: ek text file mein
        # saari part paths likhte hain, ffmpeg unhe stream-copy se
        # (no re-encode, zero RAM) ek file mein jod deta hai.
        # BGM bhi ffmpeg se hi mix karte hain — alag amix filter se.
        output_path = None

        try:
            out_tmp = tempfile.NamedTemporaryFile(
                suffix='_manga_video.mp4', delete=False)
            output_path = out_tmp.name
            out_tmp.close()
            self.temp_files.append(output_path)

            # Concat list file banao
            concat_list_f = tempfile.NamedTemporaryFile(
                mode='w', suffix='_concat.txt', delete=False)
            concat_list_path = concat_list_f.name
            self.temp_files.append(concat_list_path)
            if intro_part_path and os.path.exists(intro_part_path):
                concat_list_f.write(f"file '{intro_part_path}'\n")
            for p in part_paths:
                concat_list_f.write(f"file '{p}'\n")
            concat_list_f.close()

            # Quality scale: agar quality_height 720 se alag ho to scale karo
            vf_filter = ""
            if quality_height and quality_height != CANVAS_H:
                # 16:9 landscape — width = quality_height * (16/9)
                aspect = CANVAS_W / CANVAS_H  # 1280/720 = 1.777...
                new_w = int(quality_height * aspect)
                new_w = new_w if new_w % 2 == 0 else new_w + 1
                new_h = quality_height if quality_height % 2 == 0 else quality_height + 1
                vf_filter = f"scale={new_w}:{new_h}"

            has_bgm = bgm_enabled and os.path.exists(DEFAULT_BGM_PATH)

            def _ffmpeg_merge():
                import subprocess

                if has_bgm:
                    # Step A: concat parts → temp merged (no BGM yet)
                    merged_tmp = tempfile.NamedTemporaryFile(
                        suffix='_merged_nobgm.mp4', delete=False)
                    merged_path = merged_tmp.name
                    merged_tmp.close()
                    self.temp_files.append(merged_path)

                    cmd_concat = [
                        'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
                        '-i', concat_list_path,
                        '-c:v', 'copy',        # video stream copy — fast
                        '-c:a', 'aac', '-b:a', '128k',  # audio re-encode — consistent stream
                        merged_path
                    ]
                    subprocess.run(cmd_concat, check=True,
                                   capture_output=True)
                    logger.info("Parts concat ho gayi (no BGM)")

                    # Step B: BGM mix — amix filter
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
                    logger.info("BGM mix ho gayi")

                else:
                    # No BGM — direct concat + re-encode (never -c copy: audio
                    # stream mismatch hoti hai intro + panel clips ke beech)
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

                logger.info(f"Video ban gayi: {output_path}")

            if progress_callback:
                # Final merge shuru ho rahi hai — ek aur status update bhejo
                try:
                    await progress_callback(-1, len(image_paths))
                except Exception:
                    pass

            await loop.run_in_executor(None, _ffmpeg_merge)
            logger.info(f"Video ready: {output_path}")
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


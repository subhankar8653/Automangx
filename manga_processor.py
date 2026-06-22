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
    CompositeAudioClip, CompositeVideoClip, VideoClip, afx
)
from pdf2image import convert_from_path

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_KEY_HERE")
genai_client = genai.Client(api_key=GEMINI_API_KEY)

# Default BGM track — isi folder mein assets/default_bgm.mp3 rakhna hai
DEFAULT_BGM_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "assets", "default_bgm.mp3"
)

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')

CANVAS_W, CANVAS_H = 1280, 720  # 16:9 video canvas


class MangaProcessor:

    BEAT_PAUSE = 0.35  # beats ke beech chhota silence gap — yeh value
                        # timeline aur actual audio dono mein EXACT same
                        # honi chahiye, warna scroll/audio drift ho jaata hai

    def __init__(self):
        # gemini-2.0-flash 1 June 2026 ko shutdown ho gaya tha — isliye
        # gemini-2.5-flash use kar rahe hain (same price jo 2.0-flash ka
        # tha: $0.10/$0.40 per million tokens, behtar output limit bhi)
        self.model_name = 'gemini-2.5-flash'
        self.temp_files = []

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
    # 3. Gemini se per-panel "beats" — explainer-style script
    # ─────────────────────────────────────────
    # Har panel ke liye Gemini ko ek baar call karte hain (zyada accurate
    # rehta hai, kyunki Gemini ko sirf ek image par dhyaan dena hota hai).
    # Response: list of beats, har beat mein:
    #   - "text": Hindi/Hinglish narration (dialogue + expression + scene
    #     ka explainer-style mix — sirf bubble text nahi)
    #   - "position": 0-100 (panel ke kis vertical %% par yeh beat focus
    #     karta hai — 0 = top, 100 = bottom)
    def _make_fallback_beats(self) -> list:
        # NOTE: Yeh tab use hota hai jab Gemini call 3 baar fail ho jaaye.
        # Pehle isme 3 "beats" the jo normal content jaisa dikhte the —
        # isse pata hi nahi chalta tha ki AI fail hua ya yeh actual script
        # hai. Ab sirf EK clearly-fallback beat hai poore panel ke liye,
        # taaki agar yeh dikhe to turant samajh aaye ki Gemini fail hua,
        # aur logs check karne chahiye — fake/normal content jaisa nahi lagega.
        return [
            {"text": "Yeh panel abhi load nahi ho paya, aage badhte hain.", "position": 50},
        ]

    async def generate_panel_script(self, image_path: str, story_context: str = "") -> tuple:
        """
        story_context: ab tak ki kahani ka short summary (character names,
        jo ho chuka hai) — pichle panels se carry hota hai, taaki Gemini
        YouTube-style continuity rakh sake ("Nancy ne kaha...", "Tabhi
        vampire boy bola...") instead of generic "ladki ne kaha".

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
            "tension build ho raha hai — generic line KABHI mat de.\n\n"
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

        for attempt in range(1, 4):
            try:
                logger.info(f"Gemini panel-script call attempt {attempt}/3")
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None, lambda: genai_client.models.generate_content(
                        model=self.model_name,
                        contents=content_parts,
                        config=gen_config,
                    )
                )

                # Safety-block ya empty response check
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

                # Agar Gemini ne context nahi diya, purana hi carry karo
                if not new_context:
                    new_context = story_context

                logger.info(f"Panel se {len(beats)} beats mile, context updated")
                return beats, new_context

            except Exception as e:
                err_str = str(e)
                logger.error(f"Gemini error (attempt {attempt}/3) for {os.path.basename(image_path)}: {err_str[:300]}")

                is_rate_limit = ('429' in err_str or
                                 'quota' in err_str.lower() or
                                 'rate' in err_str.lower())
                # 503/UNAVAILABLE = Gemini temporarily overloaded (alag
                # cheez hai quota/rate-limit se) — yeh usually kuch second
                # mein clear ho jaata hai, isliye chhota retry dete hain
                # seedha fallback pe jaane se pehle (warna 8-panel manga
                # mein kayi panels seedhe generic fallback text le lete
                # the jab Gemini thodi der ke liye busy tha)
                is_overloaded = ('503' in err_str or
                                  'unavailable' in err_str.lower() or
                                  'overloaded' in err_str.lower())

                if is_rate_limit and attempt < 3:
                    wait_time = 65
                    m = re.search(r'seconds:\s*(\d+)', err_str)
                    if m:
                        wait_time = int(m.group(1)) + 10
                    logger.info(f"Rate limit — {wait_time}s wait karke retry...")
                    await asyncio.sleep(wait_time)
                elif is_overloaded and attempt < 3:
                    wait_time = 5 * attempt  # 5s, 10s — chhota backoff
                    logger.info(f"Gemini overloaded (503) — {wait_time}s wait karke retry...")
                    await asyncio.sleep(wait_time)
                else:
                    break

        logger.info("Fallback beats use kar raha hoon is panel ke liye")
        return self._make_fallback_beats(), story_context

    # ─────────────────────────────────────────
    # 4. gTTS Audio
    # ─────────────────────────────────────────
    # NOTE: gTTS sirf ek hi Hindi voice deta hai — true male/female switch
    # iske paas nahi hai. "voice" setting yahan sirf tld (accent/tone) badalti
    # hai — halka sa style difference, na ki alag gender wali awaaz.
    VOICE_TLD_MAP = {
        "hi-female": "co.in",
        "hi-male": "com",
    }

    async def text_to_speech(self, text: str, output_path: str, voice: str = "hi-female"):
        tld = self.VOICE_TLD_MAP.get(voice, "co.in")
        loop = asyncio.get_event_loop()
        def _gen():
            gTTS(text=text, lang='hi', tld=tld, slow=False).save(output_path)
        await loop.run_in_executor(None, _gen)

    # ─────────────────────────────────────────
    # 5. Panel ko CANVAS_W tak scale karna (no distortion, no stretch)
    # ─────────────────────────────────────────
    def _load_scaled_panel(self, img_path: str):
        """
        Panel ko CANVAS_W (1280) width tak scale karta hai, height
        proportionally badhti hai. Vapas: (numpy array RGB, scaled_height)

        IMPORTANT: Webtoon-style panels (jaise Solo Leveling) kabhi-kabhi
        bohot lambe hote hain (ek single PDF page mein 10+ sub-panels
        stacked). 200dpi pe extract hone ke baad aisi image ka scaled
        height 10000px+ tak ja sakta hai — itna bada uncompressed RGB
        numpy array RAM mein rakhna (especially jab kayi panels parallel
        mein process/concatenate ho rahe hain) Railway ke limited-RAM
        plan par OOM-kill aur container restart ka sabse common karan
        hai. Isliye scaled height ko ek hard cap (MAX_PANEL_H) tak limit
        karte hain — agar zyada lamba hai to thoda extra downscale karte
        hain (sirf aise extreme-tall panels ke liye, normal panels par
        koi asar nahi padega).
        """
        img = cv2.imread(img_path)
        if img is None:
            # Blank fallback frame
            img = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
        h, w = img.shape[:2]
        scale = CANVAS_W / w
        new_h = max(1, int(h * scale))

        # Extreme-tall panels ke liye extra cap — agar scaled height
        # MAX_PANEL_H se zyada hai, to thoda aur downscale karo (poora
        # panel content waise hi dikhega, sirf resolution kam hogi —
        # scroll abhi bhi proportionally sahi rahega kyunki hum sirf
        # ek hi uniform extra-scale apply kar rahe hain).
        MAX_PANEL_H = 6000  # ~23MB per RGB frame at CANVAS_W=1280
        if new_h > MAX_PANEL_H:
            extra_scale = MAX_PANEL_H / new_h
            new_w = max(1, int(CANVAS_W * extra_scale))
            new_h = MAX_PANEL_H
            resized = cv2.resize(img, (new_w, new_h))
            # Width CANVAS_W se chhota ho gaya — canvas ke beech mein rakho
            resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            canvas = np.zeros((new_h, CANVAS_W, 3), dtype=np.uint8)
            x_off = (CANVAS_W - new_w) // 2
            canvas[:, x_off:x_off + new_w] = resized_rgb
            logger.warning(
                f"Panel bohot lamba tha, {h}px -> {new_h}px tak extra-scale "
                f"kiya (memory cap ke liye)"
            )
            return canvas, new_h

        resized = cv2.resize(img, (CANVAS_W, new_h))
        resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return resized_rgb, new_h

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
        panel_rgb, scaled_h = self._load_scaled_panel(img_path)
        scroll_range = max(0, scaled_h - CANVAS_H)

        # ── Step 1: Har beat ka audio banao, duration nikaalo ──
        beat_audio_paths = []
        beat_durations = []
        for i, beat in enumerate(beats):
            audio_tmp = tempfile.NamedTemporaryFile(suffix=f'_beat{i}.mp3', delete=False)
            audio_path = audio_tmp.name
            audio_tmp.close()
            self.temp_files.append(audio_path)

            try:
                await self.text_to_speech(beat["text"], audio_path, voice=voice)
                if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
                    raise ValueError("Audio empty")
                clip = AudioFileClip(audio_path)
                dur = clip.duration + self.BEAT_PAUSE  # chhota pause beats ke beech
                clip.close()
            except Exception as e:
                logger.warning(f"Beat {i} TTS error: {e} — 1.5s silence fallback")
                dur = 1.5

            beat_audio_paths.append(audio_path)
            beat_durations.append(dur)

        total_duration = sum(beat_durations)
        if total_duration <= 0:
            total_duration = 1.5

        # ── Step 2: Beat-timeline banao — har beat ka (start_time, end_time, position_px) ──
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

        # ── Step 3: Concatenate saare beat-audios ek single audio mein ──
        # IMPORTANT: timeline (Step 2) har beat ke baad self.BEAT_PAUSE
        # ka silence gap maan ke chalti hai. Agar yahan audio mein woh
        # gap nahi daala jaaye, to actual audio chhota reh jaata hai aur
        # visual scroll se aage-peeche drift ho jaata hai (jitne zyada
        # beats, utna zyada drift) — isi wajah se "jo bola jaa raha hai
        # uske hisab se screen pe scene nahi dikhta" wala bug aata tha.
        audio_clip = None
        try:
            valid_audio_clips = []
            for idx, p in enumerate(beat_audio_paths):
                if os.path.exists(p) and os.path.getsize(p) > 0:
                    valid_audio_clips.append(AudioFileClip(p))
                # Har beat (last ko chhodke) ke baad timeline jaisa hi
                # silence gap daalo, taaki audio aur scroll exactly sync rahein
                if idx < len(beat_audio_paths) - 1:
                    silence = AudioClip(
                        lambda t: 0, duration=self.BEAT_PAUSE, fps=44100
                    )
                    valid_audio_clips.append(silence)
            if valid_audio_clips:
                audio_clip = concatenate_audioclips(valid_audio_clips)
            else:
                audio_clip = None
        except Exception as e:
            logger.warning(f"Audio concat error: {e}")
            audio_clip = None

        actual_duration = audio_clip.duration if audio_clip else total_duration

        # ── Step 4: Scroll-position-at-time function ──
        def get_y_at_time(t):
            if scroll_range <= 0:
                return 0  # panel chhota hai, scroll ki zaroorat nahi
            # Dhoondo kaunsa beat is waqt active hai
            for seg in timeline:
                if seg["start"] <= t <= seg["end"] or seg is timeline[-1]:
                    # Pichle beat ki position se is beat ki position tak
                    # smoothly move karo (transition ka time = 0.4s max)
                    transition_time = min(0.4, (seg["end"] - seg["start"]) * 0.3)
                    if t <= seg["start"] + transition_time and seg != timeline[0]:
                        prev_idx = timeline.index(seg) - 1
                        prev_y = timeline[prev_idx]["y"] if prev_idx >= 0 else seg["y"]
                        progress = (t - seg["start"]) / transition_time if transition_time > 0 else 1
                        progress = max(0, min(1, progress))
                        return int(prev_y + (seg["y"] - prev_y) * progress)
                    return seg["y"]
            return timeline[-1]["y"] if timeline else 0

        # ── Step 5: Frame-generator function (moviepy VideoClip ke liye) ──
        if scroll_range <= 0:
            # Chhota panel — halka cinematic zoom (1.0 -> 1.08 scale over duration)
            base_canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
            y_off = (CANVAS_H - scaled_h) // 2
            base_canvas[y_off:y_off + scaled_h, 0:CANVAS_W] = panel_rgb

            def make_frame(t):
                zoom = 1.0 + 0.08 * (t / max(actual_duration, 0.01))
                zh, zw = int(CANVAS_H * zoom), int(CANVAS_W * zoom)
                zoomed = cv2.resize(base_canvas, (zw, zh))
                cx, cy = (zw - CANVAS_W) // 2, (zh - CANVAS_H) // 2
                return zoomed[cy:cy + CANVAS_H, cx:cx + CANVAS_W]
        else:
            def make_frame(t):
                y = get_y_at_time(t)
                y = max(0, min(scroll_range, y))
                return panel_rgb[y:y + CANVAS_H, 0:CANVAS_W]

        video_clip = VideoClip(make_frame, duration=actual_duration)
        if audio_clip:
            video_clip = video_clip.set_audio(audio_clip)

        return video_clip

    # ─────────────────────────────────────────
    # 7. Pura video banao — saare panels ke scroll-clips jodke + BGM
    # ─────────────────────────────────────────
    async def create_video_from_panels(self, image_paths: list,
                                        panel_beats: list,
                                        quality_height: int = 720,
                                        voice: str = "hi-female",
                                        bgm_enabled: bool = True,
                                        bgm_volume: int = 30) -> str:
        """
        image_paths: har panel ki image path
        panel_beats: same length list, har entry list-of-beats hai
                     (generate_panel_script se aaya)
        """
        if not image_paths or not panel_beats:
            raise ValueError("Images ya beats empty hain!")

        panel_clips = []
        for idx, (img_path, beats) in enumerate(zip(image_paths, panel_beats)):
            try:
                clip = await self.create_panel_clip(img_path, beats, voice=voice)
                panel_clips.append(clip)
            except Exception as e:
                logger.warning(f"Panel clip error ({img_path}): {e} — skip")
            # Har panel ke baad explicit garbage collection — tall webtoon
            # panels ke intermediate cv2/numpy buffers turant free karne
            # ke liye, taaki saare panels process hote waqt RAM accumulate
            # na ho (Railway ke limited-memory plan par OOM-restart se bachne
            # ke liye)
            gc.collect()
            try:
                import resource
                peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
                logger.info(f"Panel {idx + 1}/{len(image_paths)} done — peak RAM so far: {peak_mb:.0f} MB")
            except Exception:
                pass  # resource module Windows par nahi hota, but Railway Linux hai

        if not panel_clips:
            raise ValueError("Koi panel clip nahi bani!")

        final_video = None
        bgm_clip = None
        output_path = None

        try:
            final_video = concatenate_videoclips(panel_clips, method="compose")

            # Quality scale (height ke hisaab se, aspect ratio 16:9 maintain)
            if quality_height and quality_height != CANVAS_H:
                final_video = final_video.resize(height=quality_height)

            # ── BGM mix karo agar enabled hai aur file exist karti hai ──
            if bgm_enabled and os.path.exists(DEFAULT_BGM_PATH) and final_video.audio:
                try:
                    bgm_clip = AudioFileClip(DEFAULT_BGM_PATH)
                    volume_factor = max(0, min(100, bgm_volume)) / 100.0

                    if bgm_clip.duration < final_video.duration:
                        bgm_clip = bgm_clip.fx(afx.audio_loop,
                                               duration=final_video.duration)
                    else:
                        bgm_clip = bgm_clip.subclip(0, final_video.duration)

                    bgm_clip = bgm_clip.volumex(volume_factor)
                    composite_audio = CompositeAudioClip([final_video.audio, bgm_clip])
                    final_video = final_video.set_audio(composite_audio)
                except Exception as e:
                    logger.warning(f"BGM mix nahi ho payi: {e} — bina BGM ke chalu")
            elif bgm_enabled:
                logger.warning(f"BGM file nahi mili ({DEFAULT_BGM_PATH}) — skip kar raha hoon")

            out_tmp = tempfile.NamedTemporaryFile(
                suffix='_manga_video.mp4', delete=False)
            output_path = out_tmp.name
            out_tmp.close()
            self.temp_files.append(output_path)

            temp_audio = os.path.join(
                tempfile.gettempdir(),
                f'manga_audio_tmp_{os.getpid()}.m4a')
            self.temp_files.append(temp_audio)

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: final_video.write_videofile(
                    output_path,
                    fps=24,
                    codec='libx264',
                    audio_codec='aac',
                    temp_audiofile=temp_audio,
                    remove_temp=True,
                    threads=2,
                    preset='ultrafast',
                    verbose=False,
                    logger=None,
                )
            )

            logger.info(f"Video ban gayi: {output_path}")
            return output_path

        finally:
            if bgm_clip:
                try:
                    bgm_clip.close()
                except Exception:
                    pass
            if final_video:
                try:
                    final_video.close()
                except Exception:
                    pass
            for clip in panel_clips:
                try:
                    clip.close()
                except Exception:
                    pass

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


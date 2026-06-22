import os
import re
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
import google.generativeai as genai
from gtts import gTTS
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips, CompositeAudioClip, afx
from pdf2image import convert_from_path

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_KEY_HERE")
genai.configure(api_key=GEMINI_API_KEY)

# Default BGM track — isi folder mein assets/default_bgm.mp3 rakhna hai
DEFAULT_BGM_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "assets", "default_bgm.mp3"
)

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')


class MangaProcessor:

    def __init__(self):
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        self.temp_files = []

    # ─────────────────────────────────────────
    # 1. PDF → Images
    # ─────────────────────────────────────────
    def pdf_to_images(self, pdf_path: str) -> list:
        pages = convert_from_path(pdf_path, dpi=150)
        image_paths = []
        for i, page in enumerate(pages):
            tmp = tempfile.NamedTemporaryFile(suffix=f'_page{i}.jpg', delete=False)
            page.save(tmp.name, 'JPEG', quality=85)
            image_paths.append(tmp.name)
            self.temp_files.append(tmp.name)
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

        # Naturally sorted file list (1, 2, 10 — not 1, 10, 2)
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
    # 2. Text Removal (OpenCV)
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
    # 2b. 16:9 Fit + Blur Background (YouTube-style canvas)
    # ─────────────────────────────────────────
    def apply_blur_background(self, image_paths: list, blur_strength: int) -> list:
        """
        Har panel ko 16:9 canvas ke beech mein center-fit karta hai, aspect
        ratio maintain karke. Canvas ka background side-padding wala area
        same image ke stretched+blurred version se fill hota hai (jaise
        YouTube manga-video bots mein dikhta hai).

        blur_strength: 0-100. Background blur ki intensity control karta hai.
        0 par bhi 16:9 fitting hoti hai — bas background sharp (zero blur)
        rehta hai. Yeh hamesha lagta hai, optional nahi hai, kyunki video ka
        canvas hamesha 16:9 hona chahiye.
        """
        result_paths = []

        # 0-100 ko kernel size mein convert (odd number chahiye OpenCV ko)
        # blur_strength=0 par bhi minimal kernel (1) use hota hai — yaani
        # effectively no blur, lekin same code-path se canvas banta hai.
        blur_strength = max(0, min(100, blur_strength or 0))
        if blur_strength == 0:
            k = 1
        else:
            k = int(15 + (blur_strength / 100) * 70)
            if k % 2 == 0:
                k += 1

        canvas_w, canvas_h = 1280, 720

        for img_path in image_paths:
            try:
                img = cv2.imread(img_path)
                if img is None:
                    result_paths.append(img_path)
                    continue

                h, w = img.shape[:2]

                # Background: image ko canvas size tak stretch karke blur karo
                bg = cv2.resize(img, (canvas_w, canvas_h))
                if k > 1:
                    bg = cv2.GaussianBlur(bg, (k, k), 0)
                # Thoda darken taaki foreground clearly dikhe
                bg = cv2.convertScaleAbs(bg, alpha=0.6, beta=0)

                # Foreground: original aspect ratio maintain karke center mein fit
                scale = min(canvas_w / w, canvas_h / h)
                new_w, new_h = int(w * scale), int(h * scale)
                fg = cv2.resize(img, (new_w, new_h))

                x_off = (canvas_w - new_w) // 2
                y_off = (canvas_h - new_h) // 2
                bg[y_off:y_off + new_h, x_off:x_off + new_w] = fg

                tmp = tempfile.NamedTemporaryFile(suffix='_fit169.jpg', delete=False)
                cv2.imwrite(tmp.name, bg, [cv2.IMWRITE_JPEG_QUALITY, 90])
                result_paths.append(tmp.name)
                self.temp_files.append(tmp.name)
            except Exception as e:
                logger.warning(f"16:9 fit error ({img_path}): {e} — original use kar raha hoon")
                result_paths.append(img_path)

        logger.info(f"{len(result_paths)} images 16:9 canvas mein fit hui (blur={blur_strength})")
        return result_paths


    async def generate_hindi_script(self, image_paths: list) -> list:

        def make_fallback(paths):
            pool = [
                "Is page mein scene aage badhta hai.",
                "Yahan kuch important ho raha hai.",
                "Tension is panel mein saaf dikh rahi hai.",
                "Mahaul yahan dheere dheere badal raha hai.",
                "Yeh moment kahani mein khaas hai.",
                "Sannata chha gaya hai is panel mein.",
                "Characters ki expression kuch kehna chahti hai.",
                "Yahan se kahani naya rukh leti hai.",
                "Page ka mahaul kuch alag hi hai.",
                "Kuch unexpected hone wala hai is panel mein.",
            ]
            return [
                {"image_index": i, "hindi_text": pool[i % len(pool)]}
                for i in range(len(paths))
            ]

        # Build image data (max 6 to stay within free-tier token limits)
        images_content = []
        for img_path in image_paths[:6]:
            try:
                with open(img_path, 'rb') as f:
                    images_content.append({"mime_type": "image/jpeg", "data": f.read()})
            except Exception as e:
                logger.warning(f"Image read error: {e}")

        if not images_content:
            logger.warning("Koi image load nahi hui — fallback")
            return make_fallback(image_paths)

        prompt = (
            "Tu ek OCR aur dialogue-extraction expert hai.\n"
            "Yeh manga/comic pages dekh — har page mein speech bubbles, "
            "captions, aur sound-effect text hota hai.\n\n"
            "Tera kaam:\n"
            "- Har page ke saare speech bubbles/text ko UNKE ORIGINAL ORDER mein "
            "(top-to-bottom, left-to-right jaisa padhne ka natural flow hota hai) "
            "padh aur seedha extract kar — naya kuch mat likh, sirf jo likha hai "
            "wahi nikaal\n"
            "- Agar text already Hindi/Hinglish mein hai to wahi rakh\n"
            "- Agar text pure English mein hai to natural Hindi/Hinglish mein "
            "translate kar (jaise koi voice-over artist bolega) — meaning exactly "
            "wahi rakhna, kuch add/remove mat karna\n"
            "- Agar ek page pe multiple bubbles hain to sabko jodke ek hi narration "
            "banao, comma ya naturally bolne wale pause se separate karke\n"
            "- Agar page pe koi text/dialogue nahi hai (sirf art/scene hai) to ek "
            "chhota neutral scene description Hindi mein de (jaise 'Sannata chha "
            "gaya' ya page ke visual ke hisaab se)\n"
            "- Sound effects (jaise 'Heh!', 'Wo kya h!!?') ko bhi unke tone ke "
            "hisaab se expressively bol — mat skip karo\n\n"
            "Sirf JSON array return karo, kuch aur nahi:\n"
            '[{"page":1,"narration":"..."},{"page":2,"narration":"..."}]'
        )

        for attempt in range(1, 4):
            try:
                logger.info(f"Gemini API call attempt {attempt}/3")
                loop = asyncio.get_event_loop()
                content_parts = [prompt] + [{"inline_data": img} for img in images_content]
                response = await loop.run_in_executor(
                    None, lambda: self.model.generate_content(content_parts)
                )
                text = response.text.strip()

                # Strip markdown fences
                if '```json' in text:
                    text = text.split('```json')[1].split('```')[0].strip()
                elif '```' in text:
                    text = text.split('```')[1].split('```')[0].strip()

                pages_data = json.loads(text)
                if not isinstance(pages_data, list) or len(pages_data) == 0:
                    raise ValueError("Empty JSON from Gemini")

                script_parts = []
                seen = set()
                for item in pages_data:
                    narration = (item.get("narration") or "").strip()
                    if not narration:
                        narration = "Is page mein scene aage badhta hai."
                    idx = max(0, item.get("page", 1) - 1)
                    script_parts.append({"image_index": idx, "hindi_text": narration})
                    seen.add(idx)

                # Fill missing pages with fallback
                for i in range(len(image_paths)):
                    if i not in seen:
                        script_parts.append({
                            "image_index": i,
                            "hindi_text": "Is page mein scene aage badhta hai."
                        })

                script_parts.sort(key=lambda x: x["image_index"])
                logger.info(f"Gemini se {len(script_parts)} narrations mili")
                return script_parts

            except Exception as e:
                err_str = str(e)
                logger.error(f"Gemini error (attempt {attempt}): {err_str[:200]}")

                is_rate_limit = ('429' in err_str or
                                 'quota' in err_str.lower() or
                                 'rate' in err_str.lower())

                if is_rate_limit and attempt < 3:
                    # Extract retry_delay from error, default 65s
                    wait_time = 65
                    m = re.search(r'seconds:\s*(\d+)', err_str)
                    if m:
                        wait_time = int(m.group(1)) + 10
                    logger.info(f"Rate limit — {wait_time}s wait karke retry...")
                    await asyncio.sleep(wait_time)
                else:
                    break

        logger.info("Fallback narrations use kar raha hoon")
        return make_fallback(image_paths)

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
    # 5. Video banao  (Railway-safe MoviePy)
    # ─────────────────────────────────────────
    async def create_video_with_voice(self, image_paths: list, script: list,
                                       quality_height: int = 720,
                                       voice: str = "hi-female",
                                       bgm_enabled: bool = True,
                                       bgm_volume: int = 30) -> str:

        if not script:
            raise ValueError("Script empty hai!")

        clips = []

        for item in script:
            idx = item.get("image_index", 0)
            if idx >= len(image_paths):
                continue

            img_path = image_paths[idx]
            hindi_text = (item.get("hindi_text") or "").strip() or \
                         "Is page mein scene aage badhta hai."

            audio_path = None
            audio_clip = None
            img_clip = None

            try:
                # ── 1. Audio file banao ──
                audio_tmp = tempfile.NamedTemporaryFile(
                    suffix='.mp3', delete=False)
                audio_path = audio_tmp.name
                audio_tmp.close()
                self.temp_files.append(audio_path)
                await self.text_to_speech(hindi_text, audio_path, voice=voice)

                if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
                    raise ValueError("Audio file empty/missing")

                # ── 2. AudioFileClip load karo ──
                audio_clip = AudioFileClip(audio_path)
                duration = audio_clip.duration + 0.5

                # ── 3. ImageClip banao (audio set karo BEFORE close) ──
                img_clip = ImageClip(img_path, duration=duration)
                img_clip = img_clip.set_audio(audio_clip)
                # NOTE: audio_clip intentionally NOT closed here —
                # moviepy needs the file handle open until write_videofile.
                # Will be closed in finally block after video is written.

                img_clip = img_clip.resize(height=quality_height)
                clips.append((img_clip, audio_clip))

            except Exception as e:
                logger.warning(f"Clip {idx} skip — {e}")
                if audio_clip:
                    try:
                        audio_clip.close()
                    except Exception:
                        pass
                if img_clip:
                    try:
                        img_clip.close()
                    except Exception:
                        pass

        if not clips:
            raise ValueError("Koi clip nahi bani!")

        video_clips = [c[0] for c in clips]
        audio_clips = [c[1] for c in clips]

        final_video = None
        bgm_clip = None
        composite_audio = None
        output_path = None

        try:
            final_video = concatenate_videoclips(video_clips, method="compose")

            # ── BGM mix karo agar enabled hai aur file exist karti hai ──
            if bgm_enabled and os.path.exists(DEFAULT_BGM_PATH):
                try:
                    bgm_clip = AudioFileClip(DEFAULT_BGM_PATH)
                    volume_factor = max(0, min(100, bgm_volume)) / 100.0

                    # BGM ko video duration tak loop karo
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

            # Run write_videofile in executor so Railway event loop stays alive
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
            # Close everything AFTER write is done
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
            for clip in video_clips:
                try:
                    clip.close()
                except Exception:
                    pass
            for ac in audio_clips:
                try:
                    ac.close()
                except Exception:
                    pass

    # ─────────────────────────────────────────
    # 6. Cleanup
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

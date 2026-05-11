import os
import re
import cv2
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
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips
from pdf2image import convert_from_path

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_KEY_HERE")
genai.configure(api_key=GEMINI_API_KEY)


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
    # 3. Gemini Hindi Script  (retry + fallback)
    # ─────────────────────────────────────────
    async def generate_hindi_script(self, image_paths: list) -> list:

        def make_fallback(paths):
            pool = [
                "Dekho yaar, yahan kahani ek naye mod par aa jaati hai!",
                "Aur phir ek dum se kuch aisa hua jo kisi ne soch bhi nahi tha.",
                "Characters ke beech tension badh rahi hai — mahaul garam ho raha hai!",
                "Yeh pal bahut important hai poori kahani mein.",
                "Suspense aur drama apne peak par hai is page mein!",
                "Ek dum se situation puri badal gayi — kya hoga aage?",
                "Hero ki aankhon mein determination saaf dikh rahi hai.",
                "Yahan se kahani ek nayi disha mein chali jaati hai.",
                "Drama aur action ka perfect mix hai is page mein!",
                "Aakhir mein sach saamne aa hi gaya — sabko hairani hui!",
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
            "Tu ek zabardast Hindi manga storyteller hai.\n"
            "Yeh manga pages dekh aur har page ke liye compelling Hindi narration likh.\n\n"
            "Rules:\n"
            "- Natural Hindi bolchaal (jaise koi dost bata raha ho)\n"
            "- Emotions dikhao — excitement, tension, drama!\n"
            "- Har page 2-3 sentences\n"
            "- 'Dekho', 'Yaar', 'Aur phir', 'Ek dum se' jaisi fillers use karo\n\n"
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
                        narration = "Is page mein kahani aage badhti hai."
                    idx = max(0, item.get("page", 1) - 1)
                    script_parts.append({"image_index": idx, "hindi_text": narration})
                    seen.add(idx)

                # Fill missing pages with fallback
                for i in range(len(image_paths)):
                    if i not in seen:
                        script_parts.append({
                            "image_index": i,
                            "hindi_text": "Is page mein kahani aage badhti hai."
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
    async def text_to_speech(self, text: str, output_path: str):
        loop = asyncio.get_event_loop()
        def _gen():
            gTTS(text=text, lang='hi', slow=False).save(output_path)
        await loop.run_in_executor(None, _gen)

    # ─────────────────────────────────────────
    # 5. Video banao  (Railway-safe MoviePy)
    # ─────────────────────────────────────────
    async def create_video_with_voice(self, image_paths: list, script: list) -> str:

        if not script:
            raise ValueError("Script empty hai!")

        clips = []

        for item in script:
            idx = item.get("image_index", 0)
            if idx >= len(image_paths):
                continue

            img_path = image_paths[idx]
            hindi_text = (item.get("hindi_text") or "").strip() or \
                         "Is page mein kahani aage badhti hai."

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
                await self.text_to_speech(hindi_text, audio_path)

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

                img_clip = img_clip.resize(height=720)  # 720p — faster on Railway
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
        output_path = None

        try:
            final_video = concatenate_videoclips(video_clips, method="compose")

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
                if f and os.path.exists(f):
                    os.remove(f)
            except Exception:
                pass
        self.temp_files = []

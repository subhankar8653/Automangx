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
from moviepy.editor import (
    ImageClip, AudioFileClip, concatenate_videoclips, concatenate_audioclips,
    CompositeAudioClip, CompositeVideoClip, VideoClip, afx
)
from pdf2image import convert_from_path

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_KEY_HERE")
genai.configure(api_key=GEMINI_API_KEY)

# Default BGM track — isi folder mein assets/default_bgm.mp3 rakhna hai
DEFAULT_BGM_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "assets", "default_bgm.mp3"
)

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')

CANVAS_W, CANVAS_H = 1280, 720  # 16:9 video canvas


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
        return [
            {"text": "Is panel mein scene shuru hota hai.", "position": 15},
            {"text": "Kahani yahan aage badh rahi hai.", "position": 55},
            {"text": "Yeh panel yahan khatam hota hai.", "position": 90},
        ]

    async def generate_panel_script(self, image_path: str) -> list:
        try:
            with open(image_path, 'rb') as f:
                img_bytes = f.read()
        except Exception as e:
            logger.warning(f"Image read error ({image_path}): {e}")
            return self._make_fallback_beats()

        prompt = (
            "Tu ek professional Hindi manga/comic EXPLAINER hai — jaise koi "
            "YouTuber manga explain karta hai, story ko engaging banake.\n\n"
            "Is panel (ek manga/comic page) ko dhyaan se dekh:\n"
            "- Speech bubbles, captions, sound-effects ka text padh — "
            "IMPORTANT: yeh text KISI BHI language mein ho sakta hai "
            "(English, Italian, Japanese, Korean, Spanish, ya koi aur "
            "bhasha) — jo bhi language ho, uska meaning samajh aur Hindi/ "
            "Hinglish mein convert kar. Kabhi bhi 'samajh nahi aaya' ya "
            "generic placeholder mat de — agar text chhota/unclear lage "
            "to bhi best-effort translate kar uska matlab nikaal ke\n"
            "- Characters ki facial expression, body language, background, "
            "scene ka mood bhi observe kar\n"
            "- Yeh sab milake ek natural Hindi/Hinglish explainer narration "
            "bana — jo dialogue ko bhi cover kare AUR jo dikh raha hai uska "
            "context/emotion bhi bataye (jaise 'Wo darr ke peeche dekhti hai "
            "aur kehti hai - Strange, meri mummy ne iske bare mein kabhi nahi "
            "bola tha')\n"
            "- Dialogue ka exact meaning mat badalna, bas use natural "
            "explainer flow mein pirona hai\n"
            "- Agar panel mein bilkul koi text/bubble nahi hai (sirf pure "
            "art/scene hai), tabhi sirf scene-description de — warna hamesha "
            "panel ke andar jo likha hai usi se shuru kar\n\n"
            "Panel ko TOP se BOTTOM tak alag-alag 'beats' mein todo — har "
            "beat ek chhota narration-chunk hai jo panel ke ek specific "
            "vertical hisse (top/middle/bottom ya in between) se related hai. "
            "Agar panel mein 2 bubbles hain (ek upar, ek niche), to kam se "
            "kam 2 beats banao. Agar sirf scene/art hai (bubble nahi), to "
            "1-2 beats mein scene describe kar.\n\n"
            "Har beat ke liye 'position' do — 0 se 100 ke beech ek number, "
            "jo batata hai panel ke kis vertical %% (0=sabse upar, "
            "100=sabse niche) par yeh beat ka content hai.\n\n"
            "Sirf JSON array return karo, kuch aur nahi:\n"
            '[{"position": 10, "text": "..."}, {"position": 70, "text": "..."}]'
        )

        content_parts = [
            prompt,
            {"inline_data": {"mime_type": "image/jpeg", "data": img_bytes}}
        ]

        for attempt in range(1, 4):
            try:
                logger.info(f"Gemini panel-script call attempt {attempt}/3")
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None, lambda: self.model.generate_content(
                        content_parts,
                        safety_settings={
                            'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
                            'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
                            'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
                            'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE',
                        }
                    )
                )

                # Safety-block ya empty response check
                if not response.candidates:
                    raise ValueError("Gemini ne koi candidate nahi diya (shayad safety block)")
                candidate = response.candidates[0]
                finish_reason = getattr(candidate, 'finish_reason', None)
                finish_str = str(finish_reason) if finish_reason is not None else ""
                # finish_reason==1 ya 'STOP' matlab normal completion. Kuch
                # bhi aur (SAFETY=3, RECITATION=4, MAX_TOKENS=2, etc.) means
                # response truncated/blocked tha.
                if finish_str and not any(s in finish_str.upper() for s in ('STOP', '1')):
                    raise ValueError(f"Gemini finish_reason: {finish_str} (safety/length block ho sakta hai)")

                text = response.text.strip()

                if '```json' in text:
                    text = text.split('```json')[1].split('```')[0].strip()
                elif '```' in text:
                    text = text.split('```')[1].split('```')[0].strip()

                beats_data = json.loads(text)
                if not isinstance(beats_data, list) or len(beats_data) == 0:
                    raise ValueError("Empty/invalid JSON from Gemini")

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

                # Position ke hisaab se sort karo (top-to-bottom reading order)
                beats.sort(key=lambda b: b["position"])
                logger.info(f"Panel se {len(beats)} beats mile")
                return beats

            except Exception as e:
                err_str = str(e)
                logger.error(f"Gemini error (attempt {attempt}/3) for {os.path.basename(image_path)}: {err_str[:300]}")

                is_rate_limit = ('429' in err_str or
                                 'quota' in err_str.lower() or
                                 'rate' in err_str.lower())

                if is_rate_limit and attempt < 3:
                    wait_time = 65
                    m = re.search(r'seconds:\s*(\d+)', err_str)
                    if m:
                        wait_time = int(m.group(1)) + 10
                    logger.info(f"Rate limit — {wait_time}s wait karke retry...")
                    await asyncio.sleep(wait_time)
                else:
                    break

        logger.info("Fallback beats use kar raha hoon is panel ke liye")
        return self._make_fallback_beats()

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
        """
        img = cv2.imread(img_path)
        if img is None:
            # Blank fallback frame
            img = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
        h, w = img.shape[:2]
        scale = CANVAS_W / w
        new_h = max(1, int(h * scale))
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
                dur = clip.duration + 0.35  # chhota pause beats ke beech
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
        audio_clip = None
        try:
            valid_audio_clips = []
            for p in beat_audio_paths:
                if os.path.exists(p) and os.path.getsize(p) > 0:
                    valid_audio_clips.append(AudioFileClip(p))
            if valid_audio_clips:
                # Beats ke beech chhota silence gap (0.35s) — concatenate
                # se simple rakhte hain, gap ko hum duration mein add kar
                # chuke hain audio ke baad). Seedha concatenate karte hain.
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
        for img_path, beats in zip(image_paths, panel_beats):
            try:
                clip = await self.create_panel_clip(img_path, beats, voice=voice)
                panel_clips.append(clip)
            except Exception as e:
                logger.warning(f"Panel clip error ({img_path}): {e} — skip")

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


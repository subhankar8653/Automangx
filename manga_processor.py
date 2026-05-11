import os
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
from moviepy.editor import (ImageClip, AudioFileClip, concatenate_videoclips,
                              CompositeVideoClip)
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
    def pdf_to_images(self, pdf_path: str) -> list[str]:
        """PDF ke har page ko image mein convert karo"""
        pages = convert_from_path(pdf_path, dpi=150)
        image_paths = []
        
        for i, page in enumerate(pages):
            tmp = tempfile.NamedTemporaryFile(suffix=f'_page{i}.jpg', delete=False)
            page.save(tmp.name, 'JPEG', quality=95)
            image_paths.append(tmp.name)
            self.temp_files.append(tmp.name)
        
        logger.info(f"PDF se {len(image_paths)} pages nikale")
        return image_paths
    
    # ─────────────────────────────────────────
    # 2. Text Removal (OpenCV — Fast Method)
    # ─────────────────────────────────────────
    def remove_text_from_images(self, image_paths: list[str]) -> list[str]:
        """Manga panels se text/likhawat hatao using OpenCV inpainting"""
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
                    area = cv2.contourArea(cnt)
                    if 500 < area < 50000:
                        cv2.drawContours(bubble_mask, [cnt], -1, 255, -1)
                
                final_mask = cv2.bitwise_and(dilated, bubble_mask)
                final_mask = cv2.dilate(final_mask, kernel, iterations=2)
                result = cv2.inpaint(img, final_mask, inpaintRadius=7, 
                                      flags=cv2.INPAINT_TELEA)
                
                tmp = tempfile.NamedTemporaryFile(suffix='_clean.jpg', delete=False)
                cv2.imwrite(tmp.name, result, [cv2.IMWRITE_JPEG_QUALITY, 95])
                cleaned_paths.append(tmp.name)
                self.temp_files.append(tmp.name)
            except Exception as e:
                logger.warning(f"Image clean karne mein problem ({img_path}): {e} — original use kar raha hoon")
                cleaned_paths.append(img_path)
        
        logger.info(f"{len(cleaned_paths)} images clean ho gayi")
        return cleaned_paths
    
    # ─────────────────────────────────────────
    # 3. Gemini se Hindi Script Generate karo
    #    (429 rate-limit par auto-retry + fallback)
    # ─────────────────────────────────────────
    async def generate_hindi_script(self, image_paths: list[str]) -> list[dict]:
        """
        Har image ke liye Hindi narration generate karo.
        Rate-limit (429) par retry karega, fail hone par fallback dega.
        Returns: [{"image_index": 0, "hindi_text": "..."}]
        """
        # ── Build fallback immediately so we ALWAYS have something ──
        def make_fallback(paths):
            fallback_texts = [
                "Dekho yaar, yahan kahani ek naye mod par aa jaati hai!",
                "Aur phir ek dum se kuch aisa hua jo kisi ne soch bhi nahi tha.",
                "Characters ke beech tension badh rahi hai, mahaul garam ho raha hai!",
                "Yeh pal bahut important hai poori kahani mein.",
                "Suspense aur drama apne peak par hai is page mein!",
                "Ek dum se situation puri badal gayi — kya hoga aage?",
                "Hero ki aankhon mein determination dikh rahi hai.",
                "Yahan se kahani ek nayi disha mein chali jaati hai.",
                "Drama aur action ka perfect mix hai is page mein!",
                "Aakhir mein sach saamne aa hi gaya!",
            ]
            result = []
            for i in range(len(paths)):
                result.append({
                    "image_index": i,
                    "hindi_text": fallback_texts[i % len(fallback_texts)]
                })
            return result

        script_parts = []

        # ── Prepare images (max 8 to reduce token usage) ──
        images_content = []
        for img_path in image_paths[:8]:
            try:
                with open(img_path, 'rb') as f:
                    img_data = f.read()
                images_content.append({
                    "mime_type": "image/jpeg",
                    "data": img_data
                })
            except Exception as e:
                logger.warning(f"Image read nahi hui: {e}")

        if not images_content:
            logger.warning("Koi image load nahi hui — fallback use kar raha hoon")
            return make_fallback(image_paths)

        prompt = """Tu ek zabardast Hindi manga storyteller hai. 
        
Yeh manga pages dekh aur har page ke liye ek compelling Hindi narration likh.

Rules:
- Bilkul natural Hindi bolchaal mein likh (jaise koi dost bata raha ho)
- Emotions dikhao — excitement, tension, drama!
- Har page ki narration 2-4 sentences ki honi chahiye
- "Dekho", "Yaar", "Aur phir", "Ek dum se" — aisi natural fillers use karo

Format EXACTLY aisa do (JSON array, kuch aur mat likho):
[
  {"page": 1, "narration": "Yahan se kahani shuru hoti hai..."},
  {"page": 2, "narration": "Aur phir ek dum se..."}
]"""

        # ── Retry loop: 3 attempts with exponential backoff ──
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"Gemini API call attempt {attempt}/{max_retries}")
                
                content_parts = [prompt] + [
                    {"inline_data": img} for img in images_content
                ]
                
                # Run blocking Gemini call in executor to not block async loop
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: self.model.generate_content(content_parts)
                )
                
                text = response.text.strip()
                
                # Strip markdown code fences if present
                if '```json' in text:
                    text = text.split('```json')[1].split('```')[0].strip()
                elif '```' in text:
                    text = text.split('```')[1].split('```')[0].strip()
                
                pages_data = json.loads(text)
                
                if not isinstance(pages_data, list) or len(pages_data) == 0:
                    raise ValueError("Gemini ne empty ya invalid JSON diya")
                
                for item in pages_data:
                    narration = (item.get("narration") or "").strip()
                    if not narration:
                        narration = "Is page mein kahani aage badhti hai."
                    script_parts.append({
                        "image_index": item.get("page", 1) - 1,
                        "hindi_text": narration,
                    })
                
                logger.info(f"Gemini se {len(script_parts)} page narrations mili")
                
                # ── If Gemini gave fewer pages than images, fill remaining ──
                gemini_indices = {s["image_index"] for s in script_parts}
                for i in range(len(image_paths)):
                    if i not in gemini_indices:
                        script_parts.append({
                            "image_index": i,
                            "hindi_text": "Is page mein kahani aage badhti hai."
                        })
                
                # Sort by image_index
                script_parts.sort(key=lambda x: x["image_index"])
                return script_parts

            except Exception as e:
                err_str = str(e)
                logger.error(f"Gemini error (attempt {attempt}): {err_str[:200]}")
                
                # Check if it's a rate-limit (429) error with retry hint
                if '429' in err_str or 'quota' in err_str.lower() or 'rate' in err_str.lower():
                    # Try to extract retry_delay from error message
                    wait_time = 60  # default wait
                    try:
                        import re
                        match = re.search(r'seconds:\s*(\d+)', err_str)
                        if match:
                            wait_time = int(match.group(1)) + 5  # +5s buffer
                    except Exception:
                        pass
                    
                    if attempt < max_retries:
                        logger.info(f"Rate limit hit — {wait_time} seconds wait karke retry karunga...")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        logger.warning("Max retries ho gayi — fallback use kar raha hoon")
                        break
                else:
                    # Non-rate-limit error — no point retrying
                    logger.warning("Non-rate-limit error — fallback use kar raha hoon")
                    break
        
        # ── Fallback: return generic narrations for all pages ──
        logger.info("Fallback narrations use kar raha hoon")
        return make_fallback(image_paths)
    
    # ─────────────────────────────────────────
    # 4. gTTS se Audio Generate karo (Google)
    # ─────────────────────────────────────────
    async def text_to_speech(self, text: str, output_path: str):
        """Hindi text ko Google TTS se voice mein convert karo"""
        loop = asyncio.get_event_loop()
        
        def generate():
            tts = gTTS(text=text, lang='hi', slow=False)
            tts.save(output_path)
        
        await loop.run_in_executor(None, generate)
    
    # ─────────────────────────────────────────
    # 5. Video + Voice Sync karo
    # ─────────────────────────────────────────
    async def create_video_with_voice(self, image_paths: list[str], 
                                       script: list[dict]) -> str:
        """Cleaned images + Hindi voice = Synced video"""
        
        if not script:
            raise ValueError("Script empty hai — koi narration nahi mili!")
        
        clips = []
        
        for item in script:
            idx = item.get("image_index", 0)
            if idx >= len(image_paths):
                continue
            
            img_path = image_paths[idx]
            hindi_text = (item.get("hindi_text") or "").strip()
            if not hindi_text:
                hindi_text = "Is page mein kahani aage badhti hai."
            
            try:
                # Audio generate karo
                audio_tmp = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
                self.temp_files.append(audio_tmp.name)
                await self.text_to_speech(hindi_text, audio_tmp.name)
                
                # Audio duration pata karo
                audio_clip = AudioFileClip(audio_tmp.name)
                audio_duration = audio_clip.duration
                total_duration = audio_duration + 0.5
                
                # Image clip banao
                img_clip = ImageClip(img_path, duration=total_duration)
                img_clip = img_clip.set_audio(audio_clip.set_duration(audio_duration))
                audio_clip.close()
                
                # Resize to standard height
                img_clip = img_clip.resize(height=1080)
                clips.append(img_clip)
                
            except Exception as e:
                logger.warning(f"Clip {idx} mein problem: {e} — skip kar raha hoon")
                continue
        
        if not clips:
            raise ValueError("Koi valid clip nahi bani — sab images/audio mein problem thi!")
        
        # Saari clips join karo
        final_video = concatenate_videoclips(clips, method="compose")
        
        # Output path
        output_tmp = tempfile.NamedTemporaryFile(suffix='_manga_video.mp4', delete=False)
        self.temp_files.append(output_tmp.name)
        
        # Proper temp audio file (Broken Pipe fix)
        temp_audio_path = os.path.join(
            tempfile.gettempdir(),
            f'manga_temp_audio_{os.getpid()}.m4a'
        )
        self.temp_files.append(temp_audio_path)
        
        final_video.write_videofile(
            output_tmp.name,
            fps=24,
            codec='libx264',
            audio_codec='aac',
            temp_audiofile=temp_audio_path,
            remove_temp=True,
            threads=2,
            preset='ultrafast',
            verbose=False,
            logger=None
        )
        
        # Resources properly close karo
        final_video.close()
        for clip in clips:
            try:
                clip.close()
            except Exception:
                pass
        
        logger.info(f"Video ban gayi: {output_tmp.name}")
        return output_tmp.name
    
    # ─────────────────────────────────────────
    # 6. Cleanup
    # ─────────────────────────────────────────
    def cleanup(self, *file_lists):
        """Saari temp files delete karo"""
        all_files = list(self.temp_files)
        for file_list in file_lists:
            if file_list is None:
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

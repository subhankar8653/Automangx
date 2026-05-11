import os
import cv2
import numpy as np
import asyncio
import tempfile
import logging
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
        self.model = genai.GenerativeModel('gemini-1.5-flash')
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
            img = cv2.imread(img_path)
            if img is None:
                cleaned_paths.append(img_path)
                continue
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Speech bubble text detect karna
            # White regions (speech bubbles) find karo
            _, white_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
            
            # Text ke liye: high contrast areas in white regions
            # MSER (Maximally Stable Extremal Regions) for text detection
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            
            # Edges detect karo text ke andar
            edges = cv2.Canny(gray, 50, 150)
            
            # White bubble areas mein edges = text
            text_in_bubbles = cv2.bitwise_and(edges, white_mask)
            
            # Dilate karke mask banao
            dilated = cv2.dilate(text_in_bubbles, kernel, iterations=4)
            
            # Speech bubble detect karo (large white blobs)
            contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, 
                                            cv2.CHAIN_APPROX_SIMPLE)
            
            bubble_mask = np.zeros_like(gray)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 500 < area < 50000:  # Reasonable bubble size
                    cv2.drawContours(bubble_mask, [cnt], -1, 255, -1)
            
            # Final mask: text regions inside bubbles
            final_mask = cv2.bitwise_and(dilated, bubble_mask)
            
            # Thoda aur dilate karo clean result ke liye
            final_mask = cv2.dilate(final_mask, kernel, iterations=2)
            
            # Inpainting — text ki jagah fill karo
            result = cv2.inpaint(img, final_mask, inpaintRadius=7, 
                                  flags=cv2.INPAINT_TELEA)
            
            # Save cleaned image
            tmp = tempfile.NamedTemporaryFile(suffix='_clean.jpg', delete=False)
            cv2.imwrite(tmp.name, result, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cleaned_paths.append(tmp.name)
            self.temp_files.append(tmp.name)
        
        logger.info(f"{len(cleaned_paths)} images clean ho gayi")
        return cleaned_paths
    
    # ─────────────────────────────────────────
    # 3. Gemini se Hindi Script Generate karo
    # ─────────────────────────────────────────
    async def generate_hindi_script(self, image_paths: list[str]) -> list[dict]:
        """
        Har image ke liye Hindi explanation + timing generate karo
        Returns: [{"image_index": 0, "hindi_text": "...", "duration_hint": 5}]
        """
        script_parts = []
        
        # Saari images ek saath Gemini ko bhejo context ke liye
        images_content = []
        for img_path in image_paths[:10]:  # Max 10 pages
            with open(img_path, 'rb') as f:
                img_data = f.read()
            images_content.append({
                "mime_type": "image/jpeg",
                "data": img_data
            })
        
        prompt = """Tu ek zabardast Hindi manga storyteller hai. 
        
Yeh manga pages dekh aur har page ke liye ek compelling Hindi narration likh.

Rules:
- Bilkul natural Hindi bolchaal mein likh (jaise koi dost bata raha ho)
- "AI" jaisi koi feeling nahi aani chahiye — insaan wali storytelling honi chahiye
- Emotions dikhao — excitement, tension, drama!
- Har page ki narration 2-4 sentences ki honi chahiye
- Character ke naam aur actions clearly describe karo
- "Dekho", "Yaar", "Aur phir", "Ek dum se" — aisi natural fillers use karo

Format EXACTLY aisa do (JSON):
[
  {"page": 1, "narration": "Yahan se kahani shuru hoti hai..."},
  {"page": 2, "narration": "Aur phir ek dum se..."}
]

Sirf JSON do, kuch aur mat likho."""

        try:
            content_parts = [prompt] + [
                {"inline_data": img} for img in images_content
            ]
            
            response = self.model.generate_content(content_parts)
            text = response.text.strip()
            
            # JSON parse karo
            import json
            if '```json' in text:
                text = text.split('```json')[1].split('```')[0].strip()
            elif '```' in text:
                text = text.split('```')[1].split('```')[0].strip()
            
            pages_data = json.loads(text)
            
            for item in pages_data:
                script_parts.append({
                    "image_index": item["page"] - 1,
                    "hindi_text": item["narration"],
                })
            
        except Exception as e:
            logger.error(f"Gemini error: {e}")
            # Fallback
            for i, _ in enumerate(image_paths):
                script_parts.append({
                    "image_index": i,
                    "hindi_text": f"Is page mein kahani aage badhti hai.",
                })
        
        return script_parts
    
    # ─────────────────────────────────────────
    # 4. gTTS se Audio Generate karo (Google)
    # ─────────────────────────────────────────
    async def text_to_speech(self, text: str, output_path: str):
        """Hindi text ko Google TTS se voice mein convert karo"""
        loop = asyncio.get_event_loop()
        
        def generate():
            tts = gTTS(text=text, lang='hi', slow=False)
            tts.save(output_path)
        
        # Blocking call ko thread mein run karo
        await loop.run_in_executor(None, generate)
    
    # ─────────────────────────────────────────
    # 5. Video + Voice Sync karo
    # ─────────────────────────────────────────
    async def create_video_with_voice(self, image_paths: list[str], 
                                       script: list[dict]) -> str:
        """Cleaned images + Hindi voice = Synced video"""
        
        clips = []
        
        for item in script:
            idx = item["image_index"]
            if idx >= len(image_paths):
                continue
            
            img_path = image_paths[idx]
            hindi_text = item["hindi_text"]
            
            # Audio generate karo
            audio_tmp = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
            self.temp_files.append(audio_tmp.name)
            await self.text_to_speech(hindi_text, audio_tmp.name)
            
            # Audio duration pata karo
            audio_clip = AudioFileClip(audio_tmp.name)
            audio_duration = audio_clip.duration
            
            # Extra 0.5s pause har page ke baad
            total_duration = audio_duration + 0.5
            
            # Image clip banao same duration ka
            img_clip = ImageClip(img_path, duration=total_duration)
            # Audio clip ko copy karke set karo, phir original close karo
            img_clip = img_clip.set_audio(audio_clip.set_duration(audio_duration))
            audio_clip.close()
            
            # Resize to standard YouTube size
            img_clip = img_clip.resize(height=1080)
            
            clips.append(img_clip)
        
        if not clips:
            raise ValueError("Koi valid clip nahi bani!")
        
        # Saari clips join karo
        final_video = concatenate_videoclips(clips, method="compose")
        
        # Output path
        output_tmp = tempfile.NamedTemporaryFile(suffix='_manga_video.mp4', delete=False)
        self.temp_files.append(output_tmp.name)
        
        # Proper temp audio file path - same dir mein (Broken Pipe fix)
        import os as _os
        temp_audio_path = _os.path.join(
            tempfile.gettempdir(), 
            next(tempfile._get_candidate_names()) + '_temp_audio.m4a'
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

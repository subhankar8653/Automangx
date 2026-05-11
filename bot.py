import os
import asyncio
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from pathlib import Path
import tempfile

from manga_processor import MangaProcessor

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "YOUR_BOT_TOKEN_HERE")

processor = MangaProcessor()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🎌 *Manga Hindi Explainer Bot mein swagat hai!*\n\n"
        "Mujhe bhejo:\n"
        "📸 Manga images (JPG/PNG) — ek ek karke ya saath mein\n"
        "📄 PDF file — poora chapter\n\n"
        "Main:\n"
        "✅ Text hataunga panels se\n"
        "✅ Hindi mein story explain karunga\n"
        "✅ Voice ke saath video bana ke dunga!\n\n"
        "Chalo shuru karte hain! 🔥",
        parse_mode='Markdown'
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📖 *Kaise use karein:*\n\n"
        "1. Manga ke images bhejo (JPG/PNG)\n"
        "2. Ya PDF bhejo\n"
        "3. /process command do jab saare images bhej do\n"
        "4. Wait karo — video ban raha hoga!\n\n"
        "⚠️ Note: Video banane mein 2-5 minute lag sakte hain.",
        parse_mode='Markdown'
    )

async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    
    if 'images' not in context.user_data:
        context.user_data['images'] = []
    
    # Get highest quality photo
    photo = update.message.photo[-1]
    file = await context.bot.get_file(photo.file_id)
    
    # Save to temp
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        await file.download_to_drive(tmp.name)
        context.user_data['images'].append(tmp.name)
    
    count = len(context.user_data['images'])
    await update.message.reply_text(
        f"✅ Image {count} mil gayi!\n"
        f"Aur images bhejo ya /process likho video banane ke liye 🎬"
    )

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    doc = update.message.document
    
    if doc.mime_type == 'application/pdf':
        await update.message.reply_text("📄 PDF mil gayi! Process ho rahi hai... thoda wait karo ⏳")
        
        file = await context.bot.get_file(doc.file_id)
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            await file.download_to_drive(tmp.name)
            pdf_path = tmp.name
        
        await process_and_send(update, context, pdf_path=pdf_path)
    
    elif doc.mime_type in ['image/jpeg', 'image/png']:
        file = await context.bot.get_file(doc.file_id)
        suffix = '.jpg' if doc.mime_type == 'image/jpeg' else '.png'
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            await file.download_to_drive(tmp.name)
            if 'images' not in context.user_data:
                context.user_data['images'] = []
            context.user_data['images'].append(tmp.name)
        
        count = len(context.user_data['images'])
        await update.message.reply_text(
            f"✅ Image {count} mil gayi!\n"
            f"Aur images bhejo ya /process likho 🎬"
        )
    else:
        await update.message.reply_text("⚠️ Sirf JPG, PNG ya PDF bhejo yaar!")

async def process_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    images = context.user_data.get('images', [])
    
    if not images:
        await update.message.reply_text("⚠️ Pehle kuch images bhejo bhai!")
        return
    
    await process_and_send(update, context, image_paths=images)
    context.user_data['images'] = []

async def process_and_send(update: Update, context: ContextTypes.DEFAULT_TYPE, 
                            image_paths=None, pdf_path=None):
    status_msg = await update.message.reply_text(
        "🎬 *Video ban rahi hai...*\n\n"
        "⏳ Step 1/4: Images process ho rahi hain...",
        parse_mode='Markdown'
    )
    
    try:
        # Step 1: Extract images from PDF if needed
        if pdf_path:
            await status_msg.edit_text(
                "🎬 *Video ban rahi hai...*\n\n"
                "⏳ Step 1/4: PDF se images nikal raha hoon...",
                parse_mode='Markdown'
            )
            image_paths = processor.pdf_to_images(pdf_path)
        
        # Step 2: Remove text from panels
        await status_msg.edit_text(
            "🎬 *Video ban rahi hai...*\n\n"
            "✅ Step 1/4: Images ready!\n"
            "⏳ Step 2/4: Manga text hat raha hai...",
            parse_mode='Markdown'
        )
        cleaned_images = processor.remove_text_from_images(image_paths)
        
        # Step 3: Generate Hindi explanation
        await status_msg.edit_text(
            "🎬 *Video ban rahi hai...*\n\n"
            "✅ Step 1/4: Images ready!\n"
            "✅ Step 2/4: Text hat gaya!\n"
            "⏳ Step 3/4: Hindi story likh raha hoon...",
            parse_mode='Markdown'
        )
        hindi_script = await processor.generate_hindi_script(image_paths)
        
        # Step 4: Generate video with voice
        await status_msg.edit_text(
            "🎬 *Video ban rahi hai...*\n\n"
            "✅ Step 1/4: Images ready!\n"
            "✅ Step 2/4: Text hat gaya!\n"
            "✅ Step 3/4: Script taiyar!\n"
            "⏳ Step 4/4: Voice aur video sync ho raha hai...",
            parse_mode='Markdown'
        )
        video_path = await processor.create_video_with_voice(cleaned_images, hindi_script)
        
        # Send video
        await status_msg.edit_text("✅ *Video taiyar hai! Bhej raha hoon...* 🎉", parse_mode='Markdown')
        
        with open(video_path, 'rb') as video_file:
            await update.message.reply_video(
                video=video_file,
                caption="🎌 *Tumhari Manga Video!*\n\nKaisi lagi? Aur bhejo! 🔥",
                parse_mode='Markdown',
                supports_streaming=True
            )
        
        await status_msg.delete()
        
        # Cleanup
        processor.cleanup(image_paths, cleaned_images, video_path, pdf_path)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        await status_msg.edit_text(
            f"❌ Kuch gadbad ho gayi yaar!\nError: {str(e)}\n\nDobara try karo!"
        )

def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("process", process_command))
    app.add_handler(MessageHandler(filters.PHOTO, handle_image))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    
    logger.info("Bot chal raha hai! 🚀")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()

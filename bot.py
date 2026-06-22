import os
import asyncio
import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler, CallbackQueryHandler,
    filters, ContextTypes
)
from pathlib import Path
import tempfile

from manga_processor import MangaProcessor
from db import (
    get_user_settings, update_user_setting, reset_user_settings,
    QUALITY_OPTIONS, VOICE_OPTIONS, DEFAULT_SETTINGS
)

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "YOUR_BOT_TOKEN_HERE")

processor = MangaProcessor()


# ═════════════════════════════════════════
# Basic commands
# ═════════════════════════════════════════

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🎌 *Manga Hindi Explainer Bot mein swagat hai!*\n\n"
        "Mujhe bhejo:\n"
        "📸 Manga images (JPG/PNG) — ek ek karke ya saath mein\n"
        "📄 PDF file — poora chapter\n"
        "🗂️ ZIP file — pura folder bhi chalega\n\n"
        "Main:\n"
        "✅ Text hataunga panels se\n"
        "✅ Hindi mein story explain karunga\n"
        "✅ Voice ke saath video bana ke dunga!\n\n"
        "⚙️ /settings se quality, voice, BGM, blur control karo\n\n"
        "Chalo shuru karte hain! 🔥",
        parse_mode='Markdown'
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📖 *Kaise use karein:*\n\n"
        "1. Manga ke images, PDF ya ZIP bhejo\n"
        "2. /process command do jab ready ho\n"
        "3. Wait karo — video ban raha hoga!\n\n"
        "⚙️ /settings — Quality, Voice, BGM, Blur change karo\n\n"
        "⚠️ Note: Video banane mein 2-5 minute lag sakte hain.",
        parse_mode='Markdown'
    )


# ═════════════════════════════════════════
# Settings menu
# ═════════════════════════════════════════

def build_settings_text(s: dict) -> str:
    bgm_status = f"ON ({s['bgm_volume']}%)" if s['bgm_enabled'] else "OFF"
    voice_label = VOICE_OPTIONS.get(s['voice'], {}).get('label', s['voice'])
    text_status = "Removed (clean panel)" if s.get('text_removal') else "Kept (original bubbles)"
    return (
        "⚙️ *YOUR CURRENT SETTINGS* ⚙️\n\n"
        f"🎥 Quality: `{s['quality']}`\n"
        f"🗣️ Voice: `{voice_label}`\n"
        f"🎵 BGM: `{bgm_status}`\n"
        f"📝 Panel Text: `{text_status}`\n\n"
        "👇 Select option to change:"
    )

def build_settings_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🎥 Quality", callback_data="menu_quality"),
         InlineKeyboardButton("🗣️ Voice", callback_data="menu_voice")],
        [InlineKeyboardButton("🎵 BGM", callback_data="menu_bgm"),
         InlineKeyboardButton("📝 Panel Text", callback_data="menu_text_removal")],
        [InlineKeyboardButton("🔄 Reset to Default", callback_data="reset_settings")],
    ])

async def settings_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    s = await get_user_settings(user_id)
    await update.message.reply_text(
        build_settings_text(s),
        parse_mode='Markdown',
        reply_markup=build_settings_keyboard()
    )

async def settings_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = query.from_user.id
    data = query.data
    await query.answer()

    # ── Back to main settings menu ──
    if data == "back_settings":
        s = await get_user_settings(user_id)
        await query.edit_message_text(
            build_settings_text(s), parse_mode='Markdown',
            reply_markup=build_settings_keyboard()
        )
        return

    # ── Reset ──
    if data == "reset_settings":
        await reset_user_settings(user_id)
        s = await get_user_settings(user_id)
        await query.edit_message_text(
            "✅ Settings reset ho gayi default pe!\n\n" + build_settings_text(s),
            parse_mode='Markdown',
            reply_markup=build_settings_keyboard()
        )
        return

    # ── Quality submenu ──
    if data == "menu_quality":
        buttons = [[InlineKeyboardButton(q, callback_data=f"set_quality_{q}")]
                   for q in QUALITY_OPTIONS.keys()]
        buttons.append([InlineKeyboardButton("⬅️ Back", callback_data="back_settings")])
        await query.edit_message_text(
            "🎥 *Quality select karo:*", parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup(buttons)
        )
        return

    if data.startswith("set_quality_"):
        quality = data.replace("set_quality_", "")
        await update_user_setting(user_id, "quality", quality)
        s = await get_user_settings(user_id)
        await query.edit_message_text(
            f"✅ Quality set: `{quality}`\n\n" + build_settings_text(s),
            parse_mode='Markdown',
            reply_markup=build_settings_keyboard()
        )
        return

    # ── Voice submenu ──
    if data == "menu_voice":
        buttons = [[InlineKeyboardButton(v['label'], callback_data=f"set_voice_{key}")]
                   for key, v in VOICE_OPTIONS.items()]
        buttons.append([InlineKeyboardButton("⬅️ Back", callback_data="back_settings")])
        await query.edit_message_text(
            "🗣️ *Voice select karo:*\n\n"
            "_Note: ek hi TTS engine hai, ye sirf style/tone variant hai,_\n"
            "_real alag gender ki awaaz nahi._",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup(buttons)
        )
        return

    if data.startswith("set_voice_"):
        voice = data.replace("set_voice_", "")
        await update_user_setting(user_id, "voice", voice)
        s = await get_user_settings(user_id)
        await query.edit_message_text(
            f"✅ Voice set: `{VOICE_OPTIONS[voice]['label']}`\n\n" + build_settings_text(s),
            parse_mode='Markdown',
            reply_markup=build_settings_keyboard()
        )
        return

    # ── BGM submenu ──
    if data == "menu_bgm":
        buttons = [
            [InlineKeyboardButton("🔇 OFF", callback_data="set_bgm_off")],
            [InlineKeyboardButton("🔈 20%", callback_data="set_bgm_20"),
             InlineKeyboardButton("🔉 40%", callback_data="set_bgm_40")],
            [InlineKeyboardButton("🔊 60%", callback_data="set_bgm_60"),
             InlineKeyboardButton("📢 80%", callback_data="set_bgm_80")],
            [InlineKeyboardButton("⬅️ Back", callback_data="back_settings")],
        ]
        await query.edit_message_text(
            "🎵 *BGM volume select karo:*", parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup(buttons)
        )
        return

    if data.startswith("set_bgm_"):
        val = data.replace("set_bgm_", "")
        if val == "off":
            await update_user_setting(user_id, "bgm_enabled", False)
        else:
            await update_user_setting(user_id, "bgm_enabled", True)
            await update_user_setting(user_id, "bgm_volume", int(val))
        s = await get_user_settings(user_id)
        await query.edit_message_text(
            "✅ BGM updated!\n\n" + build_settings_text(s),
            parse_mode='Markdown',
            reply_markup=build_settings_keyboard()
        )
        return

    # ── Panel Text (text removal) submenu ──
    if data == "menu_text_removal":
        buttons = [
            [InlineKeyboardButton("📝 Keep Text (original bubbles)", callback_data="set_textrm_off")],
            [InlineKeyboardButton("🧹 Remove Text (clean panel)", callback_data="set_textrm_on")],
            [InlineKeyboardButton("⬅️ Back", callback_data="back_settings")],
        ]
        await query.edit_message_text(
            "📝 *Panel text ka kya karna hai?*\n\n"
            "_Keep_ — speech bubbles waisi hi dikhengi panel mein (jaisi original "
            "manga mein hain)\n"
            "_Remove_ — OpenCV se text/bubble clean karke hata diya jayega",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup(buttons)
        )
        return

    if data.startswith("set_textrm_"):
        val = data.replace("set_textrm_", "")
        await update_user_setting(user_id, "text_removal", val == "on")
        s = await get_user_settings(user_id)
        await query.edit_message_text(
            "✅ Panel text setting updated!\n\n" + build_settings_text(s),
            parse_mode='Markdown',
            reply_markup=build_settings_keyboard()
        )
        return


# ═════════════════════════════════════════
# File handlers (image / pdf / zip)
# ═════════════════════════════════════════

async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if 'images' not in context.user_data:
        context.user_data['images'] = []

    photo = update.message.photo[-1]
    file = await context.bot.get_file(photo.file_id)

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
    fname = (doc.file_name or "").lower()

    if doc.mime_type == 'application/pdf' or fname.endswith('.pdf'):
        await update.message.reply_text("📄 PDF mil gayi! Process ho rahi hai... thoda wait karo ⏳")
        file = await context.bot.get_file(doc.file_id)
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            await file.download_to_drive(tmp.name)
            pdf_path = tmp.name
        await process_and_send(update, context, pdf_path=pdf_path)

    elif fname.endswith('.zip') or doc.mime_type in (
            'application/zip', 'application/x-zip-compressed'):
        await update.message.reply_text("🗂️ ZIP mil gayi! Extract karke process ho rahi hai... thoda wait karo ⏳")
        file = await context.bot.get_file(doc.file_id)
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp:
            await file.download_to_drive(tmp.name)
            zip_path = tmp.name
        await process_and_send(update, context, zip_path=zip_path)

    elif doc.mime_type in ('image/jpeg', 'image/png') or fname.endswith(('.jpg', '.jpeg', '.png')):
        file = await context.bot.get_file(doc.file_id)
        suffix = '.png' if fname.endswith('.png') else '.jpg'
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
        await update.message.reply_text("⚠️ Sirf JPG, PNG, PDF ya ZIP bhejo yaar!")

async def process_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    images = context.user_data.get('images', [])

    if not images:
        await update.message.reply_text("⚠️ Pehle kuch images, PDF ya ZIP bhejo bhai!")
        return

    await process_and_send(update, context, image_paths=images)
    context.user_data['images'] = []


# ═════════════════════════════════════════
# Core processing pipeline
# ═════════════════════════════════════════

async def process_and_send(update: Update, context: ContextTypes.DEFAULT_TYPE,
                            image_paths=None, pdf_path=None, zip_path=None):
    user_id = update.effective_user.id
    settings = await get_user_settings(user_id)

    status_msg = await update.message.reply_text(
        "🎬 *Video ban rahi hai...*\n\n"
        "⏳ Step 1/4: Images process ho rahi hain...",
        parse_mode='Markdown'
    )

    try:
        # Step 1: Get images (from PDF, ZIP, or direct uploads)
        # NOTE: pdf_to_images/zip_to_images CPU-bound blocking calls hain
        # (cv2, pdf2image) — run_in_executor mein chalate hain taaki bot ka
        # event loop block na ho aur Telegram ko response milte rahe
        # (warna bot "stuck" dikhta hai aur Railway timeout/restart kar deta hai)
        loop = asyncio.get_event_loop()
        if pdf_path:
            await status_msg.edit_text(
                "🎬 *Video ban rahi hai...*\n\n"
                "⏳ Step 1/4: PDF se images nikal raha hoon...",
                parse_mode='Markdown'
            )
            image_paths = await loop.run_in_executor(
                None, processor.pdf_to_images, pdf_path
            )
        elif zip_path:
            await status_msg.edit_text(
                "🎬 *Video ban rahi hai...*\n\n"
                "⏳ Step 1/4: ZIP se images nikal raha hoon...",
                parse_mode='Markdown'
            )
            image_paths = await loop.run_in_executor(
                None, processor.zip_to_images, zip_path
            )

        # Step 1b: Blank/text-only panels filter karo (jaise sirf-bubble
        # wale recap panels, koi actual artwork nahi) — yeh extraction ke
        # turant baad karte hain taaki aage Gemini script-generation aur
        # video-rendering steps mein bhi yeh panels skip ho jaayein
        # (Gemini quota bhi bachta hai aise panels pe waste hone se)
        if image_paths:
            (image_paths,) = await loop.run_in_executor(
                None, processor.filter_blank_panels, image_paths
            )
            if not image_paths:
                raise ValueError("Saare panels blank/text-only nikle — koi actual artwork nahi mila!")

        # Step 2: Remove text from panels (settings ke hisaab se — optional)
        if settings.get('text_removal'):
            await status_msg.edit_text(
                "🎬 *Video ban rahi hai...*\n\n"
                "✅ Step 1/4: Images ready!\n"
                "⏳ Step 2/4: Manga text hat raha hai...",
                parse_mode='Markdown'
            )
            cleaned_images = await loop.run_in_executor(
                None, processor.remove_text_from_images, image_paths
            )
        else:
            cleaned_images = list(image_paths)

        # Step 3: Har panel ke liye explainer-script (dialogue + expression
        # + scene) generate karo, position-tagged beats ke saath
        await status_msg.edit_text(
            "🎬 *Video ban rahi hai...*\n\n"
            "✅ Step 1/4: Images ready!\n"
            "✅ Step 2/4: Panel text settings apply ho gayi!\n"
            "⏳ Step 3/4: Har panel ka explainer-script likh raha hoon "
            f"(0/{len(cleaned_images)})...",
            parse_mode='Markdown'
        )
        panel_beats = []
        story_context = ""
        # IMPORTANT: script hamesha ORIGINAL image (image_paths) se generate
        # hota hai, kyunki agar text_removal ON hai to cleaned_images mein
        # dialogue mit chuka hota hai — Gemini ko dialogue padhne ke liye
        # original text-wali image chahiye. cleaned_images sirf VIDEO mein
        # dikhane ke liye use hoti hai.
        #
        # BATCH MODE: 2 panels ek Gemini call mein — calls aadhi ho jaati
        # hain, free-tier (10 RPM) par kaafi faster processing hoti hai.
        i = 0
        while i < len(image_paths):
            batch_paths = image_paths[i:i + 2]
            beats_list, story_context = await processor.generate_panel_scripts_batch(
                batch_paths, story_context=story_context
            )
            panel_beats.extend(beats_list)
            i += 2
            done_count = min(i, len(image_paths))
            if done_count % 2 == 0 or done_count == len(cleaned_images):
                try:
                    await status_msg.edit_text(
                        "🎬 *Video ban rahi hai...*\n\n"
                        "✅ Step 1/4: Images ready!\n"
                        "✅ Step 2/4: Panel text settings apply ho gayi!\n"
                        f"⏳ Step 3/4: Script likh raha hoon "
                        f"({done_count}/{len(cleaned_images)})...",
                        parse_mode='Markdown'
                    )
                except Exception:
                    pass  # rate-limit pe edit fail ho sakta hai, ignore karo

        # Step 4: Scroll-synced video banao (voice + BGM)
        await status_msg.edit_text(
            "🎬 *Video ban rahi hai...*\n\n"
            "✅ Step 1/4: Images ready!\n"
            "✅ Step 2/4: Panel text settings apply ho gayi!\n"
            "✅ Step 3/4: Explainer-script taiyar!\n"
            "⏳ Step 4/4: Voice, scroll-sync aur BGM se video ban rahi hai... "
            f"(0/{len(cleaned_images)})",
            parse_mode='Markdown'
        )

        async def _video_progress(done: int, total: int):
            # done == -1 is a special signal: all panels rendered,
            # final ffmpeg merge/BGM mix chal rahi hai
            if done == -1:
                try:
                    await status_msg.edit_text(
                        "🎬 *Video ban rahi hai...*\n\n"
                        "✅ Step 1/4: Images ready!\n"
                        "✅ Step 2/4: Panel text settings apply ho gayi!\n"
                        "✅ Step 3/4: Explainer-script taiyar!\n"
                        f"✅ Step 4/4: Panels render ho gaye ({total}/{total})!\n"
                        "⏳ Step 5/5: Final video merge + BGM mix ho rahi hai...",
                        parse_mode='Markdown'
                    )
                except Exception:
                    pass
                return
            # Har 2 panels ya last panel par update karo
            if done % 2 == 0 or done == total:
                try:
                    await status_msg.edit_text(
                        "🎬 *Video ban rahi hai...*\n\n"
                        "✅ Step 1/4: Images ready!\n"
                        "✅ Step 2/4: Panel text settings apply ho gayi!\n"
                        "✅ Step 3/4: Explainer-script taiyar!\n"
                        f"⏳ Step 4/4: Panel render ho rahe hain ({done}/{total})...",
                        parse_mode='Markdown'
                    )
                except Exception:
                    pass  # rate-limit pe edit fail ho sakta hai, ignore karo

        quality_height = QUALITY_OPTIONS.get(settings['quality'], 720)
        video_path = await processor.create_video_from_panels(
            cleaned_images, panel_beats,
            quality_height=quality_height,
            voice=settings['voice'],
            bgm_enabled=settings['bgm_enabled'],
            bgm_volume=settings['bgm_volume'],
            progress_callback=_video_progress,
        )

        # Send video
        await status_msg.edit_text("✅ *Video taiyar hai! Bhej raha hoon...* 🎉", parse_mode='Markdown')

        with open(video_path, 'rb') as video_file:
            await update.message.reply_video(
                video=video_file,
                caption="🎌 *Tumhari Manga Explainer Video!*\n\nKaisi lagi? Aur bhejo! 🔥\n\n⚙️ /settings se style change karo",
                parse_mode='Markdown',
                supports_streaming=True
            )

        await status_msg.delete()

        # Cleanup
        processor.cleanup(image_paths, cleaned_images,
                           video_path, pdf_path, zip_path)

    except Exception as e:
        logger.error(f"Error: {e}")
        await status_msg.edit_text(
            f"❌ Kuch gadbad ho gayi yaar!\nError: {str(e)}\n\nDobara try karo!"
        )


# ═════════════════════════════════════════
# Main
# ═════════════════════════════════════════

def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("process", process_command))
    app.add_handler(CommandHandler("settings", settings_command))
    app.add_handler(CallbackQueryHandler(settings_callback))
    app.add_handler(MessageHandler(filters.PHOTO, handle_image))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))

    logger.info("Bot chal raha hai! 🚀")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()

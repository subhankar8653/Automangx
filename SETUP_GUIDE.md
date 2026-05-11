# 🎌 Manga Hindi Bot — Setup Guide

## Step 1: API Keys Lao

### Telegram Bot Token:
1. Telegram pe @BotFather open karo
2. /newbot likho
3. Bot ka naam do (jaise: MangaHindiBot)
4. Token copy kar lo (aisa dikhega: 123456:ABC-DEF...)

### Gemini API Key:
1. https://aistudio.google.com/ pe jao
2. "Get API Key" pe click karo
3. Free mein milega!

---

## Step 2: Railway pe Deploy karo

1. https://railway.app pe account banao (GitHub se login karo)
2. "New Project" → "Deploy from GitHub repo"
3. Apna repo select karo
4. Environment Variables mein yeh add karo:
   - TELEGRAM_TOKEN = (apna token)
   - GEMINI_API_KEY = (apni key)
5. Deploy ho jayega automatically!

---

## Step 3: Bot Test karo

1. Telegram pe apna bot dhundo
2. /start likho
3. Manga image bhejo
4. /process likho
5. Video aayegi! 🎉

---

## File Structure:
```
manga_bot/
├── bot.py              # Main bot file
├── manga_processor.py  # Core processing logic
├── requirements.txt    # Dependencies
└── railway.toml        # Railway config
```

---

## Agar kuch kaam na kare:

- Railway logs check karo (Deploy → Logs)
- API keys sahi hain? Double check karo
- poppler install hona chahiye PDF ke liye (Railway pe auto hota hai)

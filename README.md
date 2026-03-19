# 📚 StudyBot v2 — AI Study Companion (Grade 6–12)

Powered by **Google Gemini (free)**. Globally hosted on Railway. No laptop needed.

---

## ✨ What's new in v2
- Grade 6–12 dropdown (ICSE & CBSE)
- All outputs printable as PDF
- Switched to Google Gemini — completely free
- Deployed to cloud — no laptop needed, works worldwide

---

## 🚀 Deploy to Railway (global, free, one-time setup)

### Step 1 — Get a free Gemini API key
1. Go to https://aistudio.google.com/apikey
2. Sign in with any Google account
3. Click **Create API Key** → copy it

### Step 2 — Put this code on GitHub
1. Create a free account at https://github.com
2. Click **New repository** → name it `studybot` → Create
3. Upload all files from this folder into that repository

### Step 3 — Deploy on Railway
1. Go to https://railway.app → sign up with GitHub
2. Click **New Project → Deploy from GitHub Repo**
3. Select your `studybot` repository
4. Click your project → go to **Variables** tab
5. Add: `GEMINI_API_KEY` = your key from Step 1
6. Railway builds and gives you a URL like:
   `https://studybot-production.up.railway.app`
7. Share this URL with family anywhere in the world!

**Railway free tier:** 500 hours/month — more than enough for family use.

---

## 💻 Run locally (optional)

```cmd
cd studyapp
copy .env.example .env
notepad .env        ← paste your GEMINI_API_KEY
npm install
npm start
```
Open http://localhost:3000

---

## 📁 Files

```
studyapp/
├── server.js          ← Express + Gemini backend
├── package.json
├── Procfile           ← Railway/Render deployment
├── .env.example
└── public/
    └── index.html     ← Full frontend
```

---

## 🖨️ PDF / Print
Every output has a **Print as PDF** button.
- In the browser print dialog, select **Save as PDF** as the destination.
- Notes, Questions, and Question Bank all print cleanly on A4.

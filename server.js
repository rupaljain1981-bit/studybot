require('dotenv').config();
const express = require('express');
const multer  = require('multer');
const cors    = require('cors');
const path    = require('path');

const app  = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.static(path.join(__dirname, 'public')));

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 20 * 1024 * 1024 },
  fileFilter: (req, file, cb) => {
    const ok = ['image/jpeg','image/png','image/gif','image/webp','application/pdf'];
    ok.includes(file.mimetype) ? cb(null, true) : cb(new Error('Images and PDFs only'));
  }
});

// ── Groq helper ───────────────────────────────────────────────────────────────
async function callGroq(prompt) {
  const response = await fetch('https://api.groq.com/openai/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${process.env.GROQ_API_KEY}`
    },
    body: JSON.stringify({
      model: 'llama-3.3-70b-versatile',
      messages: [{ role: 'user', content: prompt }],
      max_tokens: 4096,
      temperature: 0.7
    })
  });

  if (!response.ok) {
    const err = await response.json();
    throw new Error(err.error?.message || 'Groq API error');
  }

  const data = await response.json();
  return data.choices[0].message.content;
}

// ── Notes ─────────────────────────────────────────────────────────────────────
app.post('/api/generate-notes', upload.array('files', 10), async (req, res) => {
  try {
    const { subject, topic, grade } = req.body;
    const files    = req.files || [];
    const hasFiles = files.length > 0;

    if (!hasFiles && !topic) {
      return res.status(400).json({ error: 'Enter a topic or upload files.' });
    }

    const prompt = `You are an expert teacher for ${grade || 'Grade 8'} students following the ICSE/CBSE curriculum.
Subject: ${subject || ''}
Topic: ${topic || ''}
Grade: ${grade || 'Grade 8'}

Create detailed, comprehensive study notes for this topic appropriate for ${grade || 'Grade 8'} students.

Return ONLY the following HTML — no markdown, no backticks, no preamble:

<div class="notes-content">
  <section class="key-concepts">
    <h2>🔑 Key Concepts</h2>
    <!-- clear bullet list of core ideas -->
  </section>
  <section class="detailed-notes">
    <h2>📚 Detailed Notes</h2>
    <!-- thorough notes with sub-headings and examples -->
  </section>
  <section class="memory-tricks">
    <h2>💡 Memory Tricks & Mnemonics</h2>
    <!-- at least 3 clever memory aids -->
  </section>
  <section class="important-formulas">
    <h2>📐 Key Formulas / Definitions</h2>
    <!-- every important formula or definition -->
  </section>
  <section class="quick-summary">
    <h2>⚡ Quick Summary</h2>
    <!-- exactly 5 bullet points for last-minute revision -->
  </section>
  <section class="infographic">
    <h2>🗺️ Concept Map</h2>
    <!-- ASCII table or text diagram showing relationships -->
  </section>
</div>

Use simple language appropriate for ${grade || 'Grade 8'}, include real-world examples. Return ONLY the HTML.`;

    const raw     = await callGroq(prompt);
    const cleaned = raw.replace(/```html|```/g, '').trim();
    res.json({ success: true, notes: cleaned, fromTopic: !hasFiles });

  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

// ── Questions ─────────────────────────────────────────────────────────────────
app.post('/api/generate-questions', async (req, res) => {
  try {
    const { notes, subject, topic, grade } = req.body;
    if (!notes) return res.status(400).json({ error: 'Notes are required.' });

    const prompt = `You are an exam expert for ${grade || 'Grade 8'} ICSE/CBSE students.
Subject: ${subject || ''} | Topic: ${topic || ''} | Grade: ${grade || 'Grade 8'}

Based on these notes, create a question set.
NOTES:
${notes.slice(0, 3000)}

Return ONLY valid JSON — no markdown, no backticks:
{
  "easy": [
    {"q":"...","a":"...","type":"short"},
    {"q":"...","a":"...","type":"fill"},
    {"q":"...","a":"...","type":"mcq"},
    {"q":"...","a":"...","type":"short"},
    {"q":"...","a":"...","type":"fill"}
  ],
  "medium": [
    {"q":"...","a":"...","type":"short"},
    {"q":"...","a":"...","type":"long"},
    {"q":"...","a":"...","type":"short"},
    {"q":"...","a":"...","type":"long"},
    {"q":"...","a":"...","type":"short"}
  ],
  "hard": [
    {"q":"...","a":"...","type":"application"},
    {"q":"...","a":"...","type":"long"},
    {"q":"...","a":"...","type":"application"},
    {"q":"...","a":"...","type":"long"},
    {"q":"...","a":"...","type":"application"}
  ],
  "truefalse": [
    {"q":"...","a":"True","explanation":"..."},
    {"q":"...","a":"False","explanation":"..."},
    {"q":"...","a":"True","explanation":"..."},
    {"q":"...","a":"False","explanation":"..."},
    {"q":"...","a":"True","explanation":"..."}
  ]
}`;

    const raw = await callGroq(prompt);
    let questions;
    try {
      questions = JSON.parse(raw.replace(/```json|```/g, '').trim());
    } catch {
      questions = { error: 'Parse failed', raw };
    }
    res.json({ success: true, questions });

  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

// ── Question Bank ─────────────────────────────────────────────────────────────
app.post('/api/question-bank', async (req, res) => {
  try {
    const { subject, topic, difficulty, grade } = req.body;
    if (!subject || !topic) {
      return res.status(400).json({ error: 'Subject and topic are required.' });
    }

    const prompt = `You are an ICSE/CBSE exam expert. Generate a question bank for:
Grade: ${grade || 'Grade 8'}
Subject: ${subject}
Topic: ${topic}
Difficulty: ${difficulty || 'mixed'}

Create 15 exam-style questions based on the ICSE/CBSE curriculum for this topic.

Return ONLY valid JSON (no markdown):
{
  "subject": "${subject}",
  "topic": "${topic}",
  "grade": "${grade || 'Grade 8'}",
  "questions": [
    {"difficulty":"easy","question":"...","answer":"...","source":"ICSE Curriculum","year":""},
    {"difficulty":"medium","question":"...","answer":"...","source":"ICSE Curriculum","year":""},
    {"difficulty":"hard","question":"...","answer":"...","source":"ICSE Curriculum","year":""}
  ]
}

Include exactly 15 questions with mixed difficulties. Return ONLY valid JSON.`;

    const raw = await callGroq(prompt);
    let bank;
    try {
      const cleaned = raw.replace(/```json|```/g, '').trim();
      bank = JSON.parse(cleaned.slice(cleaned.indexOf('{'), cleaned.lastIndexOf('}') + 1));
    } catch {
      bank = { questions: [] };
    }
    res.json({ success: true, bank });

  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

// ── Ask ───────────────────────────────────────────────────────────────────────
app.post('/api/ask', async (req, res) => {
  try {
    const { question, context, grade } = req.body;
    const prompt = `You are a friendly tutor for ${grade || 'Grade 8'} students. Answer simply and clearly.
${context ? `Notes context: ${context.slice(0, 1000)}\n` : ''}
Student question: ${question}
Give an encouraging, clear answer with bullet points and examples.`;

    const answer = await callGroq(prompt);
    res.json({ success: true, answer });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

app.get('/api/health', (_, res) => res.json({ status: 'ok', powered_by: 'Groq (free)' }));

app.listen(PORT, () => {
  console.log(`\n🎓 StudyBot running at http://localhost:${PORT}`);
  console.log(`   Powered by Groq LLaMA (free)\n`);
});

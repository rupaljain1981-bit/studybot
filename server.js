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
async function callGroq(prompt, maxTokens = 4096) {
  const response = await fetch('https://api.groq.com/openai/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${process.env.GROQ_API_KEY}`
    },
    body: JSON.stringify({
      model: 'llama-3.3-70b-versatile',
      messages: [{ role: 'user', content: prompt }],
      max_tokens: maxTokens,
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

// ── Generate ALL in one shot ──────────────────────────────────────────────────
app.post('/api/generate-all', upload.array('files', 10), async (req, res) => {
  try {
    const { subject, topic, grade } = req.body;
    const files    = req.files || [];
    const hasFiles = files.length > 0;

    if (!topic && !hasFiles) {
      return res.status(400).json({ error: 'Please enter a topic or upload files.' });
    }

    const gradeLabel   = grade   || 'Grade 8';
    const subjectLabel = subject || '';
    const topicLabel   = topic   || '';

    // ── Prompt 1: Study Notes ────────────────────────────────────────────────
    const notesPrompt = `You are an expert teacher for ${gradeLabel} students (ICSE/CBSE curriculum).
Subject: ${subjectLabel}
Topic: ${topicLabel}
Grade: ${gradeLabel}

Create comprehensive, detailed study notes for this topic appropriate for ${gradeLabel} students.
Return ONLY the following HTML — no markdown, no backticks, no preamble:

<div class="notes-content">
  <section class="key-concepts">
    <h2>🔑 Key Concepts</h2>
    <!-- clear bullet list of all core ideas -->
  </section>
  <section class="detailed-notes">
    <h2>📚 Detailed Notes</h2>
    <!-- thorough explanation with sub-headings, real-world examples, diagrams described in text -->
  </section>
  <section class="memory-tricks">
    <h2>💡 Memory Tricks &amp; Mnemonics</h2>
    <!-- at least 3 clever memory aids specific to this topic -->
  </section>
  <section class="important-formulas">
    <h2>📐 Key Formulas / Definitions</h2>
    <!-- every important formula, law, or definition clearly laid out -->
  </section>
  <section class="quick-summary">
    <h2>⚡ Quick Summary</h2>
    <!-- exactly 5 bullet points for last-minute revision -->
  </section>
  <section class="infographic">
    <h2>🗺️ Concept Map</h2>
    <!-- ASCII table or structured text diagram showing how ideas connect -->
  </section>
</div>

Use simple language for ${gradeLabel}, include real-world examples. Return ONLY the HTML.`;

    // ── Prompt 2: Questions ──────────────────────────────────────────────────
    const questionsPrompt = `You are an expert ${gradeLabel} ICSE/CBSE examiner for ${subjectLabel} — Topic: ${topicLabel}.

Create a high-quality, varied question set. Each question must be specific, meaningful and directly test understanding of this topic.

Return ONLY valid JSON — no markdown, no backticks, no extra text:
{
  "mcq": [
    {"q":"Question with 4 options?","options":["A) ...","B) ...","C) ...","D) ..."],"a":"A) ...","explanation":"Why this is correct"},
    {"q":"...","options":["A) ...","B) ...","C) ...","D) ..."],"a":"B) ...","explanation":"..."},
    {"q":"...","options":["A) ...","B) ...","C) ...","D) ..."],"a":"C) ...","explanation":"..."},
    {"q":"...","options":["A) ...","B) ...","C) ...","D) ..."],"a":"D) ...","explanation":"..."},
    {"q":"...","options":["A) ...","B) ...","C) ...","D) ..."],"a":"A) ...","explanation":"..."}
  ],
  "fillinblanks": [
    {"q":"The process by which ________ produces glucose is called ________.","a":"chlorophyll / photosynthesis"},
    {"q":"...","a":"..."},
    {"q":"...","a":"..."},
    {"q":"...","a":"..."},
    {"q":"...","a":"..."}
  ],
  "truefalse": [
    {"q":"A specific factual statement about this topic.","a":"True","explanation":"Brief reason why"},
    {"q":"An incorrect statement that students might believe.","a":"False","explanation":"What is actually correct"},
    {"q":"...","a":"True","explanation":"..."},
    {"q":"...","a":"False","explanation":"..."},
    {"q":"...","a":"True","explanation":"..."}
  ],
  "oddonesout": [
    {"q":"Which is the odd one out: A) ... B) ... C) ... D) ...","a":"B) ...","explanation":"Because it is the only one that..."},
    {"q":"...","a":"...","explanation":"..."},
    {"q":"...","a":"...","explanation":"..."}
  ],
  "assertionreason": [
    {"assertion":"A specific claim about this topic.","reason":"The reason given for that claim.","a":"Both Assertion and Reason are true, and Reason is the correct explanation","explanation":"..."},
    {"assertion":"...","reason":"...","a":"Assertion is true but Reason is false","explanation":"..."},
    {"assertion":"...","reason":"...","a":"Both Assertion and Reason are true, but Reason is NOT the correct explanation","explanation":"..."}
  ],
  "shortanswer": [
    {"q":"Define ... in one sentence.","a":"Full answer expected from student","marks":1},
    {"q":"State two differences between ... and ...","a":"1. ... 2. ...","marks":2},
    {"q":"What happens when ...? Give one example.","a":"...","marks":2},
    {"q":"...","a":"...","marks":2},
    {"q":"...","a":"...","marks":3}
  ],
  "longanswer": [
    {"q":"Explain the process of ... in detail with a labelled diagram description.","a":"Detailed model answer covering all key points...","marks":5},
    {"q":"Compare and contrast ... and ... with examples.","a":"...","marks":5},
    {"q":"Describe ... and explain its significance.","a":"...","marks":6}
  ]
}

Make every question genuinely test understanding — no vague or trivial questions.`;

    // ── Prompt 3: Question Bank ──────────────────────────────────────────────
    const bankPrompt = `You are an expert ${gradeLabel} ICSE/CBSE examiner for ${subjectLabel} — Topic: ${topicLabel}.

Generate 18 high-quality exam questions covering all question types used in ICSE/CBSE board exams.
Mix: objective, fill in the blanks, true/false, odd one out, assertion-reason, short answer (2-3 marks), long answer (5-6 marks).

Return ONLY valid JSON — no markdown, no backticks:
{
  "subject": "${subjectLabel}",
  "topic": "${topicLabel}",
  "grade": "${gradeLabel}",
  "questions": [
    {"type":"MCQ","difficulty":"easy","question":"Full question text with options A B C D","answer":"Correct option with text","marks":1},
    {"type":"Fill in the Blank","difficulty":"easy","question":"The ___ is responsible for ___","answer":"word1 / word2","marks":1},
    {"type":"True or False","difficulty":"easy","question":"Statement to evaluate","answer":"True — because...","marks":1},
    {"type":"Odd One Out","difficulty":"medium","question":"Identify the odd one: A)... B)... C)... D)...","answer":"B — because...","marks":1},
    {"type":"Assertion-Reason","difficulty":"hard","question":"Assertion: ... Reason: ...","answer":"Both true, Reason is correct explanation","marks":2},
    {"type":"Short Answer","difficulty":"medium","question":"Question requiring 2-3 sentence answer","answer":"Model answer","marks":2},
    {"type":"Long Answer","difficulty":"hard","question":"Question requiring detailed explanation","answer":"Detailed model answer","marks":5}
  ]
}

Include exactly 18 questions with a good variety of types and difficulties. Return ONLY valid JSON.`;

    // ── Run all 3 in parallel ────────────────────────────────────────────────
    const [notesRaw, questionsRaw, bankRaw] = await Promise.all([
      callGroq(notesPrompt, 4096),
      callGroq(questionsPrompt, 2048),
      callGroq(bankPrompt, 2048)
    ]);

    // Clean notes
    const notes = notesRaw.replace(/```html|```/g, '').trim();

    // Parse questions
    let questions;
    try {
      questions = JSON.parse(questionsRaw.replace(/```json|```/g, '').trim());
    } catch {
      questions = { mcq:[], fillinblanks:[], truefalse:[], oddonesout:[], assertionreason:[], shortanswer:[], longanswer:[] };
    }

    // Parse bank
    let bank;
    try {
      const cleaned = bankRaw.replace(/```json|```/g, '').trim();
      bank = JSON.parse(cleaned.slice(cleaned.indexOf('{'), cleaned.lastIndexOf('}') + 1));
    } catch {
      bank = { questions: [] };
    }

    res.json({ success: true, notes, questions, bank, fromTopic: !hasFiles });

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

    const answer = await callGroq(prompt, 1024);
    res.json({ success: true, answer });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

app.get('/api/health', (_, res) => res.json({ status: 'ok', powered_by: 'Groq LLaMA (free)' }));

app.listen(PORT, () => {
  console.log(`\n🎓 StudyBot running at http://localhost:${PORT}`);
  console.log(`   Powered by Groq LLaMA (free)\n`);
});

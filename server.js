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
async function callGroq(prompt, maxTokens = 2048) {
  const response = await fetch('https://api.groq.com/openai/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${process.env.GROQ_API_KEY}`
    },
    body: JSON.stringify({
      model: 'llama-3.3-70b-versatile',
      messages: [
        { role: 'system', content: 'You are an expert ICSE/ISC teacher. Return ONLY valid JSON or HTML exactly as instructed. No markdown fences, no preamble, no explanation whatsoever.' },
        { role: 'user', content: prompt }
      ],
      max_tokens: maxTokens,
      temperature: 0.4
    })
  });
  if (!response.ok) {
    const err = await response.json();
    throw new Error(err.error?.message || 'Groq API error');
  }
  const data = await response.json();
  return data.choices[0].message.content;
}

// ── Safe JSON parse ───────────────────────────────────────────────────────────
function safeJSON(raw, fallback = {}) {
  try {
    const cleaned = raw.replace(/```json|```/g, '').trim();
    const start = cleaned.indexOf('{');
    const end   = cleaned.lastIndexOf('}');
    if (start === -1 || end === -1) throw new Error('No JSON found');
    return JSON.parse(cleaned.slice(start, end + 1));
  } catch(e) {
    console.error('JSON parse failed:', e.message, '\nRaw preview:', raw.slice(0, 200));
    return fallback;
  }
}

// ── Generate all 7 question types (3 parallel small calls) ────────────────────
async function generateQuestions(grade, subject, topic) {
  const ctx = `Grade: ${grade} | Subject: ${subject} | Topic: ${topic}`;

  const callA = callGroq(
`${ctx}
Generate 5 MCQs and 5 fill-in-the-blank questions about "${topic}".
Return ONLY this JSON — replace every placeholder with real content about ${topic}:
{"mcq":[{"q":"First MCQ question about ${topic}?","options":["A) option one","B) option two","C) option three","D) option four"],"a":"A) option one","explanation":"reason why A is correct"},{"q":"Second MCQ?","options":["A) opt","B) opt","C) opt","D) opt"],"a":"B) opt","explanation":"reason"},{"q":"Third MCQ?","options":["A) opt","B) opt","C) opt","D) opt"],"a":"C) opt","explanation":"reason"},{"q":"Fourth MCQ?","options":["A) opt","B) opt","C) opt","D) opt"],"a":"D) opt","explanation":"reason"},{"q":"Fifth MCQ?","options":["A) opt","B) opt","C) opt","D) opt"],"a":"A) opt","explanation":"reason"}],"fillinblanks":[{"q":"The ________ is the basic unit of ${topic}.","a":"correct answer"},{"q":"________ was first described by ________.","a":"word1 / word2"},{"q":"The process of ________ results in ________.","a":"word1 / word2"},{"q":"________ is measured in units of ________.","a":"word1 / word2"},{"q":"The main function of ________ is to ________.","a":"word1 / word2"}]}`, 1500);

  const callB = callGroq(
`${ctx}
Generate 5 true/false and 3 odd-one-out questions about "${topic}".
Return ONLY this JSON — replace every placeholder with real content about ${topic}:
{"truefalse":[{"q":"A correct factual statement about ${topic}.","a":"True","explanation":"brief reason"},{"q":"A common misconception about ${topic} that is actually wrong.","a":"False","explanation":"the correct fact is..."},{"q":"Another correct statement about ${topic}.","a":"True","explanation":"reason"},{"q":"Another incorrect statement about ${topic}.","a":"False","explanation":"correction"},{"q":"A final correct statement about ${topic}.","a":"True","explanation":"reason"}],"oddonesout":[{"q":"Which is the odd one out? A) item1 B) item2 C) item3 D) item4","a":"B) item2","explanation":"it is the only one that does not fit because..."},{"q":"Which is the odd one out? A) item1 B) item2 C) item3 D) item4","a":"C) item3","explanation":"reason"},{"q":"Which is the odd one out? A) item1 B) item2 C) item3 D) item4","a":"D) item4","explanation":"reason"}]}`, 1200);

  const callC = callGroq(
`${ctx}
Generate 3 assertion-reason, 5 short-answer, and 3 long-answer questions about "${topic}".
Return ONLY this JSON — replace every placeholder with real specific content about ${topic}:
{"assertionreason":[{"assertion":"A true factual claim about ${topic}.","reason":"The correct scientific or factual reason behind that claim.","a":"Both Assertion and Reason are true, and Reason is the correct explanation","explanation":"brief clarification"},{"assertion":"A true claim about ${topic}.","reason":"A plausible but incorrect reason for the claim.","a":"Assertion is true but Reason is false","explanation":"the correct reason is..."},{"assertion":"A true claim about ${topic}.","reason":"A true statement that is unrelated to the assertion.","a":"Both Assertion and Reason are true, but Reason is NOT the correct explanation","explanation":"what actually explains the assertion"}],"shortanswer":[{"q":"Define [key term from ${topic}] in your own words.","a":"Model answer: a clear one-sentence definition with an example.","marks":1},{"q":"State two important characteristics of [key concept in ${topic}].","a":"1. First characteristic with detail. 2. Second characteristic with detail.","marks":2},{"q":"What is the significance of [important element] in ${topic}?","a":"Clear 2-3 sentence explanation of significance with a specific example.","marks":2},{"q":"Give one real-life application of ${topic} and explain how it works.","a":"Specific real-world example with a full explanation linking it to the topic.","marks":2},{"q":"Describe the steps involved in [a key process in ${topic}].","a":"Step 1: ... Step 2: ... Step 3: ... with brief explanation of each step.","marks":3}],"longanswer":[{"q":"Explain the concept of ${topic} in detail. Include definitions, mechanisms, and real-world examples.","a":"Introduction defining ${topic}. Key concepts explained. Mechanism or process described step by step. Two or more real-world examples. Conclusion on significance.","marks":5},{"q":"Compare and contrast [two major aspects of ${topic}]. Present your answer in a tabular format followed by a concluding paragraph.","a":"Table with at least 4 points of comparison. Concluding paragraph explaining which is more significant and why.","marks":5},{"q":"Discuss the importance and real-life applications of ${topic}. How does it affect our daily lives and what is its broader significance?","a":"Importance in science/society. At least 3 real-life applications with explanation. Environmental/economic/social significance. Future relevance or challenges.","marks":6}]}`, 2000);

  const [rawA, rawB, rawC] = await Promise.all([callA, callB, callC]);
  const questions = { mcq:[], fillinblanks:[], truefalse:[], oddonesout:[], assertionreason:[], shortanswer:[], longanswer:[] };
  Object.assign(questions, safeJSON(rawA, {}));
  Object.assign(questions, safeJSON(rawB, {}));
  Object.assign(questions, safeJSON(rawC, {}));
  return questions;
}

// ── Notes prompt ──────────────────────────────────────────────────────────────
function buildNotesPrompt(grade, subject, topic) {
  return `You are an expert ${grade} ICSE/ISC teacher.
Subject: ${subject} | Topic: ${topic} | Grade: ${grade}

Create comprehensive study notes about "${topic}" for ${grade} students.
Return ONLY this HTML structure — fill every section with real content. No markdown, no backticks:

<div class="notes-content">
  <section class="key-concepts">
    <h2>🔑 Key Concepts</h2>
    <ul><li>Real concept 1 from ${topic}</li><li>Real concept 2</li><li>Real concept 3</li><li>Real concept 4</li><li>Real concept 5</li></ul>
  </section>
  <section class="detailed-notes">
    <h2>📚 Detailed Notes</h2>
    <h3>Subtopic 1 heading</h3><p>Detailed explanation with examples...</p>
    <h3>Subtopic 2 heading</h3><p>Detailed explanation...</p>
    <h3>Subtopic 3 heading</h3><p>Detailed explanation...</p>
  </section>
  <section class="memory-tricks">
    <h2>💡 Memory Tricks</h2>
    <p>Mnemonic 1 explanation:</p>
    <div class="mnemonic-row">
      <div class="mnemonic-box"><span class="mnemonic-letter">A</span><span class="mnemonic-word">Word for A</span></div>
      <div class="mnemonic-box"><span class="mnemonic-letter">B</span><span class="mnemonic-word">Word for B</span></div>
      <div class="mnemonic-box"><span class="mnemonic-letter">C</span><span class="mnemonic-word">Word for C</span></div>
    </div>
    <p>Memory tip 2: explain a visual trick or story to remember a concept.</p>
    <p>Memory tip 3: another technique.</p>
  </section>
  <section class="important-formulas">
    <h2>📐 Key Formulas / Definitions</h2>
    <div class="formula-box"><div class="formula-title">Term or Formula Name</div><div class="formula-eq">The actual formula or definition</div></div>
    <div class="formula-box"><div class="formula-title">Another Term</div><div class="formula-eq">Definition or formula</div></div>
  </section>
  <section class="quick-summary">
    <h2>⚡ Quick Summary</h2>
    <ul><li>Key point 1</li><li>Key point 2</li><li>Key point 3</li><li>Key point 4</li><li>Key point 5</li></ul>
  </section>
  <section class="infographic">
    <h2>🗺️ Concept Map</h2>
    <div class="concept-map">
      <div class="cm-center">${topic}</div>
      <div class="cm-branches">
        <div class="cm-branch">
          <div class="cm-node cm-blue">Branch 1 label</div>
          <div class="cm-children">
            <div class="cm-child">Detail A</div>
            <div class="cm-child">Detail B</div>
          </div>
        </div>
        <div class="cm-branch">
          <div class="cm-node cm-green">Branch 2 label</div>
          <div class="cm-children">
            <div class="cm-child">Detail C</div>
            <div class="cm-child">Detail D</div>
          </div>
        </div>
        <div class="cm-branch">
          <div class="cm-node cm-amber">Branch 3 label</div>
          <div class="cm-children">
            <div class="cm-child">Detail E</div>
            <div class="cm-child">Detail F</div>
          </div>
        </div>
        <div class="cm-branch">
          <div class="cm-node cm-coral">Branch 4 label</div>
          <div class="cm-children">
            <div class="cm-child">Detail G</div>
          </div>
        </div>
      </div>
    </div>
  </section>
</div>

Replace ALL placeholder text with real accurate content about "${topic}" for ${grade} ICSE/ISC students. Return ONLY the HTML.`;
}

// ── Bank prompt ───────────────────────────────────────────────────────────────
function buildBankPrompt(grade, subject, topic) {
  return `You are an ICSE/ISC board examiner. ${grade} | ${subject} | Topic: ${topic}
Generate 18 board-exam quality questions about "${topic}". Return ONLY this JSON — replace all placeholders with real content:
{"subject":"${subject}","topic":"${topic}","grade":"${grade}","questions":[
{"type":"MCQ","difficulty":"easy","question":"Real MCQ question with A) B) C) D) options about ${topic}","answer":"Correct option letter and full text","marks":1},
{"type":"MCQ","difficulty":"medium","question":"Harder MCQ about ${topic} with A) B) C) D)","answer":"Correct option","marks":1},
{"type":"Fill in the Blank","difficulty":"easy","question":"Sentence about ${topic} with ___ blank","answer":"correct word","marks":1},
{"type":"Fill in the Blank","difficulty":"medium","question":"Sentence about ${topic} with ___ and ___ blanks","answer":"word1 / word2","marks":1},
{"type":"True or False","difficulty":"easy","question":"True factual statement about ${topic}","answer":"True","marks":1},
{"type":"True or False","difficulty":"easy","question":"False statement about ${topic}","answer":"False — correct answer is X","marks":1},
{"type":"Odd One Out","difficulty":"medium","question":"A) item B) item C) item D) item — which is odd?","answer":"B) item — reason","marks":1},
{"type":"Odd One Out","difficulty":"medium","question":"A) item B) item C) item D) item — which is odd?","answer":"C) item — reason","marks":1},
{"type":"Assertion-Reason","difficulty":"medium","question":"Assertion: real claim about ${topic}. Reason: correct reason.","answer":"Both true, Reason is correct explanation","marks":2},
{"type":"Assertion-Reason","difficulty":"hard","question":"Assertion: real claim about ${topic}. Reason: incorrect reason.","answer":"Assertion true, Reason false","marks":2},
{"type":"Short Answer","difficulty":"easy","question":"Define a key term from ${topic}.","answer":"Model definition with example.","marks":1},
{"type":"Short Answer","difficulty":"medium","question":"State two differences between two concepts in ${topic}.","answer":"1. difference 2. difference","marks":2},
{"type":"Short Answer","difficulty":"medium","question":"Explain the role of a key element in ${topic}.","answer":"2-3 sentence model answer.","marks":2},
{"type":"Short Answer","difficulty":"medium","question":"Give a real-life example of ${topic}.","answer":"Example with explanation.","marks":2},
{"type":"Short Answer","difficulty":"hard","question":"Describe the process of something in ${topic}.","answer":"Step-by-step model answer.","marks":3},
{"type":"Long Answer","difficulty":"hard","question":"Explain ${topic} in detail with examples.","answer":"Full model answer covering all key aspects.","marks":5},
{"type":"Long Answer","difficulty":"hard","question":"Compare two major concepts in ${topic} using a table.","answer":"Table with 4+ comparison points and conclusion.","marks":5},
{"type":"Long Answer","difficulty":"hard","question":"Discuss the importance and applications of ${topic}.","answer":"Comprehensive answer with significance, applications, examples.","marks":6}
]}
Replace ALL placeholders with real content about "${topic}". Return ONLY valid JSON.`;
}

// ── /api/generate-all ─────────────────────────────────────────────────────────
app.post('/api/generate-all', upload.array('files', 10), async (req, res) => {
  try {
    const { subject, topic, grade } = req.body;
    const files = req.files || [];
    const hasFiles = files.length > 0;
    if (!topic && !hasFiles) return res.status(400).json({ error: 'Enter a topic or upload files.' });

    const g = grade   || 'Grade 8';
    const s = subject || '';
    const t = topic   || '';

    const [notesRaw, questions, bankRaw] = await Promise.all([
      callGroq(buildNotesPrompt(g, s, t), 4096),
      generateQuestions(g, s, t),
      callGroq(buildBankPrompt(g, s, t), 3000)
    ]);

    const notes = notesRaw.replace(/```html|```/g, '').trim();
    const bank  = safeJSON(bankRaw, { questions: [] });

    res.json({ success: true, notes, questions, bank, fromTopic: !hasFiles });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

// ── /api/questions-only ───────────────────────────────────────────────────────
app.post('/api/questions-only', async (req, res) => {
  try {
    const { subject, topic, grade } = req.body;
    if (!topic) return res.status(400).json({ error: 'Topic is required.' });
    const questions = await generateQuestions(grade || 'Grade 8', subject || '', topic);
    res.json({ success: true, questions });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

// ── /api/ask ──────────────────────────────────────────────────────────────────
app.post('/api/ask', async (req, res) => {
  try {
    const { question, context, grade } = req.body;
    const prompt = `You are a friendly encouraging tutor for ${grade || 'Grade 8'} ICSE students. Answer clearly and simply with examples.
${context ? 'Notes context: ' + context.slice(0, 800) + '\n' : ''}
Student question: ${question}`;
    const answer = await callGroq(prompt, 1024);
    res.json({ success: true, answer });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

app.get('/api/health', (_, res) => res.json({ status: 'ok', powered_by: 'Groq LLaMA (free)' }));

app.listen(PORT, () => {
  console.log(`\n🎓 StudyBot running at http://localhost:${PORT}\n`);
});

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

Create comprehensive, detailed study notes. Return ONLY the following HTML — no markdown, no backticks, no preamble:

<div class="notes-content">
  <section class="key-concepts">
    <h2>🔑 Key Concepts</h2>
    <!-- bullet list of all core ideas -->
  </section>
  <section class="detailed-notes">
    <h2>📚 Detailed Notes</h2>
    <!-- thorough explanation with sub-headings and real-world examples -->
  </section>
  <section class="memory-tricks">
    <h2>💡 Memory Tricks &amp; Mnemonics</h2>
    <!-- at least 3 memory aids with visual boxes like:
    <div class="mnemonic-box"><span class="mnemonic-letter">P</span><span class="mnemonic-word">Photosynthesis</span></div>
    -->
  </section>
  <section class="important-formulas">
    <h2>📐 Key Formulas / Definitions</h2>
    <!-- each formula in a formula-box div:
    <div class="formula-box"><div class="formula-title">Formula Name</div><div class="formula-eq">Formula here</div></div>
    -->
  </section>
  <section class="quick-summary">
    <h2>⚡ Quick Summary</h2>
    <!-- exactly 5 bullet points -->
  </section>
  <section class="infographic">
    <h2>🗺️ Concept Map</h2>
    <!-- IMPORTANT: Create a VISUAL concept map using HTML boxes and arrows like this:
    <div class="concept-map">
      <div class="cm-center">Central Topic</div>
      <div class="cm-branches">
        <div class="cm-branch">
          <div class="cm-node cm-blue">Branch 1</div>
          <div class="cm-children">
            <div class="cm-child">Detail A</div>
            <div class="cm-child">Detail B</div>
          </div>
        </div>
        <div class="cm-branch">
          <div class="cm-node cm-green">Branch 2</div>
          <div class="cm-children">
            <div class="cm-child">Detail C</div>
          </div>
        </div>
      </div>
    </div>
    Use cm-blue, cm-green, cm-amber, cm-coral, cm-mint for different branches. Make it reflect the actual topic structure.
    -->
  </section>
</div>

Use simple language for ${gradeLabel}. Return ONLY the HTML.`;

    // ── Prompt 2a: Objective Questions ───────────────────────────────────────
    const objQuestionsPrompt = `You are an expert ${gradeLabel} ICSE/CBSE examiner.
Subject: ${subjectLabel} | Topic: ${topicLabel}

Create objective questions. Return ONLY this exact JSON structure with NO markdown, NO backticks, NO extra text.
Replace all placeholder text with real questions about ${topicLabel}:

{"mcq":[{"q":"Real MCQ question about ${topicLabel}?","options":["A) first option","B) second option","C) third option","D) fourth option"],"a":"A) first option","explanation":"reason"},{"q":"Second MCQ question?","options":["A) opt1","B) opt2","C) opt3","D) opt4"],"a":"B) opt2","explanation":"reason"},{"q":"Third MCQ question?","options":["A) opt1","B) opt2","C) opt3","D) opt4"],"a":"C) opt3","explanation":"reason"},{"q":"Fourth MCQ question?","options":["A) opt1","B) opt2","C) opt3","D) opt4"],"a":"A) opt1","explanation":"reason"},{"q":"Fifth MCQ question?","options":["A) opt1","B) opt2","C) opt3","D) opt4"],"a":"D) opt4","explanation":"reason"}],"fillinblanks":[{"q":"Sentence with ________ gap about ${topicLabel}.","a":"correct word"},{"q":"Another ________ sentence.","a":"answer"},{"q":"Third ________ sentence with ________ two gaps.","a":"word1 / word2"},{"q":"Fourth ________ sentence.","a":"answer"},{"q":"Fifth ________ sentence.","a":"answer"}],"truefalse":[{"q":"True statement about ${topicLabel}.","a":"True","explanation":"reason"},{"q":"False statement students might think is true.","a":"False","explanation":"what is actually correct"},{"q":"Another true statement.","a":"True","explanation":"reason"},{"q":"Another false statement.","a":"False","explanation":"correction"},{"q":"Final true statement.","a":"True","explanation":"reason"}],"oddonesout":[{"q":"Which is the odd one out? A) item1 B) item2 C) item3 D) item4","a":"B) item2","explanation":"because it is the only one that does not belong"},{"q":"Which is the odd one out? A) item1 B) item2 C) item3 D) item4","a":"C) item3","explanation":"reason"},{"q":"Which is the odd one out? A) item1 B) item2 C) item3 D) item4","a":"A) item1","explanation":"reason"}]}`;

    // ── Prompt 2b: Subjective Questions ──────────────────────────────────────
    const subjQuestionsPrompt = `You are an expert ${gradeLabel} ICSE/CBSE examiner.
Subject: ${subjectLabel} | Topic: ${topicLabel}

Create subjective questions. Return ONLY this exact JSON with NO markdown, NO backticks, NO extra text.
Replace all placeholder text with real questions about ${topicLabel}:

{"assertionreason":[{"assertion":"A true factual claim about ${topicLabel}.","reason":"The scientific reason behind that claim.","a":"Both Assertion and Reason are true, and Reason is the correct explanation","explanation":"brief clarification"},{"assertion":"A true claim about ${topicLabel}.","reason":"A related but incorrect reason.","a":"Assertion is true but Reason is false","explanation":"what the correct reason is"},{"assertion":"Another true claim.","reason":"Another correct reason that does not explain the assertion.","a":"Both Assertion and Reason are true, but Reason is NOT the correct explanation","explanation":"clarification"}],"shortanswer":[{"q":"Define a key term from ${topicLabel} in one sentence.","a":"Complete model answer here.","marks":1},{"q":"State two important points about ${topicLabel}.","a":"1. First point. 2. Second point.","marks":2},{"q":"What is the significance of a key concept in ${topicLabel}?","a":"Model answer explaining significance.","marks":2},{"q":"Give one example of ${topicLabel} from daily life.","a":"Specific real-world example with explanation.","marks":2},{"q":"Explain how a process in ${topicLabel} works.","a":"Step-by-step explanation.","marks":3}],"longanswer":[{"q":"Explain the main concepts of ${topicLabel} in detail with examples.","a":"Comprehensive model answer covering all key aspects, definitions, processes, and examples relevant to ${topicLabel}.","marks":5},{"q":"Compare and contrast two important aspects of ${topicLabel}. Use a table or diagram in your answer.","a":"Detailed comparison with at least 4 points of difference and 2 similarities, with examples.","marks":5},{"q":"Describe the importance and applications of ${topicLabel} in real life.","a":"Detailed answer covering significance, practical applications, advantages, and any limitations.","marks":6}]}`;

    // ── Prompt 3: Question Bank ──────────────────────────────────────────────
    const bankPrompt = `You are an expert ${gradeLabel} ICSE/CBSE examiner for ${subjectLabel} — Topic: ${topicLabel}.

Generate 18 board-exam quality questions. Return ONLY valid JSON — no markdown, no backticks:
{"subject":"${subjectLabel}","topic":"${topicLabel}","grade":"${gradeLabel}","questions":[
{"type":"MCQ","difficulty":"easy","question":"MCQ question with options A) B) C) D)","answer":"Correct option","marks":1},
{"type":"MCQ","difficulty":"easy","question":"Another MCQ","answer":"Correct option","marks":1},
{"type":"Fill in the Blank","difficulty":"easy","question":"Sentence with ___ blank","answer":"correct word","marks":1},
{"type":"Fill in the Blank","difficulty":"easy","question":"Another ___ blank sentence","answer":"word","marks":1},
{"type":"True or False","difficulty":"easy","question":"A factual statement","answer":"True","marks":1},
{"type":"True or False","difficulty":"easy","question":"An incorrect statement","answer":"False — correct answer is X","marks":1},
{"type":"Odd One Out","difficulty":"medium","question":"A) item B) item C) item D) item — which is odd?","answer":"C — reason","marks":1},
{"type":"Odd One Out","difficulty":"medium","question":"A) item B) item C) item D) item — which is odd?","answer":"B — reason","marks":1},
{"type":"Assertion-Reason","difficulty":"medium","question":"Assertion: claim. Reason: reason.","answer":"Both true, Reason explains Assertion","marks":2},
{"type":"Assertion-Reason","difficulty":"hard","question":"Assertion: claim. Reason: reason.","answer":"Assertion true, Reason false","marks":2},
{"type":"Short Answer","difficulty":"easy","question":"Define key term.","answer":"Model definition.","marks":1},
{"type":"Short Answer","difficulty":"medium","question":"Explain a concept briefly.","answer":"2-3 sentence answer.","marks":2},
{"type":"Short Answer","difficulty":"medium","question":"State differences between two things.","answer":"1. diff 2. diff","marks":2},
{"type":"Short Answer","difficulty":"medium","question":"Give an example of something.","answer":"Example with explanation.","marks":2},
{"type":"Short Answer","difficulty":"hard","question":"Describe a process.","answer":"Step-by-step answer.","marks":3},
{"type":"Long Answer","difficulty":"hard","question":"Explain topic in detail.","answer":"Full model answer.","marks":5},
{"type":"Long Answer","difficulty":"hard","question":"Compare two concepts.","answer":"Detailed comparison.","marks":5},
{"type":"Long Answer","difficulty":"hard","question":"Discuss importance and applications.","answer":"Comprehensive answer.","marks":6}
]}

Replace ALL placeholder text with real content about ${topicLabel}. Return ONLY valid JSON.`;

    // ── Run all 4 in parallel ────────────────────────────────────────────────
    const [notesRaw, objQRaw, subjQRaw, bankRaw] = await Promise.all([
      callGroq(notesPrompt, 4096),
      callGroq(objQuestionsPrompt, 2048),
      callGroq(subjQuestionsPrompt, 2048),
      callGroq(bankPrompt, 3000)
    ]);

    // Clean notes
    const notes = notesRaw.replace(/```html|```/g, '').trim();

    // Parse and merge questions from both calls
    let questions = { mcq:[], fillinblanks:[], truefalse:[], oddonesout:[], assertionreason:[], shortanswer:[], longanswer:[] };
    try {
      const objQ = JSON.parse(objQRaw.replace(/```json|```/g,'').trim());
      Object.assign(questions, objQ);
    } catch(e) { console.error('Obj questions parse error:', e.message, objQRaw.slice(0,200)); }
    try {
      const subjQ = JSON.parse(subjQRaw.replace(/```json|```/g,'').trim());
      Object.assign(questions, subjQ);
    } catch(e) { console.error('Subj questions parse error:', e.message, subjQRaw.slice(0,200)); }

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

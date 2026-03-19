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
        {
          role: 'system',
          content: `You are a senior ICSE/ISC curriculum expert and examiner with 20+ years experience.
CRITICAL RULES:
1. All content MUST strictly follow the ICSE/ISC syllabus — not CBSE, not international.
2. For Science: always include SI units, scientist names with nationality and year of discovery, step-by-step derivations where applicable, solved numerical examples, and reference to ICSE practical experiments.
3. For all subjects: use ICSE-specific terminology, examples from ICSE textbooks (Selina, S.Chand, Frank), and exam-pattern questions.
4. Return ONLY valid JSON or HTML exactly as instructed. No markdown fences, no preamble, no explanation.`
        },
        { role: 'user', content: prompt }
      ],
      max_tokens: maxTokens,
      temperature: 0.3
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
    console.error('JSON parse failed:', e.message, '\nRaw:', raw.slice(0, 300));
    return fallback;
  }
}

// ── Notes prompt ──────────────────────────────────────────────────────────────
function buildNotesPrompt(grade, subject, topic) {
  const isScience = ['Physics','Chemistry','Biology'].includes(subject);
  const isMaths   = subject === 'Mathematics';

  return `ICSE ${grade} | Subject: ${subject} | Topic: ${topic}

Create thorough, exam-ready study notes following the ICSE ${grade} syllabus exactly.
${isScience ? `
SCIENCE REQUIREMENTS — include ALL of these:
- Every quantity: name, symbol, SI unit, formula. Example: Force (F), SI unit: Newton (N), F = ma
- Scientist references: full name, nationality, year. Example: "Isaac Newton (English, 1687)"
- Step-by-step derivations for every formula with each mathematical step shown
- At least 2 solved numerical examples with full working, units at each step, and final answer boxed
- ICSE practical experiment: aim, materials, procedure steps, observation, result
- Differences tables where relevant (e.g. mass vs weight, speed vs velocity)
- Laws and principles stated exactly as per ICSE syllabus
` : ''}
${isMaths ? `
MATHS REQUIREMENTS — include ALL of these:
- Every formula with proof/derivation where applicable
- At least 3 solved examples showing full step-by-step working
- Common mistakes to avoid
- ICSE board exam style problems
` : ''}

Return ONLY this HTML — no markdown, no backticks, no extra text. Fill every section with rich real ICSE content:

<div class="notes-content">

  <section class="key-concepts">
    <h2>🔑 Key Concepts</h2>
    <ul>
      <li><strong>Concept name:</strong> precise ICSE definition</li>
      <li><strong>Another concept:</strong> definition</li>
    </ul>
  </section>

  <section class="detailed-notes">
    <h2>📚 Detailed Notes</h2>

    <h3>Subtopic heading</h3>
    <p>Detailed explanation as per ICSE syllabus...</p>

    ${isScience ? `
    <h3>Key Laws / Principles</h3>
    <p><strong>Law name (Scientist Name, Nationality, Year):</strong> exact statement of the law as per ICSE.</p>

    <h3>SI Units &amp; Quantities</h3>
    <table class="notes-table">
      <thead><tr><th>Quantity</th><th>Symbol</th><th>SI Unit</th><th>Unit Symbol</th><th>Formula</th></tr></thead>
      <tbody>
        <tr><td>Quantity 1</td><td>symbol</td><td>unit name</td><td>symbol</td><td>formula</td></tr>
        <tr><td>Quantity 2</td><td>symbol</td><td>unit name</td><td>symbol</td><td>formula</td></tr>
        <tr><td>Quantity 3</td><td>symbol</td><td>unit name</td><td>symbol</td><td>formula</td></tr>
      </tbody>
    </table>

    <h3>Derivations</h3>
    <div class="derivation-block">
      <div class="derivation-title">Derivation of [formula name]</div>
      <div class="derivation-steps">
        <div class="d-step"><span class="d-num">1</span><span class="d-text">Starting equation or assumption</span></div>
        <div class="d-step"><span class="d-num">2</span><span class="d-text">Next mathematical step</span></div>
        <div class="d-step"><span class="d-num">3</span><span class="d-text">Simplification step</span></div>
        <div class="d-step"><span class="d-num">4</span><span class="d-text">Final result</span></div>
      </div>
      <div class="derivation-result">∴ Final formula here</div>
    </div>

    <h3>Solved Numericals</h3>
    <div class="numerical-block">
      <div class="num-title">Numerical 1</div>
      <div class="num-given"><strong>Given:</strong> list of given values with units</div>
      <div class="num-find"><strong>To find:</strong> what we need to calculate</div>
      <div class="num-formula"><strong>Formula:</strong> relevant formula</div>
      <div class="num-working">
        <strong>Working:</strong><br>
        Step 1: substitute values...<br>
        Step 2: calculation...<br>
        Step 3: result with units
      </div>
      <div class="num-answer">Answer: value with correct SI unit</div>
    </div>

    <h3>Comparison Table</h3>
    <table class="notes-table">
      <thead><tr><th>Basis</th><th>Concept A</th><th>Concept B</th></tr></thead>
      <tbody>
        <tr><td>Definition</td><td>definition A</td><td>definition B</td></tr>
        <tr><td>SI Unit</td><td>unit A</td><td>unit B</td></tr>
        <tr><td>Formula</td><td>formula A</td><td>formula B</td></tr>
        <tr><td>Example</td><td>example A</td><td>example B</td></tr>
      </tbody>
    </table>

    <h3>🔬 ICSE Practical Experiment</h3>
    <div class="experiment-block">
      <div class="exp-row"><span class="exp-label">Aim:</span><span class="exp-content">aim of experiment</span></div>
      <div class="exp-row"><span class="exp-label">Materials Required:</span><span class="exp-content">list of apparatus</span></div>
      <div class="exp-row"><span class="exp-label">Procedure:</span>
        <ol class="exp-steps">
          <li>Step 1</li>
          <li>Step 2</li>
          <li>Step 3</li>
        </ol>
      </div>
      <div class="exp-row"><span class="exp-label">Observation:</span><span class="exp-content">what is observed</span></div>
      <div class="exp-row"><span class="exp-label">Result / Conclusion:</span><span class="exp-content">conclusion</span></div>
      <div class="exp-row"><span class="exp-label">Precautions:</span><span class="exp-content">list precautions</span></div>
    </div>

    <h3>Labelled Diagram</h3>
    <div class="diagram-box">
      <div class="diagram-title">Diagram: [name of diagram]</div>
      <div class="diagram-description">
        <p><strong>Description:</strong> detailed description of what the diagram shows</p>
        <div class="diagram-labels">
          <div class="diagram-part"><span class="part-num">1</span><span class="part-name">Part name</span><span class="part-desc">— its function or description</span></div>
          <div class="diagram-part"><span class="part-num">2</span><span class="part-name">Part name</span><span class="part-desc">— its function</span></div>
          <div class="diagram-part"><span class="part-num">3</span><span class="part-name">Part name</span><span class="part-desc">— its function</span></div>
          <div class="diagram-part"><span class="part-num">4</span><span class="part-name">Part name</span><span class="part-desc">— its function</span></div>
        </div>
      </div>
    </div>
    ` : ''}

    ${isMaths ? `
    <h3>Solved Examples</h3>
    <div class="numerical-block">
      <div class="num-title">Example 1</div>
      <div class="num-given"><strong>Question:</strong> ICSE-style problem statement</div>
      <div class="num-working"><strong>Solution:</strong><br>Step 1: ...<br>Step 2: ...<br>Step 3: ...</div>
      <div class="num-answer">Answer: final answer</div>
    </div>
    ` : ''}

  </section>

  <section class="memory-tricks">
    <h2>💡 Memory Tricks &amp; Mnemonics</h2>
    <p><strong>Mnemonic for [concept]:</strong></p>
    <div class="mnemonic-row">
      <div class="mnemonic-box"><span class="mnemonic-letter">A</span><span class="mnemonic-word">Word for A</span></div>
      <div class="mnemonic-box"><span class="mnemonic-letter">B</span><span class="mnemonic-word">Word for B</span></div>
      <div class="mnemonic-box"><span class="mnemonic-letter">C</span><span class="mnemonic-word">Word for C</span></div>
    </div>
    <p><strong>Tip:</strong> another memory trick</p>
    <p><strong>Tip:</strong> another technique</p>
  </section>

  <section class="important-formulas">
    <h2>📐 Key Formulas &amp; Definitions</h2>
    <div class="formula-box"><div class="formula-title">Formula / Term Name</div><div class="formula-eq">Formula or exact ICSE definition</div></div>
    <div class="formula-box"><div class="formula-title">Another Formula</div><div class="formula-eq">Formula with SI unit noted</div></div>
  </section>

  <section class="quick-summary">
    <h2>⚡ Quick Revision Summary</h2>
    <ul>
      <li>Key point 1 — specific fact</li>
      <li>Key point 2 — specific fact</li>
      <li>Key point 3 — specific fact</li>
      <li>Key point 4 — specific fact</li>
      <li>Key point 5 — specific fact</li>
    </ul>
  </section>

  <section class="infographic">
    <h2>🗺️ Concept Map</h2>
    <div class="concept-map">
      <div class="cm-center">${topic}</div>
      <div class="cm-branches">
        <div class="cm-branch">
          <div class="cm-node cm-blue">Branch label 1</div>
          <div class="cm-children">
            <div class="cm-child">specific detail A</div>
            <div class="cm-child">specific detail B</div>
          </div>
        </div>
        <div class="cm-branch">
          <div class="cm-node cm-green">Branch label 2</div>
          <div class="cm-children">
            <div class="cm-child">specific detail C</div>
            <div class="cm-child">specific detail D</div>
          </div>
        </div>
        <div class="cm-branch">
          <div class="cm-node cm-amber">Branch label 3</div>
          <div class="cm-children">
            <div class="cm-child">specific detail E</div>
            <div class="cm-child">specific detail F</div>
          </div>
        </div>
        <div class="cm-branch">
          <div class="cm-node cm-coral">Branch label 4</div>
          <div class="cm-children">
            <div class="cm-child">specific detail G</div>
          </div>
        </div>
      </div>
    </div>
  </section>

</div>

Replace ALL placeholder text with specific, accurate ICSE ${grade} content about "${topic}".
Return ONLY the HTML. No markdown.`;
}

// ── Question prompts ──────────────────────────────────────────────────────────
function buildQPromptA(grade, subject, topic) {
  const isScience = ['Physics','Chemistry','Biology'].includes(subject);
  const isMaths   = subject === 'Mathematics';
  return `ICSE ${grade} | ${subject} | Topic: ${topic}

Write 5 MCQs and 5 fill-in-the-blank questions strictly based on the ICSE ${grade} ${subject} syllabus for "${topic}".
${isScience ? 'Include questions on SI units, scientist names, and definitions as per ICSE.' : ''}
${isMaths ? 'Include numerical-based MCQs with calculation required.' : ''}

Return ONLY this JSON with real ICSE-standard content — no placeholders:
{"mcq":[
{"q":"Complete question 1 about ${topic} as per ICSE?","options":["A) first option","B) second option","C) third option","D) fourth option"],"a":"A) first option","explanation":"ICSE-specific reason why this is correct"},
{"q":"Complete question 2?","options":["A) opt","B) opt","C) opt","D) opt"],"a":"B) opt","explanation":"reason"},
{"q":"Complete question 3?","options":["A) opt","B) opt","C) opt","D) opt"],"a":"C) opt","explanation":"reason"},
{"q":"Complete question 4?","options":["A) opt","B) opt","C) opt","D) opt"],"a":"D) opt","explanation":"reason"},
{"q":"Complete question 5?","options":["A) opt","B) opt","C) opt","D) opt"],"a":"A) opt","explanation":"reason"}
],"fillinblanks":[
{"q":"The SI unit of ________ is ________.","a":"quantity / unit"},
{"q":"________ stated that ________ (year).","a":"scientist name / law or principle"},
{"q":"The formula for ________ is ________.","a":"quantity / formula"},
{"q":"________ and ________ are the two types of ________.","a":"type1 / type2 / category"},
{"q":"In the ICSE experiment to ________, the main observation is ________.","a":"experiment aim / observation"}
]}`;
}

function buildQPromptB(grade, subject, topic) {
  const isScience = ['Physics','Chemistry','Biology'].includes(subject);
  return `ICSE ${grade} | ${subject} | Topic: ${topic}

Write 5 true/false statements and 3 odd-one-out questions strictly based on ICSE ${grade} ${subject} syllabus for "${topic}".
${isScience ? 'Include statements about SI units, scientists, experiments, and ICSE-specific facts.' : ''}

Return ONLY this JSON with real specific content:
{"truefalse":[
{"q":"Specific true ICSE fact about ${topic}.","a":"True","explanation":"ICSE textbook reason"},
{"q":"A common misconception about ${topic} that ICSE students make.","a":"False","explanation":"The correct ICSE fact is: ..."},
{"q":"Another specific true statement about ${topic} as per ICSE syllabus.","a":"True","explanation":"reason"},
{"q":"An incorrect statement about ${topic} that tests ICSE knowledge.","a":"False","explanation":"Correction: ..."},
{"q":"A true statement about ${topic} that requires ICSE-level understanding.","a":"True","explanation":"reason"}
],"oddonesout":[
{"q":"Which is the odd one out? A) item1 B) item2 C) item3 D) item4","a":"B) item2","explanation":"It is the only one that does not belong because..."},
{"q":"Which is the odd one out? A) item1 B) item2 C) item3 D) item4","a":"C) item3","explanation":"reason"},
{"q":"Which is the odd one out? A) item1 B) item2 C) item3 D) item4","a":"D) item4","explanation":"reason"}
]}`;
}

function buildQPromptC(grade, subject, topic) {
  const isScience = ['Physics','Chemistry','Biology'].includes(subject);
  const isMaths   = subject === 'Mathematics';
  return `ICSE ${grade} | ${subject} | Topic: ${topic}

Write 3 assertion-reason, 5 short-answer, and 3 long-answer questions strictly as per ICSE ${grade} ${subject} syllabus for "${topic}".
${isScience ? `
- Include at least 1 experiment-based question (ICSE practical).
- Include at least 1 numerical problem with given values and SI units.
- Short answers should match ICSE marking scheme (1-3 marks).
- Long answers should require diagrams, derivations, or experiments as per ICSE board pattern.
` : ''}
${isMaths ? `
- Include numerical problems requiring step-by-step working.
- Match ICSE board exam question style.
` : ''}

Return ONLY this JSON with specific real ICSE content — no placeholders:
{"assertionreason":[
{"assertion":"Specific true ICSE-level claim about ${topic}.","reason":"The correct scientific/mathematical reason behind that claim.","a":"Both Assertion and Reason are true, and Reason is the correct explanation","explanation":"Why R correctly explains A"},
{"assertion":"Another true ICSE claim about ${topic}.","reason":"A plausible but incorrect reason.","a":"Assertion is true but Reason is false","explanation":"The correct reason is: ..."},
{"assertion":"A true ICSE-level claim about ${topic}.","reason":"A true but unrelated reason.","a":"Both Assertion and Reason are true, but Reason is NOT the correct explanation","explanation":"What actually explains A"}
],"shortanswer":[
{"q":"Define [ICSE-specific term from ${topic}] and state its SI unit.","a":"ICSE definition. SI unit: [unit] ([symbol]).","marks":2},
{"q":"State [law or principle from ${topic}] and give its mathematical expression.","a":"Statement of law as per ICSE. Mathematical form: formula with all symbols defined.","marks":2},
{"q":"${isScience ? 'In the ICSE experiment related to '+topic+', state the aim and one precaution.' : 'State and prove: ICSE-standard result related to '+topic+'.'}","a":"${isScience ? 'Aim: ... Precaution: ...' : 'Statement and proof.'}","marks":2},
{"q":"Distinguish between [concept A] and [concept B] in ${topic}. [ICSE tabular format]","a":"<table class='ans-table'><tr><th>Basis</th><th>Concept A</th><th>Concept B</th></tr><tr><td>Definition</td><td>def A</td><td>def B</td></tr><tr><td>SI Unit</td><td>unit A</td><td>unit B</td></tr><tr><td>Example</td><td>eg A</td><td>eg B</td></tr></table>","marks":3},
{"q":"${isScience ? 'A numerical: A body has [given values with SI units]. Calculate [quantity to find].' : 'Solve: ICSE board-style numerical on '+topic+'.'}","a":"Given: values with units. Formula: ... Working: step-by-step. Answer: value + SI unit.","marks":3}
],"longanswer":[
{"q":"${isScience ? 'Describe the ICSE experiment to study '+topic+'. Include aim, labelled diagram (described), procedure, observation, and conclusion.' : 'Explain '+topic+' in detail with proof, examples, and ICSE board-level working.'}","a":"${isScience ? 'Aim: ... Apparatus: ... Labelled diagram: [describe parts]. Procedure: steps 1-5. Observation: ... Result: ... Precautions: ...' : 'Introduction. Theorem/concept. Proof with steps. ICSE examples. Conclusion.'}","marks":5},
{"q":"With the help of a labelled diagram, explain [main mechanism or process in ${topic}]. State all relevant laws, formulas with SI units, and give one solved numerical.","a":"Diagram description with labelled parts. Laws involved. Formulae with SI units. Derivation if applicable. Solved numerical: Given → Formula → Working → Answer.","marks":6},
{"q":"Compare and contrast [two major concepts in ${topic}] using a table with at least 5 points. Then solve: [ICSE board-style numerical].","a":"<table class='ans-table'><tr><th>Point</th><th>Concept A</th><th>Concept B</th></tr><tr><td>Definition</td><td>...</td><td>...</td></tr><tr><td>Formula</td><td>...</td><td>...</td></tr><tr><td>SI Unit</td><td>...</td><td>...</td></tr><tr><td>Example</td><td>...</td><td>...</td></tr><tr><td>Application</td><td>...</td><td>...</td></tr></table>Numerical solution: Given → Working → Answer with unit.","marks":6}
]}`;
}

// ── Bank prompt ───────────────────────────────────────────────────────────────
function buildBankPrompt(grade, subject, topic) {
  const isScience = ['Physics','Chemistry','Biology'].includes(subject);
  return `ICSE ${grade} | ${subject} | Topic: ${topic}

Generate 18 ICSE board-exam quality questions about "${topic}" strictly as per ICSE ${grade} ${subject} syllabus.
${isScience ? 'Include questions on: SI units, scientist names, ICSE practicals, derivations, and numericals.' : ''}
Use ICSE board exam language and marking scheme exactly.

Return ONLY this JSON — replace all placeholders with real ICSE ${grade} content about "${topic}":
{"subject":"${subject}","topic":"${topic}","grade":"${grade}","questions":[
{"type":"MCQ","difficulty":"easy","question":"ICSE-standard MCQ with A) B) C) D) options","answer":"Correct option","marks":1},
{"type":"MCQ","difficulty":"medium","question":"MCQ requiring ICSE-level understanding, with A) B) C) D)","answer":"Correct option with brief reason","marks":1},
{"type":"Fill in the Blank","difficulty":"easy","question":"The SI unit of ___ is ___.","answer":"quantity / unit","marks":1},
{"type":"Fill in the Blank","difficulty":"medium","question":"___ stated the law of ___ in the year ___.","answer":"scientist / law / year","marks":1},
{"type":"True or False","difficulty":"easy","question":"A specific ICSE fact about ${topic}","answer":"True","marks":1},
{"type":"True or False","difficulty":"medium","question":"A common ICSE misconception about ${topic}","answer":"False — correct ICSE fact is: ...","marks":1},
{"type":"Odd One Out","difficulty":"medium","question":"A) item B) item C) item D) item","answer":"X) item — reason based on ICSE classification","marks":1},
{"type":"Odd One Out","difficulty":"medium","question":"A) item B) item C) item D) item","answer":"X) item — reason","marks":1},
{"type":"Assertion-Reason","difficulty":"medium","question":"Assertion: ICSE fact. Reason: correct reason.","answer":"Both true, Reason is correct explanation","marks":2},
{"type":"Assertion-Reason","difficulty":"hard","question":"Assertion: ICSE fact. Reason: incorrect reason.","answer":"Assertion true, Reason false — correct reason is: ...","marks":2},
{"type":"Short Answer","difficulty":"easy","question":"Define [term] as per ICSE. State its SI unit.","answer":"ICSE definition. SI unit: ...","marks":1},
{"type":"Short Answer","difficulty":"medium","question":"State [ICSE law/principle] and write its mathematical form.","answer":"Law statement. Formula: ... where symbols mean ...","marks":2},
{"type":"Short Answer","difficulty":"medium","question":"Distinguish between [A] and [B] in tabular form. (2 points)","answer":"Table: Basis | A | B — row1 — row2","marks":2},
{"type":"Short Answer","difficulty":"medium","question":"${isScience ? 'State the aim and one precaution of the ICSE experiment on '+topic+'.' : 'State and briefly prove: key result in '+topic+'.'}","answer":"${isScience ? 'Aim: ... Precaution: ...' : 'Statement. Proof steps.'}","marks":2},
{"type":"Short Answer","difficulty":"hard","question":"${isScience ? 'A numerical problem on '+topic+' with given values and SI units.' : 'ICSE board numerical on '+topic+'.'}","answer":"Given: ... Formula: ... Working: ... Answer: value + SI unit","marks":3},
{"type":"Long Answer","difficulty":"hard","question":"${isScience ? 'Describe the ICSE experiment to study '+topic+'. Include aim, apparatus, procedure, observation and conclusion.' : 'Explain '+topic+' fully with proof and ICSE examples.'}","answer":"Full ICSE model answer with all parts.","marks":5},
{"type":"Long Answer","difficulty":"hard","question":"With a neat labelled diagram, explain [main concept in ${topic}]. State relevant laws and derive the main formula.","answer":"Diagram (described with labels). Laws. Derivation step by step. Final formula with SI unit.","marks":5},
{"type":"Long Answer","difficulty":"hard","question":"Compare [concept A] and [concept B] in ${topic} using a table (5 points). Solve a numerical: [problem statement with values].","answer":"5-row comparison table. Numerical: Given → Formula → Working → Answer.","marks":6}
]}
Replace ALL content with specific real ICSE ${grade} ${subject} content about "${topic}". Return ONLY JSON.`;
}

// ── Generate all 7 question types ─────────────────────────────────────────────
async function generateQuestions(grade, subject, topic) {
  const [rawA, rawB, rawC] = await Promise.all([
    callGroq(buildQPromptA(grade, subject, topic), 1800),
    callGroq(buildQPromptB(grade, subject, topic), 1400),
    callGroq(buildQPromptC(grade, subject, topic), 2500)
  ]);
  const questions = { mcq:[], fillinblanks:[], truefalse:[], oddonesout:[], assertionreason:[], shortanswer:[], longanswer:[] };
  Object.assign(questions, safeJSON(rawA, {}));
  Object.assign(questions, safeJSON(rawB, {}));
  Object.assign(questions, safeJSON(rawC, {}));
  return questions;
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
      callGroq(buildNotesPrompt(g, s, t), 6000),
      generateQuestions(g, s, t),
      callGroq(buildBankPrompt(g, s, t), 3500)
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
    const { question, context, grade, subject } = req.body;
    const prompt = `You are a friendly ICSE ${grade || 'Grade 8'} ${subject || ''} tutor. Answer using ICSE syllabus terminology and examples.
${context ? 'Context from notes: ' + context.slice(0, 800) + '\n' : ''}
Student question: ${question}
Give a clear, encouraging answer with ICSE-relevant examples and, where applicable, SI units or formulas.`;
    const answer = await callGroq(prompt, 1024);
    res.json({ success: true, answer });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

app.get('/api/health', (_, res) => res.json({ status: 'ok', powered_by: 'Groq LLaMA — ICSE curriculum' }));

app.listen(PORT, () => {
  console.log(`\n🎓 StudyBot (ICSE) running at http://localhost:${PORT}\n`);
});

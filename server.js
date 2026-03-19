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

// ── Groq helper with automatic fallback ──────────────────────────────────────
// Primary: llama-3.3-70b-versatile (100k tokens/day — best quality)
// Fallback: llama-3.1-8b-instant   (500k tokens/day — kicks in when primary hits limit)

const SYSTEM_PROMPT = `You are a senior ICSE/ISC curriculum expert and examiner with 20+ years experience.
CRITICAL RULES:
1. All content MUST strictly follow the ICSE/ISC syllabus — not CBSE, not international.
2. For Science: always include SI units, scientist names with nationality and year of discovery, step-by-step derivations where applicable, solved numerical examples, and reference to ICSE practical experiments.
3. For all subjects: use ICSE-specific terminology, examples from ICSE textbooks (Selina, S.Chand, Frank), and exam-pattern questions.
4. Return ONLY valid JSON or HTML exactly as instructed. No markdown fences, no preamble, no explanation.`;

async function groqCall(model, prompt, maxTokens) {
  const response = await fetch('https://api.groq.com/openai/v1/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${process.env.GROQ_API_KEY}` },
    body: JSON.stringify({
      model,
      messages: [
        { role: 'system', content: SYSTEM_PROMPT },
        { role: 'user',   content: prompt }
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

async function callGroq(prompt, maxTokens = 2048) {
  try {
    return await groqCall('llama-3.3-70b-versatile', prompt, maxTokens);
  } catch(e) {
    const isRateLimit = e.message.includes('rate_limit') || e.message.includes('quota') || e.message.includes('Limit') || e.message.includes('429');
    if (isRateLimit) {
      console.log('Primary model rate limited — switching to llama-3.1-8b-instant (500k TPD)');
      return await groqCall('llama-3.1-8b-instant', prompt, maxTokens);
    }
    throw e;
  }
}

// ── Safe JSON parse ───────────────────────────────────────────────────────────
function safeJSON(raw, fallback = {}) {
  try {
    const cleaned = raw.replace(/```json|```/g, '').trim();
    const start = cleaned.indexOf('{');
    const end   = cleaned.lastIndexOf('}');
    if (start === -1 || end === -1) throw new Error('No JSON object found');
    return JSON.parse(cleaned.slice(start, end + 1));
  } catch(e) {
    console.error('JSON parse failed:', e.message, '\nRaw preview:', raw.slice(0, 300));
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
  return `You are an ICSE ${grade} ${subject} examiner. Write questions about: ${topic}

TASK: Write exactly 5 MCQs and exactly 5 fill-in-the-blank questions based on ICSE ${grade} ${subject} syllabus.
${isScience ? 'For Science: test SI units, scientist names with years, ICSE definitions.' : ''}
${isMaths   ? 'For Maths: include calculation-based MCQs.' : ''}

Output ONLY a JSON object. Start with { and end with }. No other text before or after.

Example format (replace with real ${topic} content):
{"mcq":[{"q":"What is the SI unit of force?","options":["A) Joule","B) Newton","C) Watt","D) Pascal"],"a":"B) Newton","explanation":"Force is measured in Newtons (N) as per ICSE, named after Isaac Newton."},{"q":"Q2 here?","options":["A) opt","B) opt","C) opt","D) opt"],"a":"A) opt","explanation":"reason"},{"q":"Q3?","options":["A) opt","B) opt","C) opt","D) opt"],"a":"C) opt","explanation":"reason"},{"q":"Q4?","options":["A) opt","B) opt","C) opt","D) opt"],"a":"D) opt","explanation":"reason"},{"q":"Q5?","options":["A) opt","B) opt","C) opt","D) opt"],"a":"B) opt","explanation":"reason"}],"fillinblanks":[{"q":"The SI unit of ________ is ________.","a":"force / Newton"},{"q":"________ (nationality, year) discovered ________.","a":"scientist / concept"},{"q":"The formula for ________ is ________.","a":"quantity / formula"},{"q":"________ and ________ are two types of ________.","a":"type1 / type2 / category"},{"q":"In ICSE, the experiment to verify ________ uses ________ as the main apparatus.","a":"law / apparatus"}]}

Now write the actual questions about ${topic} for ICSE ${grade} ${subject}. Output JSON only.`;
}

function buildQPromptB(grade, subject, topic) {
  const isScience = ['Physics','Chemistry','Biology'].includes(subject);
  return `You are an ICSE ${grade} ${subject} examiner. Write questions about: ${topic}

TASK: Write exactly 5 true/false statements and exactly 3 odd-one-out questions based on ICSE ${grade} ${subject} syllabus.
${isScience ? 'Test ICSE-specific facts about SI units, scientists, experiments.' : ''}

Output ONLY a JSON object. Start with { and end with }. No other text before or after.

Example format (replace with real ${topic} content):
{"truefalse":[{"q":"The SI unit of velocity is metre per second.","a":"True","explanation":"Velocity = displacement/time, SI unit is m/s as per ICSE."},{"q":"Mass and weight are the same quantity.","a":"False","explanation":"Mass is the amount of matter (kg); weight is gravitational force on mass (N)."},{"q":"T3 statement here.","a":"True","explanation":"reason"},{"q":"T4 statement here.","a":"False","explanation":"correction"},{"q":"T5 statement here.","a":"True","explanation":"reason"}],"oddonesout":[{"q":"Which is the odd one out? A) item1  B) item2  C) item3  D) item4","a":"B) item2","explanation":"item2 is the only one that does not belong because..."},{"q":"Which is the odd one out? A) x  B) y  C) z  D) w","a":"C) z","explanation":"reason"},{"q":"Which is the odd one out? A) p  B) q  C) r  D) s","a":"D) s","explanation":"reason"}]}

Now write the actual questions about ${topic} for ICSE ${grade} ${subject}. Output JSON only.`;
}

function buildQPromptC(grade, subject, topic) {
  const isScience = ['Physics','Chemistry','Biology'].includes(subject);
  const isMaths   = subject === 'Mathematics';
  return `You are an ICSE ${grade} ${subject} examiner. Write questions about: ${topic}

TASK: Write exactly 3 assertion-reason questions, exactly 5 short-answer questions, and exactly 3 long-answer questions based on ICSE ${grade} ${subject} syllabus.
${isScience ? 'Include: 1 experiment question, 1 numerical with given values and SI units, 1 table-based distinction question.' : ''}
${isMaths   ? 'Include: numerical problems with step-by-step working required.' : ''}

IMPORTANT for short and long answers: the "a" field must contain a complete model answer, not a placeholder.
For tabular answers, write the table content as plain text like: "Basis | Concept A | Concept B --- Definition | def A | def B --- SI Unit | unit A | unit B"
For numericals, write: "Given: ... Formula: ... Working: ... Answer: value + unit"

Output ONLY a JSON object. Start with { and end with }. No other text before or after.

{"assertionreason":[{"assertion":"A true factual claim about ${topic} as per ICSE.","reason":"The correct scientific reason for that claim.","a":"Both Assertion and Reason are true, and Reason is the correct explanation","explanation":"Brief clarification of why R explains A"},{"assertion":"Another true claim about ${topic}.","reason":"A wrong reason that sounds plausible.","a":"Assertion is true but Reason is false","explanation":"The correct reason is: write it here"},{"assertion":"A third true ICSE claim about ${topic}.","reason":"A true statement that does not explain the assertion.","a":"Both Assertion and Reason are true, but Reason is NOT the correct explanation","explanation":"What actually explains the assertion"}],"shortanswer":[{"q":"Define [key ICSE term from ${topic}] and state its SI unit.","a":"Definition as per ICSE syllabus. SI unit: unit name (symbol).","marks":2},{"q":"State [law or principle from ${topic}] and write its mathematical expression.","a":"State the law exactly as in ICSE textbook. Mathematical form: write formula and define each symbol.","marks":2},{"q":"${isScience ? 'State the aim and one important precaution for the ICSE experiment on '+topic+'.' : 'State and briefly prove a key result in '+topic+'.'}","a":"${isScience ? 'Aim: write the complete aim. Precaution: write one specific precaution.' : 'State the result. Proof: step-by-step.'}","marks":2},{"q":"Distinguish between [concept A from ${topic}] and [concept B from ${topic}] on any three bases.","a":"Basis 1: A differs from B in ... | Basis 2: A has ... while B has ... | Basis 3: Example of A is ... while B is ...","marks":3},{"q":"${isScience ? 'A numerical problem: state specific values with SI units related to '+topic+' and ask student to calculate a specific quantity.' : 'Solve a step-by-step ICSE board problem on '+topic+'.'}","a":"Given: list values with units. Formula: write formula. Working: step 1 calculation, step 2 calculation. Answer: numerical value with correct SI unit.","marks":3}],"longanswer":[{"q":"${isScience ? 'Describe the ICSE experiment to study '+topic+'. Include aim, apparatus, procedure (5 steps), observation, result and two precautions.' : 'Explain '+topic+' in detail with proof, ICSE examples and board-level working.'}","a":"${isScience ? 'Aim: complete aim statement. Apparatus: list items. Procedure: 1. step one 2. step two 3. step three 4. step four 5. step five. Observation: write what is observed. Result: write conclusion. Precautions: 1. precaution one 2. precaution two.' : 'Introduction. Main concept explanation. Proof with mathematical steps. Three examples. Conclusion.'}","marks":5},{"q":"With the help of a labelled diagram, explain [main concept in ${topic}]. State the relevant law, derive the key formula showing each step, and solve one numerical.","a":"Diagram: describe each labelled part and its function. Law: state it exactly. Derivation: step 1 → step 2 → step 3 → final formula with SI unit. Numerical: Given values with units → Formula → Working → Answer with unit.","marks":6},{"q":"Compare [concept A] and [concept B] in ${topic} on five different bases in tabular form. Hence solve: write a specific numerical problem with given values.","a":"Table: Basis | Concept A | Concept B --- Definition | def A | def B --- Formula | formula A | formula B --- SI Unit | unit A | unit B --- Example | example A | example B --- Application | use A | use B. Numerical: Given → Formula → Working → Answer.","marks":6}]}

Now write the actual questions about ${topic} for ICSE ${grade} ${subject}. Output JSON only.`;
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
    callGroq(buildQPromptC(grade, subject, topic), 2800)
  ]);

  const questions = { mcq:[], fillinblanks:[], truefalse:[], oddonesout:[], assertionreason:[], shortanswer:[], longanswer:[] };

  const pA = safeJSON(rawA, null);
  const pB = safeJSON(rawB, null);
  const pC = safeJSON(rawC, null);

  if (pA) { Object.assign(questions, pA); } else { console.error('PromptA parse FAILED. Raw:', rawA.slice(0,400)); }
  if (pB) { Object.assign(questions, pB); } else { console.error('PromptB parse FAILED. Raw:', rawB.slice(0,400)); }
  if (pC) { Object.assign(questions, pC); } else { console.error('PromptC parse FAILED. Raw:', rawC.slice(0,400)); }

  console.log(`Questions generated — mcq:${questions.mcq.length} fib:${questions.fillinblanks.length} tf:${questions.truefalse.length} ooo:${questions.oddonesout.length} ar:${questions.assertionreason.length} sa:${questions.shortanswer.length} la:${questions.longanswer.length}`);

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

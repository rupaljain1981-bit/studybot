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

// ── Subject classification ────────────────────────────────────────────────────
const SCIENCE  = ['Physics','Chemistry','Biology','Environmental Science'];
const MATHS    = ['Mathematics'];
const HUMANITY = ['History & Civics','History','Geography','Economics','Commercial Studies','Accountancy','Business Studies','Psychology','Sociology','Political Science'];
const LANGUAGE = ['English Language','English Literature','Hindi','English'];
const COMP     = ['Computer Applications','Computer Science'];

function subjectType(s) {
  if (SCIENCE.includes(s))  return 'science';
  if (MATHS.includes(s))    return 'maths';
  if (HUMANITY.includes(s)) return 'humanity';
  if (LANGUAGE.includes(s)) return 'language';
  if (COMP.includes(s))     return 'computer';
  return 'general';
}

// ── Groq helper ───────────────────────────────────────────────────────────────
const SYSTEM_PROMPT = `You are a senior ICSE/ISC curriculum expert, examiner, and question paper setter with 20+ years of experience setting and marking actual ICSE board papers.
CRITICAL RULES:
1. Follow ICSE/ISC syllabus and exam paper patterns EXACTLY — not CBSE, not international boards.
2. Science: include SI units, scientist names (nationality, year), step-by-step derivations, solved numericals, ICSE practical experiments.
3. Maths: follow ICSE board paper structure — Section A short numericals, Section B long working problems. Show every step.
4. History & Civics / Geography / Economics: follow actual ICSE exam structure — source analysis, timeline questions, cause-effect, significance questions, structured essays.
5. Questions must be genuinely exam-worthy — not trivial. Match real ICSE board difficulty and language.
6. Return ONLY valid JSON or HTML exactly as instructed. No markdown fences, no preamble, no explanation.`;

async function groqCall(model, prompt, maxTokens) {
  const res = await fetch('https://api.groq.com/openai/v1/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${process.env.GROQ_API_KEY}` },
    body: JSON.stringify({
      model,
      messages: [{ role:'system', content:SYSTEM_PROMPT }, { role:'user', content:prompt }],
      max_tokens: maxTokens,
      temperature: 0.3
    })
  });
  if (!res.ok) { const e = await res.json(); throw new Error(e.error?.message || 'Groq error'); }
  const data = await res.json();
  return data.choices[0].message.content;
}

const sleep = ms => new Promise(r => setTimeout(r, ms));

function retryAfterMs(msg) {
  const m = msg.match(/try again in ([\d.]+)s/i);
  return m ? Math.ceil(parseFloat(m[1]) * 1000) + 300 : 2500;
}

function isRateLimit(msg) {
  return msg.includes('rate_limit') || msg.includes('quota') ||
         msg.includes('Limit') || msg.includes('429') ||
         msg.includes('exceeded') || msg.includes('try again in');
}

async function callGroq(prompt, maxTokens = 2048) {
  try {
    return await groqCall('llama-3.3-70b-versatile', prompt, maxTokens);
  } catch(e) {
    if (!isRateLimit(e.message)) throw e;
    console.log('70b rate limited -- switching to llama-3.1-8b-instant');
  }

  for (let attempt = 1; attempt <= 3; attempt++) {
    try {
      return await groqCall('llama-3.1-8b-instant', prompt, maxTokens);
    } catch(e) {
      if (isRateLimit(e.message) && attempt < 3) {
        const wait = retryAfterMs(e.message);
        console.log('Fallback TPM limit -- retrying in ' + wait + 'ms (attempt ' + (attempt+1) + '/3)');
        await sleep(wait);
      } else {
        throw new Error('Service is busy. Please wait 10 seconds and try again.');
      }
    }
  }
}

function safeJSON(raw, fallback = {}) {
  try {
    const cleaned = raw.replace(/```json|```/g,'').trim();
    const s = cleaned.indexOf('{'), e = cleaned.lastIndexOf('}');
    if (s===-1||e===-1) throw new Error('No JSON found');
    return JSON.parse(cleaned.slice(s, e+1));
  } catch(e) {
    console.error('JSON parse failed:', e.message, '\nRaw:', raw.slice(0,300));
    return fallback;
  }
}

// ── NOTES PROMPT ──────────────────────────────────────────────────────────────
function buildNotesPrompt(grade, subject, topic) {
  const type = subjectType(subject);

  const scienceExtra = `
SCIENCE — INCLUDE ALL:
- Every quantity: symbol, SI unit, unit symbol, formula
- Scientist name, nationality, year for each law/principle
- Full step-by-step derivations (show each algebraic step)
- Minimum 2 solved ICSE-style numericals (Given/Find/Formula/Working/Answer)
- ICSE practical experiment: Aim, Apparatus, Procedure, Observation, Result, Precautions
- Comparison table for related concepts (e.g. speed vs velocity, mass vs weight)
- Labelled diagram description with numbered parts and functions`;

  const mathsExtra = `
MATHEMATICS — INCLUDE ALL:
- Every formula/theorem with proof
- Minimum 3 solved ICSE board-style problems showing every step
- Common mistakes students make in exams
- ICSE board exam shortcuts and tips`;

  const humanityExtra = `
HISTORY & CIVICS / GEOGRAPHY / ECONOMICS — INCLUDE ALL:
- Key dates, events, people, places in a structured timeline or table
- Causes and effects clearly separated
- Significance and impact sections
- Comparison tables where relevant (e.g. Lok Sabha vs Rajya Sabha)
- Maps / diagrams described in text
- ICSE source-based question format notes (how to analyse a source)
- Key terms defined exactly as per ICSE textbook`;

  const extras = type==='science' ? scienceExtra : type==='maths' ? mathsExtra : type==='humanity' ? humanityExtra : '';

  return `ICSE ${grade} | ${subject} | Topic: ${topic}
${extras}

Create comprehensive, exam-ready study notes strictly following the ICSE ${grade} ${subject} syllabus.
Return ONLY this HTML — replace every placeholder with real ICSE content. No markdown, no backticks:

<div class="notes-content">
  <section class="key-concepts">
    <h2>🔑 Key Concepts</h2>
    <ul><li><strong>Term:</strong> ICSE definition</li><li><strong>Term:</strong> definition</li></ul>
  </section>
  <section class="detailed-notes">
    <h2>📚 Detailed Notes</h2>
    <h3>Subtopic 1</h3><p>Detailed ICSE-syllabus explanation...</p>
    ${type==='science' ? `
    <h3>Key Laws / Principles</h3>
    <p><strong>Law Name (Scientist, Nationality, Year):</strong> Exact ICSE statement.</p>
    <h3>SI Units &amp; Formulae</h3>
    <table class="notes-table"><thead><tr><th>Quantity</th><th>Symbol</th><th>SI Unit</th><th>Symbol</th><th>Formula</th></tr></thead>
    <tbody><tr><td>quantity</td><td>sym</td><td>unit</td><td>sym</td><td>formula</td></tr></tbody></table>
    <h3>Derivation</h3>
    <div class="derivation-block"><div class="derivation-title">Derivation of [formula]</div>
    <div class="derivation-steps"><div class="d-step"><span class="d-num">1</span><span class="d-text">Starting point</span></div>
    <div class="d-step"><span class="d-num">2</span><span class="d-text">Step 2</span></div>
    <div class="d-step"><span class="d-num">3</span><span class="d-text">Final step</span></div></div>
    <div class="derivation-result">∴ Final formula with SI unit</div></div>
    <h3>Solved Numericals</h3>
    <div class="numerical-block"><div class="num-title">Numerical 1</div>
    <div class="num-given"><strong>Given:</strong> values with units</div>
    <div class="num-find"><strong>To Find:</strong> quantity</div>
    <div class="num-formula"><strong>Formula:</strong> formula</div>
    <div class="num-working"><strong>Working:</strong> step-by-step calculation</div>
    <div class="num-answer">Answer: value with SI unit</div></div>
    <h3>Comparison Table</h3>
    <table class="notes-table"><thead><tr><th>Basis</th><th>Concept A</th><th>Concept B</th></tr></thead>
    <tbody><tr><td>Definition</td><td>def A</td><td>def B</td></tr><tr><td>SI Unit</td><td>unit</td><td>unit</td></tr><tr><td>Formula</td><td>formula</td><td>formula</td></tr><tr><td>Example</td><td>eg</td><td>eg</td></tr></tbody></table>
    <h3>🔬 ICSE Practical Experiment</h3>
    <div class="experiment-block">
    <div class="exp-row"><span class="exp-label">Aim:</span><span class="exp-content">aim</span></div>
    <div class="exp-row"><span class="exp-label">Apparatus:</span><span class="exp-content">list</span></div>
    <div class="exp-row"><span class="exp-label">Procedure:</span><ol class="exp-steps"><li>step 1</li><li>step 2</li><li>step 3</li></ol></div>
    <div class="exp-row"><span class="exp-label">Observation:</span><span class="exp-content">observation</span></div>
    <div class="exp-row"><span class="exp-label">Result:</span><span class="exp-content">conclusion</span></div>
    <div class="exp-row"><span class="exp-label">Precautions:</span><span class="exp-content">precautions</span></div>
    </div>
    <h3>Labelled Diagram</h3>
    <div class="diagram-box"><div class="diagram-title">Diagram: name</div>
    <div class="diagram-labels">
    <div class="diagram-part"><span class="part-num">1</span><span class="part-name">Part</span><span class="part-desc">— function</span></div>
    <div class="diagram-part"><span class="part-num">2</span><span class="part-name">Part</span><span class="part-desc">— function</span></div>
    </div></div>` : ''}
    ${type==='maths' ? `
    <h3>Solved Examples</h3>
    <div class="numerical-block"><div class="num-title">Example 1</div>
    <div class="num-given"><strong>Question:</strong> ICSE board style problem</div>
    <div class="num-working"><strong>Solution:</strong> step 1 → step 2 → step 3</div>
    <div class="num-answer">Answer: final answer</div></div>` : ''}
    ${type==='humanity' ? `
    <h3>Key Dates &amp; Events</h3>
    <table class="notes-table"><thead><tr><th>Year</th><th>Event</th><th>Significance</th></tr></thead>
    <tbody><tr><td>year</td><td>event</td><td>significance</td></tr></tbody></table>
    <h3>Causes &amp; Effects</h3>
    <table class="notes-table"><thead><tr><th>Causes</th><th>Effects / Consequences</th></tr></thead>
    <tbody><tr><td>cause 1</td><td>effect 1</td></tr><tr><td>cause 2</td><td>effect 2</td></tr></tbody></table>
    <h3>Comparison</h3>
    <table class="notes-table"><thead><tr><th>Basis</th><th>Concept A</th><th>Concept B</th></tr></thead>
    <tbody><tr><td>basis</td><td>A</td><td>B</td></tr></tbody></table>` : ''}
  </section>
  <section class="memory-tricks">
    <h2>💡 Memory Tricks</h2>
    <div class="mnemonic-row">
      <div class="mnemonic-box"><span class="mnemonic-letter">A</span><span class="mnemonic-word">word</span></div>
      <div class="mnemonic-box"><span class="mnemonic-letter">B</span><span class="mnemonic-word">word</span></div>
      <div class="mnemonic-box"><span class="mnemonic-letter">C</span><span class="mnemonic-word">word</span></div>
    </div>
    <p><strong>Tip:</strong> another memory technique</p>
  </section>
  <section class="important-formulas">
    <h2>📐 Key Formulas / Definitions</h2>
    <div class="formula-box"><div class="formula-title">Name</div><div class="formula-eq">formula or definition</div></div>
    <div class="formula-box"><div class="formula-title">Name</div><div class="formula-eq">formula or definition</div></div>
  </section>
  <section class="quick-summary">
    <h2>⚡ Quick Revision Summary</h2>
    <ul><li>Point 1</li><li>Point 2</li><li>Point 3</li><li>Point 4</li><li>Point 5</li></ul>
  </section>
  <section class="infographic">
    <h2>🗺️ Concept Map</h2>
    <div class="concept-map">
      <div class="cm-center">${topic}</div>
      <div class="cm-branches">
        <div class="cm-branch"><div class="cm-node cm-blue">Branch 1</div><div class="cm-children"><div class="cm-child">detail A</div><div class="cm-child">detail B</div></div></div>
        <div class="cm-branch"><div class="cm-node cm-green">Branch 2</div><div class="cm-children"><div class="cm-child">detail C</div><div class="cm-child">detail D</div></div></div>
        <div class="cm-branch"><div class="cm-node cm-amber">Branch 3</div><div class="cm-children"><div class="cm-child">detail E</div></div></div>
        <div class="cm-branch"><div class="cm-node cm-coral">Branch 4</div><div class="cm-children"><div class="cm-child">detail F</div></div></div>
      </div>
    </div>
  </section>
</div>
Replace ALL placeholder content with real accurate ICSE ${grade} ${subject} content about "${topic}". Return ONLY the HTML.`;
}

// ── QUESTION PROMPTS (subject-aware) ──────────────────────────────────────────
function buildQPromptA(grade, subject, topic) {
  const type = subjectType(subject);

  const scienceInstr = `For ICSE ${subject}:
- MCQ options must include plausible distractors based on common ICSE student misconceptions
- At least 2 MCQs must involve numerical calculation (e.g. "A force of 10N acts on 2kg mass. Find acceleration")
- Fill blanks must test SI units, scientist names, laws, and formulae exactly as stated in ICSE syllabus`;

  const mathsInstr = `For ICSE Mathematics:
- MCQs must be calculation-based, not definition-based (e.g. "If x² - 5x + 6 = 0, find x")
- Fill blanks must test formulae, identities, and theorems verbatim as per ICSE
- Follow ICSE board paper MCQ language and difficulty`;

  const humanityInstr = `For ICSE ${subject}:
- MCQs must test specific dates, names, events, causes, effects — not vague facts
- Fill blanks must test key terms, treaties, acts, articles, definitions exactly as per ICSE textbook
- Questions must be specific to "${topic}" — not generic history questions`;

  const instr = type==='science' ? scienceInstr : type==='maths' ? mathsInstr : humanityInstr;

  return `You are an ICSE ${grade} ${subject} examiner setting a board paper for topic: "${topic}"

${instr}

Write EXACTLY 5 MCQs and EXACTLY 5 fill-in-the-blank questions. Every question must be specific to "${topic}" in ICSE ${grade} ${subject}.

Output ONLY a JSON object — no text before or after, no markdown:
{"mcq":[
{"q":"[Write complete, specific ICSE board-style MCQ about ${topic}]","options":["A) [option]","B) [option]","C) [option]","D) [option]"],"a":"[Correct option with letter]","explanation":"[ICSE-specific explanation]"},
{"q":"[MCQ 2 — ${type==='science'||type==='maths'?'must involve a calculation or numerical':'specific fact about '+topic}]","options":["A) [opt]","B) [opt]","C) [opt]","D) [opt]"],"a":"[answer]","explanation":"[reason]"},
{"q":"[MCQ 3]","options":["A) [opt]","B) [opt]","C) [opt]","D) [opt]"],"a":"[answer]","explanation":"[reason]"},
{"q":"[MCQ 4]","options":["A) [opt]","B) [opt]","C) [opt]","D) [opt]"],"a":"[answer]","explanation":"[reason]"},
{"q":"[MCQ 5]","options":["A) [opt]","B) [opt]","C) [opt]","D) [opt]"],"a":"[answer]","explanation":"[reason]"}
],"fillinblanks":[
{"q":"[Complete sentence about ${topic} with ________ blank]","a":"[answer]"},
{"q":"[${type==='science'?'The SI unit of ________ is ________.':type==='humanity'?'The ________ was signed in the year ________.':'The formula for ________ is ________.'}]","a":"[answer]"},
{"q":"[Sentence with blank testing key term from ${topic}]","a":"[answer]"},
{"q":"[Sentence with two ________ blanks about ${topic}]","a":"[word1 / word2]"},
{"q":"[${type==='science'?'________ (nationality, year) stated that ________.':type==='humanity'?'The ________ Act was passed in ________ to ________.':'In ICSE, ________ theorem states that ________.'}]","a":"[answer]"}
]}
Write the actual questions now — all about "${topic}" for ICSE ${grade} ${subject}. Output JSON only.`;
}

function buildQPromptB(grade, subject, topic) {
  const type = subjectType(subject);

  const humanityInstr = `For ICSE ${subject} — topic "${topic}":
- True/False statements must test specific facts (dates, names, clauses, definitions) not vague statements
- Odd One Out must use ICSE classification categories relevant to ${topic}`;

  const scienceInstr = `For ICSE ${subject} — topic "${topic}":
- True/False must test specific ICSE facts about SI units, laws, properties — not general science
- Odd One Out should group by ICSE-defined categories (e.g. types of force, types of reactions)`;

  const instr = type==='humanity' ? humanityInstr : scienceInstr;

  return `You are an ICSE ${grade} ${subject} examiner setting a board paper for topic: "${topic}"

${instr}

Write EXACTLY 5 true/false statements and EXACTLY 3 odd-one-out questions. Every question must be specific to "${topic}" in ICSE ${grade} ${subject}.

Output ONLY a JSON object — no text before or after, no markdown:
{"truefalse":[
{"q":"[True factual statement about ${topic} as per ICSE ${grade} syllabus]","a":"True","explanation":"[ICSE textbook reason]"},
{"q":"[Common ICSE student misconception about ${topic} — actually false]","a":"False","explanation":"[Correct ICSE fact: ...]"},
{"q":"[Another true statement testing ICSE-level knowledge of ${topic}]","a":"True","explanation":"[reason]"},
{"q":"[Statement with a subtle error students commonly make in ${topic}]","a":"False","explanation":"[What is actually correct per ICSE: ...]"},
{"q":"[True statement requiring deeper ICSE understanding of ${topic}]","a":"True","explanation":"[reason]"}
],"oddonesout":[
{"q":"Which is the odd one out and why? A) [item from ${topic}]  B) [item]  C) [item]  D) [item]","a":"[Letter) item]","explanation":"[It does not belong because... — based on ICSE classification]"},
{"q":"Which is the odd one out? A) [item]  B) [item]  C) [item]  D) [item]","a":"[answer]","explanation":"[ICSE reason]"},
{"q":"Which is the odd one out? A) [item]  B) [item]  C) [item]  D) [item]","a":"[answer]","explanation":"[reason]"}
]}
Write the actual questions now — all about "${topic}" for ICSE ${grade} ${subject}. Output JSON only.`;
}

function buildQPromptC(grade, subject, topic) {
  const type = subjectType(subject);

  const scienceShort = `Short answer ICSE Science pattern:
Q1 (2 marks): Define [key term] and state its SI unit.
Q2 (2 marks): State [law/principle] with its mathematical form.
Q3 (2 marks): State the aim and one precaution of the ICSE experiment on ${topic}.
Q4 (3 marks): Distinguish between [A] and [B] on three bases (tabular: "Basis | A | B --- Def | ... | ... --- Unit | ... | ... --- Example | ... | ...").
Q5 (3 marks): Numerical — A [object] of [value with unit] has [another value]. Calculate [quantity]. Show full working.`;

  const scienceLong = `Long answer ICSE Science pattern:
Q1 (5 marks): Describe the ICSE experiment to study ${topic}. Include: Aim / Apparatus / Procedure (5 steps) / Observation / Result / Two precautions.
Q2 (6 marks): With a labelled diagram, explain [concept]. State relevant law. Derive the formula step-by-step. Solve: [numerical with values].
Q3 (6 marks): A word problem: [real-world scenario involving ${topic}]. Given: [specific values with SI units]. Find [two quantities]. Show complete working for each.`;

  const mathsShort = `Short answer ICSE Maths pattern (Section A style):
Q1 (2 marks): Direct application — solve a specific numerical using the formula for ${topic}.
Q2 (2 marks): A word problem — [real-world scenario], find [quantity].
Q3 (3 marks): Multi-step problem requiring use of ${topic} concepts.
Q4 (3 marks): Prove or verify [ICSE theorem related to ${topic}].
Q5 (3 marks): A problem where student must identify the method, set up the equation, and solve.`;

  const mathsLong = `Long answer ICSE Maths pattern (Section B style):
Q1 (4 marks): Multi-part word problem — (a) find [value] (b) find [another value] using ${topic}.
Q2 (4 marks): Real-world application — [scenario involving ${topic}], show complete working.
Q3 (6 marks): Complex problem requiring two different concepts from ${topic}, each with full working.`;

  const humanityShort = `Short answer ICSE ${subject} pattern:
Q1 (2 marks): Give the meaning / Define [key term from ${topic}].
Q2 (2 marks): State any two causes / effects / features of [aspect of ${topic}].
Q3 (2 marks): Mention the significance of [event/person/act] related to ${topic}.
Q4 (3 marks): Distinguish between [A] and [B] related to ${topic} on any three points (tabular: "Basis | A | B --- Point1 | ... | ... --- Point2 | ... | ... --- Point3 | ... | ...").
Q5 (3 marks): Give a short account of [specific event/concept] in ${topic}.`;

  const humanityLong = `Long answer ICSE ${subject} pattern:
Q1 (5 marks): Explain in detail [main concept in ${topic}]. Include: definition, background, key features, significance, and one ICSE-style source-based analysis if applicable.
Q2 (6 marks): "Examine the causes and consequences of [event in ${topic}]." Write a structured essay-style answer with introduction, causes (3 points), consequences (3 points), conclusion.
Q3 (6 marks): Compare and contrast [two related concepts in ${topic}] using a table (5 points), followed by a paragraph explaining which was more significant and why.`;

  const arInstr = type === 'humanity'
    ? 'Assertion-Reason: test cause-effect, significance, or classification within ' + topic + '.'
    : 'Assertion-Reason: test understanding of laws, principles, or properties in ' + topic + '.';

  const shortInstr = type === 'maths' ? mathsShort : type === 'science' ? scienceShort : humanityShort;
  const longInstr  = type === 'maths' ? mathsLong  : type === 'science' ? scienceLong  : humanityLong;

  const sa5q = (type === 'science' || type === 'maths')
    ? 'Numerical word problem: a real-world scenario with specific values and SI units. Calculate a specific quantity.'
    : 'Give a short account of a specific event or concept in ' + topic + '.';

  const sa5a = (type === 'science' || type === 'maths')
    ? 'Given: list values with units. Formula: write formula. Working: step by step. Answer: value with SI unit.'
    : 'Complete short account with key facts, dates, and significance.';

  const la2q = (type === 'science' || type === 'maths')
    ? 'Multi-step word problem or ICSE experiment for ' + topic + '.'
    : 'Examine the causes and consequences of a key event in ' + topic + '.';

  const la3q = (type === 'science' || type === 'maths')
    ? 'Real-world word problem requiring critical thinking and full working for ' + topic + '.'
    : 'Compare and contrast two related concepts in ' + topic + ' using a table, followed by a paragraph.';

  return 'You are an ICSE ' + grade + ' ' + subject + ' examiner setting a board paper for topic: "' + topic + '"\n\n'
    + arInstr + '\n\n'
    + shortInstr + '\n\n'
    + longInstr + '\n\n'
    + 'Write EXACTLY 3 assertion-reason, EXACTLY 5 short-answer, and EXACTLY 3 long-answer questions.\n'
    + 'Every question must be specific to "' + topic + '" in ICSE ' + grade + ' ' + subject + '.\n\n'
    + 'For tabular answers write: "Basis | Column A | Column B --- Row1 | val | val --- Row2 | val | val"\n'
    + 'For numericals: "Given: ... Formula: ... Step 1: ... Step 2: ... Answer: value + unit"\n'
    + 'For experiments: "Aim: ... Apparatus: ... Procedure: 1...2...3... Observation: ... Result: ..."\n\n'
    + 'Output ONLY a JSON object — no text before or after, no markdown:\n'
    + '{"assertionreason":['
    + '{"assertion":"[Specific true ICSE claim about ' + topic + ']","reason":"[Correct reason]","a":"Both Assertion and Reason are true, and Reason is the correct explanation","explanation":"[Why R explains A]"},'
    + '{"assertion":"[True ICSE claim about ' + topic + ']","reason":"[Plausible but incorrect reason]","a":"Assertion is true but Reason is false","explanation":"[Correct reason]"},'
    + '{"assertion":"[True ICSE claim about ' + topic + ']","reason":"[True but unrelated reason]","a":"Both Assertion and Reason are true, but Reason is NOT the correct explanation","explanation":"[What actually explains A]"}'
    + '],"shortanswer":['
    + '{"q":"[Short answer Q1 as per ICSE pattern above]","a":"[Complete model answer]","marks":2},'
    + '{"q":"[Short answer Q2]","a":"[Complete model answer]","marks":2},'
    + '{"q":"[Short answer Q3 — experiment aim and precaution OR significance]","a":"[Complete model answer]","marks":2},'
    + '{"q":"[Short answer Q4 — distinguish on three bases in tabular form]","a":"[Table: Basis | A | B --- row1 --- row2 --- row3]","marks":3},'
    + '{"q":"[' + sa5q + ']","a":"[' + sa5a + ']","marks":3}'
    + '],"longanswer":['
    + '{"q":"[Long answer Q1 as per ICSE pattern above]","a":"[Complete detailed model answer]","marks":5},'
    + '{"q":"[' + la2q + ']","a":"[Complete model answer with all steps and parts]","marks":6},'
    + '{"q":"[' + la3q + ']","a":"[Complete model answer]","marks":6}'
    + ']}\n'
    + 'Write the actual questions now — all specific to "' + topic + '" for ICSE ' + grade + ' ' + subject + '. Output JSON only.';
}

// ── BANK PROMPT ───────────────────────────────────────────────────────────────
function buildBankPrompt(grade, subject, topic) {
  const type = subjectType(subject);

  const subjectNote = type === 'science'
    ? 'Include: 2 numericals with full working, 1 derivation question, 1 experiment question, questions on SI units and scientist names.'
    : type === 'maths'
    ? 'Include: at least 5 word problems requiring full working, 2 proof questions, multi-step problems.'
    : 'Include: 2 source-based questions, date and event questions, cause-effect questions, significance questions, comparison questions. NO vague generic questions.';

  const saDefine   = type === 'science' ? 'Define [key term] and state its SI unit.' : 'Define or give the meaning of [key term from ' + topic + '].';
  const saLaw      = type === 'humanity' ? 'State any two causes or effects of [aspect of ' + topic + '].' : 'State [law or theorem] and write its mathematical expression.';
  const saExpt     = type === 'humanity' ? 'What was the significance of [event or person] in ' + topic + '?' : 'In the ICSE experiment on ' + topic + ', state the aim and one precaution.';
  const saNum      = (type === 'maths' || type === 'science') ? 'Numerical: [real-world scenario with values and SI units related to ' + topic + ']. Calculate [quantity]. Show full working.' : 'Give a short account of [specific event or concept in ' + topic + '].';
  const saNumAns   = (type === 'maths' || type === 'science') ? 'Given: values. Formula: formula. Working: steps. Answer: value + SI unit.' : 'Complete short account with key facts and significance.';

  const la1q = type === 'humanity'
    ? 'Examine the causes and consequences of [major event in ' + topic + ']. Give a structured answer.'
    : 'Describe the ICSE experiment to study ' + topic + '. Include aim, apparatus, procedure, observation and result.';

  const la2q = (type === 'maths' || type === 'science')
    ? 'Word problem: [real-world scenario involving ' + topic + ' with specific values]. Find two quantities. Show complete working for each.'
    : 'Compare [concept A] and [concept B] in ' + topic + ' using a table with five points of difference.';

  const la3q = type === 'humanity'
    ? 'Write a structured essay on [main theme of ' + topic + ']. Include: introduction, three main points with examples, and conclusion.'
    : 'With a labelled diagram, explain [main mechanism in ' + topic + ']. Derive the key formula step by step and solve one numerical.';

  const mcq2 = (type === 'science' || type === 'maths')
    ? 'Calculation-based MCQ — student must compute a value, with A) B) C) D) options'
    : 'Specific knowledge MCQ about a key fact, date, or term in ' + topic;

  return 'You are an ICSE ' + grade + ' ' + subject + ' board examiner. Topic: "' + topic + '"\n'
    + subjectNote + '\n\n'
    + 'Generate 20 board-exam quality questions strictly based on ICSE ' + grade + ' ' + subject + ' syllabus for "' + topic + '".\n'
    + 'Mix all question types. Every question must be specific and exam-worthy.\n\n'
    + 'Output ONLY valid JSON — no markdown:\n'
    + '{"subject":"' + subject + '","topic":"' + topic + '","grade":"' + grade + '","questions":[\n'
    + '{"type":"MCQ","difficulty":"easy","question":"[Specific MCQ with A) B) C) D) options about ' + topic + ']","answer":"[Correct option]","marks":1},\n'
    + '{"type":"MCQ","difficulty":"medium","question":"[' + mcq2 + ']","answer":"[answer]","marks":1},\n'
    + '{"type":"Fill in the Blank","difficulty":"easy","question":"[Sentence about ' + topic + ' with ___ blank]","answer":"[word]","marks":1},\n'
    + '{"type":"Fill in the Blank","difficulty":"medium","question":"[Sentence with ___ and ___ two blanks about ' + topic + ']","answer":"[word1 / word2]","marks":1},\n'
    + '{"type":"True or False","difficulty":"easy","question":"[True factual statement about ' + topic + ']","answer":"True","marks":1},\n'
    + '{"type":"True or False","difficulty":"medium","question":"[Common misconception about ' + topic + ']","answer":"False — correct answer: [what is correct]","marks":1},\n'
    + '{"type":"Odd One Out","difficulty":"medium","question":"A) [item]  B) [item]  C) [item]  D) [item]","answer":"[Letter) item — reason]","marks":1},\n'
    + '{"type":"Odd One Out","difficulty":"medium","question":"A) [item]  B) [item]  C) [item]  D) [item]","answer":"[answer — reason]","marks":1},\n'
    + '{"type":"Assertion-Reason","difficulty":"medium","question":"Assertion: [true claim about ' + topic + ']. Reason: [correct reason].","answer":"Both true, Reason explains Assertion","marks":2},\n'
    + '{"type":"Assertion-Reason","difficulty":"hard","question":"Assertion: [true claim about ' + topic + ']. Reason: [wrong reason].","answer":"Assertion true, Reason false — correct reason: [reason]","marks":2},\n'
    + '{"type":"Short Answer","difficulty":"easy","question":"' + saDefine + '","answer":"[Complete ICSE answer]","marks":2},\n'
    + '{"type":"Short Answer","difficulty":"medium","question":"' + saLaw + '","answer":"[Complete answer]","marks":2},\n'
    + '{"type":"Short Answer","difficulty":"medium","question":"Distinguish between [A] and [B] in ' + topic + ' on two bases.","answer":"Basis | A | B --- Point 1 | val | val --- Point 2 | val | val","marks":2},\n'
    + '{"type":"Short Answer","difficulty":"medium","question":"' + saExpt + '","answer":"[Complete answer]","marks":2},\n'
    + '{"type":"Short Answer","difficulty":"hard","question":"' + saNum + '","answer":"' + saNumAns + '","marks":3},\n'
    + '{"type":"Long Answer","difficulty":"hard","question":"' + la1q + '","answer":"[Full model answer]","marks":5},\n'
    + '{"type":"Long Answer","difficulty":"hard","question":"' + la2q + '","answer":"[Full model answer]","marks":5},\n'
    + '{"type":"Long Answer","difficulty":"hard","question":"' + la3q + '","answer":"[Full model answer]","marks":6},\n'
    + '{"type":"Long Answer","difficulty":"hard","question":"[A critical thinking question specific to ' + topic + ' at ICSE ' + grade + ' level]","answer":"[Complete model answer]","marks":6}\n'
    + ']}\n'
    + 'Replace ALL placeholders with real specific content about "' + topic + '". Output JSON only.';
}

// ── PAST PAPER ANALYSIS ───────────────────────────────────────────────────────
async function analyzePastPaper(fileBuffer, mimeType, grade, subject) {
  const base64 = fileBuffer.toString('base64');

  const prompt = `You have been given a scanned ICSE ${grade} ${subject} past question paper.

Analyse the paper carefully and:
1. Extract every question with its marks allocation
2. Identify the question types present (MCQ, Short Answer, Long Answer, etc.)
3. Identify the topics covered
4. Note the difficulty pattern and marking scheme

Then generate 15 NEW questions that are closely modelled on the style, difficulty, topic mix, and language of this paper — but are entirely new questions (not copies).

Output ONLY valid JSON:
{
  "paperAnalysis": {
    "totalMarks": number,
    "sections": ["Section A: ...", "Section B: ..."],
    "topicsCovered": ["topic1", "topic2"],
    "questionTypes": ["MCQ", "Short Answer", "Long Answer"],
    "difficultyObservation": "description of difficulty pattern"
  },
  "extractedQuestions": [
    {"type": "type", "question": "exact question text", "marks": number, "topic": "topic name"}
  ],
  "generatedQuestions": [
    {"type": "type", "difficulty": "easy/medium/hard", "question": "new question modelled on paper style", "answer": "complete model answer", "marks": number, "topic": "topic"}
  ]
}`;

  // Use vision-capable model for image/PDF analysis
  const res = await fetch('https://api.groq.com/openai/v1/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${process.env.GROQ_API_KEY}` },
    body: JSON.stringify({
      model: 'llama-3.3-70b-versatile',
      messages: [
        { role: 'system', content: SYSTEM_PROMPT },
        { role: 'user', content: [
          { type: 'image_url', image_url: { url: `data:${mimeType};base64,${base64}` } },
          { type: 'text', text: prompt }
        ]}
      ],
      max_tokens: 4000,
      temperature: 0.3
    })
  });

  if (!res.ok) {
    // Fallback: just generate questions if vision fails
    const fallback = await callGroq(
      `You are an ICSE ${grade} ${subject} examiner. Generate 15 high-quality board-exam questions mixing all types (MCQ, Fill Blank, Short Answer, Long Answer). Output JSON: {"paperAnalysis":{"totalMarks":100,"sections":["N/A - generated"],"topicsCovered":["${subject} general"],"questionTypes":["MCQ","Short Answer","Long Answer"],"difficultyObservation":"Mixed difficulty"},"extractedQuestions":[],"generatedQuestions":[{"type":"Short Answer","difficulty":"medium","question":"question here","answer":"answer here","marks":2,"topic":"general"}]}`,
      3000
    );
    return safeJSON(fallback, { paperAnalysis:{}, extractedQuestions:[], generatedQuestions:[] });
  }

  const data = await res.json();
  return safeJSON(data.choices[0].message.content, { paperAnalysis:{}, extractedQuestions:[], generatedQuestions:[] });
}

// ── Generate questions ────────────────────────────────────────────────────────
async function generateQuestions(grade, subject, topic) {
  const [rawA, rawB, rawC] = await Promise.all([
    callGroq(buildQPromptA(grade, subject, topic), 2000),
    callGroq(buildQPromptB(grade, subject, topic), 1600),
    callGroq(buildQPromptC(grade, subject, topic), 3000)
  ]);

  const questions = { mcq:[], fillinblanks:[], truefalse:[], oddonesout:[], assertionreason:[], shortanswer:[], longanswer:[] };

  const pA = safeJSON(rawA, null);
  const pB = safeJSON(rawB, null);
  const pC = safeJSON(rawC, null);

  if (pA) Object.assign(questions, pA); else console.error('PromptA FAILED. Raw:', rawA.slice(0,400));
  if (pB) Object.assign(questions, pB); else console.error('PromptB FAILED. Raw:', rawB.slice(0,400));
  if (pC) Object.assign(questions, pC); else console.error('PromptC FAILED. Raw:', rawC.slice(0,400));

  console.log(`Questions: mcq:${questions.mcq.length} fib:${questions.fillinblanks.length} tf:${questions.truefalse.length} ooo:${questions.oddonesout.length} ar:${questions.assertionreason.length} sa:${questions.shortanswer.length} la:${questions.longanswer.length}`);
  return questions;
}

// ── /api/generate-all ─────────────────────────────────────────────────────────
app.post('/api/generate-all', upload.array('files', 10), async (req, res) => {
  try {
    const { subject, topic, grade } = req.body;
    const files = req.files || [];
    const hasFiles = files.length > 0;
    if (!topic && !hasFiles) return res.status(400).json({ error: 'Enter a topic or upload files.' });

    const g = grade || 'Grade 8', s = subject || '', t = topic || '';
    const [notesRaw, questions, bankRaw] = await Promise.all([
      callGroq(buildNotesPrompt(g, s, t), 6000),
      generateQuestions(g, s, t),
      callGroq(buildBankPrompt(g, s, t), 4000)
    ]);

    const notes = notesRaw.replace(/```html|```/g,'').trim();
    const bank  = safeJSON(bankRaw, { questions:[] });
    res.json({ success:true, notes, questions, bank, fromTopic:!hasFiles });
  } catch(err) { console.error(err); res.status(500).json({ error: err.message }); }
});

// ── /api/questions-only ───────────────────────────────────────────────────────
app.post('/api/questions-only', async (req, res) => {
  try {
    const { subject, topic, grade } = req.body;
    if (!topic) return res.status(400).json({ error: 'Topic is required.' });
    const questions = await generateQuestions(grade||'Grade 8', subject||'', topic);
    res.json({ success:true, questions });
  } catch(err) { console.error(err); res.status(500).json({ error: err.message }); }
});

// ── /api/analyze-paper (past paper upload) ────────────────────────────────────
app.post('/api/analyze-paper', upload.single('paper'), async (req, res) => {
  try {
    const { grade, subject } = req.body;
    const file = req.file;
    if (!file) return res.status(400).json({ error: 'Please upload a past paper.' });

    const result = await analyzePastPaper(file.buffer, file.mimetype, grade||'Grade 8', subject||'');
    res.json({ success:true, ...result });
  } catch(err) { console.error(err); res.status(500).json({ error: err.message }); }
});

// ── /api/ask ──────────────────────────────────────────────────────────────────
app.post('/api/ask', async (req, res) => {
  try {
    const { question, context, grade, subject } = req.body;
    const prompt = `You are a friendly ICSE ${grade||'Grade 8'} ${subject||''} tutor. Answer using ICSE syllabus terminology.
${context ? 'Notes context: '+context.slice(0,800)+'\n' : ''}
Student question: ${question}
Give a clear answer with ICSE-relevant examples, SI units where applicable, and step-by-step working for numericals.`;
    const answer = await callGroq(prompt, 1024);
    res.json({ success:true, answer });
  } catch(err) { console.error(err); res.status(500).json({ error: err.message }); }
});

app.get('/api/health', (_, res) => res.json({ status:'ok', powered_by:'Groq LLaMA — ICSE aligned' }));
app.listen(PORT, () => console.log(`\n🎓 StudyBot ICSE running at http://localhost:${PORT}\n`));

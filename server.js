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
const BIOLOGY  = ['Biology'];
const MATHS    = ['Mathematics'];
const HUMANITY = ['History & Civics','History','Geography','Economics','Commercial Studies','Accountancy','Business Studies','Psychology','Sociology','Political Science'];
const LANGUAGE = ['English Language','English Literature','English','Hindi','French'];
const COMP     = ['Computer Applications','Computer Science'];

function subjectType(s) {
  if (BIOLOGY.includes(s))  return 'biology';
  if (SCIENCE.includes(s))  return 'science';
  if (MATHS.includes(s))    return 'maths';
  if (HUMANITY.includes(s)) return 'humanity';
  if (LANGUAGE.includes(s)) return 'language';
  if (COMP.includes(s))     return 'computer';
  return 'general';
}

// ── Gemini vision (direct API — no SDK, works in all regions) ────────────────
// Uses gemini-2.5-flash (current free tier model as of 2026), fast, and supports multimodal vision
const GEMINI_VISION_URL = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent';

// ── Gemini vision via direct fetch (no SDK — works in all regions) ────────────
async function callGeminiVision(parts, maxAttempts = 3) {
  const key = process.env.GEMINI_API_KEY;
  if (!key) return null;

  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      const res = await fetch(GEMINI_VISION_URL + '?key=' + key, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          contents: [{ parts }],
          generationConfig: { maxOutputTokens: 8192, temperature: 0.1 }
        })
      });
      const data = await res.json();
      if (!res.ok) {
        const msg = data.error?.message || ('HTTP ' + res.status);
        if ((res.status === 429 || msg.includes('quota') || msg.includes('RATE')) && attempt < maxAttempts) {
          const wait = 3000 * attempt;
          console.log('Gemini rate limit — retrying in ' + wait + 'ms (attempt ' + (attempt+1) + '/' + maxAttempts + ')');
          await sleep(wait);
          continue;
        }
        throw new Error('Gemini error: ' + msg);
      }
      const text = data.candidates?.[0]?.content?.parts?.[0]?.text || '';
      if (!text) throw new Error('Gemini returned empty response');
      return text;
    } catch(err) {
      if (attempt === maxAttempts) throw err;
      if (err.message.includes('rate') || err.message.includes('429')) {
        await sleep(3000 * attempt);
      } else {
        throw err;
      }
    }
  }
}

// Extract and consolidate text from multiple uploaded pages (camera photos or PDFs)
async function extractTextFromFiles(files, grade, subject, topic) {
  if (!files || files.length === 0) return null;

  if (!process.env.GEMINI_API_KEY) {
    console.warn('GEMINI_API_KEY not set — cannot read uploaded images. Falling back to topic-only mode.');
    return null;
  }

  try {
    const pageCount = files.length;

    // Build parts array: all images first, then the instruction text
    const imageParts = files.map(file => ({
      inline_data: {
        mime_type: file.mimetype === 'application/pdf' ? 'application/pdf' : file.mimetype,
        data: file.buffer.toString('base64')
      }
    }));

    const textPart = {
      text: `You are reading ${pageCount} page${pageCount > 1 ? 's' : ''} from an ICSE ${grade} ${subject} textbook.
${topic ? 'Topic: "' + topic + '"' : ''}

These pages may be photographed with a handheld camera — they could be slightly blurry, angled, or partially overlapping. Read everything as accurately as possible.

TASK: Transcribe ALL visible text from ALL ${pageCount} page${pageCount > 1 ? 's' : ''} into one clean document. Include:
- All headings, subheadings, and paragraph text
- All definitions (exact wording)
- All formulas and equations (use text notation: F = ma, E = mc^2)
- All numbered lists and bullet points
- All table content (preserve rows and columns)
- All diagram labels (write as: "Diagram: [name] — Label 1: [part name], Label 2: [part name]...")
- All examples and solved problems

If pages overlap, include the content only once.
Start directly with the transcribed content — no introduction or preamble.`
    };

    const extracted = await callGeminiVision([...imageParts, textPart]);
    console.log('Gemini extracted', extracted.length, 'chars from', pageCount, 'file(s)');
    return extracted;
  } catch(err) {
    console.error('Gemini vision error:', err.message);
    return null;
  }
}

// ── Groq text helper ──────────────────────────────────────────────────────────
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
function buildNotesPrompt(grade, subject, topic, extractedText = null) {
  const type      = subjectType(subject);
  const isBiology = subject === 'Biology';

  // ── When real book content is uploaded: use a SHORT focused prompt ───────────
  // The HTML template would push book content out of Groq context.
  // Instead: just ask Groq to reformat the extracted text into notes HTML.
  if (extractedText) {
    const subjectTips = isBiology
      ? 'For Biology: create diagram-box blocks (numbered parts + functions), experiment-block, comparison tables.'
      : type === 'science'
      ? 'For Physics/Chemistry: create derivation-block, numerical-block (Given/Formula/Working/Answer), SI units table, experiment-block.'
      : type === 'maths'
      ? 'For Maths: create numerical-block solved examples showing every step. No experiment blocks.'
      : type === 'humanity'
      ? 'For History/Geography/Economics: create timeline tables, cause-effect tables, key-term definitions.'
      : '';

    return `You are an expert ICSE ${grade} ${subject} teacher creating revision notes from a student\'s uploaded textbook pages.

=== UPLOADED TEXTBOOK CONTENT ===
${extractedText.slice(0, 10000)}
=== END OF CONTENT ===

Reformat the content above into clear ICSE revision notes as HTML.
${subjectTips}

Rules:
- Use ONLY the content from the upload above — do not add unrelated content
- Preserve all definitions, formulas, examples and facts exactly as in the book
- Keep exact ICSE terminology from the book
- For any diagram mentioned: create a diagram-box with numbered labels and functions
- For any table in the book: create an HTML table
- For formulas: use formula-box elements
- For experiments: use experiment-block elements
- For derivations: use derivation-block elements
- For solved numericals: use numerical-block elements

Return ONLY this HTML (no markdown, no backticks):

<div class="notes-content">
  <section class="key-concepts">
    <h2>\uD83D\uDD11 Key Concepts from Your Textbook</h2>
    <ul>
      <li><strong>[exact term from upload]:</strong> [exact definition from upload]</li>
    </ul>
  </section>
  <section class="detailed-notes">
    <h2>\uD83D\uDCDA Detailed Notes</h2>
    [Reformat each topic and subtopic from the uploaded content using proper HTML elements]
  </section>
  <section class="important-formulas">
    <h2>\uD83D\uDCD0 Key Formulas and Definitions</h2>
    <div class="formula-box"><div class="formula-title">Formula name</div><div class="formula-eq">formula from book</div></div>
  </section>
  <section class="quick-summary">
    <h2>\u26A1 Quick Revision Points</h2>
    <ul>
      <li>[key point from uploaded content]</li>
    </ul>
  </section>
</div>

Replace every placeholder with ACTUAL content from the uploaded pages. Return ONLY the HTML.`;
  }

  // ── No files — generate from ICSE curriculum knowledge ──────────────────────
  const biologyExtra = `
BIOLOGY — INCLUDE ALL (mandatory):
- At least 2 labelled diagram-box blocks with numbered parts and functions
- ICSE Biology definitions, classification tables, scientist names with years
- ICSE practical experiment block: Aim, Materials, Procedure, Observation, Result
- Comparison table (e.g. mitosis vs meiosis, aerobic vs anaerobic)`;

  const scienceExtra = `
PHYSICS/CHEMISTRY — INCLUDE ALL:
- Every quantity: symbol, SI unit, formula in a table
- Scientist name, nationality, year for each law
- Full step-by-step derivation-block
- At least 2 numerical-blocks (Given/Formula/Working/Answer)
- ICSE experiment-block: Aim, Apparatus, Procedure, Observation, Result
- At least 1 diagram-box with numbered parts`;

  const mathsExtra = `
MATHEMATICS — INCLUDE ALL:
- Every formula/theorem with proof
- At least 3 numerical-blocks (solved problems showing every step)
- Common mistakes and ICSE exam tips`;

  const humanityExtra = `
HISTORY/GEOGRAPHY/ECONOMICS — INCLUDE ALL:
- Key dates and events table (Year | Event | Significance)
- Causes and effects table
- Comparison table where relevant
- Key terms defined exactly as per ICSE textbook`;

  const extras = isBiology ? biologyExtra
    : type === 'science'   ? scienceExtra
    : type === 'maths'     ? mathsExtra
    : type === 'humanity'  ? humanityExtra
    : '';

  return `ICSE ${grade} | ${subject} | Topic: ${topic}
${extras}

Create comprehensive exam-ready ICSE revision notes for "${topic}".
Return ONLY valid HTML (no markdown, no backticks):

<div class="notes-content">
  <section class="key-concepts">
    <h2>\uD83D\uDD11 Key Concepts</h2>
    <ul><li><strong>Term:</strong> ICSE definition</li><li><strong>Term:</strong> definition</li></ul>
  </section>
  <section class="detailed-notes">
    <h2>\uD83D\uDCDA Detailed Notes</h2>
    <h3>Main subtopic</h3><p>Detailed ICSE explanation...</p>
    <h3>Key Laws / Principles</h3>
    <p><strong>Law Name (Scientist, Country, Year):</strong> Exact ICSE statement of the law.</p>
  </section>
  <section class="memory-tricks">
    <h2>\uD83D\uDCA1 Memory Tricks</h2>
    <div class="mnemonic-row">
      <div class="mnemonic-box"><span class="mnemonic-letter">A</span><span class="mnemonic-word">word</span></div>
      <div class="mnemonic-box"><span class="mnemonic-letter">B</span><span class="mnemonic-word">word</span></div>
    </div>
    <p><strong>Tip:</strong> memory technique</p>
  </section>
  <section class="important-formulas">
    <h2>\uD83D\uDCD0 Key Formulas / Definitions</h2>
    <div class="formula-box"><div class="formula-title">Name</div><div class="formula-eq">formula or definition</div></div>
  </section>
  <section class="quick-summary">
    <h2>\u26A1 Quick Revision Summary</h2>
    <ul><li>Key point 1</li><li>Key point 2</li><li>Key point 3</li><li>Key point 4</li><li>Key point 5</li></ul>
  </section>
</div>
Replace ALL placeholder content with real ICSE ${grade} ${subject} content about "${topic}". Return ONLY the HTML.`;
}

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
  const isBiology = subject === 'Biology';

  const scienceShort = isBiology
    ? `Short answer ICSE Biology pattern:
Q1 (2 marks): Define [key ICSE Biology term from ${topic}].
Q2 (2 marks): State the function of [named part/organ/structure] in ${topic}.
Q3 (2 marks): State the aim and one precaution of the ICSE Biology experiment on ${topic}.
Q4 (3 marks): Distinguish between [A] and [B] in ${topic} on three bases (tabular format).
Q5 (3 marks): Draw and label a neat diagram of [structure/organ relevant to ${topic}]. Answer must describe diagram with: title, at least 4 numbered labels with functions.`
    : `Short answer ICSE Physics/Chemistry pattern:
Q1 (2 marks): Define [key term] and state its SI unit.
Q2 (2 marks): State [law/principle] with its mathematical form.
Q3 (2 marks): State the aim and one precaution of the ICSE experiment on ${topic}.
Q4 (3 marks): Distinguish between [A] and [B] on three bases (tabular: "Basis | A | B --- Def | ... | ... --- Unit | ... | ... --- Example | ... | ...").
Q5 (3 marks): Numerical — A [object] of [value with unit] has [another value]. Calculate [quantity]. Show full working.`;

  const scienceLong = isBiology
    ? `Long answer ICSE Biology pattern:
Q1 (5 marks): Describe the ICSE Biology experiment to study ${topic}. Include Aim / Materials / Procedure (numbered steps) / Observation / Result / Precautions.
Q2 (6 marks): With the help of a neat labelled diagram, describe [main biological structure or process in ${topic}]. Include: diagram title, at least 5 numbered labels with functions, description of process.
Q3 (6 marks): Explain [key biological concept in ${topic}] in detail. Include: definition, types/stages, significance, comparison table, one real-life example.`
    : `Long answer ICSE Physics/Chemistry pattern:
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
  // Step 0: Use Gemini vision (direct fetch) to transcribe the paper
  let transcription = null;
  if (process.env.GEMINI_API_KEY) {
    try {
      const parts = [
        {
          inline_data: {
            mime_type: mimeType === 'application/pdf' ? 'application/pdf' : mimeType,
            data: fileBuffer.toString('base64')
          }
        },
        {
          text: `This is an ICSE ${grade} ${subject} past question paper.
Transcribe EVERY question EXACTLY as printed. Include:
- Section headings and their instructions
- Every question number and sub-part label (Q1, Q1(a), Q1(b), etc.)
- Marks in brackets e.g. [2] or (2 marks)
- Complete question text — every word, every value, every unit
- All MCQ options (A, B, C, D)
- Any data tables or given values

Format exactly like the original paper. Do NOT answer anything — only transcribe.`
        }
      ];
      transcription = await callGeminiVision(parts);
      console.log('Gemini transcribed paper:', transcription ? transcription.length : 0, 'chars');
    } catch(err) {
      console.log('Gemini paper transcription failed:', err.message, '— using curriculum-based generation');
    }
  } else {
    console.warn('No GEMINI_API_KEY — cannot read paper image, generating curriculum-based questions');
  }

  const paperContext = transcription
    ? `Here is the EXACT transcription of the uploaded past paper:\n\n${transcription}\n\n`
    : `Generate a typical ICSE ${grade} ${subject} board paper with realistic questions.\n\n`;

  const extractPrompt = `You are an ICSE ${grade} ${subject} examiner and model answer writer.

${paperContext}${transcription ? 'Solve EVERY question from the transcribed paper above with complete model answers.' : 'Generate a REALISTIC set of questions'} that would appear in a typical ICSE ${grade} ${subject} paper.
For EVERY question provide a COMPLETE model answer with full working.
For numericals: show Given / Formula / Working step by step / Answer with SI unit.
For long answers: give a full structured model answer.
For experiments: Aim / Apparatus / Procedure / Observation / Result.

Output ONLY valid JSON — no markdown:
{"paperAnalysis":{"totalMarks":100,"sections":["Section A (40 marks): Objective and Short Answer","Section B (60 marks): Long Answer"],"topicsCovered":["topic1","topic2","topic3"],"questionTypes":["MCQ","Fill in Blank","Short Answer","Long Answer"],"difficultyObservation":"Typical ICSE board paper — objective questions test recall, long answers require application and analysis"},"solvedQuestions":[{"qno":"1(a)","type":"MCQ","question":"Complete MCQ question with A) B) C) D) options","answer":"Correct option letter — reason why it is correct","marks":1,"topic":"topic name"},{"qno":"1(b)","type":"Fill in the Blank","question":"Sentence with ___ blank","answer":"correct word","marks":1,"topic":"topic"},{"qno":"2(a)","type":"Short Answer","question":"2-mark short answer question","answer":"Complete model answer in 2-3 sentences","marks":2,"topic":"topic"},{"qno":"2(b)","type":"Distinguish","question":"Distinguish between A and B on two bases","answer":"Basis | A | B --- Definition | def A | def B --- SI Unit | unit A | unit B","marks":2,"topic":"topic"},{"qno":"3","type":"Numerical","question":"A word problem with specific values and SI units","answer":"Given: values with units. Formula: formula. Working: step 1 calculation. Step 2 calculation. Answer: value + SI unit","marks":3,"topic":"topic"},{"qno":"4(a)","type":"Short Answer","question":"Another short answer question","answer":"Complete model answer","marks":3,"topic":"topic"},{"qno":"5","type":"Long Answer","question":"6-mark long answer question requiring detailed explanation","answer":"Introduction. Point 1 with explanation. Point 2 with explanation. Point 3 with explanation. Diagram described with numbered labels. Conclusion. All relevant formulas with SI units.","marks":6,"topic":"topic"},{"qno":"6","type":"Long Answer","question":"Another long answer — experiment or word problem","answer":"Complete model answer with all required parts","marks":6,"topic":"topic"}]}
Generate at least 10 solved questions covering all question types in ICSE ${grade} ${subject}. Replace ALL placeholders with real subject-specific content. Output JSON only.`;

  const generatePrompt = `You are an ICSE ${grade} ${subject} examiner creating new questions modelled on a typical board paper.
Generate 15 NEW ICSE board-quality questions for ${grade} ${subject}.
For EVERY question provide a COMPLETE model answer with full working.
For numericals: Given / Formula / Working / Answer with SI unit.
For long answers: full structured model answer.

Output ONLY valid JSON — no markdown:
{"generatedQuestions":[{"type":"MCQ","difficulty":"easy","question":"MCQ with A) B) C) D) options","answer":"Correct option — reason","marks":1,"topic":"topic"},{"type":"Fill in the Blank","difficulty":"easy","question":"Sentence with ___ blank","answer":"answer","marks":1,"topic":"topic"},{"type":"Short Answer","difficulty":"medium","question":"2-mark question","answer":"Complete 2-mark model answer","marks":2,"topic":"topic"},{"type":"Numerical","difficulty":"medium","question":"Word problem with specific values and SI units","answer":"Given: values. Formula: formula. Working: step by step. Answer: value + SI unit","marks":3,"topic":"topic"},{"type":"Long Answer","difficulty":"hard","question":"6-mark question","answer":"Full structured model answer covering all required points","marks":6,"topic":"topic"}]}
Generate 15 varied questions. Replace ALL placeholders with real ${grade} ${subject} content. Output JSON only.`;

  const [solvedRaw, generatedRaw] = await Promise.all([
    callGroq(extractPrompt, 4000),
    callGroq(generatePrompt, 4000)
  ]);

  const solved    = safeJSON(solvedRaw,    { paperAnalysis:{}, solvedQuestions:[] });
  const generated = safeJSON(generatedRaw, { generatedQuestions:[] });

  return {
    paperAnalysis:      solved.paperAnalysis      || {},
    solvedQuestions:    solved.solvedQuestions     || [],
    generatedQuestions: generated.generatedQuestions || []
  };
}
// ── Generate questions ────────────────────────────────────────────────────────
async function generateQuestions(grade, subject, topic, extractedText = null) {
  // Put extracted text FIRST so it's seen before the long prompts (not ignored at end)
  const pre = extractedText
    ? 'The student uploaded pages from their ICSE ' + grade + ' ' + subject + ' textbook about "' + topic + '".\n'
      + 'Base ALL questions on this exact content from the book:\n---\n'
      + extractedText.slice(0, 4000)
      + '\n---\n\n'
    : '';
  const [rawA, rawB, rawC] = await Promise.all([
    callGroq(pre + buildQPromptA(grade, subject, topic), 2200),
    callGroq(pre + buildQPromptB(grade, subject, topic), 1800),
    callGroq(pre + buildQPromptC(grade, subject, topic), 3200)
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

    // If files uploaded — use Gemini vision to extract text from all pages first
    let extractedText = null;
    if (hasFiles) {
      extractedText = await extractTextFromFiles(files, g, s, t);
      if (extractedText) {
        console.log(`Using extracted content (${extractedText.length} chars) from ${files.length} uploaded file(s)`);
      } else {
        console.log('Vision extraction failed or unavailable — using topic-only mode');
      }
    }

    const [notesRaw, questions, bankRaw] = await Promise.all([
      callGroq(buildNotesPrompt(g, s, t, extractedText), 6000),
      generateQuestions(g, s, t, extractedText),
      callGroq(buildBankPrompt(g, s, t), 4000)
    ]);

    const notes = notesRaw.replace(/```html|```/g,'').trim();
    const bank  = safeJSON(bankRaw, { questions:[] });
    const visionUsed = hasFiles && !!extractedText;
    res.json({
      success: true, notes, questions, bank,
      fromTopic: !hasFiles,
      visionUsed,
      filesRead: hasFiles ? files.length : 0
    });
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

// ── /api/generate-essay ──────────────────────────────────────────────────────
app.post('/api/generate-essay', async (req, res) => {
  try {
    const { grade, subject, topic, wordLimit, essayType, language } = req.body;
    if (!topic) return res.status(400).json({ error: 'Topic is required.' });

    const lang  = language  || 'English';
    const words = parseInt(wordLimit) || 300;
    const etype = essayType || 'descriptive';
    const isHindi  = lang === 'Hindi';
    const isFrench = lang === 'French';

    // Grade-appropriate complexity guidance
    const gradeNum = parseInt((grade.match(/\d+/) || ['8'])[0]);
    const complexity = gradeNum <= 7
      ? 'Use simple, clear sentences appropriate for a young student. Vocabulary should be accessible but rich.'
      : gradeNum <= 9
      ? 'Use varied sentence structures and a good range of vocabulary. Show awareness of tone and style.'
      : 'Use sophisticated vocabulary, complex sentence structures, and literary devices as expected in ICSE/ISC board exams.';

    // Language-specific instruction
    const langInstr = isHindi
      ? `CRITICAL: Write the ENTIRE essay in Hindi using Devanagari script (हिन्दी). Do NOT use any English words except proper nouns. Use ${gradeNum <= 8 ? 'सरल और स्पष्ट हिन्दी' : 'शुद्ध साहित्यिक हिन्दी'} appropriate for ${grade} ICSE level. Vocabulary highlights should also be in Hindi with Hindi meanings.`
      : isFrench
      ? `CRITICAL: Write the ENTIRE essay in French. Do NOT use any English words except proper nouns. Use ${gradeNum <= 8 ? 'français simple et clair' : 'français littéraire sophistiqué'} appropriate for ${grade} ICSE level. Vocabulary highlights should be French words with English meanings.`
      : `Write in English. ${complexity}`;

    // Essay type instructions
    const typeInstr = {
      descriptive:   `DESCRIPTIVE ESSAY: Paint a vivid picture of "${topic}" using sensory details (sight, sound, smell, touch, taste), imagery, and expressive language. Do not tell a story — describe.`,
      narrative:     `NARRATIVE ESSAY: Write a story about "${topic}" with a clear beginning (setting/characters), rising action, climax, and satisfying conclusion. Use first or third person consistently.`,
      argumentative: `ARGUMENTATIVE / DISCURSIVE ESSAY on "${topic}": Present arguments FOR and AGAINST. Use paragraph 1 for one side, paragraph 2 for the other, then give your clear personal stand with evidence in the conclusion.`,
      expository:    `EXPOSITORY / FACTUAL ESSAY: Explain "${topic}" clearly and informatively. Include facts, causes, effects, and examples. No personal opinion — only facts and analysis.`,
      letter:        `LETTER on "${topic}": Use correct ICSE letter format. Include: sender's address (top right), date, recipient's address (left), salutation (Dear Sir/Madam or name), body paragraphs, complimentary close, signature. Tone: ${gradeNum <= 9 ? 'semi-formal or formal' : 'formal'}.`,
      speech:        `SPEECH on "${topic}": Begin with a greeting (Ladies and gentlemen / Respected Principal / Dear friends). Use rhetorical questions, repetition for emphasis, and inclusive language ("we", "our"). End with a strong call to action or memorable closing line.`,
    }[etype] || `Write a well-structured essay on "${topic}" appropriate for ICSE ${grade} exams.`;

    // Word count guidance
    const wordGuide = words <= 200
      ? `Write approximately ${words} words. This is a short composition — be concise but complete.`
      : words <= 350
      ? `Write approximately ${words} words. 1 introduction paragraph + 2 body paragraphs + 1 conclusion.`
      : `Write approximately ${words} words. 1 introduction paragraph + 3 well-developed body paragraphs + 1 conclusion.`;

    const prompt = `You are an expert ICSE/ISC ${grade} ${isHindi ? 'Hindi' : isFrench ? 'French' : 'English Language'} teacher and examiner with 20 years of experience.

TASK: Write a complete ${etype} essay/composition for an ICSE ${grade} student.
Topic: "${topic}"
Language: ${lang}
${wordGuide}

${langInstr}

${typeInstr}

ICSE REQUIREMENTS:
- Paragraphing: clear paragraph breaks, each with one main idea
- Vocabulary: varied and appropriate for ${grade} — do not repeat the same words
- Sentences: mix of simple, compound, and complex sentences
- Opening: must immediately grab attention — start with a quote, question, or vivid image (NOT "In this essay I will...")
- Closing: memorable final line that brings the essay full circle

Structure your response using EXACTLY these markers on their own lines — in this exact order:

[ESSAY]
Write the complete essay here — nothing else in this section.

[WORD COUNT]
Approximately X words

[VOCABULARY HIGHLIGHTS]
1. word — meaning — why it works here
2. word — meaning — why it works here
3. word — meaning — why it works here

[EXAMINER TIP]
One specific ICSE examiner tip for scoring full marks on a ${etype} essay.

IMPORTANT: Start your response with [ESSAY] on the very first line.`;

    const result = await callGroq(prompt, 2500);
    res.json({ success: true, essay: result });
  } catch(err) { console.error(err); res.status(500).json({ error: err.message }); }
});

// ── /api/generate-mnemonics ───────────────────────────────────────────────────
app.post('/api/generate-mnemonics', async (req, res) => {
  try {
    const { grade, subject, topic, items } = req.body;
    if (!topic) return res.status(400).json({ error: 'Topic is required.' });

    const prompt = `You are a creative ICSE memory coach helping ${grade} ${subject} students remember difficult content.

Topic: "${topic}"
Grade: ${grade}
Subject: ${subject}
${items ? 'Specific items to remember: ' + items : ''}

Create a comprehensive set of memory aids for this topic. Make them fun, visual, and memorable for school students.

Return ONLY this JSON — no markdown:
{
  "topic": "${topic}",
  "mnemonics": [
    {
      "type": "acronym",
      "title": "Name of the mnemonic",
      "content": "The acronym or keyword",
      "expansion": "What each letter stands for",
      "example": "How to use it — e.g. VIBGYOR for rainbow colours",
      "items": ["item1", "item2", "item3"]
    }
  ],
  "memoryPalace": {
    "description": "A short vivid story or scene that links all the key concepts together",
    "story": "The actual story — make it funny, dramatic, or absurd so it sticks"
  },
  "rhymeOrSong": {
    "description": "A short rhyme, jingle or rap to remember key facts",
    "content": "The actual rhyme or jingle"
  },
  "visualAssociations": [
    {
      "concept": "concept name",
      "visual": "vivid visual image or association to remember it",
      "tip": "how to use this mental image"
    }
  ],
  "quickRecallCards": [
    {
      "front": "Question or prompt",
      "back": "Answer with memory hook"
    }
  ]
}

Create at least 3 different mnemonic types, 1 memory palace story, 1 rhyme/jingle, 3 visual associations, and 5 quick recall cards.
Make everything specific to "${topic}" in ICSE ${grade} ${subject}. Output JSON only.`;

    const raw = await callGroq(prompt, 2500);
    const data = safeJSON(raw, null);
    if (!data) return res.status(500).json({ error: 'Could not generate mnemonics. Please try again.' });
    res.json({ success: true, mnemonics: data });
  } catch(err) { console.error(err); res.status(500).json({ error: err.message }); }
});

app.get('/api/health', (_, res) => res.json({ status:'ok', powered_by:'Groq LLaMA + Gemini Vision — ICSE aligned' }));
app.listen(PORT, () => {
  console.log(`\n🎓 StudyBot ICSE running at http://localhost:${PORT}`);
  if (process.env.GEMINI_API_KEY) {
    console.log('✅ Gemini vision ready (gemini-2.5-flash via direct API) — document upload will work');
  } else {
    console.log('⚠️  GEMINI_API_KEY not set — document upload will use topic-only fallback');
  }
  if (process.env.GROQ_API_KEY) {
    console.log('✅ Groq API ready — text generation active');
  } else {
    console.log('❌ GROQ_API_KEY not set — app will not work!');
  }
  console.log('');
});

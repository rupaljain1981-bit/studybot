require('dotenv').config();
const express = require('express');
const multer  = require('multer');
const cors    = require('cors');
const path    = require('path');
const sharp   = require('sharp');

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

// ── Image compression (resize large camera photos before sending to Gemini) ───
// Gemini inline_data limit is ~5MB per image. Camera phones produce 10-20MB images.
// Resize to max 1600px on longest side, JPEG quality 82 — enough for OCR accuracy.
async function compressImage(file) {
  const MAX_BYTES = 4 * 1024 * 1024; // 4MB target
  if (file.size <= MAX_BYTES || file.mimetype === 'application/pdf') {
    return { buffer: file.buffer, mimetype: file.mimetype };
  }
  try {
    const compressed = await sharp(file.buffer)
      .rotate()                        // auto-rotate from EXIF (phone orientation)
      .resize(1600, 1600, { fit: 'inside', withoutEnlargement: true })
      .jpeg({ quality: 82, progressive: true })
      .toBuffer();
    console.log('Compressed', file.originalname || 'image',
      'from', Math.round(file.size/1024) + 'KB',
      'to', Math.round(compressed.length/1024) + 'KB');
    return { buffer: compressed, mimetype: 'image/jpeg' };
  } catch(err) {
    console.warn('Compression failed for', file.originalname, '—', err.message, '— sending original');
    return { buffer: file.buffer, mimetype: file.mimetype };
  }
}

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

// ── Page limits ──────────────────────────────────────────────────────────────
const MAX_NOTES_PAGES = 5;   // max pages for notes/questions (5 pages = one topic section)
const MAX_PAPER_PAGES = 10;  // max pages for past paper analysis

// Extract and consolidate text from uploaded pages (camera photos or PDFs)
// ─ Images:  compressed to <4MB, all sent in one Gemini call
// ─ PDFs:    each PDF file read separately (Gemini REST supports PDF inline_data)
async function extractTextFromFiles(files, grade, subject, topic, onProgress = null) {
  if (!files || files.length === 0) return null;

  if (!process.env.GEMINI_API_KEY) {
    console.warn('GEMINI_API_KEY not set — falling back to topic-only mode.');
    return null;
  }

  // Cap pages to avoid overloading Gemini and Groq context
  if (files.length > MAX_NOTES_PAGES) {
    console.warn('Too many pages (' + files.length + ') — capping at ' + MAX_NOTES_PAGES);
    files = files.slice(0, MAX_NOTES_PAGES);
  }

  const pdfs   = files.filter(f => f.mimetype === 'application/pdf');
  const images = files.filter(f => f.mimetype !== 'application/pdf');
  let allText = '';

  // ── PDFs: read each separately (Gemini REST supports PDF inline_data natively)
  for (let i = 0; i < pdfs.length; i++) {
    if (onProgress) onProgress('📄 Reading PDF ' + (i+1) + ' of ' + pdfs.length + '…');
    try {
      const pdfText = await callGeminiVision([
        { inline_data: { mime_type: 'application/pdf', data: pdfs[i].buffer.toString('base64') } },
        { text: 'This is a page from an ICSE ' + grade + ' ' + subject + ' textbook'
            + (topic ? ' about "' + topic + '"' : '') + '. '
            + 'Transcribe ALL text: headings, definitions, formulas, diagrams, tables, examples. '
            + 'Plain text only. Start directly with the content.' }
      ]);
      if (pdfText) {
        allText += (allText ? '\n\n--- PDF ' + (i+1) + ' ---\n\n' : '') + pdfText;
        console.log('PDF', i+1, 'extracted:', pdfText.length, 'chars');
        if (onProgress) onProgress('✅ PDF ' + (i+1) + ' read (' + pdfText.length + ' chars)');
      }
    } catch(err) { console.error('PDF', i+1, 'failed:', err.message); }
    if (i < pdfs.length - 1) await sleep(1200);
  }

  // ── Images: compress all then send in one Gemini call
  if (images.length > 0) {
    if (onProgress) onProgress('🗜️ Compressing ' + images.length + ' image' + (images.length > 1 ? 's' : '') + '…');
    const compressed = await Promise.all(images.map(f => compressImage(f)));
    if (onProgress) onProgress('📖 AI reading ' + images.length + ' image' + (images.length > 1 ? 's' : '') + '…');
    const imageParts = compressed.map(f => ({
      inline_data: { mime_type: f.mimetype, data: f.buffer.toString('base64') }
    }));
    try {
      const imageText = await callGeminiVision([
        ...imageParts,
        { text: 'You are reading ' + images.length + ' page' + (images.length > 1 ? 's' : '')
            + ' from an ICSE ' + grade + ' ' + subject + ' textbook'
            + (topic ? ' about "' + topic + '"' : '') + '. '
            + 'Pages may be handheld camera photos — slightly blurry or angled is fine. '
            + 'Transcribe ALL text: headings, definitions, formulas, diagrams, tables, examples. '
            + 'If pages overlap, include content once. Plain text only. Start directly with the content.' }
      ]);
      if (imageText) {
        allText += (allText ? '\n\n--- Images ---\n\n' : '') + imageText;
        console.log('Images extracted:', imageText.length, 'chars from', images.length, 'file(s)');
        if (onProgress) onProgress('✅ ' + images.length + ' image' + (images.length > 1 ? 's' : '') + ' read (' + imageText.length + ' chars)');
      }
    } catch(err) { console.error('Image extraction failed:', err.message); }
  }

  console.log('Total extracted:', allText.length, 'chars from', files.length, 'file(s)');
  return allText || null;
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

const sleep = ms => new Promise(r => setTimeout(r, ms));
const usingFallback = false; // paid tier — always use Scout, no fallback needed

async function callGroq(prompt, maxTokens = 3000) {
  // Paid Groq account — llama-4-scout, no rate limiting, no gaps
  for (let attempt = 1; attempt <= 3; attempt++) {
    try {
      const res = await fetch('https://api.groq.com/openai/v1/chat/completions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${process.env.GROQ_API_KEY}` },
        body: JSON.stringify({
          model: 'meta-llama/llama-4-scout-17b-16e-instruct',
          messages: [{ role: 'system', content: SYSTEM_PROMPT }, { role: 'user', content: prompt }],
          max_tokens: maxTokens,
          temperature: 0.3
        })
      });
      if (!res.ok) {
        const e = await res.json();
        const msg = e.error?.message || 'Groq error';
        if (attempt < 3 && (msg.includes('429') || msg.includes('rate') || msg.includes('overloaded'))) {
          console.log('Groq busy — retrying in 3s (attempt', attempt + 1, '/ 3)');
          await sleep(3000);
          continue;
        }
        throw new Error(msg);
      }
      const data = await res.json();
      return data.choices[0].message.content;
    } catch(e) {
      if (attempt === 3) throw e;
      await sleep(2000);
    }
  }
}

// groqGap is a no-op on paid tier — kept so call sites don't break
async function groqGap() {}
function safeJSON(raw, fallback = {}) {
  if (!raw) return fallback;
  try {
    // Strip all markdown fences
    let cleaned = raw.replace(/^```[a-zA-Z]*\s*/m, '').replace(/\s*```\s*$/m, '').trim();
    cleaned = cleaned.replace(/```[a-zA-Z]*\s*|\s*```/g, '').trim();

    // Fix literal newlines inside JSON string values (8b model often does this)
    // Walk char-by-char: inside a string, replace raw \n/\r/\t with escape sequences
    {
      let out = '', inStr = false, esc = false;
      for (let i = 0; i < cleaned.length; i++) {
        const c = cleaned[i];
        if (esc) { out += c; esc = false; continue; }
        if (c === '\\') { out += c; esc = true; continue; }
        if (c === '"') { inStr = !inStr; out += c; continue; }
        if (inStr) {
          if (c === '\n')      { out += '\\n'; continue; }
          if (c === '\r')      { out += '\\r'; continue; }
          if (c === '\t')      { out += '\\t'; continue; }
        }
        out += c;
      }
      cleaned = out;
    }

    // Find start of JSON
    const s = cleaned.search(/[{[]/);
    if (s === -1) throw new Error('No JSON found');
    let text = cleaned.slice(s);

    // Attempt 1: direct parse (handles well-formed JSON including pretty-printed)
    try { return JSON.parse(text); } catch(_) {}

    // Attempt 2: find balanced closing delimiter
    let depth = 0, lastClose = -1;
    const opener = text[0], closer = opener === '{' ? '}' : ']';
    for (let i = 0; i < text.length; i++) {
      if (text[i] === opener) depth++;
      else if (text[i] === closer) { depth--; if (depth === 0) { lastClose = i; break; } }
    }
    if (lastClose > 0) { try { return JSON.parse(text.slice(0, lastClose + 1)); } catch(_) {} }

    // Attempt 3: repair — cut incomplete last object, close strings/brackets
    function repairJSON(t) {
      // Remove trailing commas
      let fixed = t.replace(/,\s*([}\]])/g, '$1').replace(/,\s*$/, '');
      // Close open strings (track string state properly)
      let inStr = false, escaped = false;
      for (const c of fixed) {
        if (escaped) { escaped = false; continue; }
        if (c === '\\') { escaped = true; continue; }
        if (c === '"') inStr = !inStr;
      }
      if (inStr) fixed += '"';
      // Cut at last complete array item if there's a truncated one
      const lastComplete = fixed.lastIndexOf('},{');
      if (lastComplete > 0) {
        let trimmed = fixed.slice(0, lastComplete + 1);
        let o2 = 0, a2 = 0;
        for (const c of trimmed) { if(c==='{')o2++;else if(c==='}')o2--;if(c==='[')a2++;else if(c===']')a2--; }
        for (let i = 0; i < Math.max(0, a2); i++) trimmed += ']';
        for (let i = 0; i < Math.max(0, o2); i++) trimmed += '}';
        try { return JSON.parse(trimmed); } catch(_) {}
      }
      // Just close all open brackets
      let opens = 0, arrs = 0;
      for (const c of fixed) { if(c==='{')opens++;else if(c==='}')opens--;if(c==='[')arrs++;else if(c===']')arrs--; }
      for (let i = 0; i < Math.max(0, arrs);  i++) fixed += ']';
      for (let i = 0; i < Math.max(0, opens); i++) fixed += '}';
      try { return JSON.parse(fixed); } catch(_) { return null; }
    }
    const repaired = repairJSON(text);
    if (repaired !== null) return repaired;

    throw new Error('Could not parse JSON even after repair');
  } catch(e) {
    console.error('JSON parse failed:', e.message, '\nRaw preview:', raw.slice(0, 200));
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
// Architecture: Gemini = OCR only (plain text, never fails)
//               Groq   = solve from text + generate new questions
async function analyzePastPaper(files, grade, subject, onProgress = () => {}, onSolved = () => {}) {
  // files = array of multer file objects (supports multiple pages photographed separately)

  // Cap pages
  if (files.length > MAX_PAPER_PAGES) {
    console.warn('Too many paper pages (' + files.length + ') — capping at ' + MAX_PAPER_PAGES);
    files = files.slice(0, MAX_PAPER_PAGES);
  }

  // ── Step 1: Gemini reads ALL pages as PLAIN TEXT ──────────────────────────────
  let transcription = null;

  if (process.env.GEMINI_API_KEY) {
    try {
      const pageCount = files.length;
      const pdfs   = files.filter(f => f.mimetype === 'application/pdf');
      const images = files.filter(f => f.mimetype !== 'application/pdf');

      onProgress('📖 Preparing ' + pageCount + ' page' + (pageCount > 1 ? 's' : '') + ' for AI reading…');

      let allPageText = '';

      // PDFs: read each one separately
      for (let i = 0; i < pdfs.length; i++) {
        onProgress('📄 Reading PDF page ' + (i+1) + ' of ' + pdfs.length + '…');
        try {
          const pt = await callGeminiVision([
            { inline_data: { mime_type: 'application/pdf', data: pdfs[i].buffer.toString('base64') } },
            { text: 'This is page ' + (i+1) + ' of an ICSE ' + grade + ' ' + subject + ' exam paper. Transcribe every question exactly as printed. Include question numbers, marks, all MCQ options. Plain text only.' }
          ]);
          if (pt) allPageText += (allPageText ? '\n\n' : '') + pt;
        } catch(e) { console.error('PDF page', i+1, 'failed:', e.message); }
        if (i < pdfs.length - 1) await sleep(1000);
      }

      // Images: compress then send all together
      const compressedImages = images.length > 0
        ? await Promise.all(images.map(f => compressImage(f)))
        : [];

      if (compressedImages.length > 0) {
        onProgress('📖 AI vision reading ' + images.length + ' image page' + (images.length > 1 ? 's' : '') + '…');
      }

      // Build image parts for ALL pages — Gemini reads them together in one call
      const imageParts = compressedImages.map(f => ({
        inline_data: {
          mime_type: f.mimetype,
          data: f.buffer.toString('base64')
        }
      }));

      onProgress('📖 AI vision reading all ' + pageCount + ' page' + (pageCount > 1 ? 's' : '') + '…');
      // Read images (if any) as a combined call
      if (imageParts.length > 0) {
        try {
          const imgText = await callGeminiVision([
            ...imageParts,
            { text: 'You are reading ' + imageParts.length + ' page' + (imageParts.length > 1 ? 's' : '') + ' of an ICSE ' + grade + ' ' + subject + ' exam paper. '
                + 'Transcribe EVERY question exactly as printed. '
                + 'Include: question numbers and sub-parts, marks, full question text, all MCQ options A B C D, any given data. '
                + 'Pages are in order — combine into one complete paper. Plain text only.' }
          ], 3);
          if (imgText) allPageText += (allPageText ? '\n\n' : '') + imgText;
        } catch(e) { console.error('Image OCR failed:', e.message); }
      }

      transcription = allPageText || null;
      console.log('Gemini OCR result:', transcription ? transcription.length + ' chars from ' + pageCount + ' page(s)' : 'failed');
    } catch(err) {
      console.log('Gemini OCR failed:', err.message);
    }
  }

  // ── Step 2: Groq solves the transcribed questions (small batches) ─────────────
  let solvedQuestions = [];
  let paperAnalysis = {};

  if (transcription) {
    // First call: analyse the paper structure (tiny, always fits)
    onProgress('✅ Paper read! Analysing structure…');
    const analysePrompt = 'A student uploaded this ICSE ' + grade + ' ' + subject + ' paper:\n\n'
      + transcription.slice(0, 800) + '\n\n'
      + 'Identify: total marks, section names, topics covered, question types.\n'
      + 'Output ONLY JSON: {"totalMarks":80,"sections":["A","B"],"topicsCovered":["t1","t2"],"questionTypes":["MCQ","Short"],"difficultyObservation":"mixed"}';

    const analysisRaw = await callGroq(analysePrompt, 400);
    paperAnalysis = safeJSON(analysisRaw, { totalMarks: 80, sections: [], topicsCovered: [subject], questionTypes: [], difficultyObservation: 'mixed' });
    console.log('Paper analysis done, topics:', paperAnalysis.topicsCovered);

    await sleep(600);

    // ── Smart content splitter — used for both papers and textbooks ─────────────
    // For papers: split on Q1/Q2/1. question markers
    // For textbooks: split on section/heading markers
    function splitByQuestions(text, maxPerChunk) {
      // Match question starters: Q1, Q1(a), 1., 1) etc at start of line
      const qPattern = /(?:^|\n)\s*(?:Q\d+[.(]?|\d+[.)]\s)/gm;
      const matches = [...text.matchAll(qPattern)];
      if (matches.length === 0) {
        // No question markers — split by section headers or just use chunks
        const sections = text.split(/\n(?=Section|SECTION)/);
        return sections.filter(s => s.trim()).length > 1
          ? sections.filter(s => s.trim())
          : [text];
      }
      // Extract each question as its own string
      const questions = matches.map((m, i) => {
        const start = m.index;
        const end   = matches[i+1] ? matches[i+1].index : text.length;
        return text.slice(start, end).trim();
      });
      // Group into chunks of maxPerChunk questions
      const chunks = [];
      for (let i = 0; i < questions.length; i += maxPerChunk) {
        chunks.push(questions.slice(i, i + maxPerChunk).join('\n\n'));
      }
      return chunks;
    }

    // Chunk size depends on model:
    // 70b: 2 questions/chunk — long detailed answers, 4000 token budget
    // 8b:  1 question/chunk  — keep well within 1800 token limit
        const perChunk = 2;
    const chunks = splitByQuestions(transcription, perChunk);
    console.log('Transcription', transcription.length, 'chars →', chunks.length, 'chunk(s) (Scout paid tier)');

    for (let ci = 0; ci < chunks.length; ci++) {
      const chunkLabel = chunks.length > 1 ? ' (Q' + (ci*perChunk+1) + (perChunk>1&&ci*perChunk+2<=chunks.length*perChunk ? '–'+(ci*perChunk+perChunk) : '') + ')' : '';
      onProgress('🧠 Solving' + chunkLabel + ' (' + (ci+1) + '/' + chunks.length + ')…');

      const solvePrompt = 'You are an ICSE ' + grade + ' ' + subject + ' teacher.\n'
        + 'Solve the question(s) below from an uploaded exam paper:\n\n'
        + chunks[ci] + '\n\n'
        + 'Rules:\n'
        + '- Give a COMPLETE model answer for every question and sub-part\n'
        + '- Numericals: Given: values | Formula: formula | Working: step1; step2 | Answer: value + unit\n'
        + '- Keep answers as single-line strings — separate steps with | not with line breaks\n'
        + '- Do NOT use real newlines inside any string value\n'
        + 'Output ONLY a JSON array (no wrapper object, no markdown):\n'
        + '[{"qno":"Q1(a)","type":"Short Answer","question":"exact question","answer":"complete answer on one line","marks":2,"topic":"topic"}]\n'
        + 'JSON array only. No newlines inside string values.';

      const solveRaw = await callGroq(solvePrompt, 3000);
      let solveText = solveRaw.trim();
      if (solveText.startsWith('[')) solveText = '{"solvedQuestions":' + solveText + '}';
      const solveData = safeJSON(solveText, { solvedQuestions: [] });
      const chunkSolved = Array.isArray(solveData) ? solveData : (solveData.solvedQuestions || []);
      solvedQuestions = solvedQuestions.concat(chunkSolved);
      console.log('Chunk', ci+1, '/', chunks.length, 'solved:', chunkSolved.length, 'questions (total:', solvedQuestions.length + ')');

      // Stream results to frontend after each chunk
      if (chunkSolved.length > 0) onSolved(solvedQuestions, paperAnalysis);

      // No gap needed on paid tier — Scout has no TPM limits
    }
    console.log('Total solved:', solvedQuestions.length, 'from', chunks.length, 'chunks');

  } else {
    // No vision — generate curriculum questions
    console.log('No transcription — generating curriculum paper for', grade, subject);
    const fallbackPrompt = 'Generate 4 ICSE ' + grade + ' ' + subject + ' exam questions with model answers. '
      + 'Output JSON: {"totalMarks":80,"sections":["A","B"],"topicsCovered":["' + subject + '"],"questionTypes":["MCQ","Short Answer"],"difficultyObservation":"mixed","solvedQuestions":['
      + '{"qno":"Q1","type":"MCQ","question":"specific MCQ with A) B) C) D)","answer":"B) correct — reason","marks":1,"topic":"' + subject + '"},'
      + '{"qno":"Q2","type":"Short Answer","question":"specific 2-mark question","answer":"model answer","marks":2,"topic":"' + subject + '"}'
      + ']}';
    const fbRaw = await callGroq(fallbackPrompt, 1000);
    const fb = safeJSON(fbRaw, {});
    paperAnalysis   = { totalMarks: fb.totalMarks || 80, sections: fb.sections || [], topicsCovered: fb.topicsCovered || [subject], questionTypes: fb.questionTypes || [], difficultyObservation: fb.difficultyObservation || 'mixed' };
    solvedQuestions = fb.solvedQuestions || [];
  }

  // ── Step 3: Generate 8 NEW similar questions (always small, always fits) ──────
  await sleep(600);
  const topics = (paperAnalysis.topicsCovered || [subject]).slice(0, 3).join(', ') || subject;
  onProgress('✨ Generating similar new questions…');
  const genPrompt = 'Generate 6 NEW ICSE ' + grade + ' ' + subject + ' questions on: ' + topics + '. '
    + 'Mix MCQ, Short Answer, Numerical, Long Answer. Give model answers. '
    + 'Output JSON: {"generatedQuestions":['
    + '{"type":"MCQ","difficulty":"medium","question":"specific question A) B) C) D)","answer":"C) answer — reason","marks":1,"topic":"' + topics.split(',')[0].trim() + '"},'
    + '{"type":"Short Answer","difficulty":"medium","question":"specific 2-mark question","answer":"model answer","marks":2,"topic":"topic"},'
    + '{"type":"Long Answer","difficulty":"hard","question":"specific 5-mark question","answer":"structured answer","marks":5,"topic":"topic"}'
    + ']}';

  const genRaw = await callGroq(genPrompt, 1500);
  const generated = safeJSON(genRaw, { generatedQuestions: [] });

  return {
    paperAnalysis,
    solvedQuestions: Array.isArray(solvedQuestions) ? solvedQuestions : [],
    generatedQuestions: generated.generatedQuestions || [],
    transcribed: !!transcription
  };
}
// ── Generate questions ────────────────────────────────────────────────────────
async function generateQuestions(grade, subject, topic, extractedText = null) {
  // Cap extracted text to avoid overwhelming the prompt
  const pre = extractedText
    ? 'The student uploaded pages from their ICSE ' + grade + ' ' + subject + ' textbook about "' + topic + '".\n'
      + 'Base ALL questions on this exact content:\n---\n'
      + extractedText.slice(0, 2000)
      + '\n---\n\n'
    : '';

  const questions = { mcq:[], fillinblanks:[], truefalse:[], oddonesout:[], assertionreason:[], shortanswer:[], longanswer:[] };

  // Sequential with TPM-aware gaps between each call
  const rawA = await callGroq(pre + buildQPromptA(grade, subject, topic), 1800);
  const pA = safeJSON(rawA, null);
  if (pA) Object.assign(questions, pA); else console.error('PromptA FAILED:', rawA.slice(0,200));

  const rawB = await callGroq(pre + buildQPromptB(grade, subject, topic), 1600);
  const pB = safeJSON(rawB, null);
  if (pB) Object.assign(questions, pB); else console.error('PromptB FAILED:', rawB.slice(0,200));

  const rawC = await callGroq(pre + buildQPromptC(grade, subject, topic), 1800);
  const pC = safeJSON(rawC, null);
  if (pC) Object.assign(questions, pC); else console.error('PromptC FAILED:', rawC.slice(0,200));

  console.log('Questions: mcq:' + questions.mcq.length + ' fib:' + questions.fillinblanks.length + ' tf:' + questions.truefalse.length + ' ooo:' + questions.oddonesout.length + ' ar:' + questions.assertionreason.length + ' sa:' + questions.shortanswer.length + ' la:' + questions.longanswer.length);
  return questions;
}

// ── SSE helper ───────────────────────────────────────────────────────────────
function sseSetup(res) {
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.setHeader('X-Accel-Buffering', 'no');
  res.flushHeaders();
}
function sseSend(res, event, data) {
  res.write('event: ' + event + '\ndata: ' + JSON.stringify(data) + '\n\n');
}

// ── /api/generate-all (SSE streaming) ────────────────────────────────────────
app.post('/api/generate-all', upload.array('files', 10), async (req, res) => {
  const { subject, topic, grade } = req.body;
  const files = req.files || [];
  const hasFiles = files.length > 0;

  if (!topic && !hasFiles) return res.status(400).json({ error: 'Enter a topic or upload files.' });

  sseSetup(res);
  const g = grade || 'Grade 8', s = subject || '', t = topic || '';

  try {
    // ── Phase 1: Read each uploaded page with Gemini, stream progress per page ─
    let allExtracted = '';

    if (hasFiles && process.env.GEMINI_API_KEY) {
      // Enforce page limit — tell user if pages were trimmed
      const rawCount = files.length;
      if (rawCount > MAX_NOTES_PAGES) {
        files = files.slice(0, MAX_NOTES_PAGES);
        sseSend(res, 'progress', {
          stage: 'limit',
          message: '⚠️ ' + rawCount + ' pages uploaded — reading first ' + MAX_NOTES_PAGES + ' only. Upload in smaller batches for more content.',
          page: 0, total: MAX_NOTES_PAGES
        });
      }

      const pdfs   = files.filter(f => f.mimetype === 'application/pdf');
      const images = files.filter(f => f.mimetype !== 'application/pdf');

      // ── Read PDFs one at a time (streaming progress per PDF) ────────────────
      for (let i = 0; i < pdfs.length; i++) {
        sseSend(res, 'progress', { stage: 'reading', message: '📄 Reading PDF ' + (i+1) + ' of ' + pdfs.length + '…', page: i+1, total: files.length });
        try {
          const pdfText = await callGeminiVision([
            { inline_data: { mime_type: 'application/pdf', data: pdfs[i].buffer.toString('base64') } },
            { text: 'ICSE ' + g + ' ' + s + ' textbook page' + (t ? ' about "' + t + '"' : '') + '. Transcribe ALL text: headings, definitions, formulas, diagrams, tables. Plain text only.' }
          ]);
          if (pdfText) {
            allExtracted += (allExtracted ? '\n\n--- PDF ' + (i+1) + ' ---\n\n' : '') + pdfText;
            sseSend(res, 'progress', { stage: 'page_done', message: '✅ PDF ' + (i+1) + ' read (' + pdfText.length + ' chars)', page: i+1, total: files.length });
          } else {
            sseSend(res, 'progress', { stage: 'page_failed', message: '⚠️ Could not read PDF ' + (i+1), page: i+1 });
          }
        } catch(err) {
          sseSend(res, 'progress', { stage: 'page_failed', message: '⚠️ PDF ' + (i+1) + ': ' + err.message, page: i+1 });
        }
        if (i < pdfs.length - 1) await sleep(1200);
      }

      // ── Read images: compress then send ALL together ─────────────────────────
      if (images.length > 0) {
        sseSend(res, 'progress', { stage: 'reading', message: '🗜️ Compressing ' + images.length + ' image' + (images.length>1?'s':'') + '…', page: pdfs.length + 1, total: files.length });
        const compressed = await Promise.all(images.map(f => compressImage(f)));
        sseSend(res, 'progress', { stage: 'reading', message: '📖 AI reading ' + images.length + ' image' + (images.length>1?'s':'') + ' together…', page: pdfs.length + 1, total: files.length });
        const imageParts = compressed.map(f => ({ inline_data: { mime_type: f.mimetype, data: f.buffer.toString('base64') } }));
        try {
          const imageText = await callGeminiVision([
            ...imageParts,
            { text: 'ICSE ' + g + ' ' + s + ' textbook — ' + images.length + ' pages' + (t ? ' about "' + t + '"' : '') + '. Transcribe ALL text from ALL pages: headings, definitions, formulas, diagrams, tables. If pages overlap include content once. Plain text only.' }
          ]);
          if (imageText) {
            allExtracted += (allExtracted ? '\n\n--- Pages ---\n\n' : '') + imageText;
            // Mark each image page as done
            images.forEach((_, i) => sseSend(res, 'progress', { stage: 'page_done', message: '✅ Page ' + (pdfs.length + i + 1) + ' read', page: pdfs.length + i + 1, total: files.length }));
          } else {
            sseSend(res, 'progress', { stage: 'page_failed', message: '⚠️ Could not read images', page: pdfs.length + 1 });
          }
        } catch(err) {
          sseSend(res, 'progress', { stage: 'page_failed', message: '⚠️ Images: ' + err.message, page: pdfs.length + 1 });
        }
      }

      if (allExtracted) {
        sseSend(res, 'progress', { stage: 'all_pages_read', message: '📚 All ' + files.length + ' page' + (files.length>1?'s':'') + ' read (' + allExtracted.length + ' chars). Generating notes…', totalChars: allExtracted.length });
      }
    } else if (hasFiles && !process.env.GEMINI_API_KEY) {
      sseSend(res, 'progress', { stage: 'no_vision', message: '⚠️ GEMINI_API_KEY not set — generating from topic name only' });
    }

    const extractedText = allExtracted || null;

    // ── Phase 2: Generate notes from ALL uploaded pages ───────────────────────
    sseSend(res, 'progress', { stage: 'notes', message: '📝 Generating study notes from your book pages…' });

    const notesRaw = await callGroq(buildNotesPrompt(g, s, t, extractedText), 5000);
    const notes = notesRaw.replace(/```html|```/g,'').trim();
    sseSend(res, 'notes', { notes, visionUsed: !!extractedText, filesRead: files.length });

    // ── Phase 3: Questions — each call gets a different SLICE of book content ─────
    // Divides the full OCR text into 3 equal thirds — one per question call.
    // This guarantees questions come from every uploaded page, not just the first bit.
    function bookSlice(text, idx, total) {
      if (!text) return '';
      const size = Math.ceil(text.length / total);
      return text.slice(idx * size, (idx + 1) * size);
    }
    console.log('Book content:', extractedText ? extractedText.length + ' chars — split into 3 slices for question calls' : 'topic-only');

    sseSend(res, 'progress', { stage: 'questions', message: '❓ Generating MCQ and Fill in Blanks…' });
    const sA = bookSlice(extractedText, 0, 3);
    const ctxA = sA ? 'Base ALL questions on this ICSE textbook content:\n' + sA + '\n\n' : '';
    const rawA = await callGroq(ctxA + buildQPromptA(g, s, t), 1800);
    const pA = safeJSON(rawA, {});
    sseSend(res, 'questions_partial', { types: ['mcq','fillinblanks'], data: pA });

    sseSend(res, 'progress', { stage: 'questions', message: '✅ True/False and Odd One Out…' });
    const sB = bookSlice(extractedText, 1, 3);
    const ctxB = sB ? 'Base ALL questions on this ICSE textbook content:\n' + sB + '\n\n' : '';
    const rawB = await callGroq(ctxB + buildQPromptB(g, s, t), 1600);
    const pB = safeJSON(rawB, {});
    sseSend(res, 'questions_partial', { types: ['truefalse','oddonesout'], data: pB });

    sseSend(res, 'progress', { stage: 'questions', message: '💡 Short and Long Answer questions…' });
    const sC = bookSlice(extractedText, 2, 3);
    const ctxC = sC ? 'Base ALL questions on this ICSE textbook content:\n' + sC + '\n\n' : '';
    const rawC = await callGroq(ctxC + buildQPromptC(g, s, t), 1800);
    const pC = safeJSON(rawC, {});
    sseSend(res, 'questions_partial', { types: ['assertionreason','shortanswer','longanswer'], data: pC });

    // ── Phase 4: Question bank ───────────────────────────────────────────────
    sseSend(res, 'progress', { stage: 'bank', message: '🏦 Building question bank…' });
    const bankRaw = await callGroq(buildBankPrompt(g, s, t), 3000);
    const bank = safeJSON(bankRaw, { questions: [] });
    sseSend(res, 'bank', { bank });

    sseSend(res, 'done', { success: true });
  } catch(err) {
    console.error(err);
    sseSend(res, 'error', { error: err.message });
  } finally {
    res.end();
  }
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

// ── /api/analyze-paper (SSE streaming, multiple pages supported) ─────────────
app.post('/api/analyze-paper', upload.array('paper', 20), async (req, res) => {
  const { grade, subject } = req.body;
  const files = req.files || [];
  if (!files.length) return res.status(400).json({ error: 'Please upload at least one page.' });

  sseSetup(res);
  const g = grade || 'Grade 8', s = subject || '';
  console.log('analyze-paper received', files.length, 'file(s):', files.map(f => f.originalname + ' (' + Math.round(f.size/1024) + 'KB)').join(', '));

  try {
    const pageCount = files.length;
    sseSend(res, 'progress', {
      stage: 'reading',
      message: '📄 Reading ' + pageCount + ' page' + (pageCount > 1 ? 's' : '') + ' with AI vision…',
      total: pageCount
    });

    const result = await analyzePastPaper(files, g, s,
      // onProgress
      (msg) => { sseSend(res, 'progress', { stage: 'solving', message: msg }); },
      // onSolved — stream solved questions the moment Groq finishes them
      (solved, analysis) => {
        sseSend(res, 'solved', { solvedQuestions: solved, paperAnalysis: analysis, transcribed: true });
      }
    );

    // Stream generated questions as second wave
    sseSend(res, 'analysis', { ...result, success: true });
    sseSend(res, 'done', { success: true });
  } catch(err) {
    console.error(err);
    sseSend(res, 'error', { error: err.message });
  } finally {
    res.end();
  }
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

    const context = `ICSE ${grade} ${subject} — Topic: "${topic}"${items ? '\nItems to remember: ' + items : ''}`;

    // ── Call 1: Acronyms + Memory Story + Rhyme (smaller, faster) ──────────────
    const prompt1 = `You are a fun ICSE memory coach. ${context}

Create memory aids. Output ONLY valid JSON:
{"mnemonics":[{"type":"acronym","title":"Name of mnemonic","content":"LETTERS","expansion":"Each Letter Stands For Something","example":"e.g. VIBGYOR for rainbow","items":["item1","item2","item3"]},{"type":"keyword","title":"Another mnemonic","content":"keyword","expansion":"what it reminds you of","example":"usage example","items":["item1","item2"]}],"memoryPalace":{"description":"one sentence about the scene","story":"A short funny story linking all key concepts — make it vivid and absurd so it sticks"},"rhymeOrSong":{"description":"what the rhyme covers","content":"Short rhyme or jingle — 4 to 6 lines that rhyme and cover the key facts"}}
Make everything specific to "${topic}". Output JSON only.`;

    // ── Call 2: Visual associations + Flash cards ───────────────────────────────
    const prompt2 = `You are a fun ICSE memory coach. ${context}

Create visual memory aids. Output ONLY valid JSON:
{"visualAssociations":[{"concept":"concept from topic","visual":"vivid memorable image to associate with it","tip":"how to use this mental image"},{"concept":"another concept","visual":"vivid image","tip":"usage tip"},{"concept":"third concept","visual":"vivid image","tip":"usage tip"}],"quickRecallCards":[{"front":"Question about topic","back":"Answer with memory hook"},{"front":"Another question","back":"Answer"},{"front":"Question","back":"Answer"},{"front":"Question","back":"Answer"},{"front":"Question","back":"Answer"}]}
Make everything specific to "${topic}" in ICSE ${grade} ${subject}. Output JSON only.`;

    // Sequential calls
    const raw1 = await callGroq(prompt1, 2000);
    await sleep(800);
    const raw2 = await callGroq(prompt2, 1500);

    const part1 = safeJSON(raw1, {});
    const part2 = safeJSON(raw2, {});

    // Merge both parts into one object
    const merged = {
      topic,
      mnemonics:          part1.mnemonics          || [],
      memoryPalace:       part1.memoryPalace        || { description: '', story: '' },
      rhymeOrSong:        part1.rhymeOrSong         || { description: '', content: '' },
      visualAssociations: part2.visualAssociations  || [],
      quickRecallCards:   part2.quickRecallCards    || []
    };

    if (!merged.mnemonics.length && !merged.memoryPalace.story) {
      return res.status(500).json({ error: 'Could not generate mnemonics. Please try again.' });
    }

    res.json({ success: true, mnemonics: merged });
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
    console.log('✅ Groq paid tier active — llama-4-scout (594 TPS, no rate limits)');
  } else {
    console.log('❌ GROQ_API_KEY not set — app will not work!');
  }
  console.log('');
});

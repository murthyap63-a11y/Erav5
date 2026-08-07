#!/usr/bin/env python3
"""
clean_data.py
----------------------------------------------------------------------
Implements the 10-step cleaning pipeline from Section 5 of the
"India-first 40B" training blueprint, applied to a single local text
file, then trains/uses a Byte-Level BPE tokenizer on the cleaned
output.

Writes to <outdir>/:
    <name>_cleaned.txt              cleaned text
    <name>_tokens.json              token ids + token pieces
    <name>_human_audit_sample.txt   sample of removed lines (Step 10)
    tokenizer/vocab.json, merges.txt   the trained BPE tokenizer
    <name>_log.txt                  word/token counts + methods used +
                                     the Gopher document-level report

Usage:
    python clean_and_tokenize.py --input raw.txt --outdir cleanup

Optional:
    --vocab_size 32000               BPE vocab size (default 32000)
    --eval_file bench.txt            enables Step 7 (decontamination)
    --toxic_terms_file terms.txt     enables real filtering in Step 5
                                      (one term per line; without this,
                                      Step 5 runs in pass-through mode)
    --min_line_len 3                 min words/line to survive Step 4
    --filter_curly_braces            enable C4's curly-brace line filter
                                      (off by default -- this pipeline
                                      may see code text, and C4's rule
                                      treats { } as noise)
    --minhash_threshold 0.85         Jaccard threshold for MinHash/LSH

Dependencies:
    pip install tokenizers langdetect datasketch
----------------------------------------------------------------------
NOTE ON SCOPE: this is a runnable, single-file reference implementation
of the pipeline shape described in the report. Several steps are
heuristic stand-ins for what would be trained classifiers in a real
production pipeline (see inline notes on Steps 2, 5, and 10) --
that's flagged explicitly in the log output too. The Gopher
document-level checks are LOG-ONLY: they report pass/fail per rule but
never drop content (a deliberate choice, see gopher_document_report).
Semantic (embedding-based) deduplication is intentionally NOT
implemented here -- only exact-hash + MinHash/LSH near-dup.
"""

import argparse
import html
import json
import os
import re
import time
import unicodedata
from collections import Counter
from datetime import datetime
from lxml.html import fromstring

try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0
    HAVE_LANGDETECT = True
except ImportError:
    HAVE_LANGDETECT = False

try:
    from tokenizers import ByteLevelBPETokenizer
    HAVE_TOKENIZERS = True
except ImportError:
    HAVE_TOKENIZERS = False

try:
    from datasketch import MinHash, MinHashLSH
    HAVE_DATASKETCH = True
except ImportError:
    HAVE_DATASKETCH = False


# ======================================================================
# Step registry (mirrors Section 5 of the blueprint, in execution order)
# ======================================================================
STEP_NAMES = {
    1:  "Script normalization (NFC + HTML-unescape + control/zero-width/bidi strip)",
    2:  "Language & script identification",
    3:  "Deduplication (exact hash + MinHash/LSH near-duplicate)",
    4:  "Quality filtering (heuristic + Gopher/C4-style rules)",
    5:  "Toxicity & safety filtering",
    6:  "PII redaction (Aadhaar / PAN / phone / email)",
    7:  "Benchmark decontamination",
    8:  "Code-specific cleaning (secret scrubbing)",
    9:  "Math/science-specific cleaning (LaTeX-aware)",
    10: "Human spot-audit sampling (manual step -- sample exported only)",
}

# ----------------------------------------------------------------------
# Regex patterns -- PII / secrets / LaTeX / boilerplate
# ----------------------------------------------------------------------
AADHAAR_RE  = re.compile(r'\b\d{4}\s?\d{4}\s?\d{4}\b')
PAN_RE      = re.compile(r'\b[A-Z]{5}[0-9]{4}[A-Z]\b')
PHONE_RE    = re.compile(r'\b(?:\+91[\-\s]?)?[6-9]\d{9}\b')
EMAIL_RE    = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')
SECRET_RE   = re.compile(
    r'(?:api[_-]?key|secret|token|passwd|password)\s*[:=]\s*'
    r'[\'"]?[A-Za-z0-9\-_.]{8,}[\'"]?', re.I)
LATEX_RE    = re.compile(r'\$[^$]+\$|\\\[[^\]]+\\\]|\\begin\{.*?\}.*?\\end\{.*?\}', re.S)
SYMBOL_CHAR = re.compile(r'[^\w\s]')
BOILERPLATE_PATTERNS = [
    re.compile(r'^\s*(cookie|privacy policy|terms of use|subscribe|advertisement)\b', re.I),
    re.compile(r'^\s*(click here|read more|share this)\b', re.I),
    re.compile(r'^\s*\u00a9\s?\d{4}'),
]

# ----------------------------------------------------------------------
# Regex patterns -- Step 1 normalization
# ----------------------------------------------------------------------
CTRL_CHARS   = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
# Zero-width space (U+200B) and BOM/zero-width-no-break-space (U+FEFF) are
# pure invisible junk -- safe to strip.
# NOTE: ZWNJ (U+200C) and ZWJ (U+200D) are DELIBERATELY NOT stripped --
# they are linguistically meaningful in Devanagari/Indic conjunct
# formation, and stripping them would corrupt correct text.
ZERO_WIDTH_RE = re.compile(r'[\u200b\ufeff]')
BIDI_CTRL_RE  = re.compile(r'[\u200e\u200f\u202a-\u202e\u2066-\u2069]')
NBSP_RE       = re.compile(r'\u00a0')
WHITESPACE_RUN_RE = re.compile(r'[ \t]{2,}')

# ----------------------------------------------------------------------
# Regex patterns -- Step 4b: C4-style line rules
# ----------------------------------------------------------------------
# Terminal punctuation extended beyond ASCII to cover Devanagari danda,
# Urdu/Arabic full stop, and CJK full stop, so this rule doesn't just
# nuke every non-English sentence.
C4_TERMINAL_PUNCT = '.!?"\'\u201d\u2019)\u0964\u0965\u06d4\u3002'
C4_JAVASCRIPT_RE  = re.compile(r'\bjavascript\b', re.I)
C4_LOREM_IPSUM_RE = re.compile(r'lorem ipsum', re.I)
C4_CURLY_BRACE_RE = re.compile(r'[{}]')

# ----------------------------------------------------------------------
# Gopher document-level stopword list (English-only -- see caveat in
# gopher_document_report)
# ----------------------------------------------------------------------
GOPHER_STOPWORDS = {"the", "be", "to", "of", "and", "that", "have", "with"}


def word_count(text: str) -> int:
    return len(text.split())


def load_toxic_terms(path):
    if not path or not os.path.exists(path):
        return set()
    with open(path, encoding='utf-8') as f:
        return {line.strip().lower() for line in f if line.strip()}


# ---------------------------------------------------------------------
# Step 1 -- script normalization
#   1. unescape HTML entities (&amp; -> &, &#39; -> ', etc.)
#   2. NFC Unicode normalization
#   3. strip control characters (incl. DEL)
#   4. strip zero-width junk (ZWSP, BOM) -- but NOT ZWNJ/ZWJ, which are
#      meaningful in Devanagari/Indic conjuncts
#   5. strip bidi control characters (LRM/RLM, embedding/override/isolate)
#   6. collapse NBSP into a regular space
#   7. collapse runs of whitespace, strip leading/trailing
# ---------------------------------------------------------------------
def step1_normalize(line, stats):
    norm = html.unescape(line)
    norm = fromstring(norm).text_content()
    norm = unicodedata.normalize('NFC', norm)
    norm = CTRL_CHARS.sub('', norm)
    norm = ZERO_WIDTH_RE.sub('', norm)
    norm = BIDI_CTRL_RE.sub('', norm)
    norm = NBSP_RE.sub(' ', norm)
    norm = WHITESPACE_RUN_RE.sub(' ', norm).strip()
    if norm != line.strip():
        stats['step1_lines_normalized'] += 1
    return norm


# ---------------------------------------------------------------------
# Step 2 -- language identification (tracked, informational only here;
# a production pipeline would route text to per-language filters
# instead of the langdetect fallback used in this reference script)
# ---------------------------------------------------------------------
def step2_detect_lang(line, stats):
    if not HAVE_LANGDETECT or len(line) < 8:
        return 'unk'
    try:
        lang = detect(line)
    except Exception:
        lang = 'unk'
    stats['lang_counts'][lang] += 1
    return lang


# ---------------------------------------------------------------------
# Step 3 -- deduplication: exact hash + MinHash/LSH near-duplicate
#
# The exact-hash layer catches identical lines. The MinHash/LSH layer
# catches near-duplicates (reordered/lightly-edited text) across the
# WHOLE document -- unlike an older windowed-Jaccard approach, LSH
# doesn't lose recall just because a duplicate is far away in the file.
#
# NOTE: this is lexical near-dup detection (shingle overlap), not
# semantic/embedding-based dedup -- that's intentionally out of scope
# here.
# ---------------------------------------------------------------------
def shingles(text, k=5):
    words = text.lower().split()
    if len(words) < k:
        return {tuple(words)}
    return {tuple(words[i:i + k]) for i in range(len(words) - k + 1)}


class MinHashDeduper:
    def __init__(self, num_perm=128, threshold=0.85, k=5):
        self.available = HAVE_DATASKETCH
        self.num_perm = num_perm
        self.k = k
        self._n = 0
        if self.available:
            self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)

    def _minhash(self, text):
        m = MinHash(num_perm=self.num_perm)
        for sh in shingles(text, self.k):
            m.update(' '.join(sh).encode('utf-8'))
        return m

    def is_duplicate(self, text):
        if not self.available:
            return False, None
        m = self._minhash(text)
        hits = self.lsh.query(m)
        if hits:
            return True, hits[0]
        key = f'd{self._n}'
        self._n += 1
        self.lsh.insert(key, m)
        return False, None


# ---------------------------------------------------------------------
# Step 4 -- quality heuristics
# ---------------------------------------------------------------------
def step4_quality_ok(line, min_len=3):
    if len(line.split()) < min_len:
        return False, 'too_short'
    if line:
        symbol_ratio = len(SYMBOL_CHAR.findall(line)) / len(line)
        if symbol_ratio > 0.4:
            return False, 'symbol_heavy'
    for pat in BOILERPLATE_PATTERNS:
        if pat.search(line):
            return False, 'boilerplate'
    return True, None


# ---------------------------------------------------------------------
# Step 4b -- C4-style line rules
#   - line must end in terminal punctuation (extended past ASCII so
#     Devanagari/Urdu/CJK sentences aren't all rejected)
#   - drop lines mentioning "javascript"
#   - drop lines containing "lorem ipsum"
#   - drop lines with curly braces (code-like) -- OFF by default, since
#     this pipeline may legitimately see code text (see Step 8); turn
#     on with --filter_curly_braces
# ---------------------------------------------------------------------
def c4_line_ok(line, filter_curly_braces=False):
    stripped = line.rstrip()
    if not stripped:
        return False, 'c4_empty'
    if stripped[-1] not in C4_TERMINAL_PUNCT:
        return False, 'c4_no_terminal_punct'
    if C4_JAVASCRIPT_RE.search(line):
        return False, 'c4_javascript_mention'
    if C4_LOREM_IPSUM_RE.search(line):
        return False, 'c4_lorem_ipsum'
    if filter_curly_braces and C4_CURLY_BRACE_RE.search(line):
        return False, 'c4_curly_brace'
    return True, None


# ---------------------------------------------------------------------
# Gopher document-level report (LOG-ONLY -- see module docstring)
#
# Gopher's rules are normally an accept/reject gate on a WHOLE document,
# not individual lines. By design here they never drop content -- they
# just report pass/fail per rule so a human can review a file that
# looks structurally off (e.g. mostly bullet points, or too short).
# ---------------------------------------------------------------------
def gopher_document_report(text):
    words = text.split()
    n_words = len(words)
    lines = [l for l in text.split('\n') if l.strip()]
    checks = {}

    checks['word_count'] = {
        'value': n_words, 'pass': 50 <= n_words <= 100_000,
        'rule': 'document word count in [50, 100000]'}

    mean_wlen = (sum(len(w) for w in words) / n_words) if n_words else 0
    checks['mean_word_length'] = {
        'value': round(mean_wlen, 2), 'pass': 3 <= mean_wlen <= 10,
        'rule': 'mean word length in [3, 10] characters'}

    symbol_count = text.count('#') + text.count('...')
    symbol_ratio = (symbol_count / n_words) if n_words else 0
    checks['symbol_to_word_ratio'] = {
        'value': round(symbol_ratio, 4), 'pass': symbol_ratio <= 0.1,
        'rule': "(# count + '...' count) / word count <= 0.1"}

    bullet_lines = sum(1 for l in lines if re.match(r'^\s*[\u2022\u25cf\u25aa\-\*]\s', l))
    bullet_ratio = (bullet_lines / len(lines)) if lines else 0
    checks['bullet_line_ratio'] = {
        'value': round(bullet_ratio, 3), 'pass': bullet_ratio <= 0.9,
        'rule': 'fraction of lines starting with a bullet <= 0.9'}

    ellipsis_end = sum(1 for l in lines if l.rstrip().endswith(('...', '\u2026')))
    ellipsis_ratio = (ellipsis_end / len(lines)) if lines else 0
    checks['ellipsis_end_ratio'] = {
        'value': round(ellipsis_ratio, 3), 'pass': ellipsis_ratio <= 0.3,
        'rule': 'fraction of lines ending in an ellipsis <= 0.3'}

    alpha_words = sum(1 for w in words if any(ch.isalpha() for ch in w))
    alpha_ratio = (alpha_words / n_words) if n_words else 0
    checks['alphabetic_word_ratio'] = {
        'value': round(alpha_ratio, 3), 'pass': alpha_ratio >= 0.8,
        'rule': 'fraction of words containing >=1 letter >= 0.8'}

    lowered = {w.strip('.,!?;:').lower() for w in words}
    found_stops = GOPHER_STOPWORDS & lowered
    checks['stopword_presence'] = {
        'value': sorted(found_stops), 'pass': len(found_stops) >= 2,
        'rule': 'document contains >= 2 of a small ENGLISH stopword list '
                '(caveat: this check is English-only and will under-fire '
                'on pure Indic-language documents -- treat failures here '
                'with that in mind, not as a quality signal on its own)'}

    failed = [name for name, c in checks.items() if not c['pass']]
    return {'checks': checks, 'failed': failed, 'passed': len(failed) == 0}


# ---------------------------------------------------------------------
# Step 5 -- toxicity/safety filtering
# (keyword-level stand-in; a real pipeline uses trained classifiers,
# including ones covering caste/communal slurs as described in the
# report -- this script filters against a caller-supplied wordlist so
# no slur list ships with the code itself)
# ---------------------------------------------------------------------
def step5_toxicity_ok(line, toxic_terms):
    if not toxic_terms:
        return True
    lowered = line.lower()
    return not any(term in lowered for term in toxic_terms)


# ---------------------------------------------------------------------
# Step 6 -- PII redaction
# ---------------------------------------------------------------------
def step6_redact_pii(line, stats):
    redacted = line
    for pat, tag in [(AADHAAR_RE, '[AADHAAR]'), (PAN_RE, '[PAN]'),
                      (PHONE_RE, '[PHONE]'), (EMAIL_RE, '[EMAIL]')]:
        redacted, n = pat.subn(tag, redacted)
        if n:
            stats['step6_pii_redacted'] += n
    return redacted


# ---------------------------------------------------------------------
# Step 7 -- benchmark decontamination (n-gram overlap vs an eval file)
# ---------------------------------------------------------------------
def build_eval_ngrams(eval_file, n=13):
    grams = set()
    if not eval_file or not os.path.exists(eval_file):
        return grams
    with open(eval_file, encoding='utf-8', errors='ignore') as f:
        words = f.read().split()
    for i in range(len(words) - n + 1):
        grams.add(tuple(words[i:i + n]))
    return grams


def step7_contaminated(line, eval_ngrams, n=13):
    if not eval_ngrams:
        return False
    words = line.split()
    if len(words) < n:
        return False
    return any(tuple(words[i:i + n]) in eval_ngrams for i in range(len(words) - n + 1))


# ---------------------------------------------------------------------
# Step 8 -- code-specific: scrub obvious secrets/credentials
# ---------------------------------------------------------------------
def step8_scrub_secrets(line, stats):
    new_line, n = SECRET_RE.subn('[REDACTED_SECRET]', line)
    if n:
        stats['step8_secrets_scrubbed'] += n
    return new_line


# ---------------------------------------------------------------------
# Step 9 -- math/science: track (and preserve) LaTeX spans
# ---------------------------------------------------------------------
def step9_protect_latex(line, stats):
    hits = LATEX_RE.findall(line)
    if hits:
        stats['step9_latex_spans_protected'] += len(hits)
    return line  # left untouched deliberately


# ======================================================================
# Pipeline driver
# ======================================================================
def clean_file(input_path, eval_file=None, toxic_terms_path=None, min_line_len=3,
                filter_curly_braces=False, minhash_threshold=0.85):
    with open(input_path, encoding='utf-8', errors='ignore') as f:
        raw_lines = f.readlines()

    raw_text = ''.join(raw_lines)
    raw_word_count = word_count(raw_text)

    toxic_terms = load_toxic_terms(toxic_terms_path)
    eval_ngrams = build_eval_ngrams(eval_file)
    exact_seen = set()
    minhash_deduper = MinHashDeduper(threshold=minhash_threshold)

    stats = Counter()
    stats['lang_counts'] = Counter()
    removed_samples = []

    # ---- Step 1 runs first on every line, since the Gopher report
    # below needs the whole normalized document, not raw text ----
    normalized_lines = []
    for raw_line in raw_lines:
        stats['lines_seen'] += 1
        line = step1_normalize(raw_line, stats)
        if line:
            normalized_lines.append(line)
        else:
            stats['removed_empty'] += 1

    # ---- Gopher document-level report: LOG-ONLY, never drops content ----
    gopher_report = gopher_document_report('\n'.join(normalized_lines))

    cleaned_lines = []
    for line in normalized_lines:
        step2_detect_lang(line, stats)  # informational, does not filter

        h = hash(line.strip().lower())
        if h in exact_seen:
            stats['removed_dup_exact'] += 1
            removed_samples.append(('dedup:exact', line))
            continue
        exact_seen.add(h)

        is_dup, _ = minhash_deduper.is_duplicate(line)
        if is_dup:
            stats['removed_dup_minhash'] += 1
            removed_samples.append(('dedup:minhash', line))
            continue

        ok, reason = step4_quality_ok(line, min_line_len)
        if not ok:
            stats[f'removed_quality_{reason}'] += 1
            removed_samples.append((f'quality:{reason}', line))
            continue

        ok, reason = c4_line_ok(line, filter_curly_braces=filter_curly_braces)
        if not ok:
            stats[f'removed_{reason}'] += 1
            removed_samples.append((reason, line))
            continue

        if not step5_toxicity_ok(line, toxic_terms):
            stats['removed_toxicity'] += 1
            removed_samples.append(('toxicity', line))
            continue

        if step7_contaminated(line, eval_ngrams):
            stats['removed_contamination'] += 1
            removed_samples.append(('decontam', line))
            continue

        line = step6_redact_pii(line, stats)
        line = step8_scrub_secrets(line, stats)
        line = step9_protect_latex(line, stats)

        cleaned_lines.append(line)
        stats['lines_kept'] += 1

    cleaned_text = '\n'.join(cleaned_lines) + '\n'
    return {
        'cleaned_text': cleaned_text,
        'raw_word_count': raw_word_count,
        'cleaned_word_count': word_count(cleaned_text),
        'stats': stats,
        'removed_samples': removed_samples[:200],
        'toxic_terms_supplied': bool(toxic_terms),
        'gopher_report': gopher_report,
    }


# ======================================================================
# Tokenization
# ======================================================================
def train_or_load_bpe(cleaned_text_path, tok_dir, vocab_size=32000):
    if not HAVE_TOKENIZERS:
        raise RuntimeError("Missing dependency. Install with: pip install tokenizers")
    os.makedirs(tok_dir, exist_ok=True)
    vocab_file = os.path.join(tok_dir, 'vocab.json')
    merges_file = os.path.join(tok_dir, 'merges.txt')

    if os.path.exists(vocab_file) and os.path.exists(merges_file):
        return ByteLevelBPETokenizer(vocab_file, merges_file)

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=[cleaned_text_path],
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=['<pad>', '<unk>', '<s>', '</s>', '<mask>'],
    )
    tokenizer.save_model(tok_dir)
    return tokenizer


# ======================================================================
# Tokenizer-only mode  (--tokenize_only FILE)
# ======================================================================
def run_tokenize_only(args):
    src_path = args.tokenize_only
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"--tokenize_only file not found: {src_path}")

    with open(src_path, encoding='utf-8', errors='ignore') as f:
        text = f.read()
    src_words = word_count(text)

    tok_dir = os.path.join(args.outdir, 'tokenizer')
    existing = os.path.exists(os.path.join(tok_dir, 'vocab.json'))

    t0 = time.time()
    tokenizer = train_or_load_bpe(src_path, tok_dir, args.vocab_size)
    encoding = tokenizer.encode(text)
    ids, pieces = encoding.ids, encoding.tokens
    elapsed = time.time() - t0

    base = os.path.splitext(os.path.basename(src_path))[0]
    tokens_path = os.path.join(args.outdir, f'{base}_tokens.json')
    with open(tokens_path, 'w', encoding='utf-8') as f:
        json.dump({'ids': ids, 'pieces': pieces}, f, ensure_ascii=False)

    log_path = os.path.join(args.outdir, f'{base}_tokenize_only_log.txt')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("TOKENIZE-ONLY LOG\n")
        f.write(f"Generated : {datetime.now().isoformat()}\n")
        f.write(f"Input file: {src_path}\n")
        f.write("=" * 72 + "\n\n")
        f.write(f"Tokenizer source      : {'loaded existing' if existing else 'trained fresh'} "
                f"({tok_dir})\n")
        f.write(f"Vocab size            : {args.vocab_size}\n")
        f.write(f"Input words            : {src_words:,}\n")
        f.write(f"Tokens generated       : {len(ids):,}\n")
        if src_words:
            f.write(f"Fertility (tok/word)   : {len(ids) / src_words:.3f}\n")
        f.write(f"Elapsed                : {elapsed:.2f}s\n")

    print(f"Tokenizer {'loaded from' if existing else 'trained and saved to'}: {tok_dir}")
    print(f"Words in  : {src_words:,}")
    print(f"Tokens out: {len(ids):,}  (fertility "
          f"{(len(ids) / src_words if src_words else 0):.3f} tok/word)")
    print(f"Tokens file: {tokens_path}")
    print(f"Log        : {log_path}")


# ======================================================================
# Main
# ======================================================================
def main():
    ap = argparse.ArgumentParser(description="Clean a raw text file and BPE-tokenize it")
    ap.add_argument('--input', default=None, help='Path to raw input text file (runs full clean+tokenize pipeline)')
    ap.add_argument('--outdir', default='cleanup', help='Output folder')
    ap.add_argument('--vocab_size', type=int, default=32000)
    ap.add_argument('--eval_file', default=None, help='Enables Step 7 decontamination')
    ap.add_argument('--toxic_terms_file', default=None, help='Enables real Step 5 filtering')
    ap.add_argument('--min_line_len', type=int, default=3)
    ap.add_argument('--filter_curly_braces', action='store_true',
                     help="Enable C4's curly-brace line filter (off by default -- "
                          "this pipeline may see code text)")
    ap.add_argument('--minhash_threshold', type=float, default=0.85,
                     help='Jaccard similarity threshold for MinHash/LSH near-dup (default 0.85)')
    ap.add_argument('--tokenize_only', default=None, metavar='FILE',
                     help='Skip cleaning entirely; just BPE-tokenize FILE using the '
                          'tokenizer already saved in <outdir>/tokenizer (trains a new '
                          'one from FILE if none exists yet). Useful for tokenizing '
                          'already-cleaned text, or for quickly checking fertility on '
                          'a new file without re-running the cleaning pipeline.')
    args = ap.parse_args()

    if not args.input and not args.tokenize_only:
        ap.error('provide either --input (full pipeline) or --tokenize_only (tokenizer only)')

    os.makedirs(args.outdir, exist_ok=True)

    # -------------------------------------------------------------
    # Tokenizer-only mode: --tokenize_only FILE
    # -------------------------------------------------------------
    if args.tokenize_only:
        run_tokenize_only(args)
        return

    base = os.path.splitext(os.path.basename(args.input))[0]

    t0 = time.time()
    result = clean_file(args.input, args.eval_file, args.toxic_terms_file, args.min_line_len,
                         filter_curly_braces=args.filter_curly_braces,
                         minhash_threshold=args.minhash_threshold)
    clean_elapsed = time.time() - t0

    cleaned_path = os.path.join(args.outdir, f'{base}_cleaned.txt')
    with open(cleaned_path, 'w', encoding='utf-8') as f:
        f.write(result['cleaned_text'])

    audit_path = os.path.join(args.outdir, f'{base}_human_audit_sample.txt')
    with open(audit_path, 'w', encoding='utf-8') as f:
        f.write("# Sample of REMOVED lines for manual native-speaker review (Step 10)\n")
        f.write("# format: [removal_reason]<TAB>original_line\n\n")
        for reason, line in result['removed_samples']:
            f.write(f'[{reason}]\t{line}\n')

    #tok_dir = os.path.join(args.outdir, 'tokenizer')
    #t1 = time.time()
    #tokenizer = train_or_load_bpe(cleaned_path, tok_dir, args.vocab_size)
    #encoding = tokenizer.encode(result['cleaned_text'])
    #ids, pieces = encoding.ids, encoding.tokens
    #tok_elapsed = time.time() - t1

    #tokens_path = os.path.join(args.outdir, f'{base}_tokens.json')
    #with open(tokens_path, 'w', encoding='utf-8') as f:
    #    json.dump({'ids': ids, 'pieces': pieces}, f, ensure_ascii=False)

    # ---------------- log file ----------------
    stats = result['stats']
    log_path = os.path.join(args.outdir, f'{base}_log.txt')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("CLEANING & TOKENIZATION LOG\n")
        f.write(f"Generated : {datetime.now().isoformat()}\n")
        f.write(f"Input file: {args.input}\n")
        f.write("=" * 72 + "\n\n")

        f.write("WORD COUNTS\n")
        f.write(f"  Input words (raw)     : {result['raw_word_count']:,}\n")
        f.write(f"  Cleaned words (kept)  : {result['cleaned_word_count']:,}\n")
        pct = (result['cleaned_word_count'] / result['raw_word_count'] * 100
               if result['raw_word_count'] else 0)
        f.write(f"  Retained              : {pct:.1f}%\n\n")

        f.write("LINE-LEVEL BREAKDOWN\n")
        f.write(f"  Lines seen            : {stats['lines_seen']:,}\n")
        f.write(f"  Lines kept            : {stats['lines_kept']:,}\n")
        for k in sorted(stats):
            if k.startswith('removed_'):
                f.write(f"  {k:<24}: {stats[k]:,}\n")
        f.write("\n")

        if stats['lang_counts']:
            f.write("LANGUAGES DETECTED (Step 2, informational -- top 10)\n")
            for lang, n in stats['lang_counts'].most_common(10):
                f.write(f"  {lang:<10}: {n:,} lines\n")
            f.write("\n")

        gr = result['gopher_report']
        f.write(f"GOPHER DOCUMENT-LEVEL REPORT (log-only -- content NOT dropped based on this)\n")
        f.write(f"  Overall: {'PASS' if gr['passed'] else 'FAIL -- ' + ', '.join(gr['failed'])}\n")
        for name, c in gr['checks'].items():
            mark = 'PASS' if c['pass'] else 'FAIL'
            f.write(f"  [{mark}] {name:<22}: {c['value']}  ({c['rule']})\n")
        f.write("\n")

        #f.write("TOKENIZATION\n")
        #f.write(f"  Method                : Byte-Level BPE (vocab_size={args.vocab_size})\n")
        #f.write(f"  Tokens generated      : {len(ids):,}\n")
        #if result['cleaned_word_count']:
        #    f.write(f"  Fertility (tok/word)  : {len(ids) / result['cleaned_word_count']:.3f}\n")
        #f.write("\n")

        f.write("CLEANING METHODS APPLIED THIS RUN (Section 5, steps 1-10)\n")
        for i in range(1, 11):
            if i == 5:
                mark = "APPLIED (custom wordlist)" if result['toxic_terms_supplied'] else "PASS-THROUGH (no --toxic_terms_file given)"
            elif i == 7:
                mark = "APPLIED" if args.eval_file else "SKIPPED (no --eval_file given)"
            elif i == 10:
                mark = "MANUAL -- sample exported for review"
            else:
                mark = "APPLIED"
            f.write(f"  Step {i:>2}: {STEP_NAMES[i]:<58} [{mark}]\n")
        f.write("\n")

        f.write("OUTPUT ARTIFACTS\n")
        f.write(f"  Cleaned text          : {cleaned_path}\n")
        #f.write(f"  Tokens (ids + pieces) : {tokens_path}\n")
        #f.write(f"  Tokenizer vocab/merges: {tok_dir}\n")
        f.write(f"  Human-audit sample    : {audit_path}\n\n")

        f.write("TIMING\n")
        f.write(f"  Cleaning pass         : {clean_elapsed:.2f}s\n")
        #f.write(f"  Tokenizer train+encode: {tok_elapsed:.2f}s\n")

    print(f"Done. Outputs in: {args.outdir}/")
    print(f"  cleaned : {cleaned_path}")
    #print(f"  tokens  : {tokens_path}")
    print(f"  log     : {log_path}")


if __name__ == '__main__':
    main()

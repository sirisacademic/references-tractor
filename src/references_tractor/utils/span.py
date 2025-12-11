from tqdm.auto import tqdm

from IPython.display import HTML
from IPython.display import display

def group_adjacent_spans(spans, text):
    """
    Group adjacent model spans into larger entities.
    Conditions:
    - Same entity_group
    - Adjacent or separated only by whitespace
    - But DO NOT merge if there is a newline in between.
    """
    if not spans:
        return []

    spans = sorted(spans, key=lambda x: x["start"])
    grouped = []
    current = spans[0].copy()

    for span in spans[1:]:
        same_label = span["entity_group"] == current["entity_group"]

        # Text between spans
        between = text[current["end"]:span["start"]]

        # Check for newline
        has_newline = "\n" in between

        # Check adjacency (allow whitespace)
        touching = span["start"] <= current["end"] + 1 or between.strip() == ""

        if same_label and touching and not has_newline:
            # Merge: extend end & text
            current["end"] = span["end"]
            current["word"] = text[current["start"]:current["end"]]
        else:
            # Start new group
            grouped.append(current)
            current = span.copy()

    grouped.append(current)
    return grouped

def chunk_text_by_newline(text, max_chars=2000):
    """
    Splits text into chunks at newline boundaries, ensuring each chunk is below max_chars.
    Returns a list of (chunk_text, global_offset_in_text).
    """
    lines = text.split("\n")
    chunks = []
    
    current_chunk = ""
    current_start = 0  # global offset of current chunk start
    cursor = 0         # global cursor tracking position in original text

    for line in lines:
        line_with_n = line + "\n"
        # if adding this line exceeds the limit → flush the chunk
        if len(current_chunk) + len(line_with_n) > max_chars and current_chunk.strip():
            chunks.append((current_chunk.rstrip("\n"), current_start))
            current_start = cursor  # new chunk begins here
            current_chunk = ""

        current_chunk += line_with_n
        cursor += len(line_with_n)

    # last chunk
    if current_chunk.strip():
        chunks.append((current_chunk.rstrip("\n"), current_start))

    return chunks

def extract_entities_from_chunk(chunk, offset, ner_pipeline, full_text):
    """Run pipeline on chunk, normalize results, and offset start/end."""
    ents = ner_pipeline(chunk)
    normalized = []
    
    for e in ents:
        start = e["start"] + offset
        end = e["end"] + offset
        normalized.append({
            "entity_group": e["entity_group"],
            "start": start,
            "end": end,
            "word": full_text[start:end],     # <-- FIXED HERE
            "score": e["score"]
        })
    
    return normalized

def extract_entities_large_text(text, ner_pipeline, max_chars=2000):
    chunks = chunk_text_by_newline(text, max_chars=max_chars)
    all_entities = []

    print(f"Processing {len(chunks)} chunks...")

    for chunk_text, offset in tqdm(chunks, desc="SPAN chunks", unit=" chunk"):
        entities = extract_entities_from_chunk(chunk_text, offset, ner_pipeline, text)
        all_entities.extend(entities)

    # Sort before grouping
    all_entities = sorted(all_entities, key=lambda x: x["start"])

    # Group adjacent spans
    grouped = group_adjacent_spans(all_entities, text)
    return grouped


def extract_reference_list(grouped_entities, text, max_gap=2):
    """
    Extract references as CITATION_SPAN entries.
    If the immediately previous entity is a CITATION_ID and nearby,
    use it as the ID for the reference.
    """
    references = []

    for i, ent in enumerate(grouped_entities):
        if ent["entity_group"] != "CITATION_SPAN":
            continue

        cid = None

        # look at the previous entity in the sequence
        if i > 0:
            prev = grouped_entities[i - 1]

            if prev["entity_group"] == "CITATION_ID":
                # ensure the ID is right before the span (small distance)
                if ent["start"] - prev["end"] <= max_gap:
                    cid = text[prev["start"]:prev["end"]].translate(str.maketrans("", "", "[]()")).strip()

        # store reference
        ref_text = text[ent["start"]:ent["end"]].replace("\n", " ").strip()

        references.append({
            "id": cid,
            "text": ref_text,
            "start": ent["start"],
            "end": ent["end"]
        })

    return references


def highlight_entities_html(text, entities):
    entities = sorted(entities, key=lambda e: e['start'])
    out = ""
    last_idx = 0

    colors = ["#ff9999","#99ff99","#9999ff","#ffcc99","#cc99ff","#99ffff"]

    for ent in entities:
        start, end, label = ent["start"], ent["end"], ent["label"]
        out += text[last_idx:start]
        color = colors[hash(label) % len(colors)]
        out += f"<mark style='background-color:{color}'>{text[start:end]}</mark><sub><b>{label}</b></sub>"
        last_idx = end

    out += text[last_idx:]

    # 🔥 NEW: convert newlines to <br> for pretty markdown formatting
    out = out.replace("\n", "<br>\n")

    # Optional: preserve spaces (good for indentation in markdown)
    out = out.replace("  ", "&nbsp;&nbsp;")

    return HTML(f"<div style='font-family:monospace; white-space:pre-wrap;'>{out}</div>")

def extract_citation_groups(grouped_entities, text, max_gap=1):
    """
    Build unified citation groups:
    - AUTHOR, YEAR, CITATION_ID
    - Connected by CITATION_REF
    - No APA/numeric distinction
    """
    grouped_entities = sorted(grouped_entities, key=lambda e: e["start"])
    
    groups = []
    current = None

    def flush_current():
        nonlocal current
        if current:
            start = current["start"]
            end = current["end"]
            current["text"] = text[start:end]
            groups.append(current)
            current = None

    for i, ent in enumerate(grouped_entities):
        kind = ent["entity_group"]

        # Only these begin/continue citations
        if kind in ("AUTHOR", "YEAR", "CITATION_ID", "CITATION_REF"):

            # Start a new citation group
            if current is None:
                current = {
                    "entities": [ent],
                    "start": ent["start"],
                    "end": ent["end"]
                }
                continue

            # Continue an open citation group
            prev_end = current["end"]
            gap = ent["start"] - prev_end

            if gap <= max_gap:
                current["entities"].append(ent)
                current["end"] = ent["end"]
            else:
                # Too big a gap → end group and start new one
                flush_current()
                current = {
                    "entities": [ent],
                    "start": ent["start"],
                    "end": ent["end"]
                }
        else:
            # Entity is not citation-related → close group
            flush_current()

    # Final flush
    flush_current()

    return groups


import nltk
nltk.download("punkt")

import re

def smart_join_lines(md_text):
    """
    Extremely safe paragraph joiner to avoid breaking sentence boundaries.
    Only joins lines when:
      - line does NOT end a sentence
      - next line starts with lowercase or continuation punctuation
    NEVER joins:
      - headers
      - tables
      - list items
      - code blocks
      - next line starting with uppercase (This, How, One...)
      - next line starting with punctuation
    """
    lines = md_text.split("\n")
    cleaned = []

    end_punct = re.compile(r'[.!?]"?\'?\s*$')

    def is_header(line):
        return line.lstrip().startswith("#")

    def is_table(line):
        return line.strip().startswith("|") and line.strip().endswith("|")

    def is_list_item(line):
        return re.match(r'^(\*|-|\d+\.)\s+', line.lstrip())

    inside_code = False

    for i, line in enumerate(lines):
        curr = line.rstrip()

        # handle fenced blocks
        if curr.strip().startswith("```"):
            inside_code = not inside_code
            cleaned.append(curr)
            continue
        if inside_code:
            cleaned.append(curr)
            continue

        if curr.strip() == "":
            cleaned.append("")
            continue

        # last line
        if i == len(lines) - 1:
            cleaned.append(curr)
            continue

        nxt = lines[i+1].rstrip()

        # never join headers
        if is_header(curr):
            cleaned.append(curr)
            continue
        if is_header(nxt):
            cleaned.append(curr)
            continue

        # never join tables or lists
        if is_table(curr) or is_table(nxt):
            cleaned.append(curr)
            continue
        if is_list_item(nxt):
            cleaned.append(curr)
            continue

        # do not join if curr ends a sentence
        if end_punct.search(curr):
            cleaned.append(curr)
            continue

        # ❌ NEW SAFETY RULE
        # never join if next line starts with capital letter
        if nxt[:1].isupper():
            cleaned.append(curr)
            continue

        # never join if next begins with punctuation
        if nxt[:1] in ",.;:)]}":
            cleaned.append(curr)
            continue

        # join only if continuation (lowercase)
        if nxt[:1].islower():
            cleaned.append(curr + " ")
            continue

        # fallback: keep newline
        cleaned.append(curr)

    return "\n".join(cleaned)

def attach_sentences_to_citations(citation_groups, text):
    """
    Attach:
      - sentence string
      - absolute sentence_start
      - absolute sentence_end
    to each citation group.
    """

    # Step 1: Join broken markdown lines intelligently
    clean_md = smart_join_lines(text)

    # Step 2: Sentence tokenize the cleaned version
    sentences = nltk.sent_tokenize(clean_md)

    # Step 3: Find absolute positions in clean_md
    spans = []
    cursor = 0
    for sent in sentences:
        start = clean_md.find(sent, cursor)
        if start == -1:
            continue
        end = start + len(sent)
        spans.append((sent, start, end))
        cursor = end

    # Step 4: Now map citation spans from ORIGINAL text to clean_md
    # Since join-lines only removes some newlines, the content is still aligned.
    results = []

    for group in citation_groups:
        g_start = group["start"]
        g_end = group["end"]

        sentence, s_start, s_end = None, None, None

        # Find the sentence in which the citation falls
        for sent, ss, ee in spans:
            if g_start >= ss and g_end <= ee:
                sentence = sent
                s_start = ss
                s_end = ee
                break

        results.append({
            **group,
            "sentence": sentence,
            "sentence_start": s_start,
            "sentence_end": s_end
        })

    return results

def mask_target_citation(citation_group, text):
    """
    Return the sentence with ONLY the target citation masked as [REF].
    No other citations are removed or modified.
    Newlines inside the sentence are preserved.
    """
    sent_start = citation_group["sentence_start"]
    sent_end = citation_group["sentence_end"]

    if sent_start is None or sent_end is None:
        return None

    # extract original sentence (preserve whitespace and \n)
    sentence = text[sent_start:sent_end]

    # convert citation positions into relative offsets
    rel_start = citation_group["start"] - sent_start
    rel_end   = citation_group["end"] - sent_start

    # mask only the target citation
    masked = sentence[:rel_start] + "[REF]" + sentence[rel_end:]

    return masked

def extract_references_and_mentions(md_text, ner_pipeline, plot=False):
    """
    Full pipeline:
        1. NER over large MD text (chunk-safe)
        2. Highlight entities (optional)
        3. Extract reference list
        4. Extract citation groups (AUTHOR/YEAR/ID combinations)
        5. Attach sentence to each citation group
        6. Produce masked sentence for each mention
    """

    print("🔍 Running NER on full text...")
    grouped_entities = extract_entities_large_text(md_text, ner_pipeline)

    # Optional highlighting
    if plot:
        display(highlight_entities_html(
            md_text,
            [
                {"start": e["start"], "end": e["end"], "label": e["entity_group"]}
                for e in grouped_entities
            ]
        ))

    print("📚 Extracting reference list...")
    references = extract_reference_list(grouped_entities, md_text)

    print("🔗 Grouping in-text citations...")
    citation_groups = extract_citation_groups(grouped_entities, md_text, max_gap=3)

    print("🧩 Attaching sentences...")
    processed = attach_sentences_to_citations(citation_groups, md_text)

    print("✨ Masking citations in context...")
    mentions = []
    for cg in processed:
        masked = mask_target_citation(cg, md_text)
        mentions.append({
            **cg,
            "masked_sentence": None if masked is None else masked.replace("\n", " ")
        })

    # print("✅ Completed.")
    return {
        "references": references,
        "mentions": mentions,
    }


import re
import unidecode

# ---------------------------
# Helpers
# ---------------------------

YEAR_PATTERN = re.compile(r"\b(19|20)\d{4}?\b")  # allows 2022a, 2023b, etc.

def normalize_author(author: str) -> str:
    if not author:
        return ""
    
    author = unidecode.unidecode(author)
    author = author.lower()

    author = author.replace("&amp;", " ")
    author = re.sub(r"\bet al\.?", "", author)
    author = re.sub(r"\b[a-z]\.\b", " ", author)
    author = re.sub(r"[^a-z ]", " ", author)

    author = re.sub(r"\s+", " ", author).strip()
    return author.split()[0] if author else ""


# ---------------------------
# NER → reference key extraction
# ---------------------------

def extract_reference_keys(ref_ner):
    """
    Extract:
    - ref_surnames (set of normalized surname candidates)
    - ref_year (integer)
    """
    authors_raw = ref_ner.get("AUTHORS", [])
    years_raw = ref_ner.get("PUBLICATION_YEAR", [])

    # ---- YEAR ----
    year = None
    if years_raw:
        m = re.search(r"\b(19|20)\d{2}\b", years_raw[0])
        if m:
            year = int(m.group(0))

    # ---- AUTHORS → possible surnames ----
    surnames = set()
    if authors_raw:
        raw = authors_raw[0]
        chunks = re.split(r"[,&]| and ", raw)
        for ch in chunks:
            norm = normalize_author(ch)
            if norm:
                surnames.add(norm)

    return surnames, year


# ---------------------------
# Mention (text) → surname + year
# ---------------------------

def extract_mention_keys(mention_text: str):
    """
    From text like:
        "Ardito et al., 2023"
    return:
        ("ardito", 2023)
    """
    # YEAR
    m = re.search(r"\b(19|20)\d{2}\b", mention_text)
    year = int(m.group(0)) if m else None

    # AUTHOR = text before comma
    chunk = mention_text.split(",")[0]
    surname = normalize_author(chunk)

    return surname, year


def match_mentions_to_reference(ref, mentions):
    """
    Priority:
        1. If reference has citation_id → match only by CITATION_ID.
        2. Else → match by AUTHOR + YEAR using NER.
    """

    # ---------------------------
    # 1. Match via CITATION_ID
    # ---------------------------
    citation_id = ref.get("id")
    if citation_id is not None:
        cid = str(citation_id)
        matched = []

        for m in mentions:
            # Extract citation_id from mention entities
            for ent in m["entities"]:
                if ent["entity_group"] == "CITATION_ID":
                    if ent["word"] == cid:
                        matched.append(m["masked_sentence"])
                        break  # stop checking this mention

        return matched[:-1]  # ← STOP HERE. Numeric-ID takes priority.


    # ---------------------------
    # 2. Match via AUTHOR + YEAR
    # ---------------------------
    ref_ner = ref.get("ner", {})
    ref_surnames, ref_year = extract_reference_keys(ref_ner)

    # If NER failed → no matching possible
    if not ref_surnames or not ref_year:
        return []

    ref_year_str = str(ref_year)
    matched = []

    for m in mentions:
        text_norm = unidecode.unidecode(m["text"]).lower()

        # YEAR requirement
        if ref_year_str not in text_norm:
            continue

        # SURNAME requirement
        if any(surname in text_norm for surname in ref_surnames):
            matched.append(m["masked_sentence"])

    return matched
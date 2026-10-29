"""
Patch Note Pulse — core NLP pipeline (no UI code here).

Flow:  fetch wikitext -> clean markup -> split into one block per champion/item
       -> classify each change line (buff / nerf / rework) -> aggregate per entity
       -> (optional) LLM summary prompt.

Kept free of Streamlit so it can be imported by the app, the notebook and the tests.
"""

from __future__ import annotations

import re

import pandas as pd
import requests

WIKI_API_URL = "https://wiki.leagueoflegends.com/en-us/api.php"
REQUEST_HEADERS = {"User-Agent": "PatchNotePulse-StudentProject/1.0 (NLP course project)"}


# ============================================================
# PHASE 1 — DATA ACQUISITION
# ============================================================
def get_patch_notes_wikitext(patch_label: str, timeout: int = 15) -> str:
    """Return the raw wikitext of the wiki page for a patch label such as '26.14'."""
    page_title = f"V{patch_label.strip()}"
    params = {"action": "parse", "page": page_title, "prop": "wikitext", "format": "json"}
    response = requests.get(WIKI_API_URL, params=params, headers=REQUEST_HEADERS, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    if "error" in data:
        raise ValueError(f"Could not find wiki page '{page_title}': {data['error'].get('info', data['error'])}")
    return data["parse"]["wikitext"]["*"]


# ============================================================
# PHASE 2 — TEXT PROCESSING
# ============================================================
# The wiki formats names and numbers through templates, e.g. {{ci|Aatrox}},
# {{ai|Infernal Chains|Aatrox}}, {{fd|0.658}}, {{as|3% per 100 AP}}. Templates can
# nest ({{as|{{ap|3*2}}% per 100 AP}}), so we flatten from the inside out until
# nothing changes instead of doing a single pass.
NAME_TEMPLATE_RE = re.compile(r"\{\{(?:ai|ii|ci|si|sbc|csl|ri|ui|bi|cai|ais|tip)\|([^{}|]+)[^{}]*\}\}")
VALUE_TEMPLATE_RE = re.compile(r"\{\{(?:fd|ap|as|pp|g|gold)\|([^{}|]+)[^{}]*\}\}")
GENERIC_TEMPLATE_RE = re.compile(r"\{\{[^{}]*\}\}")
MAX_TEMPLATE_PASSES = 20


def clean_wikitext(raw_text: str) -> str:
    """Strip wiki markup down to plain, readable text while keeping numeric values."""
    text = re.sub(r"<!--.*?-->", "", raw_text, flags=re.DOTALL)
    text = re.sub(r"<ref[^>]*?/>", "", text)
    text = re.sub(r"<ref.*?>.*?</ref>", "", text, flags=re.DOTALL)
    text = re.sub(r"\[\[([^|\]]+)\|([^\]]+)\]\]", r"\2", text)  # [[target|label]] -> label
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)              # [[target]]       -> target
    for _ in range(MAX_TEMPLATE_PASSES):
        if "{{" not in text:
            break
        new_text = VALUE_TEMPLATE_RE.sub(r"\1", NAME_TEMPLATE_RE.sub(r"\1", text))
        if new_text == text:
            # Only drop unknown templates once nothing else can be unwrapped; doing it
            # in the same pass would delete an outer {{as|...}} the moment its inner
            # {{ap|...}} got flattened, losing the value.
            new_text = GENERIC_TEMPLATE_RE.sub("", text)
        if new_text == text:
            break
        text = new_text
    text = text.replace("'''", "").replace("''", "")
    return re.sub(r"\n{3,}", "\n\n", text)


# ============================================================
# PHASE 3 — ENTITY BLOCK SPLITTING
# ============================================================
# Inside "=== Champions ===" / "=== Items ===" every entity starts on its own line
# as ";{{ci|Aatrox}}" or ";{{ii|Horizon Focus}}". Splitting on those markers gives
# exact names (no fuzzy matching) and the type comes from the tag (ci / ii).
SECTION_RE = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$", re.MULTILINE)
ENTITY_RE = re.compile(r"^;\{\{(ci|ii)\|([^{}|]+?)(?:\|[^{}]*)?\}\}\s*$", re.MULTILINE)
ENTITY_TYPES = {"ci": "champion", "ii": "item"}
SECTIONS_TO_SCAN = ("Champions", "Items")


def get_section_span(text: str, section_name: str) -> tuple[int | None, int | None]:
    """(start, end) offsets of a section's body.

    The section ends at the next header of the *same or higher* level, so any
    sub-headers inside it (e.g. "==== Buffs ====") stay part of the section.
    """
    headers = list(SECTION_RE.finditer(text))
    for i, header in enumerate(headers):
        if header.group(2).strip().lower() != section_name.lower():
            continue
        level = len(header.group(1))
        end = len(text)
        for later in headers[i + 1:]:
            if len(later.group(1)) <= level:
                end = later.start()
                break
        return header.end(), end
    return None, None


def split_into_entity_blocks(raw_text: str) -> list[dict]:
    """One {'entity', 'entity_type', 'text'} dict per champion/item change block."""
    blocks = []
    for section_name in SECTIONS_TO_SCAN:
        start, end = get_section_span(raw_text, section_name)
        if start is None:
            continue
        section_text = raw_text[start:end]
        markers = list(ENTITY_RE.finditer(section_text))
        for i, marker in enumerate(markers):
            body_end = markers[i + 1].start() if i + 1 < len(markers) else len(section_text)
            body = clean_wikitext(section_text[marker.end():body_end]).strip()
            if body:
                blocks.append({
                    "entity": marker.group(2).strip(),
                    "entity_type": ENTITY_TYPES[marker.group(1)],
                    "text": body,
                })
    return blocks


# ============================================================
# PHASE 4 — BUFF / NERF CLASSIFICATION
# ============================================================
# "Cooldown reduced" is a buff but "damage reduced" is a nerf: whether a change is
# good depends on WHICH attribute moved, not only on the direction.
CHANGE_LINE_RE = re.compile(
    r"(?P<attribute>[A-Za-z' ]{3,40}?)\s+(?P<direction>increased|reduced|decreased|changed)\s+to\s+"
    r"(?P<new>[^\n]+?)\s+from\s+(?P<old>[^\n.]+)",
    re.IGNORECASE,
)
# Checked FIRST: these words decide the meaning even when combined with a "good"
# word, e.g. "shield cooldown" or "health cost" behave like a cooldown / a cost.
BUFF_WHEN_DECREASED = ("cooldown", "cost", "cast time", "delay", "recharge")
BUFF_WHEN_INCREASED = (
    "damage", "health", "range", "attack speed", "armor", "magic resist", "shield",
    "heal", "movement speed", "ratio", "duration", "ability power", "lethality",
    "penetration", "regen", "critical",
)


def classify_line(line: str) -> str | None:
    """Return 'buff', 'nerf', 'rework' or None (line isn't a recognisable change)."""
    match = CHANGE_LINE_RE.search(line)
    if not match:
        return None
    attribute = match.group("attribute").strip().lower()
    direction = match.group("direction").lower()
    if direction == "changed":
        return "rework"
    increased = direction == "increased"
    if any(key in attribute for key in BUFF_WHEN_DECREASED):
        return "nerf" if increased else "buff"
    if any(key in attribute for key in BUFF_WHEN_INCREASED):
        return "buff" if increased else "nerf"
    return None


def classify_block(block: dict) -> dict:
    """Aggregate line-level tags into one label for the whole entity."""
    lines = [line.strip("*: ").strip() for line in block["text"].split("\n") if line.strip()]
    tags = [tag for tag in map(classify_line, lines) if tag]
    buffs, nerfs, reworks = tags.count("buff"), tags.count("nerf"), tags.count("rework")

    if reworks and reworks >= max(buffs, nerfs):
        overall = "rework"
    elif buffs and nerfs:
        overall = "mixed"
    elif buffs:
        overall = "buff"
    elif nerfs:
        overall = "nerf"
    else:
        overall = "unclear"

    return {
        "entity": block["entity"],
        "entity_type": block["entity_type"],
        "classification": overall,
        "buff_lines": buffs,
        "nerf_lines": nerfs,
        "rework_lines": reworks,
        "raw_text": block["text"],
    }


def analyze_wikitext(raw_text: str) -> pd.DataFrame:
    """Phases 2-4 on text you already have (handy for tests and offline runs)."""
    return pd.DataFrame([classify_block(b) for b in split_into_entity_blocks(raw_text)])


def run_pipeline(patch_label: str) -> pd.DataFrame:
    """Phases 1-4 end to end for one patch label."""
    return analyze_wikitext(get_patch_notes_wikitext(patch_label))


# ============================================================
# PHASE 5 — LLM SUMMARY PROMPT
# ============================================================
def build_prompt(patch_df: pd.DataFrame, patch_label: str) -> str:
    rows = "\n".join(
        f"- {row.entity} ({row.entity_type}): {row.classification} "
        f"({row.buff_lines} buff line(s), {row.nerf_lines} nerf line(s))"
        for row in patch_df.itertuples()
    )
    return (
        f"You are a League of Legends patch note analyst. Summarize the following "
        f"structured list of balance changes from patch {patch_label} into a short, "
        f"friendly paragraph (4-6 sentences) a casual player could read in 30 seconds. "
        f"Call out the most notable buffs and nerfs by name. Do not invent details that "
        f"aren't in the list below.\n\nStructured changes:\n{rows}\n\nSummary:"
    )

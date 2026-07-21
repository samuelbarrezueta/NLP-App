"""
Patch Note Pulse — Streamlit prototype
========================================
League of Legends Wiki API (patch text) -> clean -> split by champion/item
entity marker -> classify buff/nerf/rework -> optional LLM summary (Anthropic Claude).

Deploy this on Streamlit Community Cloud (share.streamlit.io) or Hugging Face Spaces.
Add your Anthropic key to the platform's Secrets settings before deploying (never hard-code it).
"""

import re
import requests
import pandas as pd
import streamlit as st

# ============================================================
# CONFIG
# ============================================================
WIKI_API_URL = "https://wiki.leagueoflegends.com/en-us/api.php"
REQUEST_HEADERS = {"User-Agent": "PatchNotePulse-StudentProject/1.0 (NLP course project)"}
ANTHROPIC_SECRET_NAME = "NLP_WORK"

st.set_page_config(page_title="Patch Note Pulse", page_icon="🎮", layout="wide")


# ============================================================
# PHASE 1 — DATA ACQUISITION  (cached so we don't hammer the API on every rerun)
# ============================================================
@st.cache_data(ttl=3600)
def get_patch_notes_wikitext(patch_version_label):
    page_title = f"V{patch_version_label}"
    params = {"action": "parse", "page": page_title, "prop": "wikitext", "format": "json"}
    response = requests.get(WIKI_API_URL, params=params, headers=REQUEST_HEADERS, timeout=15)
    response.raise_for_status()
    data = response.json()
    if "error" in data:
        raise ValueError(f"Could not find wiki page '{page_title}': {data['error']}")
    return data["parse"]["wikitext"]["*"]


# ============================================================
# PHASE 2 — TEXT PROCESSING
# ============================================================
# The wiki formats numbers and names through templates, e.g. {{ci|Aatrox}},
# {{ai|Infernal Chains|Aatrox}}, {{fd|0.658}}, {{as|3% per 100 AP}}. Some of
# these nest inside each other (e.g. {{as|{{ap|3*2}}% per 100 AP}}), so we
# flatten repeatedly until nothing changes, instead of a single pass.
NAME_TEMPLATE_RE = re.compile(r"\{\{(?:ai|ii|ci|si|sbc|csl|ri|ui|bi|cai|ais|tip)\|([^\{\}\|]+)[^\{\}]*\}\}")
VALUE_TEMPLATE_RE = re.compile(r"\{\{(?:fd|ap|as|pp|g|gold)\|([^\{\}\|]+)[^\{\}]*\}\}")
GENERIC_TEMPLATE_RE = re.compile(r"\{\{[^\{\}]*\}\}")


def clean_wikitext(raw_text):
    """Strip wiki markup down to plain, readable text, keeping numeric values intact."""
    text = raw_text
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    text = re.sub(r"<ref.*?>.*?</ref>", "", text, flags=re.DOTALL)
    text = re.sub(r"<ref.*?/>", "", text)
    text = re.sub(r"\[\[([^\|\]]+)\|([^\]]+)\]\]", r"\2", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
    for _ in range(8):  # unwind nested templates from the inside out
        if "{{" not in text:
            break
        new_text = NAME_TEMPLATE_RE.sub(r"\1", text)
        new_text = VALUE_TEMPLATE_RE.sub(r"\1", new_text)
        new_text = GENERIC_TEMPLATE_RE.sub("", new_text)
        if new_text == text:
            break
        text = new_text
    text = text.replace("'''", "").replace("''", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


# ============================================================
# PHASE 3 — ENTITY BLOCK SPLITTING
# ============================================================
# Current wiki markup doesn't use one "==ChampionName==" header per entity.
# Instead, inside "=== Champions ===" / "=== Items ===" each entity starts
# a line of its own: ";{{ci|Aatrox}}" or ";{{ii|Horizon Focus}}". We split
# on those markers directly, so the entity name is exact (no fuzzy matching
# needed) and the entity type comes straight from the tag (ci=champion,
# ii=item).
SECTION_RE = re.compile(r"^={2,6}\s*(.+?)\s*={2,6}\s*$", re.MULTILINE)
ENTITY_RE = re.compile(r"^;\{\{(ci|ii)\|([^\{\}\|]+?)(?:\|[^\{\}]*)?\}\}\s*$", re.MULTILINE)


def get_section_span(text, section_name):
    """Return the (start, end) character offsets for the body of a top-level section."""
    headers = list(SECTION_RE.finditer(text))
    for i, h in enumerate(headers):
        if h.group(1).strip().lower() == section_name.lower():
            start = h.end()
            end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
            return start, end
    return None, None


def split_into_entity_blocks(raw_text):
    """Find every champion/item change block inside the Champions and Items sections."""
    blocks = []
    for section_name, entity_type in [("Champions", "champion"), ("Items", "item")]:
        start, end = get_section_span(raw_text, section_name)
        if start is None:
            continue
        section_text = raw_text[start:end]
        markers = list(ENTITY_RE.finditer(section_text))
        for i, m in enumerate(markers):
            name = m.group(2).strip()
            body_start = m.end()
            body_end = markers[i + 1].start() if i + 1 < len(markers) else len(section_text)
            body_clean = clean_wikitext(section_text[body_start:body_end]).strip()
            if body_clean:
                blocks.append({"entity": name, "entity_type": entity_type, "text": body_clean})
    return blocks


# ============================================================
# PHASE 4 — BUFF / NERF CLASSIFICATION
# ============================================================
CHANGE_LINE_RE = re.compile(
    r"(?P<attribute>[A-Za-z' ]{3,40}?)\s+(?P<direction>increased|reduced|changed)\s+to\s+"
    r"(?P<new>[^\n]+?)\s+from\s+(?P<old>[^\n\.]+)",
    re.IGNORECASE
)
BUFF_WHEN_INCREASED = {
    "damage", "health", "range", "attack speed", "armor", "magic resistance",
    "shield", "heal", "healing", "movement speed", "ap ratio", "ad ratio", "duration",
    "attack damage", "ability power"
}
BUFF_WHEN_DECREASED = {"cooldown", "cost", "mana cost", "cast time", "delay", "lethality"}


def classify_line(line):
    m = CHANGE_LINE_RE.search(line)
    if not m:
        return None
    attribute = m.group("attribute").strip().lower()
    direction = m.group("direction").lower()
    if direction == "changed":
        return "rework"
    increased = direction == "increased"
    if any(key in attribute for key in BUFF_WHEN_INCREASED):
        return "buff" if increased else "nerf"
    if any(key in attribute for key in BUFF_WHEN_DECREASED):
        return "nerf" if increased else "buff"
    return None


def classify_block(block):
    lines = [l.strip("* ").strip() for l in block["text"].split("\n") if l.strip()]
    tags = [classify_line(l) for l in lines]
    tags = [t for t in tags if t]
    buff_count = tags.count("buff")
    nerf_count = tags.count("nerf")
    rework_count = tags.count("rework")
    if rework_count >= buff_count and rework_count >= nerf_count and rework_count > 0:
        overall = "rework"
    elif buff_count > 0 and nerf_count > 0:
        overall = "mixed"
    elif buff_count > nerf_count:
        overall = "buff"
    elif nerf_count > buff_count:
        overall = "nerf"
    else:
        overall = "unclear"
    return {
        "entity": block["entity"], "entity_type": block["entity_type"],
        "classification": overall, "buff_lines": buff_count, "nerf_lines": nerf_count,
        "rework_lines": rework_count, "raw_text": block["text"]
    }


def run_pipeline(patch_label):
    raw_text = get_patch_notes_wikitext(patch_label)
    blocks = split_into_entity_blocks(raw_text)
    classified = [classify_block(b) for b in blocks]
    return pd.DataFrame(classified)


# ============================================================
# PHASE 5 — LLM SUMMARY (optional — only runs if an Anthropic key is configured)
# ============================================================
def build_prompt(patch_df, patch_label):
    rows = [
        f"- {row['entity']} ({row['entity_type']}): {row['classification']} "
        f"({row['buff_lines']} buff line(s), {row['nerf_lines']} nerf line(s))"
        for _, row in patch_df.iterrows()
    ]
    table_text = "\n".join(rows)
    return (
        f"You are a League of Legends patch note analyst. Summarize the following "
        f"structured list of balance changes from patch {patch_label} into a short, "
        f"friendly paragraph (4-6 sentences) a casual player could read in 30 seconds. "
        f"Call out the most notable buffs and nerfs by name. Do not invent details that "
        f"aren't in the list below.\n\nStructured changes:\n{table_text}\n\nSummary:"
    )


def generate_patch_summary(patch_df, patch_label):
    import anthropic
    api_key = st.secrets.get(ANTHROPIC_SECRET_NAME, None)
    if not api_key:
        return None
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=250,
        temperature=0.5,
        messages=[{"role": "user", "content": build_prompt(patch_df, patch_label)}],
    )
    return response.content[0].text.strip()


# ============================================================
# PHASE 6 — UI
# ============================================================
st.title("🎮 Patch Note Pulse")
st.caption("Turns a wall of League of Legends balance jargon into a 30-second digest.")

patch_label = st.text_input(
    "Patch version to analyze (e.g. 26.14)",
    value="26.14",
    help="Riot renamed patches to a year.patch format in 2025 (e.g. 26.14 = 2026, patch 14). "
         "Data Dragon's internal client version no longer matches this — type the patch label "
         "exactly as shown on leagueoflegends.com's official patch notes page."
)

if st.button("Analyze patch", type="primary"):
    try:
        with st.spinner(f"Pulling and analyzing patch {patch_label}..."):
            patch_df = run_pipeline(patch_label)

        if patch_df.empty:
            st.warning("No champion/item changes were detected for this patch. Try a different label.")
        else:
            summary = generate_patch_summary(patch_df, patch_label)
            if summary:
                st.subheader("📝 What this patch means")
                st.write(summary)
            else:
                st.info(f"Add your Anthropic key as '{ANTHROPIC_SECRET_NAME}' in this app's Secrets to enable the LLM summary paragraph.")

            st.subheader("📊 Buff vs. Nerf breakdown")
            counts = patch_df["classification"].value_counts()
            st.bar_chart(counts)

            st.subheader("📋 Full change table")

            def highlight(row):
                color = {"buff": "#d4f7d4", "nerf": "#f7d4d4", "mixed": "#f7f0d4", "rework": "#d4e4f7"}
                return [f"background-color: {color.get(row['classification'], '#ffffff')}"] * len(row)

            st.dataframe(
                patch_df[["entity", "entity_type", "classification", "buff_lines", "nerf_lines"]]
                .style.apply(highlight, axis=1),
                use_container_width=True
            )

            with st.expander("How this works"):
                st.write(
                    "Patch note text comes from the official League of Legends Wiki API. The text is "
                    "split into one block per champion/item using the wiki's own entity markers "
                    "(so names and types come straight from the source, no guesswork), then each "
                    "change line is classified as a buff or nerf based on which stat changed and in "
                    "which direction. An LLM (if configured) turns the structured result into a "
                    "plain-English summary."
                )
    except ValueError as e:
        st.error(f"{e}\n\nTry a different patch label — not every patch has its own wiki page.")

"""
Patch Note Pulse — Streamlit prototype
========================================
This mirrors the exact pipeline built and tested in Patch_Note_Pulse.ipynb:
Data Dragon (entity dictionary) + LoL Fandom Wiki API (patch text)
-> clean -> split by header -> fuzzy-match entities -> classify buff/nerf/rework
-> optional LLM summary.

Deploy this on Streamlit Community Cloud (share.streamlit.io) or Hugging Face Spaces.
Add OPENAI_API_KEY to the platform's Secrets settings before deploying (never hard-code it).
"""

import re
import requests
import pandas as pd
import streamlit as st
from rapidfuzz import process, fuzz

# ============================================================
# CONFIG
# ============================================================
DDRAGON_VERSIONS_URL = "https://ddragon.leagueoflegends.com/api/versions.json"
WIKI_API_URL = "https://leagueoflegends.fandom.com/api.php"
REQUEST_HEADERS = {"User-Agent": "PatchNotePulse-StudentProject/1.0 (NLP course project)"}

st.set_page_config(page_title="Patch Note Pulse", page_icon="🎮", layout="wide")


# ============================================================
# PHASE 1 — DATA ACQUISITION  (cached so we don't hammer the APIs on every rerun)
# ============================================================
@st.cache_data(ttl=3600)
def get_available_versions():
    response = requests.get(DDRAGON_VERSIONS_URL, timeout=15)
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=3600)
def get_champion_data(version):
    url = f"https://ddragon.leagueoflegends.com/cdn/{version}/data/en_US/champion.json"
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    return response.json()["data"]


@st.cache_data(ttl=3600)
def get_item_data(version):
    url = f"https://ddragon.leagueoflegends.com/cdn/{version}/data/en_US/item.json"
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    return response.json()["data"]


def ddragon_version_to_patch_label(version_string):
    parts = version_string.split(".")
    return f"{parts[0]}.{parts[1]}"


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
def clean_wikitext(raw_text):
    """Strip the most common wiki markup down to plain, readable text."""
    text = raw_text
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    text = re.sub(r"<ref.*?>.*?</ref>", "", text, flags=re.DOTALL)
    text = re.sub(r"<ref.*?/>", "", text)
    text = re.sub(r"\[\[([^\|\]]+)\|([^\]]+)\]\]", r"\2", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\{\{(?:ai|ii|ci|si|sbc)\|([^\}\|]+)[^\}]*\}\}", r"\1", text)
    text = re.sub(r"\{\{[^\{\}]*\}\}", "", text)
    text = text.replace("'''", "").replace("''", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


HEADER_RE = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$", re.MULTILINE)


def split_into_header_blocks(cleaned_text):
    matches = list(HEADER_RE.finditer(cleaned_text))
    blocks = []
    for i, m in enumerate(matches):
        header_text = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(cleaned_text)
        body = cleaned_text[start:end].strip()
        if body:
            blocks.append({"header": header_text, "text": body})
    return blocks


# ============================================================
# PHASE 3 — ENTITY RECOGNITION
# ============================================================
QUALIFIER_RE = re.compile(r"\b(buffs?|nerfs?|reworks?|changes?|adjustments?|updates?)\b", re.IGNORECASE)


def match_entity(header_text, champion_names, item_names, score_cutoff=85):
    candidate_text = QUALIFIER_RE.sub("", header_text).strip()
    if not candidate_text:
        return None, None
    combined = champion_names + item_names
    match = process.extractOne(candidate_text, combined, scorer=fuzz.WRatio, score_cutoff=score_cutoff)
    if match is None:
        return None, None
    name, score, _ = match
    entity_type = "champion" if name in champion_names else "item"
    return name, entity_type


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


def run_pipeline(patch_label, champion_names, item_names):
    raw_text = get_patch_notes_wikitext(patch_label)
    cleaned = clean_wikitext(raw_text)
    blocks = split_into_header_blocks(cleaned)
    for b in blocks:
        b["entity"], b["entity_type"] = match_entity(b["header"], champion_names, item_names)
    entity_blocks = [b for b in blocks if b["entity"] is not None]
    classified = [classify_block(b) for b in entity_blocks]
    return pd.DataFrame(classified)


# ============================================================
# PHASE 5 — LLM SUMMARY (optional — only runs if an OpenAI key is configured)
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
    from openai import OpenAI
    api_key = st.secrets.get("OPENAI_API_KEY", None)
    if not api_key:
        return None
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": build_prompt(patch_df, patch_label)}],
        temperature=0.5,
        max_tokens=250
    )
    return response.choices[0].message.content.strip()


# ============================================================
# PHASE 6 — UI
# ============================================================
st.title("🎮 Patch Note Pulse")
st.caption("Turns a wall of League of Legends balance jargon into a 30-second digest.")

with st.spinner("Loading champion/item dictionary from Data Dragon..."):
    all_versions = get_available_versions()
    latest_version = all_versions[0]
    champions_raw = get_champion_data(latest_version)
    items_raw = get_item_data(latest_version)
    champion_names = sorted([c["name"] for c in champions_raw.values()])
    item_names = sorted([i["name"] for i in items_raw.values()])

default_label = ddragon_version_to_patch_label(latest_version)
patch_label = st.text_input(
    "Patch version to analyze (e.g. 14.14)",
    value="14.14",
    help=f"Data Dragon's newest version ({latest_version}) maps to '{default_label}', "
         f"but very recent patches may not have a wiki page yet — try an older label if it 404s."
)

if st.button("Analyze patch", type="primary"):
    try:
        with st.spinner(f"Pulling and analyzing patch {patch_label}..."):
            patch_df = run_pipeline(patch_label, champion_names, item_names)

        if patch_df.empty:
            st.warning("No champion/item changes were detected for this patch. Try a different label.")
        else:
            summary = generate_patch_summary(patch_df, patch_label)
            if summary:
                st.subheader("📝 What this patch means")
                st.write(summary)
            else:
                st.info("Add OPENAI_API_KEY in this app's Secrets to enable the LLM summary paragraph.")

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
                    "Champion/item names come from Riot's Data Dragon API. Patch note text comes "
                    "from the League of Legends Fandom Wiki API. The text is cleaned, split by "
                    "section header, matched against the champion/item dictionary (fuzzy matching "
                    "via rapidfuzz), then each change line is classified as a buff or nerf based on "
                    "which stat changed and in which direction. An LLM (if configured) turns the "
                    "structured result into a plain-English summary."
                )
    except ValueError as e:
        st.error(f"{e}\n\nTry a different patch label — not every patch has its own wiki page.")

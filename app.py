"""
Patch Note Pulse — Streamlit app
================================
League of Legends Wiki API -> clean wikitext -> split per champion/item
-> classify buff / nerf / rework -> optional LLM summary (Anthropic Claude).

Run locally:   streamlit run app.py
Deploy:        Streamlit Community Cloud or Hugging Face Spaces.
LLM summary:   add ANTHROPIC_API_KEY to the platform's Secrets (see
               .streamlit/secrets.toml.example). Never hard-code keys.
"""

import requests
import streamlit as st

from patch_pulse.pipeline import build_prompt, get_patch_notes_wikitext, analyze_wikitext

SECRET_KEY_NAME = "ANTHROPIC_API_KEY"
LEGACY_SECRET_KEY_NAME = "NLP_WORK"  # name used by the original deployment; still accepted
SECRET_MODEL_NAME = "ANTHROPIC_MODEL"
DEFAULT_MODEL = "claude-sonnet-5-5"
DEFAULT_PATCH = "26.14"
ROW_COLORS = {"buff": "#2e7d32", "nerf": "#c62828", "mixed": "#f9a825", "rework": "#1565c0"}

st.set_page_config(page_title="Patch Note Pulse", page_icon="🎮", layout="wide")


@st.cache_data(ttl=3600, show_spinner=False)
def cached_wikitext(patch_label: str) -> str:
    """Cache the API response so reruns don't hammer the wiki."""
    return get_patch_notes_wikitext(patch_label)


def get_secret(name: str, default=None):
    try:
        return st.secrets.get(name, default)
    except FileNotFoundError:  # no secrets.toml at all (e.g. a fresh local run)
        return default


def generate_patch_summary(patch_df, patch_label: str) -> str | None:
    """Plain-English digest from Claude, or None when no API key is configured."""
    api_key = get_secret(SECRET_KEY_NAME) or get_secret(LEGACY_SECRET_KEY_NAME)
    if not api_key:
        return None
    import anthropic  # only needed when the summary is enabled

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=get_secret(SECRET_MODEL_NAME, DEFAULT_MODEL),
        max_tokens=300,
        messages=[{"role": "user", "content": build_prompt(patch_df, patch_label)}],
    )
    return response.content[0].text.strip()


def color_row(row):
    color = ROW_COLORS.get(row["classification"])
    style = f"background-color: {color}; color: white" if color else ""
    return [style] * len(row)


def render_results(patch_df, patch_label: str) -> None:
    try:
        summary = generate_patch_summary(patch_df, patch_label)
    except Exception as exc:  # LLM is optional: never let it break the rest of the page
        summary = None
        st.warning(f"LLM summary unavailable ({type(exc).__name__}). Showing the analysis only.")
    if summary:
        st.subheader("📝 What this patch means")
        st.write(summary)
    elif not (get_secret(SECRET_KEY_NAME) or get_secret(LEGACY_SECRET_KEY_NAME)):
        st.info(f"Add `{SECRET_KEY_NAME}` to this app's Secrets to enable the LLM summary.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Entities changed", len(patch_df))
    col2.metric("Buffs", int((patch_df["classification"] == "buff").sum()))
    col3.metric("Nerfs", int((patch_df["classification"] == "nerf").sum()))

    st.subheader("📊 Buff vs. nerf breakdown")
    st.bar_chart(patch_df["classification"].value_counts())

    st.subheader("📋 Full change table")
    table = patch_df[["entity", "entity_type", "classification", "buff_lines", "nerf_lines"]]
    st.dataframe(table.style.apply(color_row, axis=1), hide_index=True, width="stretch")

    with st.expander("🔍 Raw change text per entity"):
        for row in patch_df.itertuples():
            st.markdown(f"**{row.entity}** — {row.classification}")
            st.text(row.raw_text)

    with st.expander("How this works"):
        st.write(
            "Patch note text comes from the official League of Legends Wiki API. It is split "
            "into one block per champion/item using the wiki's own entity markers (so names and "
            "types come straight from the source), then each change line is classified as a buff "
            "or nerf based on which stat changed and in which direction. If configured, an LLM "
            "turns the structured result into a plain-English summary."
        )


# ============================================================
# UI
# ============================================================
st.title("🎮 Patch Note Pulse")
st.caption("Turns a wall of League of Legends balance jargon into a 30-second digest.")

patch_label = st.text_input(
    "Patch version to analyze (e.g. 26.14)",
    value=DEFAULT_PATCH,
    help="Riot uses a year.patch format since 2025 (26.14 = 2026, patch 14). Type the label "
         "exactly as shown on the official patch notes page.",
).strip()

if st.button("Analyze patch", type="primary"):
    try:
        with st.spinner(f"Pulling and analyzing patch {patch_label}..."):
            patch_df = analyze_wikitext(cached_wikitext(patch_label))
    except ValueError as exc:
        st.error(f"{exc}\n\nNot every patch has its own wiki page. Try another label.")
    except requests.RequestException as exc:
        st.error(f"Couldn't reach the League of Legends Wiki ({type(exc).__name__}). Try again in a moment.")
    else:
        if patch_df.empty:
            st.warning("No champion/item changes were detected for this patch. Try a different label.")
        else:
            render_results(patch_df, patch_label)

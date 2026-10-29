# 🎮 Patch Note Pulse

**Turns League of Legends patch notes into a 30-second buff/nerf digest.**

Every patch changes dozens of champions and items in dense, jargon-heavy notes
(*"Base attack damage increased to 58 from 55"*). Patch Note Pulse fetches the notes,
extracts every champion/item that changed, labels each one **Buff / Nerf / Mixed / Rework**,
and (optionally) asks an LLM for a plain-English summary grounded in that structured result.

A personal project, started as a topic idea for the NLP course of the Executive Master in
Data Science, Business Analytics and AI (2026). The group assignment ended up on a different
topic; I built this one on my own, end to end.

## How it works

```
LoL Wiki API ──► clean wikitext ──► split per champion/item ──► classify each change line ──► aggregate ──► LLM digest
 (Phase 1)         (Phase 2)            (Phase 3)                    (Phase 4)                              (Phase 5)
```

| Phase | Approach | Why |
|---|---|---|
| Acquisition | MediaWiki API of the official LoL Wiki | Structured and stable, unlike scraping HTML |
| Processing | Regex unwrapping of nested wiki templates, inside-out | `{{as|{{ap|…}}}}` nesting defeats single-pass cleaning |
| Entity extraction | Split on the wiki's own `;{{ci|…}}` / `;{{ii|…}}` markers | Exact names and types; replaced a fuzzy-matching v1 that found 1 of 9 entities |
| Classification | Attribute-aware rules: *cooldown reduced* = buff, *damage reduced* = nerf | Direction alone is meaningless without the attribute |
| Generation | Claude summarises **only** the structured table | Keeps the summary grounded, no invented changes |

The full walkthrough, including what we tried first and why it failed, is in
[`Patch_Note_Pulse.ipynb`](Patch_Note_Pulse.ipynb)
([open in Colab](https://colab.research.google.com/github/samuelbarrezueta/NLP-App/blob/main/Patch_Note_Pulse.ipynb)).

## Repository layout

```
patch_pulse/pipeline.py   # all NLP logic (no UI): shared by app, notebook and tests
app.py                    # Streamlit UI
Patch_Note_Pulse.ipynb    # step-by-step walkthrough of each phase
tests/                    # offline unit tests on a wikitext fixture
```

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py          # UI at http://localhost:8501
pytest                        # offline tests, no network needed
```

**Optional LLM summary:** copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`
and add your `ANTHROPIC_API_KEY` (on Streamlit Cloud, paste it in the app's *Secrets* settings).
Without a key the app still shows the full buff/nerf analysis.

## Limitations and next steps

- Only lines shaped like `X increased/reduced to Y from Z` are classified; anything else is
  counted as *unclear* instead of guessed.
- Magnitude is ignored (+1 and +100 damage both count as one buff line). Next: weight by relative change.
- No labelled evaluation set yet. Next: hand-label a few patches and report precision/recall,
  then compare the rules with a zero-shot LLM classifier.
- Older patches use different wiki markup; recent patches parse best.

## Author

[@samuelbarrezueta](https://github.com/samuelbarrezueta): pipeline design, parsing and classification logic,
LLM integration, Streamlit app and tests.

Data from the [League of Legends Wiki](https://wiki.leagueoflegends.com/) (CC BY-SA).
Not affiliated with Riot Games.

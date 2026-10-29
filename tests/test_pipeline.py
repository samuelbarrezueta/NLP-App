"""Offline tests: a small wikitext fixture in the same markup the LoL Wiki uses."""

import pytest

from patch_pulse.pipeline import (
    analyze_wikitext,
    build_prompt,
    classify_line,
    clean_wikitext,
    get_section_span,
)

FIXTURE = """{{Infobox patch|Image = banner.png}}
== Patch Highlights ==
Some intro text that must be ignored.
=== Champions ===
;{{ci|Aatrox}}
* {{ai|Infernal Chains|Aatrox}}
** Cooldown reduced to 10 from 12 seconds.
** Base damage increased to 60 from 50.
;{{ci|Ahri}}
* Base health reduced to 570 from 590.
<!-- editor note: double-check -->
==== Notes ====
* Health regen reduced to 5 from 6.<ref>source</ref>
;{{ci|Kai'Sa|Kai'Sa}}
* {{ai|Icathian Rain|Kai'Sa}}
** Damage increased to 40 from 35.
** Mana cost increased to 60 from 55.
=== Items ===
;{{ii|Horizon Focus}}
* Passive changed to [[Hypershot]] from ''old passive''.
;{{ii|Empty Item}}
=== Runes ===
;{{ci|ShouldNotAppear}}
* Damage increased to 1 from 0.
"""


@pytest.fixture(scope="module")
def df():
    return analyze_wikitext(FIXTURE).set_index("entity")


def test_finds_only_champions_and_items(df):
    assert list(df.index) == ["Aatrox", "Ahri", "Kai'Sa", "Horizon Focus"]
    assert df.loc["Horizon Focus", "entity_type"] == "item"


def test_subheader_does_not_cut_section(df):
    # "==== Notes ====" is deeper than "=== Champions ===", so Kai'Sa must still be found
    assert "Kai'Sa" in df.index
    assert df.loc["Ahri", "nerf_lines"] == 2


def test_block_labels(df):
    assert df.loc["Aatrox", "classification"] == "buff"
    assert df.loc["Ahri", "classification"] == "nerf"
    assert df.loc["Kai'Sa", "classification"] == "mixed"
    assert df.loc["Horizon Focus", "classification"] == "rework"


@pytest.mark.parametrize("line, expected", [
    ("Cooldown reduced to 8 from 10 seconds", "buff"),
    ("Base attack damage increased to 58 from 55", "buff"),
    ("Shield cooldown increased to 12 from 10", "nerf"),   # cooldown wins over shield
    ("Health cost increased to 20 from 15", "nerf"),       # cost wins over health
    ("Lethality increased to 18 from 15", "buff"),         # was misclassified before
    ("Range changed to 550 from 500", "rework"),
    ("Some unrelated sentence", None),
])
def test_classify_line(line, expected):
    assert classify_line(line) == expected


def test_clean_wikitext_unwinds_nested_templates():
    raw = "{{as|{{ap|3*2}}% per 100 AP}} and [[Flash|the flash]] '''bold''' {{unknown|x}}"
    assert clean_wikitext(raw) == "3*2% per 100 AP and the flash bold "


def test_missing_section():
    assert get_section_span(FIXTURE, "Systems") == (None, None)


def test_prompt_lists_every_entity(df):
    prompt = build_prompt(df.reset_index(), "26.14")
    for name in df.index:
        assert name in prompt
    assert "26.14" in prompt

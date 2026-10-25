"""prompt to requirements, rule based."""
from pathlib import Path

import pytest

from aimv import vocab
from aimv.audio.tempo import TEMPO_WORD_CLASSES
from aimv.data import load_sunocaps
from aimv.prompt_parsing import parse_dataframe, parse_prompt, split_phrases

FIXTURE = Path(__file__).parent / "fixtures" / "sunocaps_sample.csv"


def reqs_of(text, prompt_type="aspect_list"):
    """map (category, value) to polarity for a prompt."""
    return {(r.category, r.value): r.polarity for r in parse_prompt(text, prompt_type)}


def test_no_voices_instrumental():
    r = reqs_of("techno song, no voices, instrumental")
    assert r[("vocals", "vocals")] == -1
    assert not any(cat == "vocals" and pol > 0 for (cat, _), pol in r.items())


def test_fast_tempo():
    (req,) = [q for q in parse_prompt("rock, fast tempo", "aspect_list") if q.category == "tempo"]
    assert req.value == "fast" and req.check == "tempo"
    assert (req.params["bpm_min"], req.params["bpm_max"]) == vocab.TEMPO_WORDS["fast tempo"]


def test_female_singer():
    reqs = [q for q in parse_prompt("female singer, piano", "aspect_list") if q.category == "vocals"]
    assert {(q.value, q.check) for q in reqs} == {("vocals", "vocal_presence"), ("female vocals", "vocal_gender")}
    assert [q.params for q in reqs if q.check == "vocal_gender"] == [{"gender": "female"}]


def test_only_vocables_and_no_words():
    r = reqs_of("only vocables and no words")
    assert r[("vocals", "vocals")] == 1
    assert r[("vocals", "lyrics")] == -1


@pytest.mark.parametrize("text, prompt_type, expected", [
    ("There is no music nor vocals.", "caption", {("vocals", "vocals"): -1}),
    ("The song is mostly instrumental with a faint male vocal.", "caption",
     {("vocals", "vocals"): 1, ("vocals", "male vocals"): 1}),
    ("minimal instruments, no percussion", "aspect_list", {("instruments", "percussion"): -1}),
    ("didgeridoo song, no other instruments", "aspect_list",
     {("instruments", "didgeridoo"): 1, ("instruments", "other instruments"): -1}),
])
def test_negations(text, prompt_type, expected):
    r = reqs_of(text, prompt_type)
    for key, polarity in expected.items():
        assert r.get(key) == polarity, key


@pytest.mark.parametrize("text, present, absent", [
    ("shimmering hi hats", ("instruments", "hi-hat"), ("other", "shimmering hi hats")),
    ("e-bass", ("instruments", "bass"), ("other", "e-bass")),
    ("beat boxing", ("vocals", "beatbox"), ("instruments", "beat")),
    ("wide choir", ("vocals", "choir"), ("instruments", "choir")),
    ("ambient nature sounds", ("other", "nature"), ("genre", "ambient")),
    ("This clip is an electronic guitar instrumental.", ("instruments", "electric guitar"), ("genre", "electronic")),
    ("guitar string tones", ("instruments", "guitar"), ("instruments", "string")),
    ("male talking", ("vocals", "male vocals"), ("other", "talking")),
    ("easygoing", ("mood", "easygoing"), ("other", "easygoing")),
])
def test_vocab_fixes(text, present, absent):
    r = reqs_of(text, "caption" if text.endswith(".") else "aspect_list")
    assert present in r and absent not in r


def test_rap_is_vocals_and_genre():
    r = reqs_of("rap, male vocalist")
    assert ("vocals", "rap") in r and ("genre", "rap") in r


def test_imitated_instruments_are_not_required():
    r = reqs_of("The vocalist is imitating the digital drums and turntable.", "caption")
    assert not any(cat == "instruments" for cat, _ in r)


@pytest.mark.parametrize("text, word", [("mid tempo", "mid"), ("faster tempo", "faster"), ("medium fast tempo", "medium fast")])
def test_new_tempo_words_are_mapped(text, word):
    (req,) = [q for q in parse_prompt(text, "aspect_list") if q.category == "tempo"]
    assert req.value == word and word in TEMPO_WORD_CLASSES


def test_every_tempo_word_is_mapped():
    # parser drops "tempo" from words; verdict must know them all
    for word in vocab.TEMPO_WORDS:
        assert word.replace(" tempo", "") in TEMPO_WORD_CLASSES, word


def test_explicit_bpm():
    (req,) = [q for q in parse_prompt("The song is played at 120 bpm.", "caption") if q.category == "tempo"]
    assert req.value == "120 bpm" and req.params["bpm_min"] == pytest.approx(114)


def test_caption_tempo_needs_the_word_tempo():
    assert ("tempo", "slow") in reqs_of("The tempo is slow.", "caption")
    assert ("tempo", "slow") not in reqs_of("Slow strings play the melody.", "caption")


def test_split_phrases():
    assert split_phrases("rock, fast tempo,  male vocals,", "aspect_list") == ["rock", "fast tempo", "male vocals"]
    assert split_phrases("The tempo is slow.A female vocalist sings.", "caption") == \
        ["the tempo is slow", "a female vocalist sings"]


def test_parse_dataframe_on_fixture():
    df = load_sunocaps(FIXTURE)
    reqs = parse_dataframe(df)
    assert set(reqs["track_id"]) == set(df["track_id"])  # every track gets requirements
    assert not reqs.duplicated(["track_id", "category", "value", "polarity"]).any()
    assert set(reqs["polarity"]) <= {1, -1}

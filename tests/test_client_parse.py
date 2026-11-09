"""Parsing of the real Pro7 /v1/presentation/active shape, incl. arrangements
(the live slide index counts cues in the arrangement, where groups can repeat)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from propresenter.client import ProPresenterClient  # noqa: E402


def _slide(text):
    return {"enabled": True, "notes": "", "text": text, "label": ""}


def _data(arrangement_groups, current="arr-1"):
    return {"presentation": {
        "id": {"uuid": "p1", "name": "Test Song", "index": 0},
        "groups": [
            {"name": "Intro", "uuid": "g-intro", "slides": [_slide("")]},
            {"name": "Verse 1", "uuid": "g-v1", "slides": [_slide("verse one a"), _slide("verse one b")]},
            {"name": "Chorus 1", "uuid": "g-c1", "slides": [_slide("chorus a"), _slide("chorus b")]},
        ],
        "current_arrangement": current,
        "arrangements": [{"id": {"uuid": "arr-1", "name": "Mine", "index": 0}, "groups": arrangement_groups,
                          "total_cues": 99}],
        "total_cues": 5,
    }}


def parse(data):
    return ProPresenterClient("127.0.0.1", 1025)._parse_presentation(data)


def test_arrangement_repeats_groups():
    p = parse(_data(["g-intro", "g-v1", "g-c1", "g-v1", "g-c1"]))
    assert len(p.slides) == 1 + 2 + 2 + 2 + 2
    assert [s.index for s in p.slides] == list(range(9))
    assert [s.group for s in p.slides] == ["Intro", "Verse 1", "Verse 1", "Chorus 1", "Chorus 1",
                                            "Verse 1", "Verse 1", "Chorus 1", "Chorus 1"]
    assert p.slides[3].text == "chorus a" and p.slides[7].text == "chorus a"   # repeat is a separate cue
    assert p.slides[0].text == ""                                              # blank slide kept


def test_no_arrangement_uses_group_order():
    p = parse(_data([], current=""))
    assert [s.text for s in p.slides] == ["", "verse one a", "verse one b", "chorus a", "chorus b"]


def test_unknown_group_in_arrangement_falls_back():
    p = parse(_data(["g-intro", "missing", "g-c1"]))
    assert len(p.slides) == 5

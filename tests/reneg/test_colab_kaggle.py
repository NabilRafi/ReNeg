"""Kaggle notebook listing: robust parsing and slug guessing."""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reneg import colab as rc  # noqa: E402


def test_guess_slug():
    assert rc.guess_slug("Notebook_three") == "notebook-three"
    assert rc.guess_slug("notebook0") == "notebook0"
    assert rc.guess_slug("01 Build feature cache!") == "01-build-feature-cache"


def test_exact_slug_beats_lookalike():
    rows = [{"ref": "u/notebook-1", "title": "notebook-1"}, {"ref": "u/notebook1", "title": "notebook1"}]
    assert rc.resolve_slugs({"01": "notebook1"}, rows) == {"01": "notebook1"}
    assert rc.resolve_slugs({"03": "Notebook_three"}, [{"ref": "u/notebook-three", "title": "Notebook_three"}]) \
        == {"03": "notebook-three"}


def test_cli_listing_skips_lines_before_header(monkeypatch):
    text = ("Next Page Token = abc\nref,title,author,lastRunTime,totalVotes\n"
            "u/notebook1,notebook1,U,2026-09-01,0\nu/notebook-three,Notebook_three,U,2026-09-02,0\n")
    monkeypatch.setattr(rc, "run", lambda *a, **k: types.SimpleNamespace(returncode=0, stdout=text, stderr=""))
    rows = rc._kernels_via_cli(log=lambda *_: None)
    assert [r["ref"] for r in rows] == ["u/notebook1", "u/notebook-three"]
    got = rc.resolve_or_guess({"01": "notebook1", "03": "Notebook_three", "04": "Notebook_four"}, rows,
                              log=lambda *_: None)
    assert got == {"01": "notebook1", "03": "notebook-three", "04": "notebook-four"}

import re
from pathlib import Path

import pytest

HTML = Path(__file__).resolve().parent.parent / "app" / "static" / "index.html"

# Each suggested question on the page must produce the right fact from the sample documents.
EXPECTED = {
    "How many days of paid leave does a Northwind employee get?": "24 days",
    "How long does Fabrikam take to pay contractor invoices?": "45 days",
    "How do I undo the last commit but keep my changes staged?": "--soft keeps changes staged",
    "How much water do vegetable beds need each week?": "one inch of water",
    "Which debt payoff method saves the most interest?": "avalanche",
}


def page_questions() -> list[str]:
    block = re.search(r"const SAMPLE_QUESTIONS = \[(.*?)\];", HTML.read_text(), re.S).group(1)
    return re.findall(r'"([^"]+)"', block)


def test_every_suggested_question_is_covered_by_this_test():
    assert sorted(page_questions()) == sorted(EXPECTED)


@pytest.mark.parametrize("question,fact", EXPECTED.items())
def test_suggested_question_gets_the_right_fact(pipeline, question, fact):
    res = pipeline.ask(question)
    assert not res.abstained and fact in res.answer

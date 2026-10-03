"""Unit tests for Module 1: Claim Segmentation."""

import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.segmentation import segment_claims


def test_single_sentence():
    text = "0-dimensional biomaterials show inductive properties."
    claims = segment_claims(text)
    assert len(claims) == 1
    assert claims[0] == "0-dimensional biomaterials show inductive properties."


def test_multiple_sentences():
    text = "ADAR1 binds to Dicer to cleave pre-miRNA. ALDH1 expression is associated with better breast cancer outcomes."
    claims = segment_claims(text)
    assert len(claims) == 2
    assert claims[0] == "ADAR1 binds to Dicer to cleave pre-miRNA."
    assert claims[1] == "ALDH1 expression is associated with better breast cancer outcomes."


def test_scientific_abbreviations_and_decimals():
    text = (
        "Smith et al. observed a 3.14-fold increase in p53 expression (e.g. in vitro vs. in vivo). "
        "However, Dr. Watson et al. showed no significant correlation in Fig. 2."
    )
    claims = segment_claims(text)
    assert len(claims) == 2
    assert "et al." in claims[0]
    assert "3.14" in claims[0]
    assert "e.g." in claims[0]
    assert "vs." in claims[0]
    assert "Dr. Watson et al." in claims[1]
    assert "Fig. 2." in claims[1]


def test_bullet_points_and_newlines():
    text = """
    - 1/2000 in UK have abnormal PrP positivity.
    - AIRE is expressed in some skin tumors.
    * 5% of perinatal mortality is due to low birth weight.
    """
    claims = segment_claims(text)
    assert len(claims) == 3
    assert claims[0] == "1/2000 in UK have abnormal PrP positivity."
    assert claims[1] == "AIRE is expressed in some skin tumors."
    assert claims[2] == "5% of perinatal mortality is due to low birth weight."


def test_empty_and_whitespace():
    assert segment_claims("") == []
    assert segment_claims("   \n\n\t  ") == []
    assert segment_claims(None) == []


if __name__ == "__main__":
    test_single_sentence()
    test_multiple_sentences()
    test_scientific_abbreviations_and_decimals()
    test_bullet_points_and_newlines()
    test_empty_and_whitespace()
    print("All segmentation unit tests passed successfully!")


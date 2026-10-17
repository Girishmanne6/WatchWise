"""Checks for the review score normalizer."""

import math

import pytest

from src.score_normalize import normalize_score


def test_fraction_three_fifths():
    assert normalize_score("3/5") == pytest.approx(0.6)


def test_fraction_eight_out_of_ten():
    assert normalize_score("8/10") == pytest.approx(0.8)


def test_letter_a_is_high():
    score = normalize_score("A")
    assert score is not None
    assert score >= 0.9


def test_letter_f_is_zero():
    assert normalize_score("F") == 0.0


def test_zero_denominator_is_none():
    assert normalize_score("5/0") is None


def test_none_is_none():
    assert normalize_score(None) is None


def test_plain_number_is_none():
    assert normalize_score("7") is None


def test_nan_and_blank_are_none():
    assert normalize_score(float("nan")) is None
    assert normalize_score("  ") is None
    assert math.isnan(float("nan"))


def test_letter_grade_ends():
    assert normalize_score("A+") == 1.0
    assert normalize_score("a-") is not None
    assert normalize_score("A+") > normalize_score("A") > normalize_score("F")


def test_fraction_over_one_is_none():
    assert normalize_score("11/10") is None
    assert normalize_score("3.5/4") == pytest.approx(0.875)

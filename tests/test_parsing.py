"""
Unit tests for the fragile parsing / normalisation logic across the project.

Run:  pytest tests/ -v
"""

import sys
from pathlib import Path

import pytest

# Make the project root importable regardless of where pytest is invoked
sys.path.insert(0, str(Path(__file__).parent.parent))

from scraper import _to_lpa, _extract_skills_from_text
from data_pipeline import (
    _parse_salary_lpa,
    _parse_experience_years,
    _normalize_city,
    _normalize_date,
    _assign_company_tier,
)
from model import _normalize_title, _extract_seniority


# ── scraper._to_lpa ───────────────────────────────────────────────────────────

class TestToLpa:
    def test_annual_inr(self):
        assert _to_lpa(1_000_000) == "10.0 LPA"

    def test_monthly_inr(self):
        assert _to_lpa(50_000, period="MONTH") == "6.0 LPA"

    def test_usd_conversion(self):
        # 100k USD ≈ 83.5 lakh INR
        assert _to_lpa(100_000, currency="USD") == "83.5 LPA"

    def test_zero_and_negative(self):
        assert _to_lpa(0) == "Not Mentioned"
        assert _to_lpa(-5) == "Not Mentioned"

    def test_implausible_filtered(self):
        assert _to_lpa(10_000) == "Not Mentioned"        # 0.1 LPA — too low
        assert _to_lpa(50_000_000) == "Not Mentioned"    # 500 LPA — too high


# ── scraper._extract_skills_from_text ────────────────────────────────────────

class TestSkillExtraction:
    def test_whole_word_only_no_false_positives(self):
        # These words CONTAIN skill substrings but are not skills
        text = "We are interested in excellent candidates who travel internationally"
        assert _extract_skills_from_text(text) == "Not Mentioned"

    def test_java_does_not_match_javascript(self):
        skills = _extract_skills_from_text("Experience with JavaScript required")
        assert "Javascript" in skills
        assert "Java," not in skills and not skills.startswith("Java,")
        assert skills.split(", ").count("Java") == 0

    def test_r_word_boundary(self):
        assert _extract_skills_from_text("developer for our user team") == "Not Mentioned"
        assert "R" in _extract_skills_from_text("proficiency in R and Python").split(", ")

    def test_dotted_and_hyphenated_patterns(self):
        skills = _extract_skills_from_text("We use Node.js and scikit-learn daily")
        assert "Node.Js" in skills or "node.js" in skills.lower()
        assert "scikit-learn" in skills.lower()

    def test_multiword_skill(self):
        assert "Machine Learning" in _extract_skills_from_text("machine learning models")

    def test_empty(self):
        assert _extract_skills_from_text("") == "Not Mentioned"
        assert _extract_skills_from_text(None) == "Not Mentioned"

    def test_dedup(self):
        skills = _extract_skills_from_text("Python, python, PYTHON")
        assert skills.split(", ").count("Python") == 1


# ── data_pipeline._parse_salary_lpa ───────────────────────────────────────────

class TestParseSalary:
    def test_rupee_range(self):
        assert _parse_salary_lpa("₹ 3,00,000 - 5,00,000") == 4.0

    def test_single_rupee(self):
        assert _parse_salary_lpa("₹ 2,50,000") == 2.5

    def test_lpa_range(self):
        assert _parse_salary_lpa("8-12 LPA") == 10.0

    def test_monthly_annualised(self):
        # ₹50,000/month = ₹6,00,000/year = 6 LPA (was parsed as 0.5 LPA)
        assert _parse_salary_lpa("₹ 50,000 per month") == 6.0
        assert _parse_salary_lpa("₹ 50,000 /month") == 6.0
        assert _parse_salary_lpa("₹ 50,000 monthly") == 6.0

    def test_no_numbers(self):
        assert _parse_salary_lpa("Competitive salary") is None


# ── data_pipeline._parse_experience_years ─────────────────────────────────────

class TestParseExperience:
    def test_range_midpoint(self):
        assert _parse_experience_years("1 to 6 Yrs") == 3.5

    def test_single_value(self):
        assert _parse_experience_years("5 years") == 5.0

    def test_fresher(self):
        assert _parse_experience_years("Fresher") == 0.0

    def test_no_numbers(self):
        assert _parse_experience_years("Not Mentioned") is None


# ── data_pipeline._normalize_city / _normalize_date ───────────────────────────

class TestNormalizeCity:
    def test_first_comma_component(self):
        assert _normalize_city("HSR Layout, Bangalore") == "HSR Layout"

    def test_remote_variants(self):
        for raw in ["Work From Home", "remote", "WFH", "Anywhere"]:
            assert _normalize_city(raw) == "Remote"

    def test_plain_city(self):
        assert _normalize_city("Mumbai") == "Mumbai"


class TestNormalizeDate:
    def test_iso_passthrough(self):
        assert _normalize_date("2024-07-15") == "2024-07-15"

    def test_relative_days(self):
        from datetime import datetime, timedelta
        expected = (datetime.now().date() - timedelta(days=3)).isoformat()
        assert _normalize_date("3 days ago") == expected

    def test_garbage(self):
        assert _normalize_date("Just posted") == "Not Mentioned"


# ── data_pipeline._assign_company_tier ────────────────────────────────────────

class TestCompanyTier:
    def test_faang(self):
        assert _assign_company_tier("Google India") == "FAANG"

    def test_indian_it(self):
        assert _assign_company_tier("Infosys Limited") == "Indian IT"

    def test_default_startup(self):
        assert _assign_company_tier("Tiny Unknown Co") == "Startup"
        assert _assign_company_tier("") == "Startup"
        assert _assign_company_tier(None) == "Startup"


# ── model._normalize_title ────────────────────────────────────────────────────

class TestNormalizeTitle:
    def test_seniority_stripped(self):
        assert _normalize_title("Senior Data Scientist") == "Data Scientist"
        assert _normalize_title("Sr. Data Engineer") == "Data Engineer"
        assert _normalize_title("Data Analyst Intern") == "Data Analyst"

    def test_whole_words_not_mangled(self):
        # "intern" must not be stripped out of "International",
        # "lead" must not be stripped out of "Leader"
        assert _normalize_title("International Software Engineer") == "Software Engineer"
        assert _normalize_title("Internship - Data Science") == "Data Scientist"

    def test_canonical_buckets(self):
        assert _normalize_title("Machine Learning Engineer") == "ML Engineer"
        assert _normalize_title("Python Developer") == "Software Engineer"
        assert _normalize_title("DevOps Engineer") == "DevOps Engineer"

    def test_catchall(self):
        assert _normalize_title("Garment Production Coordinator") == "Other Tech"


# ── model._extract_seniority ──────────────────────────────────────────────────

class TestExtractSeniority:
    def test_levels(self):
        assert _extract_seniority("Data Science Intern") == 0
        assert _extract_seniority("Junior Developer") == 1
        assert _extract_seniority("Data Analyst") == 2
        assert _extract_seniority("Senior ML Engineer") == 3
        assert _extract_seniority("VP Engineering") == 4

    def test_sr_abbreviation(self):
        assert _extract_seniority("Sr Data Engineer") == 3
        assert _extract_seniority("Sr. Data Engineer") == 3


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

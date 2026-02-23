"""Tests for the biller onboarding engine."""

import pytest
from decimal import Decimal
from pathlib import Path

from src.core.onboarding import OnboardingEngine, OnboardingError
from src.models.biller import Biller, BillerStatus


@pytest.fixture
def engine() -> OnboardingEngine:
    return OnboardingEngine()


class TestOnboardingEngine:
    """Tests for YAML-driven biller onboarding."""

    def test_load_config(self, engine: OnboardingEngine) -> None:
        """Config file should load and contain biller definitions."""
        raw = engine.get_raw_biller_configs()
        assert len(raw) >= 2, "Expected at least 2 billers in config"

    def test_onboard_valid_biller(self, engine: OnboardingEngine) -> None:
        """A fully-configured biller should pass all validation checks."""
        raw_configs = engine.get_raw_biller_configs()
        # First biller (Georgia Power) should be valid
        biller, report = engine.onboard_biller(raw_configs[0])

        assert isinstance(biller, Biller)
        assert report.all_passed, f"Validation failed: {report.summary()}"
        assert biller.biller_id == "UTIL-GA-POWER-001"
        assert biller.biller_name == "Georgia Power & Light"

    def test_onboard_all_billers(self, engine: OnboardingEngine) -> None:
        """All billers in the config should be onboarded (some may fail checks)."""
        results = engine.onboard_all()
        assert len(results) >= 2

        for biller, report in results:
            assert isinstance(biller, Biller)
            assert biller.biller_id != ""

    def test_biller_payment_types(self, engine: OnboardingEngine) -> None:
        """Onboarded biller should have payment types populated."""
        raw_configs = engine.get_raw_biller_configs()
        biller, _ = engine.onboard_biller(raw_configs[0])

        assert len(biller.payment_types) > 0
        assert biller.accepts_payment_type("one_time")

    def test_biller_fee_structure(self, engine: OnboardingEngine) -> None:
        """Fee structure should be parsed with correct decimal values."""
        raw_configs = engine.get_raw_biller_configs()
        biller, _ = engine.onboard_biller(raw_configs[0])

        assert biller.fee_structure.convenience_fee.credit_card > Decimal("0")
        assert biller.fee_structure.late_payment_fee > Decimal("0")

    def test_biller_settlement_config(self, engine: OnboardingEngine) -> None:
        """Settlement configuration should be fully populated."""
        raw_configs = engine.get_raw_biller_configs()
        biller, _ = engine.onboard_biller(raw_configs[0])

        assert biller.settlement.method == "ach_batch"
        assert len(biller.settlement.bank_routing) == 9

    def test_activate_biller(self, engine: OnboardingEngine) -> None:
        """Activating a biller should change its status and set the onboarded date."""
        raw_configs = engine.get_raw_biller_configs()
        biller, _ = engine.onboard_biller(raw_configs[0])

        activated = engine.activate_biller(biller.biller_id)
        assert activated.status == BillerStatus.ACTIVE
        assert activated.onboarded_date is not None

    def test_activate_unknown_biller_raises(self, engine: OnboardingEngine) -> None:
        """Activating a non-existent biller should raise OnboardingError."""
        with pytest.raises(OnboardingError):
            engine.activate_biller("NONEXISTENT-001")

    def test_biller_channel_validation(self, engine: OnboardingEngine) -> None:
        """Channel acceptance should reflect the YAML configuration."""
        raw_configs = engine.get_raw_biller_configs()
        biller, _ = engine.onboard_biller(raw_configs[0])

        # Georgia Power accepts web for one_time
        assert biller.accepts_channel("one_time", "web")
        # Kiosk is not configured for Georgia Power one_time
        assert not biller.accepts_channel("one_time", "kiosk")

    def test_registry_lookup(self, engine: OnboardingEngine) -> None:
        """Onboarded billers should be retrievable from the registry."""
        engine.onboard_all()

        biller = engine.get_biller("UTIL-GA-POWER-001")
        assert biller is not None
        assert biller.biller_name == "Georgia Power & Light"

    def test_invalid_biller_config(self, engine: OnboardingEngine) -> None:
        """A biller config missing required fields should fail checks."""
        bad_config = {
            "biller_id": "TEST-BAD-001",
            "biller_name": "",
            "cis_division": "",
            "cis_vendor": "unsupported_cis",
            "payment_types": [],
            "fee_structure": {},
            "settlement": {},
            "validation_rules": {},
        }

        biller, report = engine.onboard_biller(bad_config)
        assert not report.all_passed
        assert len(report.failed_checks) > 0

    def test_defaults_applied(self, engine: OnboardingEngine) -> None:
        """Global defaults should be accessible."""
        defaults = engine.get_defaults()
        assert "max_retry_attempts" in defaults
        assert defaults["settlement_currency"] == "USD"

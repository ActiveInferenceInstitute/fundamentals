"""Tests for ``utils.logging`` — the lightweight logger factory."""

from __future__ import annotations

import logging

import pytest

from active_inference.utils.logging import get_logger


class TestGetLogger:
    def test_returns_logger_instance(self) -> None:
        log = get_logger("test.factory")
        assert isinstance(log, logging.Logger)
        assert log.name == "test.factory"

    def test_idempotent_same_name(self) -> None:
        a = get_logger("test.idempotent")
        b = get_logger("test.idempotent")
        # Same logger instance — no duplicate handlers stacked.
        assert a is b
        assert len(a.handlers) == len(b.handlers) == 1

    def test_does_not_propagate(self) -> None:
        log = get_logger("test.no_propagate")
        assert log.propagate is False

    def test_default_level_info(self) -> None:
        log = get_logger("test.default_level")
        assert log.level == logging.INFO

    def test_explicit_level(self) -> None:
        log = get_logger("test.debug_level", level=logging.DEBUG)
        assert log.level == logging.DEBUG

    def test_handler_format(self, capsys: pytest.CaptureFixture) -> None:
        log = get_logger("test.format")
        log.info("hello world")
        captured = capsys.readouterr()
        # The default format includes name, level, and message.
        assert "hello world" in captured.out
        assert "INFO" in captured.out
        assert "test.format" in captured.out

"""Acceptance checks for the quickstart tasks."""

from pathlib import Path


def test_hello() -> None:
    assert Path("greetings/hello.txt").read_text(encoding="utf-8") == "hello"


def test_goodbye() -> None:
    assert Path("greetings/goodbye.txt").read_text(encoding="utf-8") == "goodbye"

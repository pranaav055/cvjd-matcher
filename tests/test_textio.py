from pathlib import Path
import pytest
from src.textio import load_text

def test_load_text_normal(tmp_path: Path):
    p = tmp_path / "hello.txt"
    p.write_text("Hello\nWorld", encoding="utf-8")
    assert load_text(p) == "Hello\nWorld"

def test_load_text_newline(tmp_path: Path):
    p = tmp_path / "hello.txt"
    p.write_bytes(b"Hello1\r\nHello2\rHello3\n")
    assert load_text(p) == "Hello1\nHello2\nHello3\n"

def test_load_text_invalid_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_text(tmp_path)


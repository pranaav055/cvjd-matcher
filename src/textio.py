from pathlib import Path

def load_text(path: str | Path, encoding: str ="utf-8") -> str:
    p = Path(path)

    if not p.is_file():
        raise FileNotFoundError(f"Not a valid file {p}")
    
    s = p.read_text(encoding=encoding, errors="ignore")
    return s.replace("\r\n", "\n").replace("\r", "\n")

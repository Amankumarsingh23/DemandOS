"""Portable internal-link and current-scope acceptance checks."""
from pathlib import Path
import re
from urllib.parse import unquote
from demandos.serving.core import require


def introduction_checks(readme, architecture):
    require("Season 24" in readme[:4000] and "Current milestone: Season 14" not in readme[:4000], "STALE_README")
    require("Seasons 1–24" in architecture[:4000] and "Implemented scope: Seasons 1–11" not in architecture[:4000], "STALE_ARCHITECTURE")
    for text in [readme[:4000], architecture[:4000]]:
        require(not re.search(r"(?:inventory optimization|serving|monitoring)[^.]*future work", text, re.I|re.S), "STALE_IMPLEMENTED_SCOPE")


def validate(root):
    root = Path(root)
    introduction_checks((root/"README.md").read_text(encoding="utf-8"), (root/"docs/architecture.md").read_text(encoding="utf-8"))
    files = [root/"README.md", *sorted((root/"docs").rglob("*.md"))]
    checked = 0
    for path in files:
        text = path.read_text(encoding="utf-8")
        # Skip code fences when interpreting Markdown hyperlinks.
        text = re.sub(r"```.*?```", "", text, flags=re.S)
        for target in re.findall(r"\]\(([^)]+)\)", text):
            target = target.strip().strip("<>")
            require(not re.match(r"(?:file:|[A-Za-z]:[/\\])", target), "MACHINE_SPECIFIC_LINK", str(path))
            if re.match(r"[a-zA-Z]+://|mailto:|#", target):
                continue
            target = unquote(target.split("#",1)[0])
            if not target:
                continue
            resolved = root/target.lstrip("/") if target.startswith("/") else path.parent/target
            require(resolved.exists(), "BROKEN_DOCUMENTATION_LINK", f"{path.relative_to(root)} -> {target}")
            checked += 1
    return dict(markdown_files=len(files), internal_links=checked, passed=True)

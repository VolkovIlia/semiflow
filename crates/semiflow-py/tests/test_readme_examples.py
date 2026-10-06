"""Execute every ```python example of the PyPI README against the built extension.

The README (`crates/semiflow-py/README.md`) is generated from `docs/readme/` by
`cargo xtask readme`; this test keeps its examples honest. A block preceded by
an HTML comment ``readme-test: skip`` is not executed.
"""

from pathlib import Path

import pytest

README = Path(__file__).resolve().parents[1] / "README.md"


def _python_blocks(markdown: str) -> list[str]:
    blocks: list[str] = []
    current: list[str] | None = None
    keep = True
    skip_next = False
    for line in markdown.splitlines():
        if current is not None:
            if line.lstrip().startswith("```"):
                if keep:
                    blocks.append("\n".join(current))
                current = None
            else:
                current.append(line)
        elif "<!-- readme-test: skip" in line:
            skip_next = True
        elif line.lstrip().startswith("```python"):
            current, keep, skip_next = [], not skip_next, False
        elif line.strip():
            skip_next = False
    return blocks


BLOCKS = _python_blocks(README.read_text(encoding="utf-8"))


def test_readme_has_examples() -> None:
    assert len(BLOCKS) >= 3, "expected the README to carry runnable Python examples"


@pytest.mark.parametrize("source", BLOCKS, ids=[f"example-{i}" for i in range(len(BLOCKS))])
def test_readme_example_runs(source: str) -> None:
    exec(compile(source, str(README), "exec"), {"__name__": "__readme__"})  # noqa: S102  # nosec B102 - executes trusted README examples as tests

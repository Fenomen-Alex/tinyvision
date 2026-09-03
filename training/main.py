"""Legacy CLI entry point.

The canonical entry point is ``tv.cli:main`` (installed as ``tv``).
This module exists so ``uv run main.py ...`` continues to work.

Run the pipeline with:

    uv run tv <command> ...

or:

    uv run python -m main <command> ...
"""
from tv.cli import main

if __name__ == "__main__":
    main()

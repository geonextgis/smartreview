"""Command-line interface for SmartReview.

Run ``smartreview --help`` after installation, or ``python -m smartreview``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from .smartreview import rank_papers
from .embeddings import DEFAULT_MODEL, MissingAPIKeyError


def _read_interest(args: argparse.Namespace) -> str:
    if args.interest and args.interest_file:
        raise SystemExit("Pass either --interest or --interest-file, not both.")
    if args.interest_file:
        text = Path(args.interest_file).read_text(encoding="utf-8").strip()
    elif args.interest:
        text = args.interest.strip()
    else:
        raise SystemExit(
            "Provide your research interests with --interest 'text' or "
            "--interest-file path/to/interest.txt"
        )
    if not text:
        raise SystemExit("Interest text is empty.")
    return text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="smartreview",
        description=(
            "Rank Web of Science papers by semantic similarity to a "
            "free-text research interest statement."
        ),
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="Path to a Web of Science .xls or .xlsx export.",
    )
    interest_group = parser.add_argument_group("research interest (one required)")
    interest_group.add_argument(
        "--interest", help="Free-text research-interest statement.",
    )
    interest_group.add_argument(
        "--interest-file",
        help="Path to a text file containing the research-interest statement.",
    )
    parser.add_argument(
        "-o", "--output-dir", default="data",
        help="Directory for CSV / Excel / BibTeX exports (default: data).",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "-k", "--top-k", type=int,
        help="Number of top papers to keep (default: 100).",
    )
    selection.add_argument(
        "--top-percentile", type=float,
        help="Keep papers at or above this similarity percentile (e.g. 80).",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"OpenAI embedding model (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--api-key",
        help="OpenAI API key (default: $OPENAI_API_KEY).",
    )
    parser.add_argument(
        "--cache-dir",
        help="Embedding cache directory (default: <output-dir>/embeddings/cache).",
    )
    parser.add_argument(
        "--no-export", action="store_true",
        help="Skip writing CSV/Excel/BibTeX (just compute and print summary).",
    )
    parser.add_argument(
        "--version", action="store_true",
        help="Print the smartreview version and exit.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version:
        from . import __version__
        print(f"smartreview {__version__}")
        return 0

    interest_text = _read_interest(args)

    try:
        result = rank_papers(
            input_path=args.input,
            interest_text=interest_text,
            output_dir=args.output_dir,
            top_k=args.top_k,
            top_percentile=args.top_percentile,
            model=args.model,
            api_key=args.api_key,
            cache_dir=args.cache_dir,
            export=not args.no_export,
        )
    except MissingAPIKeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    df = result["dataframe"]
    print(f"\nSelected {len(df)} papers.")
    if "files" in result:
        print(f"  CSV:    {result['files']['csv']}")
        if result["files"]["excel"]:
            print(f"  Excel:  {result['files']['excel']}")
        print(f"  BibTeX: {result['bibtex']['file']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

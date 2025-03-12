"""Command line interface for the polyase package."""

import argparse
import sys
import anndata as ad
from . import __version__
from .allele_utils import calculate_allelic_ratios


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="Analyze allele-specific expression in polyploid plants")
    parser.add_argument("--version", action="version", version=f"polyase {__version__}")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Calculate ratios command
    ratio_parser = subparsers.add_parser("calc-ratios", help="Calculate allelic ratios")
    ratio_parser.add_argument("input", help="Input AnnData file (.h5ad)")
    ratio_parser.add_argument("output", help="Output AnnData file (.h5ad)")
    ratio_parser.add_argument("--layer", default="unique_counts", help="Counts layer to use")
    
    args = parser.parse_args()
    
    if args.command == "calc-ratios":
        adata = ad.read_h5ad(args.input)
        adata = calculate_allelic_ratios(adata, counts_layer=args.layer)
        adata.write_h5ad(args.output)
        print(f"Allelic ratios calculated and saved to {args.output}")
    elif not args.command:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
"""
Polyase: A package for analyzing allele-specific expression in polyploid plants.

This package provides tools for calculating and analyzing allelic ratios,
visualizing allele-specific expression patterns, and statistical testing
of allelic imbalance in polyploid plant genomes.
"""

__version__ = "0.1.4"
__author__ = "Nadja Nolte"

from .allele_utils import AlleleRatioCalculator, calculate_allelic_ratios
from .multimapping import MultimappingRatioCalculator, calculate_multi_ratios, calculate_per_allele_ratios
from .filter import filter_low_expressed_genes
from .plotting import plot_allelic_ratios, plot_allelic_ratios_comparison, plot_top_differential_syntelogs, plot_top_differential_isoforms
from .ase_data_loader import load_ase_data, aggregate_transcripts_to_genes
from .stats import test_allelic_ratios_between_conditions, test_allelic_ratios_within_conditions, get_top_differential_syntelogs, test_isoform_DIU_between_conditions, test_isoform1_DIU_between_alleles

__all__ = ["filter_low_expressed_genes","AlleleRatioCalculator", "calculate_allelic_ratios", "MultimappingRatioCalculator", "calculate_multi_ratios","plot_allelic_ratios", "plot_allelic_ratios_comparison", "load_ase_data", "aggregate_transcripts_to_genes", "test_allelic_ratios_between_conditions", "test_allelic_ratios_within_conditions","plot_top_differential_syntelogs", "plot_top_differential_isoforms", "get_top_differential_syntelogs", "test_isoform_DIU_between_conditions", "calculate_per_allele_ratios", "test_isoform1_DIU_between_alleles"]

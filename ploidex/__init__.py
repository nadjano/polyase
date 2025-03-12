"""
Polyase: A package for analyzing allele-specific expression in polyploid plants.

This package provides tools for calculating and analyzing allelic ratios,
visualizing allele-specific expression patterns, and statistical testing
of allelic imbalance in polyploid plant genomes.
"""

__version__ = "0.1.0"
__author__ = "Nadja Nolte"

from .allele_utils import AlleleRatioCalculator, calculate_allelic_ratios
from .multimapping import MultimappingRatioCalculator, calculate_multi_ratios

__all__ = ["AlleleRatioCalculator", "calculate_allelic_ratios", "MultimappingRatioCalculator", "calculate_multi_ratios"]

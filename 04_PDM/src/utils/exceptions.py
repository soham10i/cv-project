"""
Custom exception hierarchy for the MPDF pipeline.
=================================================

A small, explicit exception hierarchy makes failures self-documenting and lets
callers catch broad (``PDMError``) or narrow (``CalibrationError``) categories.
All pipeline-raised errors derive from ``PDMError`` so a top-level handler can
distinguish *our* failures from unexpected third-party crashes.
"""

from __future__ import annotations


class PDMError(Exception):
    """Base class for all pipeline-specific errors."""


class ConfigError(PDMError):
    """Invalid or inconsistent configuration."""


class DataError(PDMError):
    """Problems with raw data, slices, patches, or manifests."""


class ManifestError(DataError):
    """A manifest is missing, empty, or malformed."""


class PreprocessingError(DataError):
    """Failure while converting NIfTI volumes to slices/patches."""


class ModelError(PDMError):
    """Model construction or checkpoint loading failure."""


class CheckpointError(ModelError):
    """A checkpoint is missing or incompatible."""


class TrainingError(PDMError):
    """Failure inside the training loop."""


class CalibrationError(PDMError):
    """Calibration could not be completed (e.g. empty calibration set)."""


class ScoringError(PDMError):
    """Failure while computing anomaly scores."""


class EvaluationError(PDMError):
    """Failure during metric computation or evaluation."""


class XAIError(PDMError):
    """Failure while generating explanations."""


class DependencyError(PDMError):
    """An optional dependency is required for the requested operation."""

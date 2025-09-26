"""
Prover module for Veritor.

The prover is responsible for executing computations and generating
verification-friendly annotations and metadata.
"""

from .annotations import (
    AnnotationManager,
    AnnotationContext,
    AnnotationScope,
    AnnotationType,
    ViewType,
    Annotation,
    LSHProjectionAnnotation,
    CheckpointAnnotation,
    CrossReferenceAnnotation,
)

__all__ = [
    "AnnotationManager",
    "AnnotationContext",
    "AnnotationScope",
    "AnnotationType",
    "ViewType",
    "Annotation",
    "LSHProjectionAnnotation",
    "CheckpointAnnotation",
    "CrossReferenceAnnotation",
]
"""
Prover Annotation System for Verification.

This module provides a clean abstraction for the Prover to annotate its execution
data with verification metadata. Common patterns:

1. LSH projections for compact verification of activations/gradients
2. Challenge generation with deterministic replay
3. Cross-reference linking between traces, challenges, and data
4. Multi-view data storage (logical/distributed/device-specific)
"""

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Union

import jax.numpy as jnp
from jax import random

from veritor.db.models import ChallengeRecord, DataBundle, TensorData, TraceEvent


class AnnotationType(Enum):
    """Types of verification annotations."""
    LSH_PROJECTION = "lsh_projection"
    CHALLENGE_RESPONSE = "challenge_response"
    CHECKPOINT = "checkpoint"
    VERIFICATION_HINT = "verification_hint"
    DEVICE_ASSIGNMENT = "device_assignment"
    CROSS_REFERENCE = "cross_reference"


class ViewType(Enum):
    """Different views/perspectives of the same computation."""
    LOGICAL = "logical"           # High-level computation graph
    DISTRIBUTED = "distributed"  # Sharded across devices
    DEVICE_LOCAL = "device_local" # Single device view
    TEMPORAL = "temporal"         # Time-ordered execution view


@dataclass
class AnnotationContext:
    """Context information for creating annotations."""
    graph_id: str
    trace_id: Optional[str] = None
    device_id: Optional[str] = None
    operation_id: Optional[str] = None
    step: Optional[int] = None
    pass_idx: Optional[int] = None
    batch_idx: Optional[int] = None
    layer_idx: Optional[int] = None
    view_type: ViewType = ViewType.LOGICAL

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for metadata storage."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


class Annotation(ABC):
    """Base class for all verification annotations."""

    def __init__(self, annotation_type: AnnotationType, context: AnnotationContext):
        self.annotation_type = annotation_type
        self.context = context
        self.id = f"{annotation_type.value}_{uuid.uuid4().hex[:8]}"
        self.timestamp = datetime.now().timestamp()

    @abstractmethod
    def to_metadata(self) -> Dict[str, Any]:
        """Convert annotation to metadata dict."""
        pass

    @abstractmethod
    def store_in_database(self, database) -> str:
        """Store annotation in the database, return storage ID."""
        pass


class LSHProjectionAnnotation(Annotation):
    """Annotation for LSH projections of tensors."""

    def __init__(
        self,
        context: AnnotationContext,
        tensor: jnp.ndarray,
        projection_dim: int,
        seed: Optional[int] = None,
        projection_type: str = "activation"
    ):
        super().__init__(AnnotationType.LSH_PROJECTION, context)
        self.tensor = tensor
        self.projection_dim = projection_dim
        self.projection_type = projection_type

        # Generate deterministic seed if not provided
        if seed is None:
            # Create deterministic seed from context
            seed_parts = [
                hash(context.graph_id) % 1000,
                context.step or 0,
                context.pass_idx or 0,
                context.batch_idx or 0,
                context.layer_idx or 0,
            ]
            seed = sum(part * (10 ** i) for i, part in enumerate(seed_parts)) % (2**31)

        self.seed = seed
        self.projection = self._compute_projection()

    def _compute_projection(self) -> jnp.ndarray:
        """Compute LSH projection with normalized matrix."""
        key = random.PRNGKey(self.seed)

        # Handle multi-dimensional tensors by flattening
        if self.tensor.ndim > 1:
            # For activations: project each batch element
            if self.projection_type == "activation":
                batch_size = self.tensor.shape[0]
                feature_dim = self.tensor.shape[-1]
                proj_matrix = random.normal(key, (feature_dim, self.projection_dim))
            else:
                # For gradients/weights: flatten completely
                flat_tensor = self.tensor.flatten()
                proj_matrix = random.normal(key, (self.projection_dim, len(flat_tensor)))
                # Normalize for stable projections
                proj_matrix = proj_matrix / jnp.linalg.norm(proj_matrix, axis=1, keepdims=True)
                return jnp.dot(proj_matrix, flat_tensor)
        else:
            feature_dim = len(self.tensor)
            proj_matrix = random.normal(key, (feature_dim, self.projection_dim))

        # Normalize rows for stable projections
        proj_matrix = proj_matrix / jnp.linalg.norm(proj_matrix, axis=1, keepdims=True)

        return jnp.dot(self.tensor, proj_matrix)

    def to_metadata(self) -> Dict[str, Any]:
        return {
            **self.context.to_dict(),
            "annotation_type": self.annotation_type.value,
            "projection_type": self.projection_type,
            "projection_dim": self.projection_dim,
            "seed": self.seed,
            "tensor_shape": self.tensor.shape,
            "timestamp": self.timestamp,
        }

    def store_in_database(self, database) -> str:
        """Store as a challenge record."""
        challenge = ChallengeRecord(
            id=self.id,
            challenge_type=f"lsh_{self.projection_type}",
            timestamp=self.timestamp,
            target_operation_id=self.context.operation_id,
            seed=self.seed,
            projection_dim=self.projection_dim,
            response_value=self.projection.tolist(),
            metadata=self.to_metadata(),
        )
        database.store_challenge(challenge)
        return self.id


class CheckpointAnnotation(Annotation):
    """Annotation for model checkpoints."""

    def __init__(
        self,
        context: AnnotationContext,
        params: Dict[str, jnp.ndarray],
        loss: Optional[float] = None,
        **extra_metadata
    ):
        super().__init__(AnnotationType.CHECKPOINT, context)
        self.params = params
        self.loss = loss
        self.extra_metadata = extra_metadata

    def to_metadata(self) -> Dict[str, Any]:
        metadata = {
            **self.context.to_dict(),
            "annotation_type": self.annotation_type.value,
            "timestamp": self.timestamp,
        }
        if self.loss is not None:
            metadata["loss"] = float(self.loss)
        metadata.update(self.extra_metadata)
        return metadata

    def store_in_database(self, database) -> str:
        """Store as a checkpoint DataBundle."""
        return database.store_checkpoint(
            self.context.graph_id,
            self.params,
            step=self.context.step,
            loss=self.loss,
            **self.extra_metadata
        )


class CrossReferenceAnnotation(Annotation):
    """Annotation for linking related data across views."""

    def __init__(
        self,
        context: AnnotationContext,
        reference_type: str,
        source_id: str,
        target_id: str,
        relationship: str,
        **extra_metadata
    ):
        super().__init__(AnnotationType.CROSS_REFERENCE, context)
        self.reference_type = reference_type
        self.source_id = source_id
        self.target_id = target_id
        self.relationship = relationship
        self.extra_metadata = extra_metadata

    def to_metadata(self) -> Dict[str, Any]:
        return {
            **self.context.to_dict(),
            "annotation_type": self.annotation_type.value,
            "reference_type": self.reference_type,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relationship": self.relationship,
            "timestamp": self.timestamp,
            **self.extra_metadata,
        }

    def store_in_database(self, database) -> str:
        """Store as a graph transformation link."""
        if self.reference_type == "graph_transformation":
            database.link_graph_transformation(
                self.source_id,
                self.target_id,
                self.relationship
            )
        # Could extend to other reference types
        return self.id


class AnnotationManager:
    """Central manager for creating and storing verification annotations."""

    def __init__(self, database):
        self.database = database
        self.annotations: List[Annotation] = []
        self._context_stack: List[AnnotationContext] = []

    def push_context(self, context: AnnotationContext):
        """Push a new annotation context (for nested scopes)."""
        self._context_stack.append(context)

    def pop_context(self) -> Optional[AnnotationContext]:
        """Pop the current annotation context."""
        return self._context_stack.pop() if self._context_stack else None

    @property
    def current_context(self) -> Optional[AnnotationContext]:
        """Get the current annotation context."""
        return self._context_stack[-1] if self._context_stack else None

    def annotate_lsh(
        self,
        tensor: jnp.ndarray,
        projection_dim: int,
        projection_type: str = "activation",
        context: Optional[AnnotationContext] = None,
        **kwargs
    ) -> LSHProjectionAnnotation:
        """Create and store an LSH projection annotation."""
        ctx = context or self.current_context
        if ctx is None:
            raise ValueError("No annotation context available")

        annotation = LSHProjectionAnnotation(ctx, tensor, projection_dim, projection_type=projection_type, **kwargs)
        annotation.store_in_database(self.database)
        self.annotations.append(annotation)
        return annotation

    def annotate_checkpoint(
        self,
        params: Dict[str, jnp.ndarray],
        loss: Optional[float] = None,
        context: Optional[AnnotationContext] = None,
        **kwargs
    ) -> CheckpointAnnotation:
        """Create and store a checkpoint annotation."""
        ctx = context or self.current_context
        if ctx is None:
            raise ValueError("No annotation context available")

        annotation = CheckpointAnnotation(ctx, params, loss, **kwargs)
        annotation.store_in_database(self.database)
        self.annotations.append(annotation)
        return annotation

    def annotate_cross_reference(
        self,
        reference_type: str,
        source_id: str,
        target_id: str,
        relationship: str,
        context: Optional[AnnotationContext] = None,
        **kwargs
    ) -> CrossReferenceAnnotation:
        """Create and store a cross-reference annotation."""
        ctx = context or self.current_context
        if ctx is None:
            raise ValueError("No annotation context available")

        annotation = CrossReferenceAnnotation(ctx, reference_type, source_id, target_id, relationship, **kwargs)
        annotation.store_in_database(self.database)
        self.annotations.append(annotation)
        return annotation

    def get_annotations(
        self,
        annotation_type: Optional[AnnotationType] = None,
        context_filter: Optional[Dict[str, Any]] = None
    ) -> List[Annotation]:
        """Retrieve annotations with optional filtering."""
        annotations = self.annotations

        if annotation_type:
            annotations = [a for a in annotations if a.annotation_type == annotation_type]

        if context_filter:
            filtered = []
            for annotation in annotations:
                context_dict = annotation.context.to_dict()
                if all(context_dict.get(k) == v for k, v in context_filter.items()):
                    filtered.append(annotation)
            annotations = filtered

        return annotations

    def create_multi_view_bundle(
        self,
        base_data: Dict[str, Any],
        views: Dict[ViewType, Dict[str, Any]],
        context: Optional[AnnotationContext] = None
    ) -> Dict[ViewType, str]:
        """Create multiple data bundles for different views of the same computation."""
        ctx = context or self.current_context
        if ctx is None:
            raise ValueError("No annotation context available")

        bundle_ids = {}

        for view_type, view_data in views.items():
            # Create a context for this view
            ctx_dict = ctx.to_dict()
            ctx_dict['view_type'] = view_type
            view_context = AnnotationContext(
                graph_id=ctx_dict.get('graph_id'),
                trace_id=ctx_dict.get('trace_id'),
                device_id=ctx_dict.get('device_id'),
                operation_id=ctx_dict.get('operation_id'),
                step=ctx_dict.get('step'),
                pass_idx=ctx_dict.get('pass_idx'),
                batch_idx=ctx_dict.get('batch_idx'),
                layer_idx=ctx_dict.get('layer_idx'),
                view_type=view_type
            )

            # Create data bundle with view-specific metadata
            bundle = DataBundle(
                id=f"{view_type.value}_{uuid.uuid4().hex[:8]}",
                graph_id=ctx.graph_id,
                bundle_type="execution",
                inputs=view_data.get("inputs", {}),
                outputs=view_data.get("outputs", {}),
                weights=view_data.get("weights", {}),
                activations=view_data.get("activations", {}),
                metadata={
                    **view_context.to_dict(),
                    "base_data_id": base_data.get("id"),
                    "view_type": view_type.value,
                }
            )

            bundle_ids[view_type] = self.database.store_data_bundle(bundle)

        # Create cross-references between views
        view_items = list(bundle_ids.items())
        for i, (view_a, id_a) in enumerate(view_items):
            for view_b, id_b in view_items[i+1:]:
                self.annotate_cross_reference(
                    "view_equivalence",
                    id_a,
                    id_b,
                    f"{view_a.value}_to_{view_b.value}",
                    context=ctx
                )

        return bundle_ids


# Convenience context manager for cleaner API
class AnnotationScope:
    """Context manager for annotation scopes."""

    def __init__(self, manager: AnnotationManager, context: AnnotationContext):
        self.manager = manager
        self.context = context

    def __enter__(self) -> AnnotationManager:
        self.manager.push_context(self.context)
        return self.manager

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.manager.pop_context()
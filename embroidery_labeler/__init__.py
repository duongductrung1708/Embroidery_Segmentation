"""Public API for SVG stitch labeling."""

from .svg_labeler import (
    LabelingConfig,
    LabelingResult,
    label_svg_bytes,
    label_svg_file,
    label_svg_files,
)

__all__ = [
    "LabelingConfig",
    "LabelingResult",
    "label_svg_bytes",
    "label_svg_file",
    "label_svg_files",
]

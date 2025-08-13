"""
Orientation UV-based remote photoplethysmography analysis package.

This package provides tools for analyzing physiological signals using
UV-based remote photoplethysmography techniques with face orientation masking.
"""

# Version is imported from single source
from orientation_uv_rppg._version import __version__

# Get other metadata dynamically from package installation
try:
    from importlib import metadata
    _package_metadata = metadata.metadata("orientation-uv-rppg")
    __author__ = _package_metadata.get("Author", "Unknown")
    __email__ = _package_metadata.get("Author-email", "")
    __description__ = _package_metadata.get("Summary", "")
except (ImportError, metadata.PackageNotFoundError):
    # Fallback if package not installed or importlib.metadata not available
    __author__ = "Your Name"
    __email__ = "your.email@csiro.au"
    __description__ = "UV-based remote photoplethysmography analysis package"

# Import main classes for easy access
from orientation_uv_rppg.core.core import OrientationMaskedTextureSpaceVideoProcessor
from orientation_uv_rppg.utils.landmark_detector import MediaPipeLandmarkDetector
from orientation_uv_rppg.utils.frame_warp import GPUPiecewiseAffineTransform
from orientation_uv_rppg.utils.frame_mask import UVAngleMasker

# Core processing class (main API)
__all__ = [
    # Main processor class
    "OrientationMaskedTextureSpaceVideoProcessor",
    
    # Utility classes (for advanced users)
    "MediaPipeLandmarkDetector",
    "GPUPiecewiseAffineTransform", 
    "UVAngleMasker",
    
    # Package metadata
    "__version__",
]

# Convenience alias for the main class (shorter name)
VideoProcessor = OrientationMaskedTextureSpaceVideoProcessor

# Add convenience alias to __all__
__all__.append("VideoProcessor")


def get_version() -> str:
    """Get package version."""
    return __version__


def get_package_info() -> dict:
    """Get complete package information."""
    try:
        from importlib import metadata
        pkg_metadata = metadata.metadata("orientation-uv-rppg")
        return {
            "name": pkg_metadata.get("Name", "orientation-uv-rppg"),
            "version": __version__,
            "author": pkg_metadata.get("Author", __author__),
            "email": pkg_metadata.get("Author-email", __email__),
            "description": pkg_metadata.get("Summary", __description__),
            "license": pkg_metadata.get("License", "Unknown"),
            "homepage": pkg_metadata.get("Home-page", ""),
        }
    except (ImportError, metadata.PackageNotFoundError):
        return {
            "name": "orientation-uv-rppg",
            "version": __version__,
            "author": __author__,
            "email": __email__,
            "description": __description__,
            "license": "CSIRO Non-Commercial License (based on BSD 3-Clause)",
            "homepage": "",
        }


def get_device_info() -> dict:
    """Get available compute device information."""
    import torch
    
    info = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    
    if torch.cuda.is_available():
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_memory"] = torch.cuda.get_device_properties(0).total_memory
    
    return info


# Optional: Quick processing function for simple use cases
def quick_process(frames, **kwargs):
    """
    Quick processing function for simple use cases.
    
    Args:
        frames: Input video frames (torch.Tensor)
        **kwargs: Additional parameters for OrientationMaskedTextureSpaceVideoProcessor
        
    Returns:
        torch.Tensor: Processed UV-masked frames
        
    Example:
        >>> import orientation_uv_rppg as ouv
        >>> import torch
        >>> frames = torch.randn(100, 480, 640, 3)  # 100 frames
        >>> processed = ouv.quick_process(frames)
        >>> print(processed.shape)  # Should be [100, 64, 64, 3] by default
    """
    processor = OrientationMaskedTextureSpaceVideoProcessor(**kwargs)
    return processor(frames)

"""
Pytest configuration and fixtures for orientation-uv-rppg tests.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, MagicMock
from typing import Tuple, Optional


@pytest.fixture
def sample_frames() -> torch.Tensor:
    """Create sample video frames for testing."""
    # Create sample frames: 10 frames, 480x640 resolution, RGB
    return torch.randn(10, 480, 640, 3)


@pytest.fixture
def small_sample_frames() -> torch.Tensor:
    """Create small sample video frames for quick testing."""
    # Create small frames: 5 frames, 64x64 resolution, RGB
    return torch.randn(5, 64, 64, 3)


@pytest.fixture
def sample_landmarks() -> torch.Tensor:
    """Create sample 3D landmarks for testing."""
    # MediaPipe face mesh has 468 landmarks
    return torch.randn(10, 468, 3)


@pytest.fixture
def sample_landmarks_uv() -> torch.Tensor:
    """Create sample UV landmarks for testing."""
    return torch.randn(468, 2)


@pytest.fixture
def sample_mesh_faces() -> torch.Tensor:
    """Create sample mesh faces for testing."""
    # Simplified triangular faces
    return torch.randint(0, 468, (900, 3))


@pytest.fixture
def mock_mediapipe_detector(sample_landmarks, sample_landmarks_uv, sample_mesh_faces):
    """Mock MediaPipe landmark detector."""
    detector = Mock()
    detector.landmarks_uv = sample_landmarks_uv
    detector.mesh_faces = sample_mesh_faces
    detector.camera_normal = torch.tensor([0., 0., -1.])
    detector.return_value = sample_landmarks
    detector.__call__ = Mock(return_value=sample_landmarks)
    return detector


@pytest.fixture
def mock_gpu_warper():
    """Mock GPU piecewise affine transform."""
    warper = Mock()
    
    def mock_warp(frames, landmarks, output_size=64):
        batch_size = frames.shape[0]
        return torch.randn(batch_size, output_size, output_size, 3)
    
    warper.__call__ = Mock(side_effect=mock_warp)
    return warper


@pytest.fixture
def mock_uv_masker():
    """Mock UV angle masker."""
    masker = Mock()
    
    def mock_mask(frames_uv, landmarks_uv, landmarks_xyz, degree_threshold=60.0):
        return frames_uv  # Return frames unchanged for testing
    
    masker.__call__ = Mock(side_effect=mock_mask)
    return masker


@pytest.fixture
def device() -> str:
    """Get appropriate device for testing."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def processor_config() -> dict:
    """Default configuration for processor testing."""
    return {
        "min_detection_confidence": 0.45,
        "min_tracking_confidence": 0.45,
        "device": "cpu",  # Use CPU for testing consistency
        "output_size": 32,  # Small size for faster testing
        "degree_threshold": 60.0,
    }


@pytest.fixture(params=[
    {"output_size": 32, "degree_threshold": 30.0},
    {"output_size": 64, "degree_threshold": 45.0},
    {"output_size": 128, "degree_threshold": 60.0},
])
def processor_configs(request):
    """Parametrized processor configurations for testing."""
    base_config = {
        "min_detection_confidence": 0.45,
        "min_tracking_confidence": 0.45,
        "device": "cpu",
    }
    base_config.update(request.param)
    return base_config


@pytest.fixture
def video_batch_sizes():
    """Different batch sizes for testing."""
    return [1, 5, 10, 20]


@pytest.fixture
def video_resolutions():
    """Different video resolutions for testing."""
    return [
        (64, 64),
        (128, 128), 
        (256, 256),
        (480, 640),
        (720, 1280)
    ]


# Test data directories
@pytest.fixture
def test_data_dir(tmp_path):
    """Create temporary directory for test data."""
    test_dir = tmp_path / "test_data"
    test_dir.mkdir()
    return test_dir


@pytest.fixture
def sample_video_file(test_data_dir, sample_frames):
    """Create a sample video file for testing (mock)."""
    video_path = test_data_dir / "sample_video.mp4"
    # In real implementation, you might save actual video
    # For now, just create a placeholder
    video_path.touch()
    return str(video_path)


# Error simulation fixtures
@pytest.fixture
def error_frames():
    """Frames that might cause errors."""
    return {
        "empty": torch.empty(0, 64, 64, 3),
        "wrong_dims": torch.randn(10, 64),  # Missing dimensions
        "wrong_channels": torch.randn(10, 64, 64, 1),  # Grayscale
        "negative": torch.randn(10, 64, 64, 3) - 2,  # Negative values
        "large": torch.randn(1, 2000, 2000, 3),  # Very large frames
    }


# Performance testing fixtures
@pytest.fixture
def performance_frames():
    """Large batch of frames for performance testing."""
    return torch.randn(100, 480, 640, 3)


@pytest.fixture(scope="session")
def torch_device_info():
    """Get torch device information for the test session."""
    return {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "torch_version": torch.__version__,
    }


# Cleanup fixtures
@pytest.fixture(autouse=True)
def cleanup_cuda():
    """Clean up CUDA memory after each test if available."""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Mock entire modules for unit testing
@pytest.fixture
def mock_mediapipe_module(monkeypatch):
    """Mock the entire mediapipe module."""
    mock_mp = Mock()
    mock_solutions = Mock()
    mock_face_mesh = Mock()
    
    # Set up the mock chain
    mock_mp.solutions = mock_solutions
    mock_solutions.face_mesh = mock_face_mesh
    
    monkeypatch.setattr("mediapipe", mock_mp)
    return mock_mp


@pytest.fixture
def mock_opencv_module(monkeypatch):
    """Mock OpenCV module for testing."""
    mock_cv2 = Mock()
    
    # Mock common cv2 functions
    mock_cv2.imread = Mock(return_value=np.random.rand(480, 640, 3))
    mock_cv2.VideoCapture = Mock()
    
    monkeypatch.setattr("cv2", mock_cv2)
    return mock_cv2


# Skip markers for conditional tests
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU (skip if no CUDA available)"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Add slow marker to performance tests
        if "performance" in item.nodeid or "large" in item.nodeid:
            item.add_marker(pytest.mark.slow)
        
        # Add gpu marker to CUDA tests
        if "cuda" in item.nodeid or "gpu" in item.nodeid:
            item.add_marker(pytest.mark.gpu)
        
        # Add integration marker to integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
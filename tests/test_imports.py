"""
Test imports and basic functionality.
"""

import pytest
import torch
import sys
from unittest.mock import patch, Mock


class TestPackageImports:
    """Test package-level imports."""
    
    def test_main_package_import(self):
        """Test that main package imports correctly."""
        import orientation_uv_rppg as ouv
        
        # Test package has required attributes
        assert hasattr(ouv, '__version__')
        assert hasattr(ouv, '__all__')
        assert isinstance(ouv.__all__, list)
        assert len(ouv.__all__) > 0
    
    def test_version_format(self):
        """Test version follows semantic versioning."""
        import orientation_uv_rppg as ouv
        
        version = ouv.__version__
        assert isinstance(version, str)
        
        # Check semantic versioning pattern (major.minor.patch)
        parts = version.split('.')
        assert len(parts) >= 2, f"Version '{version}' should have at least major.minor"
        
        # Check all parts are numeric
        for part in parts:
            assert part.replace('a', '').replace('b', '').replace('rc', '').isdigit(), \
                f"Version part '{part}' should be numeric or pre-release"
    
    def test_main_classes_available(self):
        """Test that all main classes are available at package level."""
        import orientation_uv_rppg as ouv
        
        required_classes = [
            'OrientationMaskedTextureSpaceVideoProcessor',
            'VideoProcessor',  # Alias
            'MediaPipeLandmarkDetector',
            'GPUPiecewiseAffineTransform',
            'UVAngleMasker',
        ]
        
        for class_name in required_classes:
            assert hasattr(ouv, class_name), f"Package missing class: {class_name}"
    
    def test_convenience_functions_available(self):
        """Test that convenience functions are available."""
        import orientation_uv_rppg as ouv
        
        convenience_functions = [
            'quick_process',
            'get_version',
        ]
        
        for func_name in convenience_functions:
            assert hasattr(ouv, func_name), f"Package missing function: {func_name}"
            assert callable(getattr(ouv, func_name)), f"{func_name} should be callable"


class TestSubmoduleImports:
    """Test submodule imports."""
    
    def test_core_module_imports(self):
        """Test core module imports."""
        from orientation_uv_rppg.core import OrientationMaskedTextureSpaceVideoProcessor
        
        # Test class is importable
        assert OrientationMaskedTextureSpaceVideoProcessor is not None
        
        # Test it's the same as the main package import
        import orientation_uv_rppg as ouv
        assert OrientationMaskedTextureSpaceVideoProcessor is ouv.OrientationMaskedTextureSpaceVideoProcessor
    
    def test_utils_module_imports(self):
        """Test utils module imports."""
        from orientation_uv_rppg.utils import (
            MediaPipeLandmarkDetector,
            GPUPiecewiseAffineTransform,
            UVAngleMasker
        )
        
        # Test all classes are importable
        assert MediaPipeLandmarkDetector is not None
        assert GPUPiecewiseAffineTransform is not None
        assert UVAngleMasker is not None
    
    def test_individual_module_imports(self):
        """Test importing from individual modules."""
        # Test individual imports work
        from orientation_uv_rppg.core.core import OrientationMaskedTextureSpaceVideoProcessor
        from orientation_uv_rppg.utils.landmark_detector import MediaPipeLandmarkDetector
        from orientation_uv_rppg.utils.frame_warp import GPUPiecewiseAffineTransform
        from orientation_uv_rppg.utils.frame_mask import UVAngleMasker
        
        # All should be importable without errors
        assert all([
            OrientationMaskedTextureSpaceVideoProcessor,
            MediaPipeLandmarkDetector,
            GPUPiecewiseAffineTransform,
            UVAngleMasker
        ])


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_get_version_function(self):
        """Test get_version function."""
        import orientation_uv_rppg as ouv
        
        version = ouv.get_version()
        assert isinstance(version, str)
        assert version == ouv.__version__
    
    def test_get_package_info_function(self):
        """Test get_package_info function if available."""
        import orientation_uv_rppg as ouv
        
        if hasattr(ouv, 'get_package_info'):
            info = ouv.get_package_info()
            assert isinstance(info, dict)
            
            required_keys = ['name', 'version']
            for key in required_keys:
                assert key in info, f"Package info missing key: {key}"
    
    def test_get_device_info_function(self):
        """Test get_device_info function if available."""
        import orientation_uv_rppg as ouv
        
        if hasattr(ouv, 'get_device_info'):
            info = ouv.get_device_info()
            assert isinstance(info, dict)
            
            required_keys = ['torch_version', 'cuda_available']
            for key in required_keys:
                assert key in info, f"Device info missing key: {key}"


class TestClassInstantiation:
    """Test that classes can be instantiated."""
    
    @patch('orientation_uv_rppg.utils.landmark_detector.MediaPipeLandmarkDetector')
    @patch('orientation_uv_rppg.utils.frame_warp.GPUPiecewiseAffineTransform')
    @patch('orientation_uv_rppg.utils.frame_mask.UVAngleMasker')
    def test_main_processor_instantiation(self, mock_masker, mock_warper, mock_detector):
        """Test main processor can be instantiated."""
        import orientation_uv_rppg as ouv
        
        # Mock the dependencies
        mock_detector_instance = Mock()
        mock_detector_instance.landmarks_uv = torch.randn(468, 2)
        mock_detector_instance.mesh_faces = torch.randint(0, 468, (900, 3))
        mock_detector_instance.camera_normal = torch.tensor([0., 0., -1.])
        mock_detector.return_value = mock_detector_instance
        
        # Test instantiation with default parameters
        processor = ouv.OrientationMaskedTextureSpaceVideoProcessor()
        
        assert processor.output_size == 64  # Default value
        assert processor.degree_threshold == 60.0  # Default value
    
    @patch('orientation_uv_rppg.utils.landmark_detector.MediaPipeLandmarkDetector')
    @patch('orientation_uv_rppg.utils.frame_warp.GPUPiecewiseAffineTransform')
    @patch('orientation_uv_rppg.utils.frame_mask.UVAngleMasker')
    def test_processor_custom_parameters(self, mock_masker, mock_warper, mock_detector):
        """Test processor with custom parameters."""
        import orientation_uv_rppg as ouv
        
        # Mock the dependencies
        mock_detector_instance = Mock()
        mock_detector_instance.landmarks_uv = torch.randn(468, 2)
        mock_detector_instance.mesh_faces = torch.randint(0, 468, (900, 3))
        mock_detector_instance.camera_normal = torch.tensor([0., 0., -1.])
        mock_detector.return_value = mock_detector_instance
        
        # Test with custom parameters
        processor = ouv.OrientationMaskedTextureSpaceVideoProcessor(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.8,
            device="cpu",
            output_size=128,
            degree_threshold=45.0
        )
        
        assert processor.output_size == 128
        assert processor.degree_threshold == 45.0
    
    def test_video_processor_alias(self):
        """Test VideoProcessor alias works."""
        import orientation_uv_rppg as ouv
        
        # Test alias is the same as main class
        assert ouv.VideoProcessor is ouv.OrientationMaskedTextureSpaceVideoProcessor


class TestQuickProcessFunction:
    """Test the quick_process convenience function."""
    
    @patch('orientation_uv_rppg.core.core.OrientationMaskedTextureSpaceVideoProcessor')
    def test_quick_process_basic(self, mock_processor_class):
        """Test quick_process function works."""
        import orientation_uv_rppg as ouv
        
        # Mock the processor
        mock_processor = Mock()
        mock_processor.return_value = torch.randn(5, 32, 32, 3)
        mock_processor_class.return_value = mock_processor
        
        # Test frames
        frames = torch.randn(5, 64, 64, 3)
        
        # Call quick_process
        result = ouv.quick_process(frames, output_size=32)
        
        # Verify processor was created and called
        mock_processor_class.assert_called_once_with(output_size=32)
        mock_processor.assert_called_once_with(frames)
        
        # Check result
        assert result.shape == (5, 32, 32, 3)


class TestAllExports:
    """Test __all__ exports."""
    
    def test_all_items_available(self):
        """Test that all items in __all__ are actually available."""
        import orientation_uv_rppg as ouv
        
        all_items = ouv.__all__
        
        for item in all_items:
            assert hasattr(ouv, item), f"Item '{item}' in __all__ but not available"
    
    def test_all_items_not_private(self):
        """Test that __all__ doesn't contain private items."""
        import orientation_uv_rppg as ouv
        
        all_items = ouv.__all__
        
        for item in all_items:
            assert not item.startswith('_'), f"Private item '{item}' should not be in __all__"
    
    def test_main_classes_in_all(self):
        """Test that main classes are in __all__."""
        import orientation_uv_rppg as ouv
        
        required_in_all = [
            'OrientationMaskedTextureSpaceVideoProcessor',
            'VideoProcessor',
            '__version__',
        ]
        
        for item in required_in_all:
            assert item in ouv.__all__, f"Required item '{item}' missing from __all__"


if __name__ == "__main__":
    # Run tests manually
    pytest.main([__file__, "-v"])
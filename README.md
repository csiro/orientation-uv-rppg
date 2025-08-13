<h1>Orientation UV rPPG</h1>

A self-contained Python package containing the video processing module similar to the paper <a href="https://samcantrill.github.io/orientation-uv-rppg/">Orientation-conditioned Facial Texture Mapping for Video-based Facial Remote Photoplethysmography Estimation</a>.

<h2>:wrench: Installation</h2>

<h3>Prerequisites</h3>

<ul>
    <li>Python 3.10 or higher</lu>
    <li>CUDA-compatible GPU (optional, but recommended for performance)
</ul>


<h3>Install from GitHub</h3>

```Bash
pip install git+https://github.com/csiro-internal/orientation-uv-rppg.git@package
```


<h2>:computer: Quick Start</h2>

<h3>Basic Usage</h3>

The simplest way to use the package:
```Python
import torch
import orientation_uv_rppg as ouv

# Load your video frames (replace with your video loading code)
frames = torch.randn(100, 480, 640, 3)  # 100 frames, 480x640 resolution, RGB

# Quick processing with default parameters
processed_frames = ouv.quick_process(frames)
print(f"Processed {frames.shape} → {processed_frames.shape}")
```

For more control over the video processing operations:
```Python
import torch
import orientation_uv_rppg as ouv

# Create video processor with custom parameters
processor = ouv.OrientationMaskedTextureSpaceVideoProcessor(
    min_detection_confidence=0.7,    # Higher confidence threshold
    min_tracking_confidence=0.8,     # More stable tracking
    device="cuda",                   # Use GPU acceleration
    output_size=128,                 # Higher resolution output
    degree_threshold=45.0            # Stricter orientation filtering
)

# Load your video frames
frames = torch.randn(200, 720, 1280, 3)  # HD video frames

# Process the video
result = processor(frames)
print(f"Input: {frames.shape}")
print(f"Output: {result.shape}")  # Should be [200, 128, 128, 3]
```

Please see the <code>examples/</code> directory for usage examples and visualizations.


<h2>:scroll: Citation</h2>

If you find this [paper](https://arxiv.org/abs/2404.09378) useful please cite our work.

```
@inproceedings{cantrill2024orientationconditionedfacialtexturemapping,
      title={Orientation-conditioned Facial Texture Mapping for Video-based Facial Remote Photoplethysmography Estimation}, 
      author={Sam Cantrill and David Ahmedt-Aristizabal and Lars Petersson and Hanna Suominen and Mohammad Ali Armin},
      booktitle={Proceedings of the IEEE/CVF Computer Vision and Pattern Recognition Workshops}
      year={2024},
      url={https://openaccess.thecvf.com/content/CVPR2024W/CVPM/papers/Cantrill_Orientation-conditioned_Facial_Texture_Mapping_for_Video-based_Facial_Remote_Photoplethysmography_Estimation_CVPRW_2024_paper.pdf}, 
}
```
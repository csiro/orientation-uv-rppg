<h1>Orientation-conditioned Facial Texture Mapping for Video-based Facial Remote Photoplethysmography Estimation</h1>

To reproduce the results from the paper please follow the steps listed below. If you have an questions or concerns please create an issue.

<h2>:file_folder: Resources</h2> 

<h3>Download Datasets</h3>

Please download the [MMPD](https://github.com/McJackTang/MMPD_rPPG_dataset) and [PURE](https://www.tu-ilmenau.de/universitaet/fakultaeten/fakultaet-informatik-und-automatisierung/profil/institute-und-fachgebiete/institut-fuer-technische-informatik-und-ingenieurinformatik/fachgebiet-neuroinformatik-und-kognitive-robotik/data-sets-code/pulse-rate-detection-dataset-pure) into your designated data directory. Note, access to these datasets typically requires a user/licence agreement, please see the specific datasets for further access details.


<h2>:wrench: Setup</h2> 

<h3>Create Environment</h3>

<ol>
    <li>Run <code>setup_conda.sh</code> or <code>setup_uv.sh</code> to create the virtual environment and install the relevant dependencies.</li>
    <li>Create a <code>.env</code> file containing <code>PROJECT_ROOT=&ltyour-project-root-directory&gt</code> and <code>DATA_ROOT=&ltyour-data-root-directory&gt</code>. An example file has been provded <code>.venv.example</code>.</li>
</ol>


<h2>:computer: Data Pre-processing</h2>

<h3>Step 1: Dataset Formatting</h3>

We re-format several datasets to allow for more dynamic experimentation. Perform formatting of the PURE and MMPD datasets by running:

```Bash
python scripts/process.py --config-name=format_pure
```

```Bash
python scripts/process.py --config-name=format_mmpd
```


<h3>Step 2: Perform Landmark Detection</h3>

Perform 3D landmark detection on the PURE and MMPD datasets using the [mediapipe](https://github.com/google-ai-edge/mediapipe) FaceMesh module by running:

```Bash
python scripts/process.py --config-name=detect_pure
```

```Bash
python scripts/process.py --config-name=format_mmpd
```


<h3>Step 3: Generate UV Video Frames</h3>

Leverage the detected 3D landmarks and the video to generate the UV coordinate video frames for the PURE and MMPD datasets by running:

```Bash
python scripts/process.py --config-name=process_pure_uv
```

```Bash
python scripts/process.py --config-name=process_mmpd_uv
```

Additionally perform re-sampling of the ground-truth BVP signal for the PURE dataset using the same process as the [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox/tree/main) by running:

```Bash
python scripts/process.py --config-name=process_pure_bvp_rppgtoolbox
```


<h2>:sparkles: Experiments</h2>

We use the [hydra](https://github.com/facebookresearch/hydra) to dynamically configure the experiment being run, we provide the configuration files used for all of the main results and ablation experiments. To override a file to run a specific experiment please consult the hydra documentation on configuration overrides.


<h3>Step 1: PhysNet-XY</h3>

<h4>Step 1A. Train PhysNet on PURE using XY-video</h4>

Launch training for fold-1 by running the following command, please note to run training for the remaining folds e.g. fold-2 please modify the subjects used during training e.g. <code>datamodule.dataset.construct.operations.split_index.train.filters.include_filter.match.subject=[1,2,3,4,5,6,9,10]</code>.

```Bash
src/train.py datamodule/dataset=pure_rppg_toolbox model=physnet_rppg_toolbox model/process/inputs=physnet_rppgtoolbox trainer.devices=[0] datamodule.dataset.construct.operations.split_index.train.filters.include_filter.match.subject=[1,2,3,4,5,6,7,8] note=PhysNetXY_Fold1
```

<h4>1B. Evaluate trained model on MMPD</h4>

Launch testing for a trained model by running the following command, please head over to the configured W&B folder to see the generated UUID or read through the training logs.

```Bash
src/test.py datamodule/dataset=pure_rppg_toolbox model=physnet_rppgtoolbox model/process/inputs=physnet_baseline trainer.devices=[0] +checkpoint=<model-dir>/model.ckpt
```

For convenience you can simply provide the UUID generated for a given training run, launch testing in this manner by running the following command:

```Bash
python scripts/test.py --uuid <UUID> datamodule/dataset=pure_rppg_toolbox model=physnet_rppgtoolbox model/process/inputs=physnet_baseline trainer.devices=[0]
```

<h4>1C. Extract the Pulse Rate from Predictions</h4>

Launch the post-processing pipeline to extract the pulse rate from the predicted rPPG signals.

```Bash
python scripts/process.py --config-name=evaluate inputs.regex="PhysNet3DCNN_MMPD_<UUID>.HDF5"
```

<h4>1D. Analyse the Errors</h4>

Head into the <code>notebooks/analysis.ipynb</code> and read the data contained in the generated <code>PROCESSED_PhysNet3DCNN_MMPD_&lt;UUID&gt;.HDF5</code> file.


<h3>Step 2: PhysNet-UV</h3>

<h4>2A. Train PhysNet on PURE using UV-video: Masking &theta;&geq;45&deg;</h4>

Launch training for fold-1 by running the following command, please note to run training for the remaining folds e.g. fold-2 please modify the subjects used during training e.g. <code>datamodule.dataset.construct.operations.split_index.train.filters.include_filter.match.subject=[1,2,3,4,5,6,9,10]</code>.

```Bash
src/train.py model=physnet_rppg_toolbox datamodule/dataset=pure_uv model/process/inputs=physnet_uv_mask45 trainer.devices=[0] datamodule.dataset.construct.operations.split_index.train.filters.include_filter.match.subject=[1,2,3,4,5,6,7,8] note=PhysNetUV45_Fold1
```

<h4>2B. Evaluate trained model on MMPD</h4>

Launch testing for a trained model by running the following command, please head over to the configured W&B folder to see the generated UUID or read through the training logs.

```Bash
src/test.py model=physnet_rppg_toolbox model/process/inputs=physnet_uv_mask45 datamodule/dataset=mmpd_uv trainer.devices=[0] +checkpoint=<model-dir>/model.ckpt
```

For convenience you can simply provide the UUID generated for a given training run, launch testing in this manner by running the following command:

```Bash
python scripts/test.py --uuid <UUID> model=physnet_rppg_toolbox model/process/inputs=physnet_uv_mask45 datamodule/dataset=mmpd_uv trainer.devices=[0]
```

<h4>2C. Extract the Pulse Rate from Predictions</h4>

Launch the post-processing pipeline to extract the pulse rate from the predicted rPPG signals.

```Bash
python scripts/process.py --config-name=evaluate inputs.regex="PhysNet3DCNN_PURE_rPPGToolbox_<UUID>.HDF5"
```

<h4>2D. Analyse the Errors</h4>

Head into the <code>notebooks/analysis.ipynb</code> and read the data contained in the generated <code>PROCESSED_PhysNet3DCNN_PURE_rPPGToolbox_&lt;UUID&gt;.HDF5</code> file.


<h2>:scroll: Citation</h2>

## :scroll: Citation

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
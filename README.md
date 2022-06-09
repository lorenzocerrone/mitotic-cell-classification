# Mitotic cell classification

## Requirements
* Linux
* Anaconda Python

## Install environment
```bash
conda create -n mitotic -c pytorch -c conda-forge  cudatoolkit=11.3 tifffile h5py pyyaml numba pytorch scipy napari notebook torchvision pytorch-lightning matplotlib
```

## Usage
1. Clone the `mitotic-cell-classification` repository
2. Start a Jupyer notebook
```bash
conda activate mitotic
jupyter-notebook
```
3. Open the `process-files.ipynb` notebook
4. Edit the inputs paths
```python
raw_path = '~/Mitotic-cells/raw/1136/1136_stain.tif'
seg_path = '~/Mitotic-cells/raw/1136/1136_segmented_flipped.tif'
```
5. Run all cells 

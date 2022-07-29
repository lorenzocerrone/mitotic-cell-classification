# Mitotic cell classification

## Requirements
* Linux
* Anaconda Python

## Install environment
```bash
conda create -n mitotic -c pytorch -c conda-forge  cudatoolkit=11.3 tifffile h5py pyyaml numba pytorch scipy napari notebook torchvision pytorch-lightning matplotlib
```

## Usage (with napari)
1. Clone the `mitotic-cell-classification` repository
2. Open a terminal inside `mitotic-cell-classification` and start the viewer
using 
```bash
conda activate mitotic
python run_viewer.py
``` 
3. Execute the widget in order (top to bottom) and one-by-one
4. Once you have a `Mitotic` layer in your viewer, you can over with the mouse
over wrongly classified cells and correct them using the key-binding `p`.
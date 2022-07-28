import h5py
import napari
import numpy as np
from pathlib import Path
from magicgui import magicgui, magic_factory
from src.utils.patches import process_data, process_tiff2h5
from napari.types import ImageData, LabelsData, LayerDataTuple
from typing import List, Dict
from dataclasses import dataclass
from concurrent.futures import Future
from src.classification.train import simple_predict
from src.utils.io import load_raw, load_segmentation
from src.utils.utils import map_cell_features2segmentation


def mitotic_viewer():
    @dataclass
    class ViewerState:
        models: List[str]
        results: Dict = None
        path: str = None
        patches_path: str = None

    @magic_factory(
        call_button="Open and Preprocessing",
        raw_path={"label": "Pick nuclei stain file (tiff)"},
        seg_path={"label": "Pick nuclei segmentation file (tiff)"},
        labels_path={"label": "(Optional) label as csv file"})
    def open_file(raw_path: Path = '/home/lcerrone/data/Mitotic-cells/raw/608/608_stain.tif',
                  seg_path: Path = '/home/lcerrone/data/Mitotic-cells/raw/608/608_segmented.tif',
                  labels_path: Path = None,
                  flip=False) -> Future[List[LayerDataTuple]]:
        from napari.qt.threading import thread_worker

        @thread_worker
        def func():
            path, (raw, seg) = process_tiff2h5(raw_path, segmentation_path=seg_path, flip=flip)
            patches_path = process_data(path,
                                        labels_csv_path=labels_path,
                                        shape=(3, 128, 128),
                                        sigma=(0, 1, 1))
            viewer_state.path = path
            viewer_state.patches_path = patches_path
            return [(raw, {}, "Image"),
                    (seg, {}, "Labels")
                    ]

        future = Future()

        def on_done(result):
            future.set_result(result)

        worker = func()
        worker.returned.connect(on_done)
        worker.start()

        return future


    @magic_factory(call_button="Run Predictions")
    def run_predictions(segmentation: LabelsData) -> Future[LayerDataTuple]:
        from napari.qt.threading import thread_worker

        @thread_worker
        def func():
            results = simple_predict(viewer_state.patches_path, 'place_holder')
            viewer_state.results = results

            list_outs = []
            for res in results['outputs'].values():
                list_outs.append(res)

            outs = np.array(list_outs)
            predictions = np.mean(outs, axis=0)

            predictions_image = map_cell_features2segmentation(segmentation,
                                                               results['cell_idx'],
                                                               predictions,
                                                               data_type='float64')
            viewer.add_image(predictions_image, visible=True)
            return predictions_image, {}, "Labels"


        future = Future()

        def on_done(result):
            future.set_result(result)

        worker = func()
        worker.returned.connect(on_done)
        worker.start()

        return future


    viewer_state = ViewerState(models='None')
    viewer = napari.Viewer()
    viewer.window.add_dock_widget(open_file())
    viewer.window.add_dock_widget(run_predictions())

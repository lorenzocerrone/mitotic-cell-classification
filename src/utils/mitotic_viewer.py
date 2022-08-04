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
from src.classification.train import simple_predict, export_labels_csv
from src.utils.io import load_raw, load_segmentation
from src.utils.utils import map_cell_features2segmentation


def predictions2mask(segmentation, cell_idx, predictions, threshold=0.5):
    return map_cell_features2segmentation(segmentation,
                                          cell_idx,
                                          predictions > threshold)


def mitotic_viewer():
    @dataclass
    class ViewerState:
        results: Dict = None
        path: str = None
        patches_path: str = None
        sigma: int = 0.5
        segmentation = None
        predictions = None
        cell_idx = None

    @magic_factory(
        call_button="Open and Preprocessing",
        raw_path={"label": "Pick nuclei stain file (tiff)"},
        seg_path={"label": "Pick nuclei segmentation file (tiff)"})
    def open_file(raw_path: Path = Path.home(),
                  seg_path: Path = Path.home(),
                  flip=False) -> Future[List[LayerDataTuple]]:
        from napari.qt.threading import thread_worker

        @thread_worker
        def func():
            path, (raw, seg) = process_tiff2h5(raw_path,
                                               segmentation_path=seg_path,
                                               flip=flip)
            patches_path = process_data(path,
                                        labels_csv_path=None,
                                        shape=(3, 128, 128),
                                        sigma=(0, 1, 1))
            viewer_state.path = path
            viewer_state.patches_path = patches_path
            return [(raw, {'name': 'Stain'}, 'image'),
                    (seg, {'name': 'Nuclei'}, 'labels')
                    ]

        future = Future()

        def on_done(result):
            future.set_result(result)

        worker = func()
        worker.returned.connect(on_done)
        worker.start()

        return future

    @magic_factory(call_button="Run Predictions", model_paths={"label": "Trained models directory", 'mode': 'd'})
    def run_predictions(segmentation: LabelsData, model_paths: Path) -> Future[List[LayerDataTuple]]:
        from napari.qt.threading import thread_worker
        list_checkpoints = list(model_paths.glob('*.ckpt'))

        @thread_worker
        def func():
            results = simple_predict(viewer_state.patches_path, list_checkpoints)
            viewer_state.results = results

            list_outs = []
            for res in results['outputs'].values():
                list_outs.append(res)

            outs = np.array(list_outs)
            predictions = np.mean(outs, axis=0)

            viewer_state.segmentation = segmentation
            viewer_state.predictions = predictions
            viewer_state.cell_idx = results['cell_idx']

            mask = predictions2mask(segmentation=segmentation,
                                    cell_idx=viewer_state.cell_idx,
                                    predictions=predictions,
                                    threshold=0.5)
            return [(mask, {'name': 'Mitotic'}, 'labels')]

        future = Future()

        def on_done(result):
            future.set_result(result)

        worker = func()
        worker.returned.connect(on_done)
        worker.start()

        return future

    @magicgui(call_button='Update', sigma={"widget_type": "FloatSlider", "max": 1., 'min': 0.})
    def threshold_pred(sigma: float = .5) -> Future[List[LayerDataTuple]]:
        from napari.qt.threading import thread_worker
        if viewer_state.predictions is None:
            return None

        @thread_worker
        def func():
            viewer_state.sigma = sigma
            mask = predictions2mask(segmentation=viewer_state.segmentation,
                                    cell_idx=viewer_state.cell_idx,
                                    predictions=viewer_state.predictions,
                                    threshold=sigma)
            return [(mask, {'name': 'Mitotic'}, 'labels')]

        future = Future()

        def on_done(result):
            future.set_result(result)

        worker = func()
        worker.returned.connect(on_done)
        worker.start()

        return future

    @magicgui(call_button='Export')
    def export_pred(mitotic: int = 1, normal: int = 10) -> None:
        out_path = str(viewer_state.patches_path).replace('.h5', '_predictions.csv')
        predictions = np.where(viewer_state.predictions > viewer_state.sigma,
                               mitotic, normal)
        export_labels_csv(viewer_state.cell_idx,
                          predictions,
                          path=out_path)

    viewer_state = ViewerState()
    viewer = napari.Viewer()

    viewer.window.add_dock_widget(open_file())
    viewer.window.add_dock_widget(run_predictions())
    viewer.window.add_dock_widget(threshold_pred)
    viewer.window.add_dock_widget(export_pred)

    @viewer.bind_key('p')
    def print_names(viewer):
        from napari.qt.threading import thread_worker

        @thread_worker
        def mitotic_proofread():
            pos = viewer.cursor.position
            z, x, y = viewer.layers['Nuclei'].world_to_data(pos)
            idx = viewer.layers['Nuclei'].data[int(z), int(x), int(y)]
            loc_idx = np.where(viewer_state.cell_idx == idx)[0]
            if viewer_state.predictions[loc_idx] > viewer_state.sigma:
                viewer_state.predictions[loc_idx] = 0.
            else:
                viewer_state.predictions[loc_idx] = 1.
            mask = predictions2mask(segmentation=viewer_state.segmentation,
                                    cell_idx=viewer_state.cell_idx,
                                    predictions=viewer_state.predictions,
                                    threshold=viewer_state.sigma)
            viewer.layers['Mitotic'].data = mask

        if 'Mitotic' in viewer.layers:
            worker = mitotic_proofread()
            worker.start()

    return viewer

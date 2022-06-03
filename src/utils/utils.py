from scipy.ndimage import zoom


def create_features_mapping(features_ids, features, all_ids=None, default_value=10):
    mapping = {}
    for key, value in zip(features_ids, features):
        mapping[key] = value

    if all_ids is not None:
        for idx in all_ids:
            mapping.setdefault(idx, default_value)
    return mapping


def scale_image(image, in_shape, out_shape, order=0):
    zoom_factor = [i / o for i, o in zip(in_shape, out_shape)]
    return zoom(image, zoom=zoom_factor, order=order)


def scale_image_voxel_size(image, current_voxel_size, out_voxel_size, order=0):
    zoom_factor = [i / o for i, o in zip(current_voxel_size, out_voxel_size)]
    return zoom(image, zoom=zoom_factor, order=order)

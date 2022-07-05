try:
    from numba import njit
except ImportError:
    print('NUMBA NOT FOUND! Program might be slower.')
    def njit(func):
        def inner(*args, **kwargs):
            return func(*args, **kwargs)
        return inner
import numpy as np
np.set_printoptions(suppress=True, linewidth=1000)
import tqdm
import skimage.morphology as skim
from skimage.morphology._util import (
    _offsets_to_raveled_neighbors,
    _resolve_neighborhood,
    _set_border_values,
)
from typing import List
import multiprocessing
from multiprocessing import Pool
from functools import partial


@njit
def _fill_threshold(
    image,
    flags,
    neighbor_offsets,
    start_index,
    seed_value,
    low_tol,
    high_tol,
    comparison,
):
    FILL = 1
    UNKNOWN = 0

    check_queue = []
    check_queue.append(start_index)
    flags[start_index] = FILL
    while check_queue:
        current_index = check_queue.pop()
        for i in range(neighbor_offsets.shape[0]):
            neighbor = current_index + neighbor_offsets[i]

            if comparison[neighbor] == UNKNOWN:
                if flags[neighbor] == UNKNOWN:
                    if image[neighbor] > seed_value:
                        return False
                    elif low_tol <= image[neighbor] <= high_tol:
                        flags[neighbor] = FILL
                        check_queue.append(neighbor)
            else:
                return False
    return True

def get_avg_pos(classes: int, regions: np.array, region_max_value: List, image: np.array):
    maxima_coords = []
    for cl in classes:
        coords = np.where(regions == cl)
        weights = image[coords]

        reg_max = region_max_value[cl - 1]  # np.max(weights)
        if len(weights) == 0:
            pass

        # weights = np.exp(2 * weights) - np.exp(-2)
        # weights = softmax(weights)
        try:
            avgs = []
            for c in coords:
                avg = np.average(c)  # , weights=weights)
                avgs.append(avg)
        except ZeroDivisionError:
            print("Zero devision. Pass this entry. Some dbug infos:")
            print("len(coords):", len(coords))
            print("Weights:", weights)
            pass

        p = tuple(avgs)
        # p = region_max_pos[cl-1]
        maxima_coords.append((p, len(coords[0]), reg_max))  # region_max_value[cl-1]))
    return maxima_coords

def find_maxima(volume: np.array, tolerance: float, global_min: float = 0.5) -> tuple[list, np.array]:

    """
    :param volume: 3D volume
    :param tolerance: Tolerance for detection
    :param global_min: Minimum value for peaks
    :return: List of maximas and flood filling mask
    """
    image = volume.astype(np.float32)
    if 0 in image.shape:
        return np.zeros(image.shape, dtype=bool)

    if image.flags.f_contiguous is True:
        order = "F"
    elif image.flags.c_contiguous is True:
        order = "C"
    else:
        image = np.ascontiguousarray(image)
        order = "C"

    ## Ignore the border
    if len(image.shape) == 3:
        output_slice = np.s_[1:-1, 1:-1, 1:-1]
        image[0, :, 0] = -1 * np.inf
        image[-1, :, 0] = -1 * np.inf
        image[0, :, -1] = -1 * np.inf
        image[-1, :, -1] = -1 * np.inf

        image[0, 0, :] = -1 * np.inf
        image[-1, 0, :] = -1 * np.inf
        image[0, -1, :] = -1 * np.inf
        image[-1, -1, :] = -1 * np.inf

        image[:, 0, 0] = -1 * np.inf
        image[:, -1, 0] = -1 * np.inf
        image[:, 0, -1] = -1 * np.inf
        image[:, -1, -1] = -1 * np.inf
    elif len(image.shape) == 2:
        output_slice = np.s_[1:-1, 1:-1]
        image[0, :] = -1 * np.inf
        image[:, 0] = -1 * np.inf
        image[-1, :] = -1 * np.inf
        image[:, -1] = -1 * np.inf
    elif len(image.shape) == 1:
        output_slice = np.s_[1:-1]
        image[0] = -1 * np.inf
        image[-1] = -1 * np.inf
    else:
        print("Dimension > 3 not supported!")

    local_maxima = skim.local_maxima(image, indices=True, allow_borders=False)
    max_sorted = np.argsort(-1 * image[local_maxima])

    # Start flood filling
    coords_sorted = [
        tuple([arr[max_index] for arr in local_maxima]) for max_index in max_sorted
    ]

    if global_min == None:
        global_min = np.min(image) + tolerance
        print("effective global min:", global_min)

    footprint = None
    connectivity = None

    working_image = np.pad(image, 1, mode="constant", constant_values=image.min())
    footprint = _resolve_neighborhood(footprint, connectivity, image.ndim)
    neighbor_offsets = _offsets_to_raveled_neighbors(
        working_image.shape, footprint, center=((1,) * image.ndim), order=order
    )
    flags = np.zeros(working_image.shape, dtype=np.uint8, order=order)
    regions = np.zeros(working_image.shape, dtype=np.int32, order=order)
    tmp_flags = flags.copy()
    _set_border_values(flags, value=2)
    try:
        max_value = np.finfo(working_image.dtype).max
        min_value = np.finfo(working_image.dtype).min
    except ValueError:
        max_value = np.iinfo(working_image.dtype).max
        min_value = np.iinfo(working_image.dtype).min

    k = 0
    region_max_pos = []
    region_max_value = []
    working_image_raveled = working_image.ravel(order)
    for idx, seed_point in enumerate(coords_sorted):
        try:
            iter(seed_point)
        except TypeError:
            seed_point = (seed_point,)

        seed_value = image[seed_point]
        seed_point = tuple(np.asarray(seed_point) % image.shape)
        ravelled_seed_idx = np.ravel_multi_index(
            [i + 1 for i in seed_point], working_image.shape, order=order
        )
        if not (seed_value > global_min and regions.ravel(order)[ravelled_seed_idx] == 0):
            continue

        tmp_flags[...] = flags[...]
        high_tol = min(max_value, seed_value + tolerance)
        low_tol = max(min_value, seed_value - tolerance)

        keep_region = _fill_threshold(
            working_image_raveled,
            tmp_flags.ravel(order),
            neighbor_offsets,
            ravelled_seed_idx,
            seed_value,
            low_tol,
            high_tol,
            regions.ravel(order),
        )
        tmp_flags[tmp_flags == 2] = 0
        if not keep_region:
            continue

        k = k + 1
        regions[tmp_flags.astype(bool)] = k
        region_max_pos.append(seed_point)
        region_max_value.append(seed_value)

    regions = regions[output_slice]

    num_cores = multiprocessing.cpu_count()
    region_list = list(range(1, k + 1))
    chunked_arrays = np.array_split(region_list, num_cores)
    from concurrent.futures import ProcessPoolExecutor as Pool
    with Pool() as pool:
        maxima_coords = pool.map(partial(get_avg_pos, regions=regions, region_max_value=region_max_value, image=image),
                     chunked_arrays)
        #maxima_coords = pool.map(get_avg_pos, repeat(regions), repeat(region_max_value), repeat(image), chunked_arrays)
    import itertools

    maxima_coords = list(itertools.chain.from_iterable(maxima_coords))


    return maxima_coords, regions
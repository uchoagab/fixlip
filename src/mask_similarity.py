from collections.abc import Sequence
from itertools import combinations
import random
from typing import Union

import numpy as np
import torch
from PIL import Image

from .attribution import _load_clip


def compute_mask_similarity(
    image_pil: Image.Image,
    text: str,
    mask_array: np.ndarray,
    visible_regions: Sequence[int],
    device: Union[str, torch.device] = "cpu",
) -> float:
    """Return CLIP cosine similarity for an image with only selected regions visible."""
    device_name = str(device)
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device '{device_name}', but CUDA is unavailable")

    image = image_pil.convert("RGB")
    mask = np.asarray(mask_array)
    if mask.ndim != 2:
        raise ValueError("mask_array must be a two-dimensional numpy array")
    if mask.shape != (image.height, image.width):
        raise ValueError(
            "mask_array dimensions must match the image dimensions "
            f"({image.height}x{image.width})"
        )

    visible_mask = np.isin(mask, tuple(int(region_id) for region_id in visible_regions))
    masked_pixels = np.asarray(image).copy()
    masked_pixels[~visible_mask] = 0
    masked_image = Image.fromarray(masked_pixels)

    model, processor = _load_clip(device_name)
    inputs = processor(images=masked_image, text=text, return_tensors="pt")
    inputs = {name: value.to(device_name) for name, value in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs)
        similarity = outputs.logits_per_image[0, 0] / model.logit_scale.exp()

    return float(similarity.item())


def predict_mask_similarity(
    phi_base: np.ndarray,
    M: np.ndarray,
    visible_regions: Sequence[int],
    phi_0: float = 0.0,
) -> float:
    """Predict mask similarity with the requested additive attribution approximation."""
    phi = np.asarray(phi_base, dtype=np.float64)
    matrix = np.asarray(M, dtype=np.float64)
    if phi.ndim != 1:
        raise ValueError("phi_base must be a one-dimensional array")
    if matrix.shape != (phi.size, phi.size):
        raise ValueError("M must be a square matrix matching the size of phi_base")

    region_ids = tuple(dict.fromkeys(int(region_id) for region_id in visible_regions))
    if any(region_id < 1 or region_id > phi.size for region_id in region_ids):
        raise ValueError("visible region IDs must be between 1 and len(phi_base)")

    indices = np.asarray([region_id - 1 for region_id in region_ids], dtype=int)
    regional_sum = float(phi[indices].sum()) if indices.size else 0.0
    interaction_sum = (
        float(matrix[np.ix_(indices, indices)].sum()) if indices.size else 0.0
    )
    return float(phi_0 + regional_sum + interaction_sum)


def quadratic_difference(v_real: float, v_pred: float) -> float:
    """Return the squared difference between real and predicted mask similarity."""
    return float((v_real - v_pred) ** 2)


def calibrate_coefficients(
    image_pil: Image.Image,
    text: str,
    mask_array: np.ndarray,
    n_regioes: int,
    device: Union[str, torch.device] = "cpu",
) -> tuple[float, np.ndarray, np.ndarray]:
    """Fit CLIP scores; split each unordered pair effect across symmetric matrix cells."""
    if n_regioes < 0:
        raise ValueError("n_regioes must be non-negative")

    mask = np.asarray(mask_array)
    if mask.ndim != 2:
        raise ValueError("mask_array must be a two-dimensional numpy array")
    if mask.shape != (image_pil.height, image_pil.width):
        raise ValueError(
            "mask_array dimensions must match the image dimensions "
            f"({image_pil.height}x{image_pil.width})"
        )

    region_ids = tuple(
        int(region_id)
        for region_id in np.unique(mask)
        if int(region_id) != 0
    )
    if not region_ids:
        raise ValueError("mask_array must contain at least one non-background region")
    if len(region_ids) != n_regioes:
        raise ValueError(
            f"n_regioes ({n_regioes}) must match the {len(region_ids)} "
            "non-background region IDs in mask_array"
        )

    region_pairs = tuple(combinations(region_ids, 2))
    if n_regioes > 6:
        sample_size = min(30, (1 << len(region_ids)) - 1)
        rng = random.Random(0)
        sampled_subsets: list[tuple[int, ...]] = []
        seen_subsets: set[tuple[int, ...]] = set()
        while len(sampled_subsets) < sample_size:
            subset = tuple(
                region_id
                for region_id in region_ids
                if rng.getrandbits(1)
            )
            if subset and subset not in seen_subsets:
                sampled_subsets.append(subset)
                seen_subsets.add(subset)
        visible_subsets = sampled_subsets
    else:
        visible_subsets = [
            subset
            for subset_size in range(1, len(region_ids) + 1)
            for subset in combinations(region_ids, subset_size)
        ]
    visible_subsets.insert(0, ())

    region_columns = {region_id: index + 1 for index, region_id in enumerate(region_ids)}
    pair_offset = 1 + len(region_ids)
    pair_columns = {
        pair: pair_offset + index for index, pair in enumerate(region_pairs)
    }
    design_matrix = np.zeros(
        (len(visible_subsets), 1 + len(region_ids) + len(region_pairs)),
        dtype=np.float64,
    )
    target_values = np.empty(len(visible_subsets), dtype=np.float64)

    for row_index, visible_regions in enumerate(visible_subsets):
        design_matrix[row_index, 0] = 1.0
        for region_id in visible_regions:
            design_matrix[row_index, region_columns[region_id]] = 1.0
        for pair in combinations(visible_regions, 2):
            design_matrix[row_index, pair_columns[pair]] = 1.0
        target_values[row_index] = compute_mask_similarity(
            image_pil,
            text,
            mask,
            visible_regions,
            device=device,
        )

    coefficients, _, _, _ = np.linalg.lstsq(
        design_matrix,
        target_values,
        rcond=None,
    )
    intercept = float(coefficients[0])
    phi = coefficients[1:pair_offset].copy()
    matrix_size = max(n_regioes, max(region_ids))
    phi_by_region_id = np.zeros(matrix_size, dtype=np.float64)
    pair_matrix = np.zeros((matrix_size, matrix_size), dtype=np.float64)
    region_indices = {region_id: index for index, region_id in enumerate(region_ids)}
    for region_id, region_index in region_indices.items():
        phi_by_region_id[region_id - 1] = phi[region_index]
    for (region_a, region_b), column in pair_columns.items():
        coefficient = coefficients[column] / 2.0
        index_a = region_a - 1
        index_b = region_b - 1
        pair_matrix[index_a, index_b] = coefficient
        pair_matrix[index_b, index_a] = coefficient

    return intercept, phi_by_region_id, pair_matrix

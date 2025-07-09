import numpy as np
import torch
from PIL import Image
import SimpleITK as sitk
from skimage import measure

def process_image(
        image: np.ndarray, 
        image_size: int, 
        mean: tuple[float, float, float], 
        std: tuple[float, float, float],
        lower_bound: float,
        upper_bound: float
    ) -> torch.Tensor:
    """
    Convert a 3D grayscale NumPy array to a normalized CUDA torch tensor of RGB images.
    Applies intensity clipping and normalization before processing.

    Args:
        image (np.ndarray): Input array of shape (d, h, w).
        image_size (int): Desired size for the width and height.
        mean (tuple[float, float, float]): Mean values for each channel.
        std (tuple[float, float, float]): Standard deviation values for each channel.
        lower_bound (float): Lower bound for intensity clipping.
        upper_bound (float): Upper bound for intensity clipping.
    Returns:
        torch.Tensor: Normalized tensor of shape (d, 3, image_size, image_size) on CUDA.
    """
    # Intensity clipping and normalization
    image = np.clip(image, lower_bound, upper_bound)
    image = (image - lower_bound) / (upper_bound - lower_bound) * 255.0
    image = image.astype(np.uint8)

    d, h, w = image.shape
    resized_array = np.zeros((d, 3, image_size, image_size), dtype=np.float32)

    for i in range(d):
        img_pil = Image.fromarray(image[i])
        img_rgb = img_pil.convert("RGB")
        img_resized = img_rgb.resize((image_size, image_size))
        img_array = np.array(img_resized).transpose(2, 0, 1)  # (3, image_size, image_size)
        resized_array[i] = img_array

    # Normalize to [0, 1]
    resized_array /= 255.0

    # Convert to torch tensor and move to CUDA
    img_tensor = torch.from_numpy(resized_array).cuda()

    # Standardize
    img_mean = torch.tensor(mean, dtype=torch.float32, device=img_tensor.device)[:, None, None]
    img_std = torch.tensor(std, dtype=torch.float32, device=img_tensor.device)[:, None, None]
    img_tensor -= img_mean
    img_tensor /= img_std

    return img_tensor


def mask3D_to_bbox(gt3D: np.ndarray, max_shift: int = 20) -> np.ndarray:
    """Convert a 3D binary mask to a bounding box with optional random padding.

    This function takes a 3D binary mask and computes its bounding box coordinates.
    It can optionally add random padding to the x and y dimensions up to max_shift pixels.
    The z dimension is not padded.

    Args:
        gt3D: A 3D numpy array containing a binary mask (values > 0 are considered foreground)
        max_shift: Maximum number of pixels to randomly pad the x and y dimensions (default: 20)

    Returns:
        np.ndarray: A 1D array of 6 coordinates [x_min, y_min, z_min, x_max, y_max, z_max]
                   defining the bounding box that contains the mask
    
    Note:
        The returned coordinates are guaranteed to be within the original array bounds
        even after applying the random padding.

    Credit:
        Adapted from MedSAM2 by Bo Wang Lab
        https://github.com/bowang-lab/MedSAM2/blob/main/medsam2_infer_3D_CT.py
    """
    # Find the indices of non-zero elements
    z_indices, y_indices, x_indices = np.where(gt3D > 0)
    
    # Get min and max coordinates in each dimension
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    z_min, z_max = np.min(z_indices), np.max(z_indices)
    
    # Get array dimensions
    D, H, W = gt3D.shape
    
    # Generate random padding amount
    bbox_shift = np.random.randint(0, max_shift + 1, 1)[0]
    
    # Apply padding to x and y dimensions while keeping within bounds
    x_min = max(0, x_min - bbox_shift)
    x_max = min(W-1, x_max + bbox_shift)
    y_min = max(0, y_min - bbox_shift)
    y_max = min(H-1, y_max + bbox_shift)
    
    # Keep z dimension unchanged but ensure within bounds
    z_min = max(0, int(z_min))
    z_max = min(D-1, int(z_max))
    
    # Return coordinates as 1D array
    boxes3d = np.array([x_min, y_min, z_min, x_max, y_max, z_max])

    return boxes3d


def getLargestCC(segmentation: np.ndarray) -> np.ndarray:
    """
    Get the largest connected component of a segmentation.

    Args:
        segmentation: The segmentation array.

    Returns:
        largestCC: The largest connected component of the segmentation.
    """
    if np.max(segmentation) == 0:
        return segmentation
    
    labels = measure.label(segmentation)
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC

def infer(
        predictor, 
        image: torch.Tensor, 
        propagate_with_box: bool, 
        bbox: np.ndarray, 
        original_shape: tuple[int, int, int],
        key_slice: int
    ) -> np.ndarray:
    """
    Infer the mask for the image.

    Args:
        predictor: The predictor object.
        image: The image tensor.
        propagate_with_box: Whether to propagate the mask with the bounding box.
        bbox: The bounding box in format [x_min, y_min, x_max, y_max].
        original_shape: The original shape of the image.
        key_slice: The key slice index for the bounding box.
    
    Returns:
        segs_3D: The inferred mask.
    """

    segs_3D = np.zeros(original_shape, dtype=np.uint8)
    video_height, video_width = original_shape[1:]
    
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        inference_state = predictor.init_state(image, video_height, video_width)

        # Forward pass
        if propagate_with_box:
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=key_slice,
                obj_id=1,
                box=bbox,
            )

        for out_frame_idx, _, out_mask_logits in predictor.propagate_in_video(inference_state):
            segs_3D[out_frame_idx, (out_mask_logits[0] > 0.0).cpu().numpy()[0]] = 1

        predictor.reset_state(inference_state)

        # Backward pass
        if propagate_with_box:
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=key_slice,
                obj_id=1,
                box=bbox,
            )

        for out_frame_idx, _, out_mask_logits in predictor.propagate_in_video(inference_state, reverse=True):
            segs_3D[out_frame_idx, (out_mask_logits[0] > 0.0).cpu().numpy()[0]] = 1

        predictor.reset_state(inference_state)
        
    if np.max(segs_3D) > 0:
        segs_3D = getLargestCC(segs_3D)
        segs_3D = np.uint8(segs_3D) 

    return segs_3D
from pathlib import Path
import json

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
import SimpleITK as sitk
from sam2.build_sam import build_sam2_video_predictor_npz

from utils import infer, mask3D_to_bbox, process_image


class MedSAM3DRunnerConfig(BaseModel):
    """Configuration for MedSAM3D runner."""

    dataset_csv: str = Field(description="Path to the dataset CSV file.")
    model_config_path: str = Field(description="Path to the model configuration file.")
    checkpoint_path: str = Field(description="Path to the checkpoint file.")
    output_dir: str = Field(description="Path to the output directory.")

    image_size: int = Field(default=512, description="Size of the image to be processed.")  
    lower_bound: float = Field(default=-500, description="Lower bound for intensity clipping.")
    upper_bound: float = Field(default=1000, description="Upper bound for intensity clipping.")
    mean: tuple[float, float, float] = Field(default=(0.485, 0.456, 0.406), description="Mean values for each channel.")
    std: tuple[float, float, float] = Field(default=(0.229, 0.224, 0.225), description="Standard deviation values for each channel.")

    propagate_with_bbox: bool = Field(default=True, description="Whether to propagate the mask with the bounding box.")

    def post_init(self):
        """Post-initialization hook."""
        # Check if dataset file exists
        if not Path(self.dataset_csv).exists():
            raise FileNotFoundError(f"Dataset CSV file not found: {self.dataset_csv}")
        
        # Check if model config file exists
        if not Path(self.model_config_path).exists():
            raise FileNotFoundError(f"Model config file not found: {self.model_config_path}")
        
        # Check if checkpoint file exists
        if not Path(self.checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint file not found: {self.checkpoint_path}")

class MedSAM3DRunner:
    """Class for running MedSAM3D segmentation model."""

    def __init__(self, config: MedSAM3DRunnerConfig):
        self.config = config

        self.predictor = build_sam2_video_predictor_npz(
            config_file=self.config.model_config_path,
            checkpoint_path=self.config.checkpoint_path,
        )

    def run(self):
        """Run the MedSAM3D segmentation model."""
        
        dataset = pd.read_csv(self.config.dataset_csv)

        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        masks_dir = Path(self.config.output_dir) / "masks"
        masks_dir.mkdir(parents=True, exist_ok=True)

        with open(Path(self.config.output_dir) / "config.json", "w") as f:
            json.dump(self.config.model_dump(), f, indent=4)

        for _, row in dataset.iterrows():
            image_path = Path(str(row["image_path"]))
            mask_path = Path(str(row["mask_path"]))
            
            # Check if image and mask files exist
            if not image_path.exists():
                print(f"Warning: Image file not found: {image_path}, skipping...")
                continue
            if not mask_path.exists():
                print(f"Warning: Mask file not found: {mask_path}, skipping...")
                continue
            
            try:
                image = sitk.GetArrayFromImage(sitk.ReadImage(str(image_path)))
                mask_image = sitk.ReadImage(str(mask_path))
                mask = sitk.GetArrayFromImage(mask_image)
            except Exception as e:
                print(f"Error reading image/mask for {str(row['ID'])}: {e}, skipping...")
                continue

            original_shape = image.shape
            image = process_image(
                image, 
                self.config.image_size, 
                self.config.mean, 
                self.config.std, 
                self.config.lower_bound, 
                self.config.upper_bound
            )

            x_min, y_min, z_min, x_max, y_max, z_max = mask3D_to_bbox(mask)
            bbox = np.array([x_min, y_min, x_max, y_max])
            key_frame_idx = z_min + (z_max - z_min) // 2

            print("Bounding Box Info:")
            print(f"x_min: {x_min}, y_min: {y_min}, x_max: {x_max}, y_max: {y_max}")
            print(f"Key Frame Index: {key_frame_idx}")
            print(f"z_min: {z_min}, z_max: {z_max}")
            
            segs_3D = infer(self.predictor, image, self.config.propagate_with_bbox, bbox, original_shape, key_frame_idx)

            # Calculate Dice score
            intersection = np.logical_and(segs_3D, mask)
            dice_score = 2.0 * intersection.sum() / (segs_3D.sum() + mask.sum())
            print(f"Dice score for {str(row['ID'])}: {dice_score:.4f}")

            sitk_mask = sitk.GetImageFromArray(segs_3D)
            sitk_mask.CopyInformation(mask_image)

            sitk.WriteImage(sitk_mask, masks_dir / f"{str(row['ID'])}.nii.gz")   


if __name__ == "__main__":
    # Get the directory where this script is located

    config = MedSAM3DRunnerConfig(
        dataset_csv="data/test_dataset.csv",
        checkpoint_path="MedSAM2/checkpoints/MedSAM2_latest.pt",
        model_config_path="configs/sam2.1_hiera_t512.yaml",
        output_dir="output",
    )
    runner = MedSAM3DRunner(config)
    runner.run()

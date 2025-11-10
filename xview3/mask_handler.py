import numpy as np
import torch
import cv2
from typing import Tuple
from .constants import MASK_WATER, MASK_NONWATER, VALID_MASK_VALUES

class MaskHandler:
    """Enhanced mask handler for SAR image processing with binary water/non-water classification"""
    def __init__(self, mask: np.ndarray):
        self.mask = mask
        self.water_mask = (mask == MASK_WATER)
        self.invalid_mask = (mask == MASK_NONWATER)
    
    def is_valid_location(self, y: int, x: int, window_size: int = 7) -> bool:
        """Check if a location is valid (in water) with surrounding context"""
        half = window_size // 2
        y1, y2 = max(0, y - half), min(self.mask.shape[0], y + half + 1)
        x1, x2 = max(0, x - half), min(self.mask.shape[1], x + half + 1)
        
        region = self.invalid_mask[y1:y2, x1:x2]
        return not np.any(region)
    
    def get_tile_mask(self, coords: Tuple[int, int, int, int], sar_to_mask_ratio: int = 4) -> np.ndarray:
        """Convert SAR tile coordinates to mask space and return upsampled mask"""
        x, y, w, h = coords
        
        # Convert to mask coordinates with proper rounding
        mask_x = x // sar_to_mask_ratio
        mask_y = y // sar_to_mask_ratio
        mask_w = max(1, w // sar_to_mask_ratio)
        mask_h = max(1, h // sar_to_mask_ratio)
        
        # Get mask region
        mask_region = self.invalid_mask[
            mask_y:mask_y+mask_h, 
            mask_x:mask_x+mask_w
        ]
        
        # Upsample to SAR resolution using nearest neighbor
        return cv2.resize(
            mask_region.astype(np.uint8),
            (w, h),
            interpolation=cv2.INTER_NEAREST
        )
    
    def validate_mask(self, mask: torch.Tensor) -> bool:
        """Validate mask values against valid mask values"""
        device_type = str(mask.device).split(':')[0]
        
        if device_type == 'mps':
            with torch.autocast(device_type='mps', enabled=True):
                mask = mask.to(device='mps', dtype=torch.float16)
                unique_values = torch.unique(mask).cpu().numpy()
        else:
            unique_values = torch.unique(mask).cpu().numpy()
            
        return set(unique_values).issubset(VALID_MASK_VALUES)
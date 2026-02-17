"""
ML Model Service for Pneumonia Detection.
Includes lung segmentation validation and pneumonia classification.
Singleton pattern to load models once at startup.
"""

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50
from PIL import Image
import numpy as np
from django.conf import settings
import segmentation_models_pytorch as smp


class LungSegmentor:
    """
    Singleton class for lung segmentation using U-Net with ResNet34 encoder.
    Validates that uploaded image contains lungs before classification.
    """
    _instance = None
    _model = None
    _transform = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize the segmentation model and transforms."""
        print(f"Loading lung segmentation model on {settings.DEVICE}...")
        
        # Create U-Net model with ResNet34 encoder (matching training config)
        self._model = smp.Unet(
            encoder_name='resnet34',
            encoder_weights=None,  # Will load from checkpoint
            in_channels=1,         # Grayscale X-ray
            classes=1              # Binary mask
        )
        
        # Load trained weights
        checkpoint_path = settings.SEGMENTATION_CHECKPOINT_PATH
        if checkpoint_path and checkpoint_path.exists():
            checkpoint = torch.load(
                checkpoint_path,
                map_location=settings.DEVICE,
                weights_only=True
            )
            if 'model_state_dict' in checkpoint:
                self._model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self._model.load_state_dict(checkpoint)
            print(f"Segmentation model loaded from {checkpoint_path}")
        else:
            print(f"WARNING: Segmentation checkpoint not found at {checkpoint_path}")
            print("Lung validation will be skipped!")
        
        self._model.to(settings.DEVICE)
        self._model.eval()
        
        # Create inference transform (grayscale input)
        self._transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((settings.SEGMENTATION_IMAGE_SIZE, settings.SEGMENTATION_IMAGE_SIZE)),
            transforms.ToTensor(),
        ])
        
        print("Lung segmentation model ready!")
    
    def _is_xray_like(self, image):
        """
        Check if an image has sufficient contrast (not a solid color).
        
        Returns:
            (is_valid, details_dict)
        """
        details = {}
        
        # Convert to RGB for analysis
        if image.mode != 'RGB':
            rgb_img = image.convert('RGB')
        else:
            rgb_img = image
        
        img_array = np.array(rgb_img).astype(np.float32)
        gray = np.mean(img_array, axis=2)
        
        # Check 1: Not too uniform (solid color images)
        std_dev = np.std(gray)
        details['std_dev'] = round(std_dev, 2)
        
        MIN_STD_DEV = 20.0  # Minimum standard deviation for sufficient contrast
        if std_dev < MIN_STD_DEV:
            details['reason'] = f'Image too uniform (std dev: {std_dev:.1f} < {MIN_STD_DEV})'
            return False, details
        
        # Check 2: Appropriate intensity range
        min_val, max_val = np.min(gray), np.max(gray)
        intensity_range = max_val - min_val
        details['intensity_range'] = round(intensity_range, 2)
        
        MIN_INTENSITY_RANGE = 50.0  # Minimum range for reasonable contrast
        if intensity_range < MIN_INTENSITY_RANGE:
            details['reason'] = f'Insufficient contrast (range: {intensity_range:.1f} < {MIN_INTENSITY_RANGE})'
            return False, details
        
        return True, details
    
    def segment(self, image):
        """
        Segment lungs from a chest X-ray image.
        
        Args:
            image: PIL Image
            
        Returns:
            dict with:
                - mask: PIL Image of lung mask
                - lung_area_ratio: float (0-1) of image covered by lungs
                - is_valid_lung: bool indicating if lungs were detected
                - masked_image: PIL Image with non-lung areas darkened
        """
        # Pre-check: Validate image looks like a grayscale X-ray
        is_xray_like, xray_details = self._is_xray_like(image)
        if not is_xray_like:
            # Return early with invalid result
            mask_pil = Image.new('L', image.size, 0)
            return {
                'mask': mask_pil,
                'lung_area_ratio': 0.0,
                'lung_area_percent': 0.0,
                'mean_confidence': 0.0,
                'is_valid_lung': False,
                'validation_details': {
                    'pre_check_failed': True,
                    'reason': xray_details.get('reason', 'Not an X-ray image'),
                    **xray_details
                },
                'masked_image': image.convert('RGB') if image.mode != 'RGB' else image
            }
        
        # Store original for masked image creation
        if image.mode != 'RGB':
            original_rgb = image.convert('RGB')
        else:
            original_rgb = image
        
        original_size = image.size  # (width, height)
        
        # Preprocess (converts to grayscale)
        img_tensor = self._transform(image).unsqueeze(0).to(settings.DEVICE)
        
        # Inference
        with torch.no_grad():
            logits = self._model(img_tensor)
            mask_tensor = torch.sigmoid(logits)  # Apply sigmoid to get probabilities
        
        # Convert mask to numpy (probability map)
        prob_map = mask_tensor.squeeze().cpu().numpy()
        
        # Threshold to binary mask
        binary_mask = (prob_map > 0.5).astype(np.uint8)
        
        # Calculate lung area ratio
        lung_area_ratio = np.sum(binary_mask) / binary_mask.size
        
        # Calculate mean confidence in detected lung regions
        if np.sum(binary_mask) > 0:
            mean_confidence = np.mean(prob_map[binary_mask == 1])
        else:
            mean_confidence = 0.0
        
        # Validate lung structure (not just area)
        is_valid_lung, validation_details = self._validate_lung_structure(
            binary_mask, lung_area_ratio, mean_confidence
        )
        
        # Resize mask back to original image size
        mask_pil = Image.fromarray((binary_mask * 255).astype(np.uint8), mode='L')
        mask_pil = mask_pil.resize(original_size, Image.BILINEAR)
        
        # Create masked image (darken non-lung areas)
        mask_array = np.array(mask_pil) / 255.0
        image_array = np.array(original_rgb)
        
        # Apply mask - keep lung areas bright, darken others
        masked_array = image_array.copy()
        for c in range(3):
            masked_array[:, :, c] = image_array[:, :, c] * (0.3 + 0.7 * mask_array)
        
        masked_image = Image.fromarray(masked_array.astype(np.uint8))
        
        return {
            'mask': mask_pil,
            'lung_area_ratio': float(lung_area_ratio),
            'lung_area_percent': round(lung_area_ratio * 100, 2),
            'mean_confidence': round(float(mean_confidence) * 100, 2),
            'is_valid_lung': is_valid_lung,
            'validation_details': validation_details,
            'masked_image': masked_image
        }
    
    def _validate_lung_structure(self, binary_mask, lung_area_ratio, mean_confidence):
        """
        Validate that the detected mask actually looks like lungs.
        Checks for:
        1. Minimum area threshold (15-40% typical for lungs)
        2. Model confidence in detected regions (>75%)
        3. Two distinct lung regions (left and right)
        4. Regions are in expected positions
        5. Each lung has reasonable size
        6. Mask compactness (real lungs have smooth boundaries)
        """
        from scipy import ndimage
        
        details = {
            'area_check': False,
            'confidence_check': False,
            'region_count_check': False,
            'position_check': False,
            'size_balance_check': False,
            'compactness_check': False,
            'num_regions': 0,
            'mean_confidence': round(mean_confidence * 100, 2)
        }
        
        # Check 1: Minimum area (real lungs typically cover 15-40% of image)
        min_area = settings.MIN_LUNG_AREA_RATIO
        max_area = 0.50  # Lungs shouldn't cover more than 50%
        if lung_area_ratio < min_area or lung_area_ratio > max_area:
            return False, details
        details['area_check'] = True
        
        # Check 2: Model confidence should be VERY high for real X-rays
        # Increased to 92% to reject more false positives
        MIN_CONFIDENCE = 0.92  # 92% average confidence in detected regions
        if mean_confidence < MIN_CONFIDENCE:
            return False, details
        details['confidence_check'] = True
        
        # Check 3: Find connected components (should have ~2 main lung regions)
        labeled_mask, num_features = ndimage.label(binary_mask)
        details['num_regions'] = num_features
        
        # Too many regions = noise, not lungs (lungs have limited distinct regions)
        if num_features == 0 or num_features > 10:
            return False, details
        
        # Get region properties
        region_sizes = ndimage.sum(binary_mask, labeled_mask, range(1, num_features + 1))
        total_lung_area = np.sum(binary_mask)
        
        # Filter significant regions (>15% of total lung area each)
        significant_regions = [s for s in region_sizes if s > 0.15 * total_lung_area]
        details['num_significant_regions'] = len(significant_regions)
        
        # Should have 1-3 significant regions (sometimes lungs connect at bottom)
        if len(significant_regions) < 1 or len(significant_regions) > 3:
            return False, details
        details['region_count_check'] = True
        
        # Check 4: Regions should be roughly in the upper-middle portion of image
        # Find centroids of significant regions
        h, w = binary_mask.shape
        centroids = ndimage.center_of_mass(binary_mask, labeled_mask, range(1, num_features + 1))
        
        valid_positions = 0
        for centroid in centroids:
            if centroid and not np.isnan(centroid[0]):
                cy, cx = centroid
                # Lung centroids should be in roughly the middle portion vertically
                if 0.2 * h < cy < 0.8 * h:  # Not at very top or bottom
                    valid_positions += 1
        
        if valid_positions < 1:
            return False, details
        details['position_check'] = True
        
        # Check 5: If there are 2 regions, they should be roughly balanced in size
        if len(significant_regions) >= 2:
            sorted_regions = sorted(significant_regions, reverse=True)
            size_ratio = sorted_regions[1] / sorted_regions[0] if sorted_regions[0] > 0 else 0
            # The smaller lung should be at least 40% the size of the larger
            if size_ratio >= 0.4:
                details['size_balance_check'] = True
        else:
            # Single connected region is acceptable (lungs may connect)
            details['size_balance_check'] = True
        
        if not details['size_balance_check']:
            return False, details
        
        # Check 6: Compactness check - real lungs have smooth, regular boundaries
        # Calculate perimeter to area ratio (circularity/compactness)
        from scipy.ndimage import binary_erosion
        
        # Get perimeter by eroding and subtracting
        eroded = binary_erosion(binary_mask)
        perimeter = np.sum(binary_mask) - np.sum(eroded)
        area = np.sum(binary_mask)
        
        if area > 0:
            # Compactness = 4 * pi * area / perimeter^2
            # Circle = 1.0, more complex shapes < 1.0
            # Real lungs should have reasonable compactness (not too jagged)
            compactness = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
            details['compactness'] = round(compactness, 3)
            
            # Very low compactness means jagged/noisy boundary (not real lungs)
            # Real lungs typically have compactness > 0.1
            if compactness < 0.05:
                return False, details
            details['compactness_check'] = True
        else:
            return False, details
        
        # All checks passed
        is_valid = all([
            details['area_check'],
            details['confidence_check'],
            details['region_count_check'],
            details['position_check'],
            details['size_balance_check'],
            details['compactness_check']
        ])
        
        return is_valid, details


# Global segmentor instance (lazy loaded)
_segmentor = None

def get_segmentor():
    """Get the singleton segmentor instance."""
    global _segmentor
    if _segmentor is None:
        _segmentor = LungSegmentor()
    return _segmentor


class PneumoniaPredictor:
    """
    Singleton class for pneumonia prediction using ResNet50.
    Loads the model once and reuses for all predictions.
    """
    _instance = None
    _model = None
    _transform = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize the model and transforms."""
        print(f"Loading pneumonia detection model on {settings.DEVICE}...")
        
        # Create model architecture
        self._model = self._create_model()
        
        # Load trained weights
        checkpoint_path = settings.MODEL_CHECKPOINT_PATH
        if checkpoint_path.exists():
            checkpoint = torch.load(
                checkpoint_path, 
                map_location=settings.DEVICE,
                weights_only=True
            )
            if 'model_state_dict' in checkpoint:
                self._model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self._model.load_state_dict(checkpoint)
            print(f"Model loaded from {checkpoint_path}")
        else:
            print(f"WARNING: Checkpoint not found at {checkpoint_path}")
            print("Model will use random weights!")
        
        self._model.to(settings.DEVICE)
        self._model.eval()
        
        # Create inference transform
        self._transform = transforms.Compose([
            transforms.Resize((settings.IMAGE_SIZE, settings.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=settings.IMAGENET_MEAN, 
                std=settings.IMAGENET_STD
            )
        ])
        
        print("Pneumonia detection model ready!")
    
    def _create_model(self):
        """Create ResNet50 model with custom classification head."""
        model = resnet50(weights=None)
        num_features = model.fc.in_features  # 2048
        
        model.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, len(settings.CLASS_NAMES))
        )
        
        return model
    
    def predict(self, image, validate_lung=True):
        """
        Predict pneumonia from a chest X-ray image.
        
        Args:
            image: PIL Image or file path
            validate_lung: If True, first validate that image contains lungs
            
        Returns:
            dict with prediction, confidence, probabilities, and lung validation info
        """
        # Load image if path provided
        if isinstance(image, str):
            image = Image.open(image)
        
        # Convert to RGB if needed (X-rays might be grayscale)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Lung segmentation validation
        segmentation_result = None
        if validate_lung:
            segmentor = get_segmentor()
            segmentation_result = segmentor.segment(image)
            
            if not segmentation_result['is_valid_lung']:
                return {
                    'success': False,
                    'error': 'lung_not_detected',
                    'message': 'No lung tissue detected in the image. Please upload a valid chest X-ray.',
                    'lung_area_percent': segmentation_result['lung_area_percent'],
                    'segmentation': segmentation_result
                }
        
        # Preprocess
        img_tensor = self._transform(image).unsqueeze(0).to(settings.DEVICE)
        
        # Inference
        with torch.no_grad():
            outputs = self._model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
        
        # Get results
        prediction = settings.CLASS_NAMES[pred_idx]
        confidence = probs[0, pred_idx].item()
        
        result = {
            'success': True,
            'prediction': prediction,
            'confidence': confidence,
            'confidence_percent': round(confidence * 100, 2),
            'probabilities': {
                'NORMAL': round(probs[0, 0].item() * 100, 2),
                'PNEUMONIA': round(probs[0, 1].item() * 100, 2)
            },
            'is_pneumonia': pred_idx == 1
        }
        
        if segmentation_result:
            result['lung_validated'] = True
            result['lung_area_percent'] = segmentation_result['lung_area_percent']
            result['segmentation'] = segmentation_result
        else:
            result['lung_validated'] = False
        
        return result


# Global predictor instance (lazy loaded)
_predictor = None

def get_predictor():
    """Get the singleton predictor instance."""
    global _predictor
    if _predictor is None:
        _predictor = PneumoniaPredictor()
    return _predictor

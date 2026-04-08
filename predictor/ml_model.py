"""
ML Model Service for Pneumonia Detection - 3-Stage Pipeline.

Stage 1: Chest X-ray Validation (EfficientNet-B0 classifier)
Stage 2: Lung Segmentation (U-Net with ResNet34 encoder)
Stage 3: Pneumonia Detection (ResNet50 classifier on segmented image)

Singleton pattern to load models once at startup.
"""

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50, efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
import numpy as np
import logging
from django.conf import settings
import segmentation_models_pytorch as smp


logger = logging.getLogger(__name__)


# ============================================
# Stage 1: Chest X-ray Validator (EfficientNet-B0)
# ============================================

class ChestXRayValidatorModel(nn.Module):
    """
    Binary classifier to validate if an image is a chest X-ray.
    Uses EfficientNet-B0 for efficiency.
    """
    
    def __init__(self, pretrained=False):
        super().__init__()
        
        if pretrained:
            self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        else:
            self.backbone = efficientnet_b0(weights=None)
        
        # Replace classifier
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, 1)  # Binary output
        )
    
    def forward(self, x):
        return self.backbone(x)
    
    def predict_proba(self, x):
        """Return probability that image is a chest X-ray."""
        with torch.inference_mode():
            logits = self.forward(x)
            return torch.sigmoid(logits)


class ChestXRayValidator:
    """
    Singleton class for validating if an image is a chest X-ray.
    Uses EfficientNet-B0 binary classifier.
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
        """Initialize the validation model and transforms."""
        logger.info('Loading chest X-ray validator model on %s...', settings.DEVICE)
        
        # Create EfficientNet-B0 model
        self._model = ChestXRayValidatorModel(pretrained=False)
        
        # Load trained weights
        checkpoint_path = settings.XRAY_VALIDATOR_PATH
        if checkpoint_path and checkpoint_path.exists():
            checkpoint = torch.load(
                checkpoint_path,
                map_location=settings.DEVICE,
                weights_only=False  # Our own trusted checkpoint
            )
            if 'model_state_dict' in checkpoint:
                self._model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self._model.load_state_dict(checkpoint)
            logger.info('Validator model loaded from %s', checkpoint_path)
        else:
            logger.warning('Validator checkpoint not found at %s', checkpoint_path)
            logger.warning('X-ray validation will be skipped.')
        
        self._model.to(settings.DEVICE)
        self._model.eval()
        
        # Create inference transform (RGB input, ImageNet normalization)
        self._transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        logger.info('Chest X-ray validator model ready.')
    
    def _basic_image_check(self, image):
        """
        Basic check if image has sufficient contrast (not a solid color).
        Also detects composite images (e.g., X-ray on one side, other content on the other).
        """
        details = {}
        
        if image.mode != 'RGB':
            rgb_img = image.convert('RGB')
        else:
            rgb_img = image
        
        img_array = np.array(rgb_img).astype(np.float32)
        gray = np.mean(img_array, axis=2)
        
        std_dev = np.std(gray)
        details['std_dev'] = round(std_dev, 2)
        
        MIN_STD_DEV = 20.0
        if std_dev < MIN_STD_DEV:
            details['reason'] = f'Image too uniform (std dev: {std_dev:.1f} < {MIN_STD_DEV})'
            return False, details
        
        min_val, max_val = np.min(gray), np.max(gray)
        intensity_range = max_val - min_val
        details['intensity_range'] = round(intensity_range, 2)
        
        MIN_INTENSITY_RANGE = 50.0
        if intensity_range < MIN_INTENSITY_RANGE:
            details['reason'] = f'Insufficient contrast (range: {intensity_range:.1f} < {MIN_INTENSITY_RANGE})'
            return False, details
        
        # Check for composite images (different content in left vs right halves)
        is_composite, composite_details = self._detect_composite_image(gray)
        details.update(composite_details)
        if is_composite:
            details['reason'] = 'Image appears to be a composite (different content in left/right halves)'
            return False, details
        
        return True, details
    
    def _detect_composite_image(self, gray_image):
        """
        Detect if an image is a composite (e.g., X-ray on one side, other image on the other).
        
        Checks for significant differences between left/right and top/bottom halves,
        plus detection of visible dividing lines.
        """
        details = {}
        h, w = gray_image.shape
        
        # ===== Check both horizontal (left/right) and vertical (top/bottom) splits =====
        
        # Horizontal split analysis (left vs right)
        left_half = gray_image[:, :w//2]
        right_half = gray_image[:, w//2:]
        lr_composite, lr_indicators, lr_details = self._analyze_split(left_half, right_half, 'lr')
        details.update(lr_details)
        
        # Vertical split analysis (top vs bottom)
        top_half = gray_image[:h//2, :]
        bottom_half = gray_image[h//2:, :]
        tb_composite, tb_indicators, tb_details = self._analyze_split(top_half, bottom_half, 'tb')
        details.update(tb_details)
        
        # ===== Check for visible dividing lines =====
        has_vertical_line, vline_details = self._detect_dividing_line(gray_image, 'vertical')
        has_horizontal_line, hline_details = self._detect_dividing_line(gray_image, 'horizontal')
        details.update(vline_details)
        details.update(hline_details)
        
        # ===== Final composite decision =====
        # Flag as composite if any split shows strong indicators OR dividing line detected
        details['lr_indicators'] = lr_indicators
        details['tb_indicators'] = tb_indicators
        details['has_vertical_line'] = has_vertical_line
        details['has_horizontal_line'] = has_horizontal_line
        
        is_composite = (
            lr_indicators >= 2 or 
            tb_indicators >= 2 or 
            has_vertical_line or 
            has_horizontal_line or
            (lr_indicators >= 1 and has_vertical_line) or
            (tb_indicators >= 1 and has_horizontal_line)
        )
        
        return is_composite, details
    
    def _analyze_split(self, half1, half2, prefix):
        """Analyze difference between two halves of an image."""
        details = {}
        
        # Statistics for each half
        mean1, mean2 = np.mean(half1), np.mean(half2)
        std1, std2 = np.std(half1), np.std(half2)
        
        details[f'{prefix}_mean1'] = round(mean1, 2)
        details[f'{prefix}_mean2'] = round(mean2, 2)
        details[f'{prefix}_std1'] = round(std1, 2)
        details[f'{prefix}_std2'] = round(std2, 2)
        
        # Check 1: Mean intensity difference (lowered threshold)
        mean_diff = abs(mean1 - mean2)
        details[f'{prefix}_mean_diff'] = round(mean_diff, 2)
        MAX_MEAN_DIFF = 40.0  # Lowered from 60
        
        # Check 2: Standard deviation ratio (lowered threshold)
        std_ratio = max(std1, std2) / (min(std1, std2) + 1e-8)
        details[f'{prefix}_std_ratio'] = round(std_ratio, 2)
        MAX_STD_RATIO = 2.0  # Lowered from 3.0
        
        # Check 3: Histogram distribution difference
        hist1, _ = np.histogram(half1.flatten(), bins=32, range=(0, 255))
        hist2, _ = np.histogram(half2.flatten(), bins=32, range=(0, 255))
        hist1 = hist1.astype(np.float32) / (hist1.sum() + 1e-8)
        hist2 = hist2.astype(np.float32) / (hist2.sum() + 1e-8)
        hist_intersection = np.sum(np.minimum(hist1, hist2))
        details[f'{prefix}_hist_intersection'] = round(hist_intersection, 3)
        MIN_HIST_INTERSECTION = 0.5  # Raised from 0.4 (more strict)
        
        # Check 4: Edge density difference
        edges1 = np.abs(np.diff(half1, axis=1)).mean()
        edges2 = np.abs(np.diff(half2, axis=1)).mean()
        edge_ratio = max(edges1, edges2) / (min(edges1, edges2) + 1e-8)
        details[f'{prefix}_edge_ratio'] = round(edge_ratio, 2)
        MAX_EDGE_RATIO = 3.0  # Lowered from 4.0
        
        # Count indicators
        indicators = 0
        if mean_diff > MAX_MEAN_DIFF:
            indicators += 1
        if std_ratio > MAX_STD_RATIO:
            indicators += 1
        if hist_intersection < MIN_HIST_INTERSECTION:
            indicators += 1
        if edge_ratio > MAX_EDGE_RATIO:
            indicators += 1
        
        is_composite = indicators >= 2
        return is_composite, indicators, details
    
    def _detect_dividing_line(self, gray_image, direction):
        """
        Detect if there's a visible dividing line (often black or white) in the image.
        Composite images often have a clear seam.
        """
        details = {}
        h, w = gray_image.shape
        
        if direction == 'vertical':
            # Check middle vertical strip for uniform values (dividing line)
            center_strip = gray_image[:, w//2-5:w//2+5]
            strip_std = np.std(center_strip, axis=1).mean()  # Variation along the line
            strip_mean = np.mean(center_strip)
            
            # A dividing line would have low std (uniform) and be very dark or light
            details['vertical_strip_std'] = round(strip_std, 2)
            details['vertical_strip_mean'] = round(strip_mean, 2)
            
            # Check if strip is significantly different from surroundings
            left_of_strip = gray_image[:, w//2-20:w//2-5].mean()
            right_of_strip = gray_image[:, w//2+5:w//2+20].mean()
            strip_contrast = abs(strip_mean - (left_of_strip + right_of_strip) / 2)
            details['vertical_strip_contrast'] = round(strip_contrast, 2)
            
            # Dividing line: low variation, high contrast with surroundings
            has_line = strip_std < 15 and strip_contrast > 30
            
        else:  # horizontal
            center_strip = gray_image[h//2-5:h//2+5, :]
            strip_std = np.std(center_strip, axis=0).mean()
            strip_mean = np.mean(center_strip)
            
            details['horizontal_strip_std'] = round(strip_std, 2)
            details['horizontal_strip_mean'] = round(strip_mean, 2)
            
            top_of_strip = gray_image[h//2-20:h//2-5, :].mean()
            bottom_of_strip = gray_image[h//2+5:h//2+20, :].mean()
            strip_contrast = abs(strip_mean - (top_of_strip + bottom_of_strip) / 2)
            details['horizontal_strip_contrast'] = round(strip_contrast, 2)
            
            has_line = strip_std < 15 and strip_contrast > 30
        
        return has_line, details
    
    def validate(self, image):
        """
        Validate if an image is a chest X-ray.
        
        Returns:
            dict with is_valid_xray, confidence, validation_details
        """
        is_quality_ok, quality_details = self._basic_image_check(image)
        if not is_quality_ok:
            return {
                'is_valid_xray': False,
                'confidence': 0.0,
                'confidence_percent': 0.0,
                'validation_details': {
                    'pre_check_failed': True,
                    'reason': quality_details.get('reason', 'Image quality check failed'),
                    **quality_details
                }
            }
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        img_tensor = self._transform(image).unsqueeze(0).to(settings.DEVICE)
        
        with torch.inference_mode():
            prob = self._model.predict_proba(img_tensor)
            confidence = prob.squeeze().item()
        
        xray_threshold = getattr(settings, 'XRAY_VALIDATION_THRESHOLD', 0.5)
        is_valid_xray = confidence >= xray_threshold
        
        return {
            'is_valid_xray': is_valid_xray,
            'confidence': float(confidence),
            'confidence_percent': round(float(confidence) * 100, 2),
            'validation_details': {
                'pre_check_failed': False,
                'model_confidence': round(float(confidence) * 100, 2),
                'threshold': xray_threshold,
                'threshold_percent': round(float(xray_threshold) * 100, 2),
                **quality_details
            }
        }


# Global validator instance (lazy loaded)
_validator = None

def get_validator():
    """Get the singleton validator instance."""
    global _validator
    if _validator is None:
        _validator = ChestXRayValidator()
    return _validator


# ============================================
# Stage 2: Lung Segmentation (U-Net with ResNet34)
# ============================================

class LungSegmentor:
    """
    Singleton class for lung segmentation using U-Net with ResNet34 encoder.
    Segments the lung area from a chest X-ray.
    Includes validation to reject non-anatomical segmentations.
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
        logger.info('Loading lung segmentation model on %s...', settings.DEVICE)
        
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
                weights_only=False  # Our own trusted checkpoint
            )
            if 'model_state_dict' in checkpoint:
                self._model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self._model.load_state_dict(checkpoint)
            logger.info('Segmentation model loaded from %s', checkpoint_path)
        else:
            logger.warning('Segmentation checkpoint not found at %s', checkpoint_path)
            logger.warning('Lung segmentation will produce empty masks.')
        
        self._model.to(settings.DEVICE)
        self._model.eval()
        
        # Create inference transform (grayscale input)
        self._transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((settings.SEGMENTATION_IMAGE_SIZE, settings.SEGMENTATION_IMAGE_SIZE)),
            transforms.ToTensor(),
        ])
        
        logger.info('Lung segmentation model ready.')
    
    def _validate_lung_anatomy(self, binary_mask, prob_map):
        """
        Validate that the segmented mask looks like actual lung anatomy.
        
        Checks for:
        1. Bilateral presence (lungs should be on both sides)
        2. Connected component analysis (should be 1-2 major regions)
        3. Central position (lungs should be centered, not at edges)
        4. Shape aspect ratio (lungs are taller than wide)
        
        Returns:
            (is_valid, details_dict)
        """
        from scipy import ndimage
        
        details = {}
        h, w = binary_mask.shape
        
        # Check 1: Bilateral presence - lungs should be on both sides of image
        left_half_mask = binary_mask[:, :w//2]
        right_half_mask = binary_mask[:, w//2:]
        
        left_area = np.sum(left_half_mask)
        right_area = np.sum(right_half_mask)
        total_area = left_area + right_area
        
        if total_area == 0:
            details['reason'] = 'No segmentation detected'
            return False, details
        
        left_ratio = left_area / total_area
        right_ratio = right_area / total_area
        details['left_right_ratio'] = round(min(left_ratio, right_ratio) / (max(left_ratio, right_ratio) + 1e-8), 3)
        
        # Lungs should have at least some content on both sides (allow for asymmetry)
        MIN_SIDE_RATIO = 0.15  # At least 15% of total on each side
        if left_ratio < MIN_SIDE_RATIO or right_ratio < MIN_SIDE_RATIO:
            details['reason'] = f'Segmentation not bilateral (left: {left_ratio:.2f}, right: {right_ratio:.2f})'
            details['bilateral_check'] = False
        else:
            details['bilateral_check'] = True
        
        # Check 2: Connected components - should be 1-2 major regions (lungs)
        labeled_mask, num_components = ndimage.label(binary_mask)
        details['num_components'] = num_components
        
        if num_components == 0:
            details['reason'] = 'No connected regions found'
            return False, details
        
        # Get sizes of all components
        component_sizes = ndimage.sum(binary_mask, labeled_mask, range(1, num_components + 1))
        if isinstance(component_sizes, (int, float)):
            component_sizes = [component_sizes]
        else:
            component_sizes = list(component_sizes)
        
        # Sort by size (descending)
        component_sizes.sort(reverse=True)
        
        # Count significant components (>5% of total area)
        significant_components = sum(1 for size in component_sizes if size > total_area * 0.05)
        details['significant_components'] = significant_components
        
        # Real lungs should have 1-2 significant components, maybe 3 if middle parts
        # Many small scattered components suggest noise/non-lung segmentation
        if significant_components > 4:
            details['reason'] = f'Too many scattered regions ({significant_components}), not lung-like'
            details['component_check'] = False
        else:
            details['component_check'] = True
        
        # Check 3: Central position - lungs should not be at extreme edges
        rows_with_content = np.any(binary_mask, axis=1)
        cols_with_content = np.any(binary_mask, axis=0)
        
        if np.sum(rows_with_content) > 0 and np.sum(cols_with_content) > 0:
            row_indices = np.where(rows_with_content)[0]
            col_indices = np.where(cols_with_content)[0]
            
            # Check vertical position (lungs should be in upper-middle region typically)
            top_margin = row_indices[0] / h
            bottom_margin = 1 - (row_indices[-1] / h)
            details['top_margin'] = round(top_margin, 3)
            details['bottom_margin'] = round(bottom_margin, 3)
            
            # Check horizontal margins (should have some margin on sides)
            left_margin = col_indices[0] / w
            right_margin = 1 - (col_indices[-1] / w)
            details['left_margin'] = round(left_margin, 3)
            details['right_margin'] = round(right_margin, 3)
            
            # Lungs extend from edge to edge horizontally is suspicious
            if left_margin < 0.02 and right_margin < 0.02:
                details['position_check'] = False
                details['reason'] = 'Segmentation spans entire width - not lung-like'
            else:
                details['position_check'] = True
        else:
            details['position_check'] = False
        
        # Check 4: Confidence consistency - real lungs have high confidence in center
        if np.sum(binary_mask) > 0:
            center_region = binary_mask[h//4:3*h//4, w//4:3*w//4]
            center_prob = prob_map[h//4:3*h//4, w//4:3*w//4]
            
            if np.sum(center_region) > 0:
                center_confidence = np.mean(center_prob[center_region == 1])
                details['center_confidence'] = round(center_confidence, 3)
                
                # Low center confidence suggests uncertain/wrong segmentation
                if center_confidence < 0.5:
                    details['confidence_check'] = False
                else:
                    details['confidence_check'] = True
            else:
                details['center_confidence'] = 0
                details['confidence_check'] = False
        else:
            details['confidence_check'] = False
        
        # Overall validation: require multiple checks to pass
        checks = [
            details.get('bilateral_check', False),
            details.get('component_check', True),  # Default to True if not set
            details.get('position_check', True),
            details.get('confidence_check', True)
        ]
        
        passed_checks = sum(checks)
        details['passed_checks'] = f'{passed_checks}/4'
        
        # Make strictness tunable per deployment while keeping a safe range.
        min_checks_required = max(1, min(4, int(getattr(settings, 'MIN_SEGMENTATION_CHECKS', 3))))
        details['min_checks_required'] = min_checks_required
        is_valid_anatomy = passed_checks >= min_checks_required
        
        if not is_valid_anatomy and 'reason' not in details:
            details['reason'] = f'Segmentation failed anatomical validation ({passed_checks}/4 checks passed)'
        
        return is_valid_anatomy, details
    
    def segment(self, image):
        """
        Segment lungs from a chest X-ray image.
        
        Args:
            image: PIL Image
            
        Returns:
            dict with mask, lung_area_ratio, masked_image, etc.
        """
        # Store original for masked image creation
        if image.mode != 'RGB':
            original_rgb = image.convert('RGB')
        else:
            original_rgb = image
        
        original_size = image.size  # (width, height)
        
        # Preprocess (converts to grayscale)
        img_tensor = self._transform(image).unsqueeze(0).to(settings.DEVICE)
        
        # Inference
        with torch.inference_mode():
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
        
        # Check if valid lung detected (minimum area)
        min_area_ratio = getattr(settings, 'MIN_LUNG_AREA_RATIO', 0.10)
        area_valid = lung_area_ratio >= min_area_ratio
        
        # Validate lung anatomy (bilateral presence, connected components, etc.)
        anatomy_valid, anatomy_details = self._validate_lung_anatomy(binary_mask, prob_map)
        
        # Both area and anatomy checks must pass
        is_valid_lung = area_valid and anatomy_valid
        
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
        
        # Create segmented image (only lung regions, black background)
        segmented_array = np.zeros_like(image_array)
        for c in range(3):
            segmented_array[:, :, c] = image_array[:, :, c] * mask_array
        
        segmented_image = Image.fromarray(segmented_array.astype(np.uint8))
        
        # Determine rejection reason
        rejection_reason = None
        if not area_valid:
            rejection_reason = f'Insufficient lung area ({round(lung_area_ratio*100, 1)}% < {min_area_ratio*100}%)'
        elif not anatomy_valid:
            rejection_reason = anatomy_details.get('reason', 'Segmentation does not resemble lung anatomy')
        
        return {
            'mask': mask_pil,
            'lung_area_ratio': float(lung_area_ratio),
            'lung_area_percent': round(lung_area_ratio * 100, 2),
            'mean_confidence': round(float(mean_confidence) * 100, 2),
            'is_valid_lung': is_valid_lung,
            'rejection_reason': rejection_reason,
            'validation_details': {
                'lung_area_ratio': round(lung_area_ratio, 4),
                'min_required_ratio': min_area_ratio,
                'area_valid': area_valid,
                'anatomy_valid': anatomy_valid,
                'mean_confidence': round(float(mean_confidence), 4),
                **anatomy_details
            },
            'masked_image': masked_image,
            'segmented_image': segmented_image  # For pneumonia detection
        }


# Global segmentor instance (lazy loaded)
_segmentor = None

def get_segmentor():
    """Get the singleton segmentor instance."""
    global _segmentor
    if _segmentor is None:
        _segmentor = LungSegmentor()
    return _segmentor


# ============================================
# Stage 3: Pneumonia Detection (ResNet50)
# ============================================

class PneumoniaPredictor:
    """
    Singleton class for pneumonia prediction using ResNet50.
    Loads the model once and reuses for all predictions.
    Includes Grad-CAM visualization support.
    """
    _instance = None
    _model = None
    _transform = None
    _grad_cam = None
    _grad_cam_pp = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize the model, transforms, and Grad-CAM."""
        logger.info('Loading pneumonia detection model on %s...', settings.DEVICE)
        
        # Create model architecture
        self._model = self._create_model()
        
        # Load trained weights
        checkpoint_path = settings.MODEL_CHECKPOINT_PATH
        if checkpoint_path and checkpoint_path.exists():
            checkpoint = torch.load(
                checkpoint_path, 
                map_location=settings.DEVICE,
                weights_only=False  # Our own trusted checkpoint
            )
            if 'model_state_dict' in checkpoint:
                self._model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self._model.load_state_dict(checkpoint)
            logger.info('Pneumonia model loaded from %s', checkpoint_path)
        else:
            logger.warning('Pneumonia model checkpoint not found at %s', checkpoint_path)
            logger.warning('Model will use random weights.')
        
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
        
        # Initialize Grad-CAM on layer4 (last conv block of ResNet50)
        from .grad_cam import GradCAM, GradCAMPlusPlus
        target_layer = self._model.layer4[-1]  # Last bottleneck block
        self._grad_cam = GradCAM(self._model, target_layer)
        self._grad_cam_pp = GradCAMPlusPlus(self._model, target_layer)
        
        logger.info('Pneumonia detection model ready with Grad-CAM support.')
    
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

    def _predict_probabilities(self, image, use_tta=False):
        """Run classifier and return class probabilities as a CPU tensor."""
        # Training used grayscale chest X-rays, so normalize all inputs to grayscale->RGB.
        base_image = image.convert('L').convert('RGB')
        candidate_images = [base_image]

        if use_tta:
            candidate_images.append(base_image.transpose(Image.FLIP_LEFT_RIGHT))

        probs_list = []
        with torch.inference_mode():
            for candidate in candidate_images:
                img_tensor = self._transform(candidate).unsqueeze(0).to(settings.DEVICE)
                outputs = self._model(img_tensor)
                probs = torch.softmax(outputs, dim=1).squeeze(0).cpu()
                probs_list.append(probs)

        return torch.stack(probs_list, dim=0).mean(dim=0)
    
    def predict(self, image, validate_xray=True, segment_lungs=True, generate_gradcam=True, gradcam_type='gradcam'):
        """
        Predict pneumonia from a chest X-ray image using 3-stage pipeline.
        
        Stage 1: Validate image is a chest X-ray
        Stage 2: Segment lungs from the image
        Stage 3: Detect pneumonia from segmented image (with optional Grad-CAM)
        
        Args:
            image: PIL Image or file path
            validate_xray: If True, first validate that image is a chest X-ray
            segment_lungs: If True, segment lungs before detection
            generate_gradcam: If True, generate Grad-CAM visualization
            gradcam_type: 'gradcam' or 'gradcam++' 
            
        Returns:
            dict with prediction, confidence, probabilities, and pipeline info
        """
        # Load image if path provided
        if isinstance(image, str):
            image = Image.open(image)
        
        # Convert to RGB if needed (X-rays might be grayscale)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Stage 1: X-ray validation
        validation_result = None
        if validate_xray:
            validator = get_validator()
            validation_result = validator.validate(image)
            
            if not validation_result['is_valid_xray']:
                return {
                    'success': False,
                    'error': 'not_xray',
                    'message': 'This does not appear to be a chest X-ray image. Please upload a valid chest X-ray.',
                    'validation': validation_result,
                    'segmentation': None
                }
        
        # Stage 2: Lung segmentation
        segmentation_result = None
        image_for_detection = image  # Default to original image
        
        if segment_lungs:
            segmentor = get_segmentor()
            segmentation_result = segmentor.segment(image)
            
            if not segmentation_result['is_valid_lung']:
                # Use specific rejection reason if available
                rejection_reason = segmentation_result.get('rejection_reason')
                if rejection_reason:
                    message = f'Image validation failed: {rejection_reason}'
                else:
                    message = 'No valid lung tissue detected in the image. Please upload a valid chest X-ray.'
                
                return {
                    'success': False,
                    'error': 'lung_not_detected',
                    'message': message,
                    'lung_area_percent': segmentation_result['lung_area_percent'],
                    'validation': validation_result,
                    'segmentation': segmentation_result
                }
            
            # Use segmented image for pneumonia detection
            image_for_detection = segmentation_result['segmented_image']
        
        # Stage 3: Pneumonia detection on segmented image
        probs = self._predict_probabilities(
            image_for_detection,
            use_tta=getattr(settings, 'USE_TTA', False)
        )
        
        # Apply threshold for classification
        pneumonia_prob = probs[1].item()
        threshold = getattr(settings, 'PNEUMONIA_THRESHOLD', 0.5)
        
        if pneumonia_prob >= threshold:
            pred_idx = 1  # PNEUMONIA
        else:
            pred_idx = 0  # NORMAL
        
        # Get results
        prediction = settings.CLASS_NAMES[pred_idx]
        confidence = probs[pred_idx].item()
        
        result = {
            'success': True,
            'prediction': prediction,
            'confidence': confidence,
            'confidence_percent': round(confidence * 100, 2),
            'probabilities': {
                'NORMAL': round(probs[0].item() * 100, 2),
                'PNEUMONIA': round(probs[1].item() * 100, 2)
            },
            'is_pneumonia': pred_idx == 1,
            'threshold_used': threshold
        }
        
        # Generate Grad-CAM visualization
        if generate_gradcam:
            # Use ORIGINAL image for Grad-CAM, not segmented image
            # The segmented image has black background which causes the model
            # to focus on edges/boundaries rather than actual lung content.
            # We then mask the heatmap to only show within the lung region.
            lung_mask = segmentation_result['mask'] if segmentation_result else None
            gradcam_result = self.generate_gradcam(
                image,  # Use original image, not image_for_detection
                gradcam_type=gradcam_type,
                target_class=1,  # Always show pneumonia attention
                lung_mask=lung_mask  # Mask heatmap to lung region
            )
            result['gradcam'] = gradcam_result
        
        # Add validation info
        if validation_result:
            result['xray_validated'] = True
            result['xray_confidence'] = validation_result['confidence_percent']
            result['validation'] = validation_result
        else:
            result['xray_validated'] = False
        
        # Add segmentation info
        if segmentation_result:
            result['lung_validated'] = True
            result['lung_area_percent'] = segmentation_result['lung_area_percent']
            result['segmentation'] = segmentation_result
        else:
            result['lung_validated'] = False
        
        return result
    
    def generate_gradcam(self, image, gradcam_type='gradcam', target_class=1, lung_mask=None):
        """
        Generate Grad-CAM visualization for an image.
        
        Args:
            image: PIL Image
            gradcam_type: 'gradcam' or 'gradcam++'
            target_class: Class to visualize (0=NORMAL, 1=PNEUMONIA)
            lung_mask: Optional PIL Image mask to constrain heatmap to lung region
            
        Returns:
            dict with overlay image, heatmap, and base64 encoded versions
        """
        from .grad_cam import GradCAM
        
        # Convert color images to grayscale first, then to RGB
        # This ensures consistent processing for color-tinted X-rays (like green ones)
        # since the model was trained on grayscale X-rays
        grayscale_image = image.convert('L').convert('RGB')
        
        # Store original for overlay (keep as grayscale RGB for cleaner visualization)
        original_image = grayscale_image.copy()
        
        # Preprocess for model
        img_tensor = self._transform(grayscale_image).unsqueeze(0).to(settings.DEVICE)
        
        # Select Grad-CAM variant
        if gradcam_type == 'gradcam++':
            cam_generator = self._grad_cam_pp
        else:
            cam_generator = self._grad_cam
        
        # Generate visualization with lung mask
        visualization = cam_generator.generate_visualization(
            original_image,
            img_tensor,
            target_class=target_class,
            lung_mask=lung_mask
        )
        
        return {
            'heatmap': visualization['cam'],
            'overlay_image': visualization['overlay'],
            'overlay_base64': GradCAM.pil_to_base64(visualization['overlay']),
            'grayscale_image': grayscale_image,  # Include grayscale version
            'target_class': 'PNEUMONIA' if target_class == 1 else 'NORMAL'
        }


# Global predictor instance (lazy loaded)
_predictor = None

def get_predictor():
    """Get the singleton predictor instance."""
    global _predictor
    if _predictor is None:
        _predictor = PneumoniaPredictor()
    return _predictor

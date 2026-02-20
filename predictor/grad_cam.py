"""
Grad-CAM implementation for pneumonia detection visualization.

Grad-CAM (Gradient-weighted Class Activation Mapping) highlights which regions
of the chest X-ray the model focuses on when making predictions.

Works with the ResNet50 pneumonia classifier.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import io
import base64


class GradCAM:
    """
    Grad-CAM implementation for ResNet50 pneumonia classifier.
    Highlights regions the model focuses on for pneumonia detection.
    """
    
    def __init__(self, model, target_layer=None):
        """
        Initialize Grad-CAM.
        
        Args:
            model: PyTorch CNN model (ResNet50)
            target_layer: Layer to compute Grad-CAM for (default: layer4[-1])
        """
        self.model = model
        self.model.eval()
        self.device = next(model.parameters()).device
        
        # For ResNet50, use layer4[-1] (last bottleneck block)
        if target_layer is None:
            self.target_layer = self.model.layer4[-1]
        else:
            self.target_layer = target_layer
            
        self.gradients = None
        self.activations = None
        self._hooks = []
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks on target layer."""
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self._hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self._hooks.append(self.target_layer.register_full_backward_hook(backward_hook))
    
    def remove_hooks(self):
        """Remove registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
    
    def generate_cam(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap.
        
        Args:
            input_tensor: Preprocessed input image tensor [1, C, H, W]
            target_class: Class to generate CAM for (0=NORMAL, 1=PNEUMONIA, None=predicted)
            
        Returns:
            cam: Grad-CAM heatmap as numpy array [H, W]
            prediction: Model prediction (0 or 1)
            confidence: Prediction confidence
            probabilities: Dict with class probabilities
        """
        self.model.eval()
        
        # Create a copy that requires grad
        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        input_tensor = input_tensor.to(self.device)
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Get prediction (2-class softmax)
        prob = F.softmax(output, dim=1)
        prediction = prob.argmax(dim=1).item()
        confidence = prob[0, prediction].item()
        probabilities = {
            'NORMAL': prob[0, 0].item(),
            'PNEUMONIA': prob[0, 1].item()
        }
        
        # Use PNEUMONIA class (1) by default for visualization
        if target_class is None:
            target_class = 1  # Always show pneumonia attention
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for target class
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot)
        
        # Get gradients and activations
        gradients = self.gradients  # [1, C, H, W]
        activations = self.activations  # [1, C, H, W]
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1, keepdim=True)  # [1, 1, H, W]
        
        # ReLU to keep only positive contributions
        cam = F.relu(cam)
        
        # Normalize
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        return cam, prediction, confidence, probabilities
    
    def generate_heatmap_overlay(self, original_image, cam, alpha=0.5, colormap=cv2.COLORMAP_JET, lung_mask=None):
        """
        Overlay Grad-CAM heatmap on original image.
        
        Args:
            original_image: Original image as numpy array or PIL Image
            cam: Grad-CAM heatmap [H, W]
            alpha: Transparency of heatmap overlay
            colormap: OpenCV colormap for heatmap
            lung_mask: Optional PIL Image mask to constrain heatmap to lung region
            
        Returns:
            overlay: Image with heatmap overlay as PIL Image
        """
        # Convert PIL to numpy if needed
        if isinstance(original_image, Image.Image):
            original_image = np.array(original_image)
        
        # Ensure image is RGB
        if len(original_image.shape) == 2:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
        elif original_image.shape[2] == 4:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_RGBA2RGB)
        
        # Resize CAM to match image size
        cam_resized = cv2.resize(cam, (original_image.shape[1], original_image.shape[0]))
        
        # Apply lung mask to constrain heatmap to lung region
        if lung_mask is not None:
            # Convert PIL mask to numpy
            if isinstance(lung_mask, Image.Image):
                mask_array = np.array(lung_mask.resize((original_image.shape[1], original_image.shape[0])))
            else:
                mask_array = cv2.resize(lung_mask, (original_image.shape[1], original_image.shape[0]))
            
            # Normalize mask to 0-1 range
            if mask_array.max() > 1:
                mask_array = mask_array / 255.0
            
            # Apply mask to CAM - zero out areas outside lungs
            cam_resized = cam_resized * mask_array
            
            # RE-NORMALIZE after masking to ensure visibility within lung region
            # This is critical when the model's attention was outside the lung area
            if cam_resized.max() > 0:
                cam_resized = (cam_resized - cam_resized.min()) / (cam_resized.max() - cam_resized.min())
        
        # Convert CAM to heatmap
        heatmap = np.uint8(255 * cam_resized)
        heatmap = cv2.applyColorMap(heatmap, colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # For masked heatmap, only blend where there's actual heatmap intensity
        if lung_mask is not None:
            # Create blend mask based on heatmap intensity
            blend_mask = cam_resized[:, :, np.newaxis]  # Add channel dimension
            overlay = np.uint8(original_image * (1 - alpha * blend_mask) + heatmap * (alpha * blend_mask))
        else:
            # Standard overlay
            overlay = np.uint8(original_image * (1 - alpha) + heatmap * alpha)
        
        return Image.fromarray(overlay)
    
    def generate_visualization(self, original_image, input_tensor, target_class=None, lung_mask=None):
        """
        Generate complete Grad-CAM visualization.
        
        Args:
            original_image: Original image (PIL or numpy)
            input_tensor: Preprocessed input tensor
            target_class: Target class for CAM (0=NORMAL, 1=PNEUMONIA)
            lung_mask: Optional PIL Image mask to constrain heatmap to lung region
            
        Returns:
            dict with cam, overlay, prediction, confidence
        """
        # Generate CAM
        cam, prediction, confidence, probabilities = self.generate_cam(input_tensor, target_class)
        
        # Generate overlay with lung mask
        overlay = self.generate_heatmap_overlay(original_image, cam, lung_mask=lung_mask)
        
        return {
            'cam': cam,
            'overlay': overlay,
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': probabilities,
            'class_name': 'PNEUMONIA' if prediction == 1 else 'NORMAL'
        }
    
    @staticmethod
    def pil_to_base64(image):
        """Convert PIL Image to base64 string for web display."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))
        
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)
        
        return base64.b64encode(buffer.getvalue()).decode('utf-8')


class GradCAMPlusPlus(GradCAM):
    """
    Grad-CAM++ for better localization with multiple regions.
    Improves on standard Grad-CAM for images with multiple instances of the target class.
    """
    
    def generate_cam(self, input_tensor, target_class=None):
        """Generate Grad-CAM++ heatmap."""
        self.model.eval()
        
        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        input_tensor = input_tensor.to(self.device)
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Get prediction (2-class softmax)
        prob = F.softmax(output, dim=1)
        prediction = prob.argmax(dim=1).item()
        confidence = prob[0, prediction].item()
        probabilities = {
            'NORMAL': prob[0, 0].item(),
            'PNEUMONIA': prob[0, 1].item()
        }
        
        # Use PNEUMONIA class by default
        if target_class is None:
            target_class = 1
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot)
        
        gradients = self.gradients
        activations = self.activations
        
        # Grad-CAM++ weights calculation
        grad_2 = gradients ** 2
        grad_3 = gradients ** 3
        
        sum_activations = torch.sum(activations, dim=(2, 3), keepdim=True)
        
        alpha_num = grad_2
        alpha_denom = 2 * grad_2 + sum_activations * grad_3 + 1e-8
        alpha = alpha_num / alpha_denom
        
        weights = torch.sum(alpha * F.relu(gradients), dim=(2, 3), keepdim=True)
        
        # Weighted combination
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Normalize
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        return cam, prediction, confidence, probabilities

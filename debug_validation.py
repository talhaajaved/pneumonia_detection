"""Debug: Check what confidence values the model outputs for various images"""
import os
os.environ['DJANGO_SETTINGS_MODULE'] = 'pneumonia_django.settings'

import django
django.setup()

from predictor.ml_model import get_segmentor
from PIL import Image
import numpy as np

segmentor = get_segmentor()

def test_image(name, img):
    result = segmentor.segment(img)
    print(f"\n{'='*50}")
    print(f"Image: {name}")
    print(f"  Lung area: {result['lung_area_percent']:.2f}%")
    print(f"  Mean confidence: {result['mean_confidence']:.2f}%")
    print(f"  Is valid lung: {result['is_valid_lung']}")
    print(f"  Validation details: {result['validation_details']}")
    return result

# Test 1: Pure white
img = Image.new('RGB', (512, 512), color='white')
test_image("Pure white", img)

# Test 2: Pure black
img = Image.new('RGB', (512, 512), color='black')
test_image("Pure black", img)

# Test 3: Random noise (uniform)
noise = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
test_image("Random noise", Image.fromarray(noise))

# Test 4: Gray gradient (simulates some soft contrast)
gradient = np.zeros((512, 512, 3), dtype=np.uint8)
for i in range(512):
    gradient[i, :, :] = int(i / 512 * 255)
test_image("Gray gradient", Image.fromarray(gradient))

# Test 5: Two dark blobs (fake "lungs")
img = np.ones((512, 512, 3), dtype=np.uint8) * 200  # Light gray background
# Add two dark oval shapes
for y in range(150, 350):
    for x in range(80, 200):
        if ((x - 140)**2 / 60**2 + (y - 250)**2 / 100**2) < 1:
            img[y, x] = [50, 50, 50]  # Dark
for y in range(150, 350):
    for x in range(300, 420):
        if ((x - 360)**2 / 60**2 + (y - 250)**2 / 100**2) < 1:
            img[y, x] = [50, 50, 50]  # Dark
test_image("Two dark blobs (fake lungs)", Image.fromarray(img))

# Test 6: Natural photo pattern (simulate)
img = np.zeros((512, 512, 3), dtype=np.uint8)
img[:, :, 0] = np.clip(np.random.normal(150, 30, (512, 512)), 0, 255).astype(np.uint8)  # Red
img[:, :, 1] = np.clip(np.random.normal(100, 20, (512, 512)), 0, 255).astype(np.uint8)  # Green
img[:, :, 2] = np.clip(np.random.normal(50, 15, (512, 512)), 0, 255).astype(np.uint8)   # Blue
test_image("Random colored (photo-like)", Image.fromarray(img))

# Test 7: Very low contrast dark image
img = np.random.randint(10, 50, (512, 512, 3), dtype=np.uint8)
test_image("Low contrast dark", Image.fromarray(img))

print("\n" + "="*50)
print("THRESHOLD CHECK: Images with confidence < 85% should be rejected")
print("="*50)

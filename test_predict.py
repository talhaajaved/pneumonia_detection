"""Test the full predict pipeline with various images"""
import os
os.environ['DJANGO_SETTINGS_MODULE'] = 'pneumonia_django.settings'

import django
django.setup()

from predictor.ml_model import get_predictor
from PIL import Image
import numpy as np

predictor = get_predictor()

def test_image(name, img):
    result = predictor.predict(img, validate_lung=True)
    print(f"\n=== {name} ===")
    print(f"Success: {result.get('success', 'N/A')}")
    if result.get('success'):
        print(f"Prediction: {result.get('prediction')}")
        print(f"Lung validated: {result.get('lung_validated')}")
    else:
        print(f"Error: {result.get('error')}")
        print(f"Message: {result.get('message')}")
    if 'segmentation' in result:
        details = result['segmentation'].get('validation_details', {})
        print(f"Validation details: {details}")

# Test 1: White image
print("\n" + "="*50)
print("Testing predict() with non-X-ray images")
print("="*50)

img = Image.new('RGB', (256, 256), color='white')
test_image("Plain white image", img)

# Test 2: Random noise
noise = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
img = Image.fromarray(noise)
test_image("Random noise", img)

# Test 3: A colored image (like a photo)
gradient = np.zeros((256, 256, 3), dtype=np.uint8)
gradient[:, :, 0] = np.linspace(0, 255, 256).reshape(1, -1)
gradient[:, :, 1] = 128
gradient[:, :, 2] = np.linspace(255, 0, 256).reshape(-1, 1)
img = Image.fromarray(gradient)
test_image("Colored gradient (like a photo)", img)

print("\n" + "="*50)
print("If 'Success: False' for all above, validation works!")
print("="*50)

"""
Django settings for pneumonia_django project.
Pneumonia Detection Web Application using ResNet50.
"""

import os
from pathlib import Path

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# Path to the detection folder (parent of pneumonia_django)
DETECTION_DIR = BASE_DIR.parent

# Security settings from environment variables
SECRET_KEY = os.environ.get('DJANGO_SECRET_KEY', 'django-insecure-pneumonia-detection-change-this-in-production')

DEBUG = os.environ.get('DJANGO_DEBUG', 'True').lower() in ('true', '1', 'yes')

ALLOWED_HOSTS = os.environ.get('DJANGO_ALLOWED_HOSTS', '*').split(',')

# CSRF trusted origins for production
CSRF_TRUSTED_ORIGINS = os.environ.get('CSRF_TRUSTED_ORIGINS', 'http://localhost:8000,http://127.0.0.1:8000').split(',')

# Application definition
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'predictor',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'pneumonia_django.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'pneumonia_django.wsgi.application'

# Database
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# Password validation
AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator'},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]

# Internationalization
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

# Static files (CSS, JavaScript, Images)
STATIC_URL = 'static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'  # For collectstatic in production
STATICFILES_DIRS = [BASE_DIR / 'static'] if (BASE_DIR / 'static').exists() else []

# Media files (Uploaded images)
MEDIA_URL = 'media/'
MEDIA_ROOT = BASE_DIR / 'media'

# Default primary key field type
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# ============================================
# ML Model Configuration
# ============================================
# Check multiple locations for the model checkpoint
_env_model_path = os.environ.get('MODEL_CHECKPOINT_PATH', '')
_checkpoint_locations = [
    Path(_env_model_path) if _env_model_path else None,  # Environment variable
    BASE_DIR / 'best_pneumonia_resnet50.pth',  # Model in project root
    BASE_DIR / 'checkpoints' / 'best_pneumonia_resnet50.pth',  # Docker volume mount
    DETECTION_DIR / 'checkpoints' / 'best_pneumonia_resnet50.pth',  # Local development
    Path('/app/checkpoints/best_pneumonia_resnet50.pth'),  # Docker absolute path
]

MODEL_CHECKPOINT_PATH = None
for path in _checkpoint_locations:
    if path and path.exists():
        MODEL_CHECKPOINT_PATH = path
        break

if MODEL_CHECKPOINT_PATH is None:
    MODEL_CHECKPOINT_PATH = DETECTION_DIR / 'checkpoints' / 'best_pneumonia_resnet50.pth'

# Image preprocessing constants (same as training)
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Class names
CLASS_NAMES = ['NORMAL', 'PNEUMONIA']

# ============================================
# Lung Segmentation Model Configuration
# ============================================
_env_seg_path = os.environ.get('SEGMENTATION_CHECKPOINT_PATH', '')
_seg_checkpoint_locations = [
    Path(_env_seg_path) if _env_seg_path else None,
    BASE_DIR / 'best_lung_segmentation_unet.pth',
    BASE_DIR / 'checkpoints' / 'best_lung_segmentation_unet.pth',
    DETECTION_DIR / 'checkpoints' / 'best_lung_segmentation_unet.pth',
    Path('/app/checkpoints/best_lung_segmentation_unet.pth'),
]

SEGMENTATION_CHECKPOINT_PATH = None
for path in _seg_checkpoint_locations:
    if path and path.exists():
        SEGMENTATION_CHECKPOINT_PATH = path
        break

if SEGMENTATION_CHECKPOINT_PATH is None:
    SEGMENTATION_CHECKPOINT_PATH = BASE_DIR / 'best_lung_segmentation_unet.pth'

# Segmentation settings
SEGMENTATION_IMAGE_SIZE = 256
MIN_LUNG_AREA_RATIO = 0.10  # Minimum 10% of image must be lung

# Device (CPU by default, change to 'cuda' if GPU available)
import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

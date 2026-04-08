"""
Django settings for pneumonia_django project.
Pneumonia Detection Web Application using ResNet50.
"""

import os
import secrets
from pathlib import Path
from django.core.exceptions import ImproperlyConfigured


def _env_bool(name, default=False):
    """Read a boolean environment variable safely."""
    return os.environ.get(name, str(default)).lower() in ('true', '1', 'yes')


def _env_list(name, default=''):
    """Read a comma-separated env var into a stripped list."""
    value = os.environ.get(name, default)
    return [item.strip() for item in value.split(',') if item.strip()]

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# Path to the detection folder (parent of pneumonia_django)
DETECTION_DIR = BASE_DIR.parent

# Security settings from environment variables
DEBUG = _env_bool('DJANGO_DEBUG', False)

_secret_from_env = os.environ.get('DJANGO_SECRET_KEY')
if _secret_from_env:
    SECRET_KEY = _secret_from_env
else:
    # Ephemeral fallback avoids shipping a hardcoded secret in source control.
    SECRET_KEY = secrets.token_urlsafe(64)

ALLOWED_HOSTS = _env_list('DJANGO_ALLOWED_HOSTS', 'localhost,127.0.0.1')
if not DEBUG and ('*' in ALLOWED_HOSTS or not ALLOWED_HOSTS):
    raise ImproperlyConfigured('Set DJANGO_ALLOWED_HOSTS explicitly in production (wildcard is not allowed).')

# CSRF trusted origins for production
CSRF_TRUSTED_ORIGINS = _env_list(
    'CSRF_TRUSTED_ORIGINS',
    'http://localhost:8000,http://127.0.0.1:8000'
)

# Upload limits to protect memory usage on malformed or oversized requests.
FILE_UPLOAD_MAX_MEMORY_SIZE = int(os.environ.get('FILE_UPLOAD_MAX_MEMORY_SIZE', str(10 * 1024 * 1024)))
DATA_UPLOAD_MAX_MEMORY_SIZE = int(os.environ.get('DATA_UPLOAD_MAX_MEMORY_SIZE', str(12 * 1024 * 1024)))

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
    'whitenoise.middleware.WhiteNoiseMiddleware',
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

# WhiteNoise for serving static files in production
STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'

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
    BASE_DIR / 'best_pneumonia_pipeline_v2.pth',  # Model in project root (v2)
    BASE_DIR / 'best_pneumonia_resnet50.pth',  # Legacy model name
    BASE_DIR / 'checkpoints' / 'best_pneumonia_pipeline_v2.pth',  # Docker volume mount
    DETECTION_DIR / 'checkpoints' / 'best_pneumonia_pipeline_v2.pth',  # Local development
    Path('/app/checkpoints/best_pneumonia_pipeline_v2.pth'),  # Docker absolute path
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

# Pneumonia classification threshold
# If pneumonia probability >= threshold, predict PNEUMONIA
PNEUMONIA_THRESHOLD = 0.90

# Runtime toggles
ENABLE_XRAY_VALIDATION = _env_bool('ENABLE_XRAY_VALIDATION', True)
ENABLE_LUNG_SEGMENTATION = _env_bool('ENABLE_LUNG_SEGMENTATION', True)
ENABLE_GRADCAM = _env_bool('ENABLE_GRADCAM', True)
USE_TTA = _env_bool('USE_TTA', True)

# API throttling (simple per-IP fixed window in cache)
API_RATE_LIMIT = int(os.environ.get('API_RATE_LIMIT', '20'))
API_RATE_WINDOW_SECONDS = int(os.environ.get('API_RATE_WINDOW_SECONDS', '60'))

# ============================================
# Chest X-ray Validator Model Configuration (Stage 1)
# ============================================
_env_validator_path = os.environ.get('XRAY_VALIDATOR_PATH', '')
_validator_checkpoint_locations = [
    Path(_env_validator_path) if _env_validator_path else None,
    BASE_DIR / 'best_chest_xray_validator.pth',
    BASE_DIR / 'checkpoints' / 'best_chest_xray_validator.pth',
    DETECTION_DIR / 'checkpoints' / 'best_chest_xray_validator.pth',
    Path('/app/checkpoints/best_chest_xray_validator.pth'),
]

XRAY_VALIDATOR_PATH = None
for path in _validator_checkpoint_locations:
    if path and path.exists():
        XRAY_VALIDATOR_PATH = path
        break

if XRAY_VALIDATOR_PATH is None:
    XRAY_VALIDATOR_PATH = BASE_DIR / 'best_chest_xray_validator.pth'

# X-ray validation threshold (higher = stricter, rejects more non-X-rays)
# 0.7 is recommended to reject composite/dual images more aggressively
XRAY_VALIDATION_THRESHOLD = 0.90

# ============================================
# Lung Segmentation Model Configuration (Stage 2)
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
MIN_SEGMENTATION_CHECKS = int(os.environ.get('MIN_SEGMENTATION_CHECKS', '3'))

# Device (CPU by default, change to 'cuda' if GPU available)
import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# Production security headers and cookie policies
if not DEBUG:
    SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')
    SECURE_SSL_REDIRECT = _env_bool('SECURE_SSL_REDIRECT', False)
    SESSION_COOKIE_SECURE = True
    CSRF_COOKIE_SECURE = True
    SECURE_CONTENT_TYPE_NOSNIFF = True
    X_FRAME_OPTIONS = 'DENY'
    SECURE_REFERRER_POLICY = 'strict-origin-when-cross-origin'

    SECURE_HSTS_SECONDS = int(os.environ.get('SECURE_HSTS_SECONDS', '31536000'))
    SECURE_HSTS_INCLUDE_SUBDOMAINS = _env_bool('SECURE_HSTS_INCLUDE_SUBDOMAINS', True)
    SECURE_HSTS_PRELOAD = _env_bool('SECURE_HSTS_PRELOAD', True)

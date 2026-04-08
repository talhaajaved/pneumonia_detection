from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.core.files.base import ContentFile
from django.core.cache import cache
from PIL import Image, ImageOps, UnidentifiedImageError
import logging
import io
import time
import uuid

from .forms import ImageUploadForm
from .models import PredictionHistory
from .ml_model import get_predictor


logger = logging.getLogger(__name__)
MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024


def _as_bool(raw_value, default=False):
    if raw_value is None:
        return default
    return str(raw_value).strip().lower() in {'1', 'true', 'yes', 'on'}


def _get_client_ip(request):
    forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if forwarded_for:
        return forwarded_for.split(',')[0].strip()
    return request.META.get('REMOTE_ADDR', 'unknown')


def _check_rate_limit(request, scope='api_predict'):
    """Simple fixed-window per-IP limiter backed by Django cache."""
    limit = max(1, int(getattr(settings, 'API_RATE_LIMIT', 20)))
    window_seconds = max(1, int(getattr(settings, 'API_RATE_WINDOW_SECONDS', 60)))
    client_ip = _get_client_ip(request)

    cache_key = f'rate-limit:{scope}:{client_ip}'
    now = int(time.time())
    bucket = cache.get(cache_key) or {'start': now, 'count': 0}

    elapsed = now - int(bucket.get('start', now))
    if elapsed >= window_seconds:
        bucket = {'start': now, 'count': 0}
        elapsed = 0

    bucket['count'] = int(bucket.get('count', 0)) + 1
    cache.set(cache_key, bucket, timeout=window_seconds)

    if bucket['count'] > limit:
        retry_after = max(1, window_seconds - elapsed)
        response = JsonResponse({
            'error': 'Rate limit exceeded. Please retry later.',
            'retry_after_seconds': retry_after,
        }, status=429)
        response['Retry-After'] = str(retry_after)
        return response

    return None


def home(request):
    """Home page with upload form."""
    form = ImageUploadForm()
    recent_predictions = PredictionHistory.objects.only(
        'id', 'prediction', 'confidence', 'created_at'
    ).order_by('-created_at')[:5]
    
    return render(request, 'predictor/home.html', {
        'form': form,
        'recent_predictions': recent_predictions
    })


def predict_view(request):
    """Handle image upload and prediction."""
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        
        if form.is_valid():
            image_file = form.cleaned_data['image']

            # Normalize orientation and force full decode early to fail fast on bad files.
            try:
                image_file.seek(0)
                with Image.open(image_file) as uploaded_image:
                    image = ImageOps.exif_transpose(uploaded_image).copy()
            except (UnidentifiedImageError, OSError):
                return render(request, 'predictor/home.html', {
                    'form': form,
                    'error': 'The uploaded file is not a readable image.',
                })

            # Get prediction with pipeline options from settings.
            predictor = get_predictor()
            result = predictor.predict(
                image,
                validate_xray=getattr(settings, 'ENABLE_XRAY_VALIDATION', True),
                segment_lungs=getattr(settings, 'ENABLE_LUNG_SEGMENTATION', True),
                generate_gradcam=getattr(settings, 'ENABLE_GRADCAM', True),
            )

            logger.info(
                'Web prediction completed (success=%s, prediction=%s, confidence=%.4f)',
                result.get('success', False),
                result.get('prediction', 'n/a'),
                float(result.get('confidence', 0.0)),
            )

            # Check if lung validation failed
            if not result.get('success', True):
                logger.warning(
                    'Web prediction rejected (error=%s, message=%s)',
                    result.get('error', 'unknown'),
                    result.get('message', 'n/a'),
                )
                return render(request, 'predictor/invalid_image.html', {
                    'result': result,
                    'lung_area_percent': result.get('lung_area_percent', 0)
                })

            # Save to history
            image_file.seek(0)  # Reset file pointer
            probabilities = result.get('probabilities', {})
            prediction_record = PredictionHistory.objects.create(
                image=image_file,
                prediction=result['prediction'],
                confidence=result['confidence'],
                normal_probability=float(probabilities.get('NORMAL', 0.0)) / 100,
                pneumonia_probability=float(probabilities.get('PNEUMONIA', 0.0)) / 100,
                lung_validated=result.get('lung_validated', False),
                lung_area_ratio=result.get('lung_area_percent', 0) / 100
            )

            # Save segmented image if available
            if result.get('segmentation') and result['segmentation'].get('segmented_image'):
                segmented_img = result['segmentation']['segmented_image']
                # Convert PIL Image to bytes
                img_buffer = io.BytesIO()
                segmented_img.save(img_buffer, format='PNG')
                img_buffer.seek(0)
                # Save to model
                filename = f"segmented_{uuid.uuid4().hex[:8]}.png"
                prediction_record.segmented_image.save(filename, ContentFile(img_buffer.read()), save=True)
            
            # Save Grad-CAM image if available
            if result.get('gradcam') and result['gradcam'].get('overlay_image'):
                gradcam_img = result['gradcam']['overlay_image']
                # Convert PIL Image to bytes
                img_buffer = io.BytesIO()
                gradcam_img.save(img_buffer, format='PNG')
                img_buffer.seek(0)
                # Save to model
                filename = f"gradcam_{uuid.uuid4().hex[:8]}.png"
                prediction_record.gradcam_image.save(filename, ContentFile(img_buffer.read()), save=True)
            
            # Save grayscale image if available (for color X-rays)
            if result.get('gradcam') and result['gradcam'].get('grayscale_image'):
                grayscale_img = result['gradcam']['grayscale_image']
                # Convert PIL Image to bytes
                img_buffer = io.BytesIO()
                grayscale_img.save(img_buffer, format='PNG')
                img_buffer.seek(0)
                # Save to model
                filename = f"grayscale_{uuid.uuid4().hex[:8]}.png"
                prediction_record.grayscale_image.save(filename, ContentFile(img_buffer.read()), save=True)
            
            return render(request, 'predictor/result.html', {
                'result': result,
                'prediction_record': prediction_record
            })
        else:
            return render(request, 'predictor/home.html', {
                'form': form,
                'error': 'Please upload a valid image file.'
            })
    
    return redirect('home')


@csrf_exempt
@require_http_methods(["POST"])
def api_predict(request):
    """
    REST API endpoint for prediction.
    
    POST /api/predict/
    Request: multipart/form-data with 'image' file
    Response: JSON with prediction results
    """
    rate_limit_response = _check_rate_limit(request, scope='api_predict')
    if rate_limit_response is not None:
        return rate_limit_response

    if 'image' not in request.FILES:
        return JsonResponse({
            'error': 'No image file provided. Use "image" field in form-data.'
        }, status=400)

    image_file = request.FILES['image']

    if image_file.size > MAX_UPLOAD_SIZE_BYTES:
        return JsonResponse({'error': 'Image file too large. Max size is 10MB.'}, status=400)
    
    # Validate file type
    allowed_types = {'image/jpeg', 'image/png'}
    if image_file.content_type and image_file.content_type not in allowed_types:
        return JsonResponse({
            'error': 'Invalid image format. Use JPEG or PNG.'
        }, status=400)

    try:
        # Open image with PIL, normalize EXIF orientation, and fully decode it.
        image_file.seek(0)
        with Image.open(image_file) as uploaded_image:
            image = ImageOps.exif_transpose(uploaded_image).copy()

        # Get prediction with optional pipeline controls.
        validate_xray = _as_bool(
            request.POST.get('validate_xray'),
            getattr(settings, 'ENABLE_XRAY_VALIDATION', True)
        )
        # Support legacy parameter name
        if 'validate_lung' in request.POST:
            validate_xray = _as_bool(request.POST.get('validate_lung'), validate_xray)

        segment_lungs = _as_bool(
            request.POST.get('segment_lungs'),
            getattr(settings, 'ENABLE_LUNG_SEGMENTATION', True)
        )
        generate_gradcam = _as_bool(request.POST.get('generate_gradcam'), False)

        predictor = get_predictor()
        result = predictor.predict(
            image,
            validate_xray=validate_xray,
            segment_lungs=segment_lungs,
            generate_gradcam=generate_gradcam,
        )

        # Check if lung validation failed
        if not result.get('success', True):
            return JsonResponse({
                'success': False,
                'error': result.get('error', 'validation_failed'),
                'message': result.get('message', 'Image validation failed'),
                'lung_area_percent': result.get('lung_area_percent', 0)
            }, status=400)

        # Optionally save to history
        save_history = request.POST.get('save_history', 'false').lower() == 'true'
        if save_history:
            image_file.seek(0)
            probabilities = result.get('probabilities', {})
            PredictionHistory.objects.create(
                image=image_file,
                prediction=result['prediction'],
                confidence=result['confidence'],
                normal_probability=float(probabilities.get('NORMAL', 0.0)) / 100,
                pneumonia_probability=float(probabilities.get('PNEUMONIA', 0.0)) / 100,
                lung_validated=result.get('lung_validated', False),
                lung_area_ratio=result.get('lung_area_percent', 0) / 100
            )

        logger.info(
            'API prediction completed (ip=%s, prediction=%s, confidence=%.4f)',
            _get_client_ip(request),
            result.get('prediction', 'n/a'),
            float(result.get('confidence', 0.0)),
        )

        return JsonResponse({
            'success': True,
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'confidence_percent': result['confidence_percent'],
            'probabilities': result['probabilities'],
            'is_pneumonia': result['is_pneumonia'],
            'lung_validated': result.get('lung_validated', False),
            'lung_area_percent': result.get('lung_area_percent', 0)
        })

    except (UnidentifiedImageError, OSError):
        return JsonResponse({'error': 'Invalid image file. Use a readable JPEG or PNG image.'}, status=400)
    except ValueError:
        logger.exception('Input validation error during API prediction')
        return JsonResponse({'error': 'Prediction failed due to invalid input.'}, status=400)
    except Exception:
        logger.exception('Unexpected API prediction failure')
        return JsonResponse({
            'error': 'Prediction failed due to an internal server error.'
        }, status=500)


def history_view(request):
    """View prediction history."""
    predictions = PredictionHistory.objects.only(
        'id',
        'image',
        'prediction',
        'confidence',
        'normal_probability',
        'pneumonia_probability',
        'created_at',
    ).order_by('-created_at')[:50]

    # Calculate statistics
    total = PredictionHistory.objects.count()
    pneumonia_count = PredictionHistory.objects.filter(prediction='PNEUMONIA').count()
    normal_count = total - pneumonia_count

    stats = {
        'total': total,
        'pneumonia': pneumonia_count,
        'normal': normal_count,
        'pneumonia_percent': round(pneumonia_count / total * 100, 1) if total > 0 else 0
    }

    return render(request, 'predictor/history.html', {
        'predictions': predictions,
        'stats': stats
    })

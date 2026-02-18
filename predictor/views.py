from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.core.files.base import ContentFile
from PIL import Image
import io
import uuid

from .forms import ImageUploadForm
from .models import PredictionHistory
from .ml_model import get_predictor


def home(request):
    """Home page with upload form."""
    form = ImageUploadForm()
    recent_predictions = PredictionHistory.objects.all()[:5]
    
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
            
            # Open image with PIL
            image = Image.open(image_file)
            
            # Get prediction with X-ray validation
            predictor = get_predictor()
            result = predictor.predict(image, validate_xray=True)
            
            # Debug logging
            print(f"[DEBUG] Prediction result: success={result.get('success')}, error={result.get('error')}")
            print(f"[DEBUG] X-ray validation: validated={result.get('xray_validated')}, confidence={result.get('xray_confidence')}")
            if result.get('validation'):
                print(f"[DEBUG] Validation details: {result['validation'].get('validation_details')}")
            
            # Check if lung validation failed
            if not result.get('success', True):
                print(f"[DEBUG] Redirecting to invalid_image.html")
                return render(request, 'predictor/invalid_image.html', {
                    'result': result,
                    'lung_area_percent': result.get('lung_area_percent', 0)
                })
            
            # Save to history
            image_file.seek(0)  # Reset file pointer
            prediction_record = PredictionHistory.objects.create(
                image=image_file,
                prediction=result['prediction'],
                confidence=result['confidence'],
                normal_probability=result['probabilities']['NORMAL'] / 100,
                pneumonia_probability=result['probabilities']['PNEUMONIA'] / 100,
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
    if 'image' not in request.FILES:
        return JsonResponse({
            'error': 'No image file provided. Use "image" field in form-data.'
        }, status=400)
    
    image_file = request.FILES['image']
    
    # Validate file type
    allowed_types = ['image/jpeg', 'image/png']
    if image_file.content_type not in allowed_types:
        return JsonResponse({
            'error': 'Invalid image format. Use JPEG or PNG.'
        }, status=400)
    
    try:
        # Open image with PIL
        image = Image.open(image_file)
        
        # Get prediction with X-ray validation
        validate_xray = request.POST.get('validate_xray', 'true').lower() == 'true'
        # Support legacy parameter name
        if 'validate_lung' in request.POST:
            validate_xray = request.POST.get('validate_lung', 'true').lower() == 'true'
        predictor = get_predictor()
        result = predictor.predict(image, validate_xray=validate_xray)
        
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
            PredictionHistory.objects.create(
                image=image_file,
                prediction=result['prediction'],
                confidence=result['confidence'],
                normal_probability=result['probabilities']['NORMAL'] / 100,
                pneumonia_probability=result['probabilities']['PNEUMONIA'] / 100,
                lung_validated=result.get('lung_validated', False),
                lung_area_ratio=result.get('lung_area_percent', 0) / 100
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
        
    except Exception as e:
        return JsonResponse({
            'error': f'Prediction failed: {str(e)}'
        }, status=500)


def history_view(request):
    """View prediction history."""
    predictions = PredictionHistory.objects.all()[:50]
    
    # Calculate statistics
    total = predictions.count()
    pneumonia_count = predictions.filter(prediction='PNEUMONIA').count()
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

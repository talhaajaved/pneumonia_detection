# Pneumonia Detection - Django Web Application

AI-powered chest X-ray analysis for pneumonia detection using ResNet50.

## Quick Start (Local)

```bash
# Activate virtual environment
source /path/to/env/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Run migrations
python manage.py migrate

# Start server
python manage.py runserver 0.0.0.0:8000
```

Access at: http://localhost:8000

## Docker Deployment

### Build and Run

```bash
# Build and start
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### With Nginx (Production)

```bash
# Start with nginx profile
docker-compose --profile production up -d --build
```

Access at: http://localhost (port 80)

### Environment Variables

Create `.env` file from template:
```bash
cp .env.example .env
```

Edit `.env`:
```
DJANGO_SECRET_KEY=your-secure-random-key
DJANGO_DEBUG=False
DJANGO_ALLOWED_HOSTS=your-domain.com,www.your-domain.com
CSRF_TRUSTED_ORIGINS=https://your-domain.com
```

## API Usage

### Predict Endpoint

```bash
curl -X POST -F "image=@chest_xray.jpg" http://localhost:8000/api/predict/
```

Response:
```json
{
  "success": true,
  "prediction": "PNEUMONIA",
  "confidence": 0.9847,
  "confidence_percent": 98.47,
  "probabilities": {
    "NORMAL": 1.53,
    "PNEUMONIA": 98.47
  },
  "is_pneumonia": true
}
```

## Model Information

- **Architecture**: ResNet50 (pretrained on ImageNet)
- **Input Size**: 224x224 RGB
- **Classes**: NORMAL, PNEUMONIA
- **Validation Accuracy**: 98.28%

## Project Structure

```
pneumonia_django/
├── manage.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── pneumonia_django/
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
└── predictor/
    ├── ml_model.py      # Model loading & inference
    ├── views.py         # Web & API views
    ├── models.py        # Database models
    └── templates/
        └── predictor/
            ├── home.html
            ├── result.html
            └── history.html
```

## License

For research and educational purposes only.

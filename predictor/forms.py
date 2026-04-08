from django import forms
from PIL import Image, UnidentifiedImageError


MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024
MIN_IMAGE_DIMENSION = 224
MAX_IMAGE_DIMENSION = 8192
ALLOWED_CONTENT_TYPES = {'image/jpeg', 'image/png'}
ALLOWED_PIL_FORMATS = {'JPEG', 'PNG'}


class ImageUploadForm(forms.Form):
    """Form for uploading chest X-ray images."""
    
    image = forms.ImageField(
        label='Upload Chest X-ray Image',
        help_text='Supported formats: JPEG, PNG. Max size: 10MB',
        widget=forms.FileInput(attrs={
            'class': 'form-control',
            'accept': 'image/*'
        })
    )
    
    def clean_image(self):
        image = self.cleaned_data.get('image')
        if image:
            # Check file size (10MB max)
            if image.size > MAX_UPLOAD_SIZE_BYTES:
                raise forms.ValidationError('Image file too large. Max size is 10MB.')

            # MIME type is user-controlled, but still useful as an early filter.
            if image.content_type not in ALLOWED_CONTENT_TYPES:
                raise forms.ValidationError('Invalid image format. Use JPEG or PNG.')

            image.seek(0)
            try:
                # Verify the binary stream is a valid image.
                with Image.open(image) as img:
                    img.verify()

                image.seek(0)
                with Image.open(image) as img:
                    if img.format not in ALLOWED_PIL_FORMATS:
                        raise forms.ValidationError('Invalid image format. Use JPEG or PNG.')

                    width, height = img.size
                    if width < MIN_IMAGE_DIMENSION or height < MIN_IMAGE_DIMENSION:
                        raise forms.ValidationError(
                            f'Image is too small. Minimum size is {MIN_IMAGE_DIMENSION}x{MIN_IMAGE_DIMENSION}px.'
                        )
                    if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
                        raise forms.ValidationError(
                            f'Image is too large. Maximum size is {MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION}px.'
                        )
            except UnidentifiedImageError as exc:
                raise forms.ValidationError('Uploaded file is not a valid image.') from exc
            except OSError as exc:
                raise forms.ValidationError('Could not read the uploaded image file.') from exc
            finally:
                image.seek(0)
        
        return image

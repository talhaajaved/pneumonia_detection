from django import forms


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
            if image.size > 10 * 1024 * 1024:
                raise forms.ValidationError('Image file too large. Max size is 10MB.')
            
            # Check file type
            allowed_types = ['image/jpeg', 'image/png']
            if image.content_type not in allowed_types:
                raise forms.ValidationError('Invalid image format. Use JPEG or PNG.')
        
        return image

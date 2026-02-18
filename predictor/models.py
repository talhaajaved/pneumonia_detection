from django.db import models


class PredictionHistory(models.Model):
    """Store prediction history for audit and analytics."""
    
    image = models.ImageField(upload_to='predictions/%Y/%m/%d/')
    segmented_image = models.ImageField(upload_to='predictions/%Y/%m/%d/segmented/', null=True, blank=True)
    gradcam_image = models.ImageField(upload_to='predictions/%Y/%m/%d/gradcam/', null=True, blank=True)
    prediction = models.CharField(max_length=20)
    confidence = models.FloatField()
    normal_probability = models.FloatField()
    pneumonia_probability = models.FloatField()
    lung_validated = models.BooleanField(default=False)
    lung_area_ratio = models.FloatField(default=0.0)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        verbose_name = 'Prediction'
        verbose_name_plural = 'Prediction History'
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.prediction} ({self.confidence_percent}%) - {self.created_at.strftime('%Y-%m-%d %H:%M')}"
    
    @property
    def confidence_percent(self):
        return round(self.confidence * 100, 2)
    
    @property
    def lung_area_percent(self):
        return round(self.lung_area_ratio * 100, 2)
    
    @property
    def is_pneumonia(self):
        return self.prediction == 'PNEUMONIA'

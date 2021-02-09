from django.db import models
from django.core.validators import FileExtensionValidator


# Create your models here.
class ImageUpload(models.Model):
    image_file = models.ImageField(validators=[FileExtensionValidator(['png', 'jpg', 'jpeg'])])
    upload_time = models.DateTimeField(auto_now_add=True)

class CaptionUpload(models.Model):
    # image_file = models.ImageField(validators=[FileExtensionValidator(['png', 'jpg', 'jpeg'])])
    caption_flag = models.BooleanField()

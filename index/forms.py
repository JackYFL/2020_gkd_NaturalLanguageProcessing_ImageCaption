from django import forms
from index.models import ImageUpload, CaptionUpload


class ImageForm(forms.ModelForm):
    """Form handling uploaded image"""

    class Meta:
        model = ImageUpload
        fields = [
            'image_file'
        ]

class CaptionForm(forms.ModelForm):
    """Form handling captioning image"""

    class Meta:
        model = CaptionUpload
        fields = [
            'caption_flag'
        ]

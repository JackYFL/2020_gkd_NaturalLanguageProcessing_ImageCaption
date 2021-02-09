from django.shortcuts import render
from .forms import ImageForm, CaptionForm
from index.caption import *

# Create your views here.
def index(request):
    context = {'flag': False}
    context = {'image_show': False}
    form = ImageForm()
    caption_form = CaptionForm()

    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            my_image = form.save(commit=False)
            my_image.image_file = request.FILES['image_file']
            image_name = my_image.image_file  # image_file attribute of myImage model object
            img = Image.open(image_name).convert('RGB')
            img.save(base_dir/'media/caption_img.jpg')
            my_image.save()

            with open('myDataBase.txt', 'w') as my_database:
                my_database.write('True')
            context['image_show'] = True
            # context['caption_in_chs'] = caption_in_chs(image_name)
            # context['caption_in_eng'] = caption_in_eng(image_name)
            context['image_name'] = image_name
            context['form'] = form

            form_obj = ImageForm(request.POST)
            return render(request, 'index.html', context)
        caption_form = CaptionForm(request.POST)
        if "caption" in request.POST:
            context['flag'] = True
            context['caption_in_chs'] = caption_in_chs(base_dir/'media/caption_img.jpg')
            context['caption_in_eng'] = caption_in_eng(base_dir/'media/caption_img.jpg')
            context['form'] = ImageForm()
            context['image_name'] = base_dir/'media/caption_img.jpg'
            context['image_show'] = True
            return render(request, 'index.html', context)
        # return render(request, 'index.html', context)

    context['form'] = form
    context['caption_form'] = caption_form
    return render(request, 'index.html', context)

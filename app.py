import re

from flask import Flask, request, render_template, send_from_directory
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration
import os

app = Flask(__name__)

processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        else:
            filename = file.filename
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(img_path)
            image = Image.open(img_path).convert('RGB')  # Open and process the image
            text = "An image of"
            inputs = processor(images=image, text=text, return_tensors="pt")
            outputs = model.generate(**inputs, max_length=50)
            caption = processor.decode(outputs[0], skip_special_tokens=True)
            caption = re.sub(r'^\w', lambda x: x.group(0).upper(),caption)
            return render_template('result.html', caption=caption, filename=filename)
    return render_template('upload.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    if not os.path.exists('uploads'): # Make sure the 'uploads' directory exists
        os.makedirs('uploads')

    app.run(debug=False)

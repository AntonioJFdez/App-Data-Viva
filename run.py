# ejecutar.py (script principal Flask completo)
from flask import Flask, render_template, request, send_from_directory
import os
from werkzeug.utils import secure_filename
from PIL import Image

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def compare():
    result_image = None
    if request.method == 'POST':
        image1 = request.files['image1']
        image2 = request.files['image2']
        if image1 and image2:
            img1 = Image.open(image1.stream).convert("RGBA")
            img2 = Image.open(image2.stream).convert("RGBA")
            img1 = img1.resize((600, 400))
            img2 = img2.resize((600, 400))
            blended = Image.blend(img1, img2, alpha=0.5)
            result_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result.png')
            blended.save(result_path)
            result_image = 'result.png'
    return render_template('visual_compare.html', result_image=result_image)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)


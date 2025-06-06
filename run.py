from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
from werkzeug.utils import secure_filename
from PIL import Image, ImageChops

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def visual_compare():
    result_image = None
    if request.method == 'POST':
        file1 = request.files['image1']
        file2 = request.files['image2']
        if file1 and file2:
            path1 = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file1.filename))
            path2 = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file2.filename))
            file1.save(path1)
            file2.save(path2)

            image1 = Image.open(path1).convert('RGB')
            image2 = Image.open(path2).convert('RGB')

            diff = ImageChops.difference(image1, image2)
            diff_path = os.path.join(app.config['UPLOAD_FOLDER'], 'diff_result.png')
            diff.save(diff_path)
            result_image = 'diff_result.png'

    return render_template('visual_compare.html', result_image=result_image)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    # Cambia esto para soportar Render (y sigue funcionando en local)
    port = int(os.environ.get("PORT", 5000))   # Render te da PORT, local usa 5000
    app.run(host="0.0.0.0", port=port, debug=True)


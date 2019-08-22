from flask import Flask, render_template, request, redirect, url_for, send_from_directory, session
from werkzeug import secure_filename
import os
import shutil
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image
from typing import Tuple, List, Dict
from sklearn.feature_extraction import image
from sklearn.decomposition import NMF
import pickle
import numbers
from itertools import product
import seaborn as sns
import scipy as sp
import io
import base64

app = Flask(__name__)

ALLOWED_EXTENSIONS = set(['png', 'jpg'])
UPLOAD_FOLDER = './static/images/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = os.urandom(24)

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

# 保存したモデルをロードする
filename = 'NMF_model_n40.sav'
loaded_model = pickle.load(open(filename, 'rb'))

def extract_patches_2d(img: np.ndarray, patch_size: Tuple[int, int],
                       extraction_step: int):
    img_height, img_width = img.shape[:2]
    img = img.reshape((img_height, img_width, -1))
    colors = img.shape[-1]
    patch_height, patch_width = patch_size[:2]
    patches = image.extract_patches(img, patch_shape=(patch_height, patch_width, colors),
                                    extraction_step=extraction_step)
    patches = patches.reshape(-1, patch_height, patch_width, colors)
    if patches.shape[-1] == 1:
        return patches.reshape((-1, patch_height, patch_width))
    else:
        return patches

def reconstruct_from_patches_2d(patches: np.ndarray, image_size: Tuple[int, int], extraction_step: int):
    array_ndimension = len(image_size)
    if isinstance(extraction_step, numbers.Number):
        extraction_step = tuple([extraction_step] * array_ndimension)
    img_height, img_width = image_size[:2]
    patch_height, patch_width = patches.shape[1:3]
    extract_height, extract_width = extraction_step[:2]
    img = np.zeros(image_size)
    counter = np.zeros(image_size)
    n_height = (img_height - patch_height) // extract_height + 1
    n_width = (img_width - patch_width) // extract_width + 1
    for p, (i, j) in zip(patches, product(range(n_height), range(n_width))):
        img[extract_height * i:extract_height * i + patch_height, extract_width * j:extract_width * j + patch_width] += p
        counter[extract_height * i:extract_height * i + patch_height, extract_width * j:extract_width * j + patch_width] += 1.
    counter[counter == 0] = 1.
    return img / counter

def expected_popularity(error_value):

    diff=abs(error_value-4007.523)
    different_rate=diff/(515.818*2)
    if different_rate<=1.0:
        rate=(1.0-different_rate)*100
    else:
        rate=np.random.randint(1, 5)+np.random.rand()

    return round(rate, 1)

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/how')
def howto():
    return render_template('howto.html')


@app.route('/upload', methods=['POST', 'GET'])
def result():
    if request.method == 'POST':
        img_file = request.files['img_file']
        if img_file and allowed_file(img_file.filename):
            image = Image.open(img_file)
            buf = io.BytesIO()
            image.save(buf, 'png')
            qr_b64str = base64.b64encode(buf.getvalue()).decode("utf-8")
            qr_b64data = "data:image/png;base64,{}".format(qr_b64str)
            #img_file.save(os.path.join(app.config['UPLOAD_FOLDER'], "original.jpg"))
            #img_url = '../static/images/' + "original.jpg"

            # 画像の読み込み
            selected_img=np.array(Image.open(img_file).resize((600, 600)).crop((100, 100, 500, 500)).convert('L'))/255
            #selected_img=np.array(Image.open(UPLOAD_FOLDER+"original.jpg").resize((600, 600)).crop((100, 100, 500, 500)).convert('L'))/255

            # パッチ分割
            patch_list=[]
            patches_array=np.ndarray
            patches=extract_patches_2d(selected_img, (20, 20), 20)
            for idx in range(len(patches)):
                    patch_list.append(patches[idx].flatten())
            patches_array=np.array(patch_list)

            #パッチの再構成
            total_error=0
            transformed_patch_list=[]
            # reconstructed img
            for i in range(len(patches_array)):
                H=loaded_model.components_
                W=loaded_model.transform(patches_array[i].reshape(1, -1))
                X=np.dot(W, H).reshape(20, -1)
                transformed_patch_list.append(X)
                total_error+=abs(patches_array[i].reshape(20, -1)-X).sum()
            transformed_patches=np.array(transformed_patch_list)

            #パッチを画像に再構成
            reconstructed_img=reconstruct_from_patches_2d(transformed_patches, (400, 400), 20)

            plt.figure(figsize=(10, 5), facecolor="#FFCC33")
            # orginal img
            plt.subplot(1, 2, 1)
            plt.gray()
            plt.title("extracted image (from your Ramen)", fontsize=20)
            plt.imshow(selected_img)
            # reconstructed img
            plt.subplot(1, 2, 2)
            plt.gray()
            plt.title("compared image (made by AI)", fontsize=20)
            plt.imshow(reconstructed_img)
            plt.tight_layout()
            plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'], "result_comparison.jpg"))

            score=expected_popularity(total_error)

            image_result="../static/images/result_comparison.jpg"

            return render_template('result.html', img_url=qr_b64data, score=score, image_result=image_result)
        else:
            return redirect(url_for('index'))
    else:
        return redirect(url_for('index'))

@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

if __name__ == "__main__":
    app.run(debug=True)
# run by ">python index.py"

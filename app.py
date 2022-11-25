from flask import Flask, redirect, url_for, render_template, request, session, flash, jsonify, send_from_directory
from datetime import timedelta

from werkzeug.utils import secure_filename
UPLOAD_FOLDER = r'C:\Users\HP\Documents\flask_app\static\upload_imgs'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

import pickle

app = Flask(__name__)
app.secret_key = "hello"
app.permanent_session_lifetime = timedelta(minutes=5)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


from tensorflow.keras.layers import Conv2D,Flatten,Dense,MaxPool2D,BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
model = tf.keras.models.load_model(r"C:\Users\HP\Documents\flask_app\models\my_model.h5")


    

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/login", methods=["POST", "GET"])
def login():
    if request.method == "POST":
        session.permanent = True
        user = request.form["nm"]
        session["user"] = user
        flash("Login successful!")
        return redirect(url_for("user"))
    else:
        if "user" in session:
            flash("Already Logged In!")
            return redirect(url_for("user"))
        return render_template("login.html")


@app.route("/user", methods=["POST", "GET"])
def user():
    email = None
    if "user" in session:
        user = session["user"]

        if request.method == "POST":
            email = request.form["email"]
            session["email"] = email
            flash("Email was saved!")
        else:
            if "email" in session:
                email = session["email"]

        return render_template("user.html", email=email)
    else:
        flash("You are not logged in!")
        return redirect(url_for("login"))

def pre_dict(img):
    import keras
    img = tf.keras.preprocessing.image.load_img(img)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    class_names = ['Mild', 'Moderate', 'No_DR', 'Proliferate_DR', 'Severe']
    score = tf.nn.softmax(predictions[0])
    return render_template('user.html', pred='Predicted Class : {}'.format(class_names[np.argmax(score)]))#class_names[np.argmax(score)]

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return pre_dict(os.path.join(app.config['UPLOAD_FOLDER'], filename))

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file',filename=filename))
    return redirect(url_for("user"))



"""
@app.route("/result", methods = ["GET","POST"])
def result():
    if request.method == "POST":
        file_upload = request.form["imgupload"]
        result = pre_dict(file_upload)
        if result == "Mild":
            prediction = 'STAGE OF DIABETIC RETINOPATHY : Mild'
        if result == "Moderate":
            prediction = 'STAGE OF DIABETIC RETINOPATHY : Moderate'
        if result == "No_DR":
            prediction = 'STAGE OF DIABETIC RETINOPATHY : No Diabetic Retinopathy'
        if result == "Proliferate":
            prediction = 'STAGE OF DIABETIC RETINOPATHY : Proliferate'
        if result == "Severe":
            prediction = 'STAGE OF DIABETIC RETINOPATHY : Severe'
        return render_template("result.html", prediction = prediction)	
"""
@app.route("/logout")
def logout():
    flash("You have been logged out", "info")
    session.pop("user", None)
    session.pop("email", None)
    return redirect(url_for("login"))

if __name__ == '__main__':
    app.run(debug=True)
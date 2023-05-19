import io, base64
import numpy as np
from PIL import Image
from keras.models import load_model

model = load_model('model.h5')

def preprocess_input_predict(base64_image):
    decoded_img = base64.b64decode((base64_image))
    img = Image.open(io.BytesIO(decoded_img))
    np_img = [np.array(img)]
    np_img = np.array(np_img)
    np_img = np_img.reshape(np_img.shape[0],48, 48, 1)
    np_img = np_img / 255.0
    return np_img

def predict_image(base64_image):
    np_img = preprocess_input_predict(base64_image)
    prediction = model.predict(np_img)
    return prediction
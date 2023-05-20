import io, base64
import cv2
import numpy as np
from PIL import Image
from keras.models import load_model

model = load_model('model.h5')

def preprocess_input_predict(base64_image):
    # decoded_img = base64.b64decode((base64_image))
    # img_rgb = Image.open(io.BytesIO(decoded_img))
    # img_gray = img_rgb.convert('L').resize((48,48))
    # np_img = [np.array(img_gray)]
    # np_img = np.array(np_img)
    # print(np_img.shape)
    # np_img = np_img.reshape(np_img.shape[0],48, 48, 1)
    # np_img = np_img / 255.0
    # return np_img
    
    decoded_img = base64.b64decode((base64_image))
    img_rgb = Image.open(io.BytesIO(decoded_img))
    img_rgb = np.array(img_rgb)
    img_gray = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_BGR2GRAY)
    cropped_image = preprocess_crop_face(img_rgb, img_gray)
    final_image = Image.fromarray(cropped_image).convert('L')
    final_image = np.array(final_image)
    final_image = cv2.resize(final_image, (48, 48))
    final_image = np.expand_dims(final_image, axis=0)
    final_image = np.expand_dims(final_image, axis=-1)
    final_image = final_image / 255.0
    return final_image

def preprocess_crop_face(img_rgb, img_gray):
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = faceCascade.detectMultiScale(img_gray, 1.1, 4)
    for (x, y, w, h) in faces:
        roi_gray = img_gray[y:y+h, x:x+w]
        roi_rgb = img_rgb[y:y+h, x:x+w]
        cv2.rectangle(img_rgb, (x, y), (x+w, y+h), (255, 0, 0), 2)
        facess = faceCascade.detectMultiScale(roi_gray)
        # if len(facess) == 0:
        #     return 
        # else:
        for (ex, ey, ew, eh) in facess:
            face_roi = roi_rgb[ey: ey+eh, ex:ex + ew]
    return face_roi

def predict_image(base64_image):
    np_img = preprocess_input_predict(base64_image)
    prediction = model.predict(np_img)
    return prediction
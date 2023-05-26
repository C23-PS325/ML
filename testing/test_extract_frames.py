import io, base64, os
import cv2
import numpy as np
import tqdm
from datetime import timedelta
from PIL import Image
from keras.models import load_model
from io import BytesIO

model = load_model('model.h5')

def preprocess_image(base64_image):
    decoded_img = base64.b64decode((base64_image))
    img_rgb = Image.open(io.BytesIO(decoded_img))
    img_rgb = np.array(img_rgb)
    img_gray = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_BGR2GRAY)
    try:
        cropped_image = preprocess_crop_face(img_rgb, img_gray)
    except:
        cropped_image = img_gray
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
    np_img = preprocess_image(base64_image)
    prediction = model.predict(np_img)
    return prediction

def extract_frames(video):
    SAVING_FRAMES_PER_SECOND = 30
    raw_filename, _ = os.path.splitext(video)
    filename =  "extracted_frames/" + raw_filename + "_frames"
    # make a folder by the name of the video file
    if not os.path.isdir(filename):
        os.mkdir(filename)
    # read the video file    
    capture_video = cv2.VideoCapture(video)
    fps = capture_video.get(cv2.CAP_PROP_FPS)
    curr_frame = 0
    while(True):
        ret, frame = capture_video.read()
        if not ret:
            break
        if curr_frame % fps == 0:
            img_name = filename + '/' + str(curr_frame) + ".jpg"
            cv2.imwrite(img_name, frame)
        curr_frame += 1
    
    capture_video.release()
    cv2.destroyAllWindows()
    
    return filename
    
def predict_all_frames(path):
    path += '/'
    all_frames = os.listdir(path)
    
    total_predictions = dict()
    for frame in all_frames:
        with open(path + frame, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read())
        prediction = predict_image(base64_image)
        list_prediction = prediction.tolist()
        prediction_dict  = {list_prediction[0][0] : "angry", list_prediction[0][1] : "fear", list_prediction[0][2] : "happy", list_prediction[0][3] : "sad", list_prediction[0][4] : "surprise"}
        result = prediction.argmax()
        expression = prediction_dict[prediction[0][result]]
        
        if expression in total_predictions:
            total_predictions[expression] += 1
        else:
            total_predictions[expression] = 1 
    
    return total_predictions

def predict_video(video):
    frames_folder = extract_frames(video)
    total_predictions = predict_all_frames(frames_folder)
    return total_predictions

if __name__ == "__main__":
    video = "test.mp4"
    print(predict_video(video))
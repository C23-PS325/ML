import io, base64, os, cv2, librosa
import numpy as np
from PIL import Image
from keras.models import load_model
from moviepy.editor import VideoFileClip

model_gambar = load_model('model/model_gambar.h5')
model_suara = load_model('model/model_suara.h5')

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
        for (ex, ey, ew, eh) in facess:
            face_roi = roi_rgb[ey: ey+eh, ex:ex + ew]
    return face_roi

def predict_image(base64_image):
    np_img = preprocess_image(base64_image)
    prediction = model_gambar.predict(np_img)
    return prediction

def extract_frames(video):
    SAVING_FRAMES_PER_SECOND = 30
    raw_filename, _ = os.path.splitext(video)
    raw_filename = raw_filename.split('/')[-1]
    filename =  "extracted_frames/" + raw_filename + "_frames"

    if not os.path.isdir(filename):
        os.mkdir(filename)

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

def extract_audio_from_video(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    video = VideoFileClip(video_path)
    audio = video.audio
    filename = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(output_dir, f"{filename}.wav")
    audio.write_audiofile(output_path, codec='pcm_s16le')
    audio.close()
    video.close()
    return output_path

def extract_mfcc(filename):
    audio, sample_rate = librosa.load(filename, duration=3, offset=0.5)
    mfcss = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T,axis=0)
    return mfcss

def predict_sound(filename):
    mfcc = extract_mfcc(filename)
    mfcc = np.expand_dims(mfcc, axis=0)
    mfcc = np.expand_dims(mfcc, axis=2)
    prediction = model_suara.predict(mfcc)
    list_prediction = prediction.tolist()
    prediction_dict  = {list_prediction[0][0] : "angry", list_prediction[0][1] : "disgust", list_prediction[0][2] : "fear", list_prediction[0][3] : "happy", list_prediction[0][4] : "natural", list_prediction[0][5] : "surprise", list_prediction[0][6] : "sad"}
    result = prediction.argmax()
    expression = prediction_dict[prediction[0][result]]
    return expression

def predict_video(video):
    frames_folder = extract_frames(video)
    frames_prediction = predict_all_frames(frames_folder)
    extracted_audio = extract_audio_from_video(video, "extracted_audio")
    audio_prediction = predict_sound(extracted_audio)
    total_predictions = dict()
    total_predictions["frames"] = frames_prediction
    total_predictions["audio"] = audio_prediction
    return total_predictions
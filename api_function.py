import io, base64, os, cv2, librosa, datetime
import numpy as np
from PIL import Image
from keras.models import load_model
from moviepy.editor import VideoFileClip
from moviepy.editor import AudioFileClip
from google.cloud import storage
from io import BytesIO

class APIHelper:
    def __init__(self):
        self.model_gambar = load_model('model/model_gambar.h5')
        self.model_suara = load_model('model/model_suara.h5')
        self.blob_path = ""
        credential_path = "./service-account.json"
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path
        self.google_project_id = "communicare-staging"
        self.google_bucket_name = "communicare-proc-dev"
        self.client = storage.Client()

    def preprocess_image(self,base64_image):
        decoded_img = base64.b64decode((base64_image))
        img_rgb = Image.open(io.BytesIO(decoded_img))
        img_rgb = np.array(img_rgb)
        img_gray = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_BGR2GRAY)
        try:
            cropped_image = self.preprocess_crop_face(img_rgb, img_gray)
        except:
            cropped_image = img_gray
        final_image = Image.fromarray(cropped_image).convert('L')
        final_image = np.array(final_image)
        final_image = cv2.resize(final_image, (48, 48))
        final_image = np.expand_dims(final_image, axis=0)
        final_image = np.expand_dims(final_image, axis=-1)
        final_image = final_image / 255.0
        return final_image

    def preprocess_crop_face(self,img_rgb, img_gray):
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

    def predict_image(self,base64_image):
        np_img = self.preprocess_image(base64_image)
        prediction = self.model_gambar.predict(np_img)
        return prediction

    def extract_frames(self,url_video, frames_folder):
        SAVING_FRAMES_PER_SECOND = 30
        # raw_filename, _ = os.path.splitext(video)
        # raw_filename = raw_filename.split('/')[-1]
        directory =  f"extracted_frames/{frames_folder}" 
        # if not os.path.isdir(filename):
        #     os.mkdir(filename)
        list_images = []
            
        capture_video = cv2.VideoCapture(url_video)
        fps = capture_video.get(cv2.CAP_PROP_FPS)
        curr_frame = 0
        while(True):
            ret, frame = capture_video.read()
            if not ret:
                break
            if curr_frame % fps == 0:
                # img_name = directory + '/' + str(curr_frame) + ".jpg"
                # cv2.imwrite(img_name, frame)
                # await self.upload_frames_to_gcs(img_name)
                retval, buffer = cv2.imencode('.jpg', frame)
                base64_image = base64.b64encode(buffer)
                list_images.append(base64_image)
            curr_frame += 1

        capture_video.release()
        cv2.destroyAllWindows()
        
        return list_images
        
    # def predict_all_frames(self,path):
    #     path += '/'
    #     all_frames = os.listdir(path)
        
    #     total_predictions = dict()
    #     for frame in all_frames:
    #         with open(path + frame, "rb") as image_file:
    #             base64_image = base64.b64encode(image_file.read())
    #         prediction = self.predict_image(base64_image)
    #         list_prediction = prediction.tolist()
    #         prediction_dict  = {list_prediction[0][0] : "angry", list_prediction[0][1] : "fear", list_prediction[0][2] : "happy", list_prediction[0][3] : "sad", list_prediction[0][4] : "surprise"}
    #         result = prediction.argmax()
    #         expression = prediction_dict[prediction[0][result]]
            
    #         if expression in total_predictions:
    #             total_predictions[expression] += 1
    #         else:
    #             total_predictions[expression] = 1 
                
    #     final_result = {key: round(((value / len(all_frames)) * 100),2) for key, value in total_predictions.items()}
        
    #     return final_result
    
    def predict_all_frames(self,lst_base_64):
        total_predictions = dict()
        for base64_image in lst_base_64:
            prediction = self.predict_image(base64_image)
            list_prediction = prediction.tolist()
            prediction_dict  = {list_prediction[0][0] : "angry", list_prediction[0][1] : "fear", list_prediction[0][2] : "happy", list_prediction[0][3] : "sad", list_prediction[0][4] : "surprise"}
            result = prediction.argmax()
            expression = prediction_dict[prediction[0][result]]
            
            if expression in total_predictions:
                total_predictions[expression] += 1
            else:
                total_predictions[expression] = 1 
                
        final_result = {key: round(((value / len(lst_base_64)) * 100),2) for key, value in total_predictions.items()}
        
        return final_result

    def extract_audio_from_video(self,url_video):
        # os.makedirs(output_dir, exist_ok=True)

        video = VideoFileClip(url_video)
        audio = video.audio
        filename = self.blob_path
        output_path = f"{filename}.wav"
        audio.write_audiofile(output_path, codec='pcm_s16le')
        # audio_blob_path = self.upload_audio_to_gcs(output_path)
        audio.close()
        video.close()
        return output_path

    def extract_mfcc(self,filename):
        audio, sample_rate = librosa.load(filename, duration=3, offset=0.5)
        mfcss = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T,axis=0)
        return mfcss

    def predict_sound(self,audio):
        # video = VideoFileClip(url_audio)
        # audio = video.audio
        mfcc = self.extract_mfcc(audio)
        mfcc = np.expand_dims(mfcc, axis=0)
        mfcc = np.expand_dims(mfcc, axis=2)
        prediction = self.model_suara.predict(mfcc)
        list_prediction = prediction.tolist()
        prediction_dict  = {list_prediction[0][0] : "angry", list_prediction[0][1] : "disgust", list_prediction[0][2] : "fear", list_prediction[0][3] : "happy", list_prediction[0][4] : "natural", list_prediction[0][5] : "surprise", list_prediction[0][6] : "sad"}
        result = prediction.argmax()
        expression = prediction_dict[prediction[0][result]]
        return expression

    def predict_video(self, video_blob_path):
        frames_folder = video_blob_path.split('/')[-1].replace('.mp4','')
        self.blob_path = frames_folder
        signed_url_video = self.generate_signed_url(video_blob_path)
        frames_folder = self.extract_frames(signed_url_video, frames_folder)
        frames_prediction = self.predict_all_frames(frames_folder)
        extracted_audio = self.extract_audio_from_video(signed_url_video)
        # signed_url_audio = self.generate_signed_url(extracted_audio)
        audio_prediction = self.predict_sound(extracted_audio)
        os.remove(extracted_audio)
        total_predictions = dict()
        total_predictions["frames"] = frames_prediction
        total_predictions["audio"] = audio_prediction
        return total_predictions
    
    def generate_signed_url(self, blob_path):
        bucket = self.client.get_bucket(self.google_bucket_name)                                                     
        blob = bucket.blob(blob_path)
        return blob.generate_signed_url(datetime.timedelta(seconds=300), method='GET')
    
    def get_audio_from_gcs(self, audio_name):
        bucket = self.client.get_bucket(self.google_bucket_name)
        blob = bucket.blob(audio_name)
        blob.download_to_filename(audio_name)
        return True
    
    def upload_audio_to_gcs(self, audio_name):
        bucket = self.client.get_bucket(self.google_bucket_name)
        audio_blob_path = f"audio/{audio_name}"
        blob = bucket.blob(audio_blob_path)
        blob.upload_from_filename(audio_name)
        return audio_blob_path
    
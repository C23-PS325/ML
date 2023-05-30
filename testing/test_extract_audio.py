from moviepy.editor import VideoFileClip
import os

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


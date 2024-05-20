import numpy as np
import cv2
import face_recognition
import sounddevice as sd
from scipy.io.wavfile import write
from moviepy.editor import VideoFileClip, AudioFileClip
import threading
import time
import os

def record_audio(sample_rate, output_file, stop_event, start_event):
    try:
        print("Recording audio...")
        audio_data = []

        def callback(indata, frames, time, status):
            if status:
                print(status)
            audio_data.extend(indata.copy())

            if stop_event.is_set():
                raise sd.CallbackStop

        start_event.set()  # 비디오 녹화 시작을 알림
        with sd.InputStream(samplerate=sample_rate, channels=1, callback=callback):
            stop_event.wait()  # 오디오 녹음을 중지할 때까지 대기

        write(output_file, sample_rate, np.array(audio_data))
        print("Recording finished.")
    except Exception as e:
        print("Error in recording audio:", e)

def input_image(img_path):
    img = face_recognition.load_image_file(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def blur_face(image, face):
    (startY, endX, endY, startX) = face
    face_img = image[startY:endY, startX:endX]
    blurred_face = cv2.GaussianBlur(face_img, (99, 99), 30)
    image[startY:endY, startX:endX] = blurred_face
    return image

def webcam_face_blur(exclusion_img, save_video_path, audio_path):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 6)  # FPS 20 설정
    encode_exclusion_img = face_recognition.face_encodings(exclusion_img)[0]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_video_path, fourcc, 6.0, (640, 480))  # FPS 20.0

    stop_event = threading.Event()  # 스레드 종료 이벤트
    start_event = threading.Event()  # 오디오 녹음 시작 이벤트
    audio_thread = threading.Thread(target=record_audio, args=(44100, audio_path, stop_event, start_event))  # 오디오 녹음 스레드 시작
    audio_thread.start()

    start_event.wait()  # 오디오 녹음 시작 대기
    print("Recording video...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for enc, loc in zip(face_encodings, face_locations):
            match = face_recognition.compare_faces([encode_exclusion_img], enc, tolerance=0.4)
            if not match[0]:
                frame = blur_face(frame, loc)

        out.write(frame)
        cv2.imshow('Webcam', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    stop_event.set()  # 오디오 녹음을 중지하도록 이벤트 설정
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    audio_thread.join()

    print("Video recording finished.")
    combine_audio_video(save_video_path, audio_path, save_video_path.replace('.mp4', '_with_audio.mp4'))

def combine_audio_video(video_path, audio_path, output_path):
    try:
        video_clip = VideoFileClip(video_path)
        audio_clip = AudioFileClip(audio_path)
        if audio_clip.duration > video_clip.duration:
            audio_clip = audio_clip.subclip(0, video_clip.duration)
        final_clip = video_clip.set_audio(audio_clip)
        final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac', fps=20)  # FPS 20 설정
        print("Video and audio have been successfully combined.")
    except Exception as e:
        print("Failed to combine video and audio:", e)
    finally:
        video_clip.close()
        audio_clip.close()

exclusion_img_path = "/Users/anchaelim/PycharmProjects/ssblur0509/Face-Out/face_out/img/cr.jpeg"
save_video_path = "/Users/anchaelim/PycharmProjects/ssblur0509/Face-Out/face_out/outputvideo/output_video.mp4"
audio_path = "/Users/anchaelim/PycharmProjects/ssblur0509/Face-Out/face_out/outputvideo/output_audio.wav"

exclusion_img = input_image(exclusion_img_path)
webcam_face_blur(exclusion_img, save_video_path, audio_path)

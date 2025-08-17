from flask import Flask, render_template, Response, jsonify
import cv2
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import threading
import pyttsx3

app = Flask(__name__)

VIDEO_URL = "http://192.168.1.2:4444"   

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

latest_frame = None
lock = threading.Lock()

def generate_frames():
    global latest_frame
    cap = cv2.VideoCapture(VIDEO_URL, cv2.CAP_FFMPEG)
    while True:
        success, frame = cap.read()
        if not success:
            break

        with lock:
            latest_frame = frame.copy()

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/caption', methods=['POST'])
def caption():
    global latest_frame
    try:
        with lock:
            if latest_frame is None:
                return jsonify({'caption': "No frame available yet"})

            image = cv2.cvtColor(latest_frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(image)

        inputs = processor(pil_img, return_tensors="pt").to(device)
        out = model.generate(**inputs)
        caption_text = processor.decode(out[0], skip_special_tokens=True)

        pyttsx3.speak(caption_text)

        return jsonify({'caption': caption_text})

    except Exception as e:
        return jsonify({'caption': f"Error: {str(e)}"})

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")

from flask import Flask, render_template, Response, request, jsonify
import cv2
import requests
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import io
import torch
import threading
import pyttsx3

app = Flask(__name__)

VIDEO_URL = "<IP>:<PORT>/video"
SNAPSHOT_URL = "<IP>:<PORT>/shot.jpg"

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

def generate_frames():
    cap = cv2.VideoCapture(VIDEO_URL)
    while True:
        success, frame = cap.read()
        if not success:
            break
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
    try:
        img_bytes = requests.get(SNAPSHOT_URL, timeout=5).content
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        inputs = processor(image, return_tensors="pt").to(device)
        out = model.generate(**inputs)
        caption_text = processor.decode(out[0], skip_special_tokens=True)

        pyttsx3.speak(caption_text)

        return jsonify({'caption': caption_text})

    except Exception as e:
        return jsonify({'caption': f"Error: {str(e)}"})

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")


from flask import Flask, render_template, request, jsonify
import threading
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import io
import base64
import torch
import pyttsx3


app = Flask(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/caption', methods=['POST'])
def caption():
    data = request.json
    img_data = data['image']
    header, encoded = img_data.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(img_bytes)).convert('RGB')

    inputs = processor(image, return_tensors="pt").to(device)
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)

    threading.Thread(target=lambda: pyttsx3.speak(caption)).start()

    return jsonify({'caption': caption})


if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, request, jsonify
import numpy as np
import cv2
import os
import sys

# Adiciona o diretório pai ao sys.path para importar o módulo corretamente
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from object_color_detector import SpecificObjectDetector

app = Flask(__name__)
detector = SpecificObjectDetector()

def desenhar_objetos(frame, objetos):
    for obj in objetos:
        x, y, w, h = obj.get('bounding_box', [0,0,0,0])
        cor = (0,255,0)
        cv2.rectangle(frame, (x, y), (x+w, y+h), cor, 2)
        cv2.putText(frame, obj.get('cor', ''), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor, 2)
    return frame

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Lê o arquivo de imagem
    img_array = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    if image is None:
        return jsonify({'error': 'Formato de imagem inválido'}), 400
    
    # Detecta objetos na imagem
    objects = detector.detect_target_objects(image)
    
    return jsonify(objects)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Não foi possível acessar a webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        objetos = detector.detect_target_objects(frame)
        frame = desenhar_objetos(frame, objetos)

        cv2.imshow('Detecção ao Vivo', frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC para sair
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
    main()
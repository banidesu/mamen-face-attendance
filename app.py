from flask import Flask, request, jsonify, render_template
from ultralytics import YOLO
from datetime import datetime
import cv2
import numpy as np
import mysql.connector

app = Flask(__name__)
model = YOLO("best.pt")  # pastikan path benar

# Konfigurasi koneksi MySQL
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",            # ganti sesuai usermu
        password="",            # ganti sesuai passwordmu
        database="absensi_db"   # ganti sesuai nama database-mu
    )

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'frame' not in request.files:
        return jsonify({'error': 'No frame received'}), 400

    file = request.files['frame']
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({'error': 'Invalid image'}), 400

    results = model(img)
    detections = []

    for box in results[0].boxes.data.tolist():
        x1, y1, x2, y2, score, cls = box
        class_name = results[0].names[int(cls)]
        confidence = float(score) * 100
        detections.append({
            'bbox': [float(x1), float(y1), float(x2), float(y2)],
            'score': confidence,
            'class_id': int(cls),
            'class_name': class_name
        })

    return jsonify({'detections': detections})

@app.route('/absen', methods=['POST'])
def absen():
    data = request.get_json()

    if not data or 'nama' not in data or 'confidence' not in data:
        return jsonify({'error': 'Data tidak lengkap'}), 400

    class_name = data['nama']
    confidence = data['confidence']
    waktu = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        query_check = "SELECT id FROM absensi WHERE nama = %s AND DATE(waktu) = CURDATE()"
        cursor.execute(query_check, (class_name,))
        result = cursor.fetchone()

        if not result:
            query_insert = "INSERT INTO absensi (nama, confidence, waktu) VALUES (%s, %s, %s)"
            cursor.execute(query_insert, (class_name, confidence, waktu))
            conn.commit()
            status = {
                'status': 'success',
                'message': f'{class_name} berhasil absen hari ini.',
                'class_name': class_name,
                'confidence': confidence
            }
        else:
            status = {
                'status': 'info',
                'message': f'{class_name} sudah absen hari ini.',
                'class_name': class_name,
                'confidence': confidence
            }

        cursor.close()
        conn.close()
        return jsonify(status)

    except Exception as e:
        return jsonify({'error': f'Gagal menyimpan ke database: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)

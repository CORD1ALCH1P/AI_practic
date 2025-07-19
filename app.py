from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
from datetime import datetime
import sqlite3
import os
from ultralytics import YOLO

app = Flask(__name__)
model = YOLO("yolov8n.pt")  # Предобученная модель

if not os.path.exists('static'):
    os.makedirs('static')

# Инициализация БД
def init_db():
    conn = sqlite3.connect('history.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            result TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    file = request.files['image']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    results = model(img)
    output_img = results[0].plot()
    cv2.imwrite('static/result.jpg', output_img)

    boxes = results[0].boxes
    names = results[0].names
    tables = []
    people = []

    for box in boxes:
        cls_id = int(box.cls)
        label = names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if 'table' in label:
            tables.append(((x1, y1, x2, y2), False))
        elif 'person' in label:
            people.append((x1, y1, x2, y2))

    def is_near(box1, box2, threshold=100):
        x1 = (box1[0] + box1[2]) / 2
        y1 = (box1[1] + box1[3]) / 2
        x2 = (box2[0] + box2[2]) / 2
        y2 = (box2[1] + box2[3]) / 2
        return ((x2 - x1)**2 + (y2 - y1)**2) ** 0.5 < threshold

    for i, (table_box, _) in enumerate(tables):
        for person_box in people:
            if is_near(table_box, person_box):
                tables[i] = (table_box, True)
                break

    occupied = sum(1 for _, status in tables if status)
    free = len(tables) - occupied

    conn = sqlite3.connect('history.db')
    cursor = conn.cursor()
    cursor.execute('INSERT INTO requests (timestamp, result) VALUES (?, ?)',
                   (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    f'Занято: {occupied}, Свободно: {free}'))
    conn.commit()
    conn.close()

    return jsonify(occupied=occupied, free=free, total=len(tables))

@app.route('/history')
def history():
    conn = sqlite3.connect('history.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM requests ORDER BY timestamp DESC')
    data = cursor.fetchall()
    conn.close()
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)

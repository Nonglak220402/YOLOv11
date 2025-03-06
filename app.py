from flask import Flask, request, jsonify
import torch
from PIL import Image
import io
from ultralytics import YOLO  

# โหลดโมเดล YOLOv1
MODEL_PATH = r"C:\Users\ADVICE_003\Downloads\best (0.9).pt"
model = YOLO(MODEL_PATH)  # โหลดโมเดลจากไฟล์ .pt

# สร้างแอป Flask
app = Flask(__name__)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # ตรวจสอบว่ามีไฟล์รูปใน request หรือไม่
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No file selected for uploading'}), 400

        # โหลดและแปลงรูปภาพ
        image = Image.open(io.BytesIO(file.read()))

        # ใช้ YOLO ทำการพยากรณ์
        confidence_threshold = request.args.get('confidence', 0.7, type=float)
        nms_threshold = request.args.get('nms', 0.5, type=float)
        results = model(image, conf=confidence_threshold, iou=nms_threshold)

        # ตรวจสอบว่า `results` มีผลลัพธ์ที่ต้องการหรือไม่
        if len(results) == 0 or not hasattr(results[0], 'boxes') or results[0].boxes is None:
            return jsonify({'predictions': 'No Data'})  # Return 'No Data' if no detections

        # ดึงข้อมูลจากผลลัพธ์
        predictions = []
        for box in results[0].boxes:
            if float(box.conf) < 0.5:  # ตรวจสอบค่า confidence
                continue

            coords = box.xyxy.tolist()  # แปลง tensor เป็น list
            predictions.append({
                'class': results[0].names[int(box.cls)],  # แปลง class ID เป็นชื่อ class
                'confidence': float(box.conf),  # Confidence score
                'coordinates': {
                    'x_min': coords[0][0],  # Top-left x
                    'y_min': coords[0][1],  # Top-left y
                    'x_max': coords[0][2],  # Bottom-right x
                    'y_max': coords[0][3],  # Bottom-right y
                }
            })

        # ถ้าไม่มี prediction ที่มี confidence >= 0.5
        if not predictions:
            return jsonify({'predictions': 'No Data'})

        return jsonify({'predictions': predictions})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

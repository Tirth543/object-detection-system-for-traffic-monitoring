from flask import Flask, render_template, request, jsonify, Response
from ultralytics import YOLO
import os
import cv2
import numpy as np
import math
from werkzeug.utils import secure_filename
from collections import Counter
import torch
import time

app = Flask(__name__)

os.makedirs('static/uploads', exist_ok=True)
os.makedirs('static/results', exist_ok=True)

# ---------------------------------------------------------
# CORE AI ENGINE (YOLOV8m Model)
# ---------------------------------------------------------
model = YOLO('best.pt') 
TARGET_DEVICE = 0 if torch.cuda.is_available() else 'cpu'
#
TARGET_CLASSES = [0, 1, 2, 5]  

session_stats = {}

# --- MATHEMATICAL ALGORITHMS ---

def calculate_iou(box1, box2):
    x_left   = max(box1[0], box2[0])
    y_top    = max(box1[1], box2[1])
    x_right  = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    denominator = float(box1_area + box2_area - intersection_area)
    if denominator == 0:
        return 0.0
    return intersection_area / denominator


def get_max_cluster_size(boxes, threshold_distance):
    if not boxes:
        return 0

    centroids = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in boxes]
    n = len(centroids)
    parent = list(range(n))

    def find(i):
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]

    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j

    for i in range(n):
        for j in range(i + 1, n):
            dist = math.hypot(
                centroids[i][0] - centroids[j][0],
                centroids[i][1] - centroids[j][1]
            )
            if dist < threshold_distance:
                union(i, j)

    counts = {}
    for i in range(n):
        root = find(i)
        counts[root] = counts.get(root, 0) + 1

    return max(counts.values()) if counts else 0


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400

    file = request.files['file']
    filename = secure_filename(file.filename)
    filepath = os.path.join('static/uploads', filename)
    file.save(filepath)

    session_stats[filename] = {'total': 0, 'current_load': 0, 'counts': {}, 'status': 'Initializing...'}

    ext = filename.rsplit('.', 1)[1].lower()
    is_video = ext in ['mp4', 'avi', 'mov', 'mkv', 'webm']

    if not is_video:
        orig_img = cv2.imread(filepath)

        model.predict(
            source=orig_img, device=TARGET_DEVICE, imgsz=1280,
            conf=0.55, iou=0.45, classes=TARGET_CLASSES, verbose=False 
        )
        results = model.predict(
            source=orig_img,
            device=TARGET_DEVICE,
            imgsz=1280,
            conf=0.55, 
            iou=0.45,
            classes=TARGET_CLASSES,
            save=False,
            verbose=False
        )

        classes_detected = []
        valid_boxes_for_clustering = []

        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.int().cpu().tolist()
            c_ids = results[0].boxes.cls.int().cpu().tolist()

            for box, cid in zip(boxes, c_ids):
                raw_name = model.names[cid]
                
                #------------------------------------------
                if raw_name.lower() in ["truck", "mini-truck"]:
                    display_name = "Car"
                else:
                    display_name = raw_name.capitalize()
                # ------------------------------------------

                classes_detected.append(display_name)
                valid_boxes_for_clustering.append(box)

                x1, y1, x2, y2 = box
                color = (0, 255, 0)
                thickness = max(2, int(orig_img.shape[0] / 400))
                font_scale = max(0.6, orig_img.shape[0] / 800)

                cv2.rectangle(orig_img, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(
                    orig_img, display_name,
                    (x1, max(15, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, thickness
                )

        display_img = cv2.resize(
            orig_img, (800, int(800 * orig_img.shape[0] / orig_img.shape[1]))
        )
        res_path = os.path.join('static/results', 'res_' + filename)
        cv2.imwrite(res_path, display_img)

        total_objects = len(classes_detected)
        cluster_distance_threshold = orig_img.shape[1] * 0.20
        max_jam_size = get_max_cluster_size(valid_boxes_for_clustering, cluster_distance_threshold)

        if total_objects >= 20 or max_jam_size >= 15:
            status = "Heavy Traffic 🔴"
        elif total_objects >= 12 or max_jam_size >= 8:
            status = "Medium Traffic 🟡"
        else:
            status = "Normal Flow 🟢"

        session_stats[filename].update({
            'total': total_objects,
            'current_load': total_objects,
            'counts': dict(Counter(classes_detected)),
            'status': status
        })

        return jsonify({
            'media_url': '/' + res_path,
            'is_video': False,
            'total': total_objects,
            'current_load': total_objects,
            'counts': dict(Counter(classes_detected)),
            'status': status,
            'filename': filename
        })

    else:
        return jsonify({
            'media_url': f'/video_stream/{filename}',
            'is_video': True,
            'filename': filename
        })


def generate_video_frames(filepath):
    filename = os.path.basename(filepath)
    cap = cv2.VideoCapture(filepath)

    unique_ids = set()
    class_counts = {}
    id_lifespan = {}
    recent_boxes_memory = {}
    frame_number = 0

    ret, test_frame = cap.read()
    if not ret:
        return

    frame_area = test_frame.shape[0] * test_frame.shape[1]
    cluster_distance_threshold = test_frame.shape[1] * 0.20
    MIN_AREA_THRESHOLD = frame_area * 0.0005

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1

        if frame_number % 100 == 0:
            recent_boxes_memory = {k: v for k, v in recent_boxes_memory.items() if (frame_number - v['frame']) < 100}

        results = model.track(
            source=frame,
            persist=True,
            tracker="botsort.yaml",
            device=TARGET_DEVICE,
            imgsz=1280,
            conf=0.55, 
            iou=0.45,
            classes=TARGET_CLASSES,
            verbose=False
        )

        current_frame_load = 0
        active_boxes_for_clustering = []

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.int().cpu().tolist()
            t_ids = results[0].boxes.id.int().cpu().tolist()
            c_ids = results[0].boxes.cls.int().cpu().tolist()

            for box, tid, cid in zip(boxes, t_ids, c_ids):
                x1, y1, x2, y2 = box
                box_area = (x2 - x1) * (y2 - y1)

                if box_area > MIN_AREA_THRESHOLD:
                    raw_name = model.names[cid]
                    
                    # --- BULLETPROOF CASE-INSENSITIVE LOGIC ---
                    if raw_name.lower() in ["truck", "mini-truck"]:
                        display_name = "Car"
                    else:
                        display_name = raw_name.capitalize()
                    # ------------------------------------------

                    id_lifespan[tid] = id_lifespan.get(tid, 0) + 1

                    if id_lifespan[tid] >= 5 and tid not in unique_ids:
                        unique_ids.add(tid)

                        is_duplicate = False
                        for old_tid, old_data in recent_boxes_memory.items():
                            if old_tid != tid and old_data['cls'] == display_name:
                                if (frame_number - old_data['frame']) < 90:
                                    overlap = calculate_iou(box, old_data['box'])
                                    if overlap > 0.50:
                                        is_duplicate = True
                                        break

                        if not is_duplicate:
                            class_counts[display_name] = class_counts.get(display_name, 0) + 1

                    recent_boxes_memory[tid] = {
                        'box': box, 'cls': display_name, 'frame': frame_number
                    }

                    if id_lifespan[tid] >= 3:
                        current_frame_load += 1
                        active_boxes_for_clustering.append(box)

                        color = (0, 255, 0)
                        thickness = max(2, int(frame.shape[0] / 400))
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                        cv2.putText(
                            frame, display_name,
                            (x1, max(15, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, color, thickness
                        )

        total_valid_counts = sum(class_counts.values())

        max_jam_size = get_max_cluster_size(active_boxes_for_clustering, cluster_distance_threshold)

        if filename in session_stats:
            session_stats[filename]['current_load'] = current_frame_load
            session_stats[filename]['total'] = total_valid_counts
            session_stats[filename]['counts'] = class_counts

            if current_frame_load >= 20 or max_jam_size >= 15:
                session_stats[filename]['status'] = "Heavy Traffic 🔴"
            elif current_frame_load >= 12 or max_jam_size >= 8:
                session_stats[filename]['status'] = "Medium Traffic 🟡"
            else:
                session_stats[filename]['status'] = "Normal Flow 🟢"

        stream_frame = cv2.resize(
            frame, (800, int(800 * frame.shape[0] / frame.shape[1]))
        )
        ret2, buffer = cv2.imencode('.jpg', stream_frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()
    if filename in session_stats:
        session_stats[filename]['status'] = "Analysis Complete ✅"
        session_stats[filename]['current_load'] = 0


@app.route('/video_stream/<filename>')
def video_stream(filename):
    filepath = os.path.join('static/uploads', filename)
    return Response(
        generate_video_frames(filepath),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/live_stats')
def get_stats():
    filename = request.args.get('filename')
    if filename and filename in session_stats:
        return jsonify(session_stats[filename])
    
    if session_stats:
        latest_file = list(session_stats.keys())[-1]
        return jsonify(session_stats[latest_file])
        
    return jsonify({'total': 0, 'current_load': 0, 'counts': {}, 'status': 'Standby'})


if __name__ == '__main__':
    app.run(port=3000, debug=True)
import os
import csv
import time
import base64
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from utils import register_user, recognize

FRONTEND_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
app = Flask(__name__, template_folder=FRONTEND_FOLDER, static_folder=FRONTEND_FOLDER)
CORS(app)

ATTENDANCE_CSV = "attendance.csv"
if not os.path.exists(ATTENDANCE_CSV):
    with open(ATTENDANCE_CSV, mode="w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["timestamp", "name", "score"])

@app.route("/")
def root_index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/style.css")
def css_file():
    return send_from_directory(app.static_folder, "style.css")

@app.route("/madeby")
def madeby_page():
    return send_from_directory(app.static_folder, "madeby.html")

@app.route("/attendance_page")
def attendance_page():
    return send_from_directory(app.static_folder, "attendance.html")

@app.route("/home_page")
def home_page():
    return send_from_directory(app.static_folder, "home.html")

@app.route("/register_page")
def register_page():
    return send_from_directory(app.static_folder, "register_camera.html")

@app.route("/register_upload_page")
def register_upload_page():
    return send_from_directory(app.static_folder, "register_upload.html")

@app.route("/admin_page")
def admin_page():
    return send_from_directory(app.static_folder, "admin.html")

@app.route("/register_camera", methods=["POST"])
def api_register_camera():
    data = request.get_json()
    if not data or "nim" not in data or "name" not in data or "image" not in data:
        return jsonify({"success": False, "message": "Data tidak lengkap"}), 400
    
    full_identity = f"{data['nim'].strip()}_{data['name'].strip()}"
    try:
        image_base64 = data["image"].split(",")[1] if "," in data["image"] else data["image"]
        img_bytes = base64.b64decode(image_base64)
    except:
        return jsonify({"success": False, "message": "Base64 Error"}), 400
        
    ok, msg = register_user(full_identity, img_bytes)
    return jsonify({"success": ok, "message": msg})

@app.route("/register", methods=["POST"])
def api_register():
    if "nim" not in request.form or "name" not in request.form or "photo" not in request.files:
        return jsonify({"success": False, "message": "Data kurang"}), 400
    full_identity = f"{request.form['nim'].strip()}_{request.form['name'].strip()}"
    ok, msg = register_user(full_identity, request.files["photo"].read())
    return jsonify({"success": ok, "message": msg})

@app.route("/attendance", methods=["POST"])
def api_attendance():
    file = request.files.get("photo") or request.files.get("image")
    if not file: return jsonify({"success": False, "message": "No Image"}), 400
    
    name, score, msg = recognize(file.read())
    if name:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(ATTENDANCE_CSV, mode="a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([timestamp, name, f"{score:.4f}"])
        return jsonify({"success": True, "name": name, "score": score, "message": msg})
    else:
        return jsonify({"success": False, "name": None, "score": score, "message": msg})

@app.route("/attendance_log", methods=["GET"])
def attendance_log():
    logs = []
    if os.path.exists(ATTENDANCE_CSV):
        with open(ATTENDANCE_CSV, newline="", encoding="utf-8") as f:
            logs = list(csv.DictReader(f))
    return jsonify({"logs": logs})

@app.route("/check_user", methods=["POST"])
def check_user_exists():
    data = request.get_json()
    if not data or "nim" not in data:
        return jsonify({"exists": False}), 400
        
    target_nim = data["nim"].strip()
    faces_dir = "faces" 
    
    if os.path.exists(faces_dir):
        for fname in os.listdir(faces_dir):
            if fname.endswith(".npy"):
                parts = fname.split('_')
                existing_nim = parts[0]
                if existing_nim == target_nim:
                    existing_name = parts[1].replace(".npy", "") if len(parts) > 1 else "User"
                    return jsonify({
                        "exists": True, 
                        "old_name": existing_name
                    })
                    
    return jsonify({"exists": False})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
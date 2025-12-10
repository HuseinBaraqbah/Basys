import os
import numpy as np
from PIL import Image
import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mtcnn = MTCNN(keep_all=False, device=DEVICE)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

FACES_DIR = "faces"
IMG_DIR = os.path.join(FACES_DIR, "images")
os.makedirs(FACES_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

def image_bytes_to_cv2(img_bytes):
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def detect_and_crop(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    face = mtcnn(pil_img)
    if face is None:
        return None
    face_np = (face.permute(1,2,0).mul(255).byte().cpu().numpy())
    return Image.fromarray(face_np)

def get_embedding_from_pil(face_pil):
    face_arr = np.asarray(face_pil).astype(np.float32)
    face_arr = (face_arr / 255.0 - 0.5) / 0.5
    face_arr = np.transpose(face_arr, (2,0,1))
    tensor = torch.tensor(face_arr, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = resnet(tensor)
    emb_np = emb.cpu().numpy().reshape(-1)
    return emb_np / np.linalg.norm(emb_np)

def register_user(full_identity, img_bytes):
    """
    full_identity format: "NIM_Nama"
    Fitur: Hapus data lama jika NIM sama, lalu simpan yang baru.
    """
    try:
        img = image_bytes_to_cv2(img_bytes)
    except Exception as e:
        return False, f"Error gambar: {e}"

    face_pil = detect_and_crop(img)
    if face_pil is None:
        return False, "Wajah tidak terdeteksi!"

    emb = get_embedding_from_pil(face_pil)
    
    nim_baru = full_identity.split('_')[0]
    status_msg = "Berhasil Didaftarkan"

    for fname in os.listdir(FACES_DIR):
        if fname.endswith(".npy"):
            nim_lama = fname.split('_')[0]
            if nim_lama == nim_baru:
                try:
                    os.remove(os.path.join(FACES_DIR, fname))
                    old_img = os.path.join(IMG_DIR, fname.replace(".npy", ".jpg"))
                    if os.path.exists(old_img):
                        os.remove(old_img)
                except:
                    pass
                status_msg = "Data Wajah Diperbarui"
                break

    np.save(os.path.join(FACES_DIR, f"{full_identity}.npy"), emb)
    face_pil.save(os.path.join(IMG_DIR, f"{full_identity}.jpg"))
    
    return True, status_msg

def recognize(img_bytes, threshold=0.65):
    try:
        img = image_bytes_to_cv2(img_bytes)
    except:
        return None, None, "Gambar rusak"
        
    face_pil = detect_and_crop(img)
    if face_pil is None:
        return None, None, "Wajah tidak terdeteksi"

    emb = get_embedding_from_pil(face_pil)
    best_name, best_score = None, -1.0
    
    for fname in os.listdir(FACES_DIR):
        if not fname.endswith(".npy"): continue
        user_emb = np.load(os.path.join(FACES_DIR, fname))
        score = float(np.dot(emb, user_emb) / (np.linalg.norm(emb) * np.linalg.norm(user_emb)))
        if score > best_score:
            best_score = score
            best_name = fname[:-4]

    if best_score >= threshold:
        return best_name, best_score, "Matched"
    return None, best_score, "Unknown"
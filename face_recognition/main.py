import cv2
from deepface import DeepFace
import os
import shutil
import numpy as np

# === Configuration ===
video_path = r"C:\Users\Hp\OneDrive\Desktop\AI\videos\WhatsApp Video 2025-07-03 at 23.42.10_092aaead.mp4"
suspect_path = r"C:\Users\Hp\OneDrive\Desktop\AI\known\suspect.jpg"
detector_backend = "opencv"           # Faster than retinaface
model_name = "VGG-Face"               # Lightweight model
frame_skip = 5                        # Process every 5th frame
distance_threshold = 0.4              # Adjust if too strict/loose

# === Output folder ===
matched_folder = "matched_faces"
os.makedirs(matched_folder, exist_ok=True)

# === Load suspect embedding ===
print("[INFO] Loading suspect face embedding...")
suspect_embedding = DeepFace.represent(
    img_path=suspect_path,
    model_name=model_name,
    detector_backend=detector_backend,
    enforce_detection=True
)[0]["embedding"]

# === Process CCTV video ===
print("[INFO] Starting video analysis...")
cap = cv2.VideoCapture(video_path)
frame_id = 0
match_id = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    if frame_id % frame_skip != 0:
        continue

    try:
        # Detect faces in current frame
        faces = DeepFace.extract_faces(
            img_path=frame,
            detector_backend=detector_backend,
            enforce_detection=False
        )

        for face in faces:
            facial_area = face["facial_area"]
            face_img = face["face"]

            if face_img is None or face_img.size == 0:
                continue

            # Get embedding of detected face
            embedding = DeepFace.represent(
                img_path=face_img,
                model_name=model_name,
                detector_backend=detector_backend,
                enforce_detection=False
            )[0]["embedding"]

            # Compute cosine distance
            distance = np.dot(suspect_embedding, embedding) / (np.linalg.norm(suspect_embedding) * np.linalg.norm(embedding))

            # Cosine similarity: closer to 1 means similar
            if distance > (1 - distance_threshold):
                # Draw bounding box & label
                x, y, w, h = facial_area["x"], facial_area["y"], facial_area["w"], facial_area["h"]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"MATCH: {distance:.2f}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Save the matched face
                face_filename = os.path.join(matched_folder, f"match_{match_id}.jpg")
                cv2.imwrite(face_filename, face_img)
                match_id += 1

    except Exception as e:
        print(f"[ERROR] Frame {frame_id}: {e}")
        continue

    # Display annotated frame
    cv2.imshow("CCTV Face Matching", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print(f"\n[INFO] Completed. Matches saved in: {matched_folder}")

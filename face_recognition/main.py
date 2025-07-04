import cv2, os, numpy as np, csv, smtplib, mimetypes
from deepface import DeepFace
from email.message import EmailMessage
from email.utils import formatdate

# === Configuration ===
video_path = r"C:\Users\Hp\OneDrive\Desktop\AI\videos\WhatsApp Video 2025-07-03 at 23.42.10_092aaead.mp4"
known_folder = r"C:\Users\Hp\OneDrive\Desktop\AI\known"
matched_folder = "matched_faces"
os.makedirs(matched_folder, exist_ok=True)

detector_backend = "opencv"         # Accurate & fast
model_name = "VGG-Face"             # Accurate (but slower)
frame_skip = 5                      # Moderate speed
distance_threshold = 0.4            # Match threshold

# === Email Setup ===
def send_match_email(to_email, subject, body, attachment_path):
    sender_email = "priyamani5122005@gmail.com"
    sender_password = "vwet gekw itvu ozzl"  # Gmail app password (keep secret)

    msg = EmailMessage()
    msg["From"] = sender_email
    msg["To"] = to_email
    msg["Subject"] = subject
    msg["Date"] = formatdate(localtime=True)
    msg.set_content(body)

    mime_type, _ = mimetypes.guess_type(attachment_path)
    maintype, subtype = mime_type.split('/')
    with open(attachment_path, 'rb') as f:
        msg.add_attachment(f.read(), maintype=maintype, subtype=subtype,
                           filename=os.path.basename(attachment_path))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(sender_email, sender_password)
        smtp.send_message(msg)
        print(f"[EMAIL] Alert sent to {to_email}")

# === Load suspect embeddings ===
print("[INFO] Loading suspect embeddings...")
suspects = []
for file in os.listdir(known_folder):
    if file.lower().endswith((".jpg", ".jpeg", ".png")):
        path = os.path.join(known_folder, file)
        name = os.path.splitext(file)[0].replace("_", " ").title()
        embedding = DeepFace.represent(img_path=path, model_name=model_name,
                                       detector_backend=detector_backend, enforce_detection=True)[0]["embedding"]
        suspects.append((name, embedding))
        print(f"[INFO] Loaded: {name}")
print(f"[INFO] Total suspects loaded: {len(suspects)}")

# === Logging Setup ===
log_file = open("match_log.csv", mode="w", newline="")
csv_writer = csv.writer(log_file)
csv_writer.writerow(["Match_ID", "Frame_ID", "Timestamp", "Suspect_Name", "Similarity"])

# === Process Video ===
cap = cv2.VideoCapture(video_path)
frame_id = 0
match_id = 0
email_sent = set()

print("[INFO] Starting video analysis...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    if frame_id % frame_skip != 0:
        continue

    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # in seconds

    try:
        faces = DeepFace.extract_faces(frame, detector_backend=detector_backend, enforce_detection=False)

        for face in faces:
            face_img = face["face"]
            if face_img is None or face_img.size == 0:
                continue

            if face_img.dtype != "uint8":
                face_img = (face_img * 255).astype("uint8")

            face_embedding = DeepFace.represent(img_path=face_img, model_name=model_name,
                                                detector_backend=detector_backend, enforce_detection=False)[0]["embedding"]

            for name, suspect_embedding in suspects:
                similarity = np.dot(suspect_embedding, face_embedding) / (
                    np.linalg.norm(suspect_embedding) * np.linalg.norm(face_embedding))

                if similarity > (1 - distance_threshold) and name not in email_sent:
                    facial_area = face["facial_area"]
                    x, y, w, h = facial_area["x"], facial_area["y"], facial_area["w"], facial_area["h"]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    cv2.putText(frame, f"{name} ({similarity:.2f})", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    face_path = os.path.join(matched_folder, f"match_{match_id}_{name}.jpg")
                    cv2.imwrite(face_path, face_img)

                    csv_writer.writerow([match_id, frame_id, f"{timestamp:.2f}", name, f"{similarity:.2f}"])

                    # Send email
                    body = f"""
🚨 Suspect Match Detected!

🧍 Name: {name}
⏱️ Time: {timestamp:.2f}s
📈 Similarity: {similarity:.2f}
📸 Frame ID: {frame_id}
                    """.strip()

                    send_match_email("priyamani5122005@gmail.com", f"ALERT: Match Found - {name}", body, face_path)

                    email_sent.add(name)
                    match_id += 1

    except Exception as e:
        print(f"[ERROR] Frame {frame_id}: {e}")
        continue

    cv2.imshow("CCTV Face Matching", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
log_file.close()
print("[INFO] Done. Matches saved in:", matched_folder)

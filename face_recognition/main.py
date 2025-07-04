import cv2
from deepface import DeepFace
import os
import numpy as np
import csv
import smtplib
from email.message import EmailMessage
from email.utils import formatdate
import mimetypes

# === Configuration ===
video_path = r"C:\Users\Hp\OneDrive\Desktop\AI\videos\WhatsApp Video 2025-07-03 at 23.42.10_092aaead.mp4"
suspect_path = r"C:\Users\Hp\OneDrive\Desktop\AI\known\actor vijay.jpg"
detector_backend = "opencv"
model_name = "VGG-Face"
frame_skip = 5
distance_threshold = 0.4

# === Output folders ===
matched_folder = "matched_faces"
os.makedirs(matched_folder, exist_ok=True)

# === Email Sender Function ===
def send_match_email(to_email, subject, body, attachment_path):
    sender_email = "priyamani5122005@gmail.com"
    sender_password = "vwet gekw itvu ozzl"  # Replace with your App Password

    msg = EmailMessage()
    msg['From'] = sender_email
    msg['To'] = to_email
    msg['Subject'] = subject
    msg['Date'] = formatdate(localtime=True)
    msg.set_content(body)

    mime_type, _ = mimetypes.guess_type(attachment_path)
    mime_type, mime_subtype = mime_type.split('/')
    with open(attachment_path, 'rb') as file:
        msg.add_attachment(file.read(), maintype=mime_type,
                           subtype=mime_subtype,
                           filename=os.path.basename(attachment_path))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(sender_email, sender_password)
        smtp.send_message(msg)
        print(f"[EMAIL] Alert sent to {to_email}")

# === Load suspect embedding ===
print("[INFO] Loading suspect face embedding...")
suspect_embedding = DeepFace.represent(
    img_path=suspect_path,
    model_name=model_name,
    detector_backend=detector_backend,
    enforce_detection=True
)[0]["embedding"]

# === Setup logging to CSV ===
log_file = open("match_log.csv", mode="w", newline="")
csv_writer = csv.writer(log_file)
csv_writer.writerow(["Match_ID", "Frame_ID", "Similarity", "Timestamp", "Person_Name"])

# === Process video ===
print("[INFO] Starting video analysis...")
cap = cv2.VideoCapture(video_path)
frame_id = 0
match_id = 0
email_sent = False
face_saved = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    if frame_id % frame_skip != 0:
        continue

    try:
        frame_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

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

            embedding = DeepFace.represent(
                img_path=face_img,
                model_name=model_name,
                detector_backend=detector_backend,
                enforce_detection=False
            )[0]["embedding"]

            distance = np.dot(suspect_embedding, embedding) / (
                np.linalg.norm(suspect_embedding) * np.linalg.norm(embedding))

            if distance > (1 - distance_threshold):
                x, y, w, h = facial_area["x"], facial_area["y"], facial_area["w"], facial_area["h"]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"MATCH: {distance:.2f}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Fix black image issue
                if face_img.dtype != "uint8":
                    face_img = (face_img * 255).astype("uint8")

                # Skip dark images
                if np.mean(face_img) < 10:
                    print(f"[SKIPPED] Face too dark at frame {frame_id}")
                    continue

                # Save and email only once
                if not face_saved:
                    face_filename = os.path.join(matched_folder, f"match_{match_id}.jpg")
                    frame_filename = os.path.join(matched_folder, f"frame_{match_id}.jpg")
                    cv2.imwrite(face_filename, face_img)
                    cv2.imwrite(frame_filename, frame)

                    person_name = os.path.splitext(os.path.basename(suspect_path))[0].replace("_", " ").title()

                    # Log match
                    csv_writer.writerow([match_id, frame_id, f"{distance:.2f}", f"{frame_time:.2f}", person_name])

                    # Send alert once
                    if not email_sent:
                        subject = f"ALERT: Suspect Match - {person_name}"
                        body = f"""
🚨 Face Match Detected!

📍 Suspect Name: {person_name}
🕒 Time: {frame_time:.2f}s
🎞️ Frame ID: {frame_id}
📌 Similarity Score: {distance:.2f}

Check attached matched face.
                        """.strip()

                        send_match_email(
                            to_email="priyamani5122005@gmail.com",
                            subject=subject,
                            body=body,
                            attachment_path=face_filename
                        )
                        email_sent = True

                    face_saved = True
                    match_id += 1

        # Timestamp overlay
        cv2.putText(frame, f"Time: {frame_time:.2f}s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    except Exception as e:
        print(f"[ERROR] Frame {frame_id}: {e}")
        continue

    cv2.imshow("CCTV Face Matching", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
log_file.close()
print(f"\n[INFO] Completed. Matches saved in: {matched_folder}")

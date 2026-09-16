import cv2
import time
from tracker import CentroidTracker
from analytics import RetailMetrics, CsvLogger
from deepface import DeepFace

# OpenCVに同梱されているHaar Cascadeを利用するため、
# モデルファイルをリポジトリに含める必要はありません。
face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(face_cascade_path)


def detect_faces(img, face_cascade_classifier):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces_xywh = face_cascade_classifier.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=8, minSize=(80, 80)
    )
    results = []
    for (x, y, w, h) in faces_xywh:
        results.append((x, y, x + w, y + h))
    return results


tracker = CentroidTracker()
metrics = RetailMetrics()
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError(
        "カメラを開けませんでした。Webカメラが接続されているか確認してください。"
    )

frame_counter = 0

with CsvLogger("logs") as logger:
    print("[q] キーで終了します")
    print("初回起動時はDeepFaceのモデルが自動ダウンロードされます。")
    print("初回のみ時間がかかる場合があります。")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("カメラから映像を取得できませんでした。")
            break

        faces = detect_faces(frame, face_cascade)
        objects, deregistered_ids = tracker.update(faces)

        for obj_id in deregistered_ids:
            summary = metrics.get_person_summary(obj_id)
            if summary:
                logger.log(summary)
            metrics.finalize_person(obj_id)

        for obj_id, (x1, y1, x2, y2) in objects.items():
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # 人物ごとに5フレームに1回DeepFace推論を実行
            if obj_id % 5 == frame_counter % 5:
                try:
                    analysis_results = DeepFace.analyze(
                        crop,
                        actions=["age", "gender", "emotion"],
                        enforce_detection=False,
                        detector_backend="skip",
                    )

                    if analysis_results:
                        result = analysis_results[0]
                        age = result["age"]
                        gender = result["dominant_gender"]
                        emotion = result["dominant_emotion"]
                        metrics.update(obj_id, emotion, age, gender)

                except Exception:
                    # 一時的な推論失敗では映像処理を停止しない
                    pass

            stable_attrs = metrics.get_current_stable_attributes(obj_id)
            stable_age = stable_attrs.get("age", "?")
            stable_gender = stable_attrs.get("gender", "?")
            stable_expr = stable_attrs.get("expression", "?")

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"ID:{obj_id} {stable_age} {stable_gender} {stable_expr}"

            y_pos = y1 + 15 if y1 < 30 else y1 - 10
            cv2.putText(
                frame,
                label,
                (x1, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        frame_counter += 1
        cv2.imshow("Retail Analytics", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
print("CSV 保存先:", logger.path)

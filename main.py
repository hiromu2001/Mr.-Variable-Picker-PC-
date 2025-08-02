import cv2
import time
from tracker import CentroidTracker
from analytics import RetailMetrics, CsvLogger
from deepface import DeepFace

face_cascade_path = 'models/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(face_cascade_path)

def detect_faces(img, face_cascade_classifier):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 顔検出の感度を少し上げるためminNeighborsを8に調整
    faces_xywh = face_cascade_classifier.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=8, minSize=(80, 80))
    results = []
    for (x, y, w, h) in faces_xywh:
        results.append((x, y, x + w, y + h))
    return results

tracker  = CentroidTracker()
metrics  = RetailMetrics()
cap      = cv2.VideoCapture(0)

# フレームカウンター
frame_counter = 0

with CsvLogger('logs') as logger:
    print('[q] キーで終了します')
    print('初回起動時はモデルのダウンロードが自動で始まります。しばらくお待ちください...')
    while True:
        ret, frame = cap.read()
        if not ret: break

        # 顔の位置を検出・追跡
        faces = detect_faces(frame, face_cascade)
        objects, deregistered_ids = tracker.update(faces)

        # 画面から消えた人を記録
        for obj_id in deregistered_ids:
            summary = metrics.get_person_summary(obj_id)
            if summary:
                logger.log(summary)
            metrics.finalize_person(obj_id)

        # 画面にいる人を処理
        for obj_id, (x1, y1, x2, y2) in objects.items():
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0: continue

            if obj_id % 5 == frame_counter % 5:
                try:
                    # 検出した顔画像(crop)を直接DeepFaceに渡して分析
                    analysis_results = DeepFace.analyze(
                        crop, 
                        actions=['age', 'gender', 'emotion'],
                        enforce_detection=False,
                        detector_backend='skip' # 顔の再検出をスキップして高速化
                    )
                    
                    if analysis_results and len(analysis_results) > 0:
                        result = analysis_results[0]
                        age = result['age']
                        gender = result['dominant_gender'] # Man / Woman
                        emotion = result['dominant_emotion']
                        
                        metrics.update(obj_id, emotion, age, gender)

                except Exception as e:
                    pass # 分析失敗時はスキップ

            stable_attrs = metrics.get_current_stable_attributes(obj_id)
            stable_age = stable_attrs.get("age", "?")
            stable_gender = stable_attrs.get("gender", "?")
            stable_expr = stable_attrs.get("expression", "?")

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f'ID:{obj_id} {stable_age} {stable_gender} {stable_expr}'
            
            if y1 < 30:
                y_pos = y1 + 15
            else:
                y_pos = y1 - 10
            cv2.putText(frame, label, (x1, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        frame_counter += 1
        cv2.imshow('Retail Analytics', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print('CSV 保存先:', logger.path)

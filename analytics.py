import csv
import time
from collections import deque, Counter
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

DWELL_THRESHOLD = 2.0

class RetailMetrics:
    def __init__(self):
        self.face_records = {} 

    def update(self, obj_id: int, expression: str, age: int, gender: str):
        if obj_id not in self.face_records:
            self.face_records[obj_id] = deque(maxlen=150)
        
        # 性別表記を 'Male'/'Female' に統一
        gender = gender.replace("Man", "Male").replace("Woman", "Female")
        self.face_records[obj_id].append((time.time(), expression, age, gender))

    def get_person_summary(self, obj_id: int) -> Dict[str, Any]:
        records = self.face_records.get(obj_id)
        if not records:
            return {}

        timestamps, expressions, ages, genders = zip(*records)
        dwell_sec = timestamps[-1] - timestamps[0]
        result = 'stay' if dwell_sec >= DWELL_THRESHOLD else 'pass'
        top_expression = Counter(expressions).most_common(1)[0][0]
        top_gender = Counter(genders).most_common(1)[0][0]
        stable_age = int(np.median(ages))

        return {
            "gender": top_gender,
            "age": stable_age,
            "expression": top_expression,
            "result": result,
            "dwell_sec": dwell_sec
        }
    
    def get_current_stable_attributes(self, obj_id: int) -> Dict[str, Any]:
        records = self.face_records.get(obj_id)
        if not records or len(records) < 1:
            return {"age": "?", "gender": "?", "expression": "?"}
        
        recent_records = list(records)[-5:]
        _, expressions, ages, genders = zip(*recent_records)

        return {
            "age": int(np.median(ages)),
            "gender": Counter(genders).most_common(1)[0][0],
            "expression": Counter(expressions).most_common(1)[0][0]
        }

    def finalize_person(self, obj_id: int):
        if obj_id in self.face_records:
            del self.face_records[obj_id]

class CsvLogger:
    def __init__(self, out_dir='logs', flush_interval=10):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        ts = time.strftime('%Y%m%d_%H%M')
        self.path = Path(out_dir) / f'analytics_{ts}.csv'
        self.file = open(self.path, 'w', newline='', encoding='utf-8')
        self.writer = csv.writer(self.file)
        self.writer.writerow(
            ['end_timestamp', 'gender', 'age_stable', 'top_expression', 'result', 'total_dwell_sec'])
        
        self.flush_interval = flush_interval
        self.write_count = 0

    def log(self, summary: Dict):
        self.writer.writerow([
            time.strftime('%Y-%m-%d %H:%M:%S'),
            summary.get("gender", "N/A"),
            summary.get("age", "N/A"),
            summary.get("expression", "N/A"),
            summary.get("result", "N/A"),
            f'{summary.get("dwell_sec", 0):.2f}'
        ])
        
        self.write_count += 1
        if self.write_count % self.flush_interval == 0:
            self.file.flush()

    def close(self):
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

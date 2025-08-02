import time
from collections import OrderedDict
from scipy.spatial import distance as dist
import numpy as np

class CentroidTracker:
    def __init__(self, max_disappeared: int = 30):
        self.next_id = 0
        self.objects = OrderedDict()
        self.bboxes = OrderedDict()
        self.disappeared = OrderedDict()
        self.timestamps = OrderedDict()
        self.max_disappeared = max_disappeared

    def register(self, centroid, bbox):
        self.objects[self.next_id] = centroid
        self.bboxes[self.next_id] = bbox
        self.disappeared[self.next_id] = 0
        self.timestamps[self.next_id] = time.time()
        self.next_id += 1

    def deregister(self, obj_id):
        del self.objects[obj_id]
        del self.bboxes[obj_id]
        del self.disappeared[obj_id]
        del self.timestamps[obj_id]

    def update(self, rects):
        deregistered_ids = []  # このフレームで消えたIDを記録するリスト

        if len(rects) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    deregistered_ids.append(obj_id)
                    self.deregister(obj_id)
            return self.bboxes, deregistered_ids

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            input_centroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i], rects[i])
        else:
            obj_ids = list(self.objects.keys())
            obj_centroids = list(self.objects.values())
            D = dist.cdist(np.array(obj_centroids), input_centroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            used_rows, used_cols = set(), set()
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                
                obj_id = obj_ids[row]
                self.objects[obj_id] = input_centroids[col]
                self.bboxes[obj_id] = rects[col]
                self.disappeared[obj_id] = 0
                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            if D.shape[0] >= D.shape[1]:
                for row in unused_rows:
                    obj_id = obj_ids[row]
                    self.disappeared[obj_id] += 1
                    if self.disappeared[obj_id] > self.max_disappeared:
                        deregistered_ids.append(obj_id)
                        self.deregister(obj_id)
            else:
                for col in unused_cols:
                    self.register(input_centroids[col], rects[col])

        return self.bboxes, deregistered_ids

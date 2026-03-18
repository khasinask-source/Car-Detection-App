from scipy.spatial import distance
import numpy as np

class Tracker:
    def __init__(self):
        self.center_points = {}
        self.id_count = 0

    def update(self, objects_rect):

        objects_bbs_ids = []

        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            same_object_detected = False

            for id, pt in self.center_points.items():
                dist = distance.euclidean((cx, cy), pt)

                if dist < 50:
                    self.center_points[id] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    break

            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        return objects_bbs_ids
from ultralytics import YOLO
import numpy as np
from typing import List, Dict, Tuple
import cv2
from sklearn.cluster import DBSCAN

class PoolDetector:
    def __init__(self, model_path: str = 'models/pool_keypoints_detection.pt', confidence_threshold: float = 0.4):
        """
        Initialize the PoolDetector with a YOLO model for detecting pool elements.
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.class_names = ['15-meter-mark', '5-meter-mark', 'wall']
        self.tracked_elements = {class_name: {} for class_name in self.class_names}
        self.next_id = {class_name: 0 for class_name in self.class_names}
        self.max_distance = 50

        # For coloring lines in draw_tracked_elements
        self.color_map = {
            'wall': (0, 0, 255),         # Red
            '5-meter-mark': (0, 255, 0), # Green
            '15-meter-mark': (255, 0, 0) # Blue
        }

    # ---------------------------------------------------------------------
    # (1) Detection + Tracking
    # ---------------------------------------------------------------------
    
    def _assign_tracks(self, current_detections: Dict[str, List[Tuple[float, float]]],
                       previous_elements: Dict[int, Tuple[float, float]]) -> Dict[int, Tuple[float, float]]:
        if not previous_elements:
            return {}
        assignments = {}
        used_prev_ids = set()
        for curr_pos in current_detections:
            min_dist = float('inf')
            best_id = None
            for prev_id, prev_pos in previous_elements.items():
                if prev_id in used_prev_ids:
                    continue
                dist = np.linalg.norm(np.array(curr_pos) - np.array(prev_pos))
                if dist < min_dist and dist < self.max_distance:
                    min_dist = dist
                    best_id = prev_id
            if best_id is not None:
                assignments[best_id] = curr_pos
                used_prev_ids.add(best_id)
        return assignments

    def detect_and_track(self, frame) -> Dict[str, Dict[int, Tuple[float, float]]]:
        results = self.model(frame, conf=self.confidence_threshold)[0]
        current_detections = {c: [] for c in self.class_names}
        
        for box in results.boxes:
            if box.conf.item() >= self.confidence_threshold:
                class_id = int(box.cls.item())
                class_name = self.class_names[class_id]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx = 0.5*(x1 + x2)
                cy = 0.5*(y1 + y2)
                current_detections[class_name].append((cx, cy))
        
        new_tracked = {c: {} for c in self.class_names}
        for c in self.class_names:
            assignments = self._assign_tracks(current_detections[c], self.tracked_elements[c])
            new_tracked[c].update(assignments)
            assigned_positions = set(assignments.values())
            for pos in current_detections[c]:
                if pos not in assigned_positions:
                    new_tracked[c][self.next_id[c]] = pos
                    self.next_id[c] += 1
        
        self.tracked_elements = new_tracked
        return self.tracked_elements

    def process_video(self, frames: List[np.ndarray]) -> List[Dict[str, Dict[int, Tuple[float, float]]]]:
        return [self.detect_and_track(f) for f in frames]

    # ---------------------------------------------------------------------
    # (2) DBSCAN Clustering + Line Fitting
    # ---------------------------------------------------------------------
    
    def _cluster_points(self, points: List[Tuple[float, float]], eps: float = 100) -> List[List[Tuple[float, float]]]:
        """
        DBSCAN to group points that belong to the same line.
        """
        if not points:
            return []
        arr = np.array(points)
        clustering = DBSCAN(eps=eps, min_samples=2).fit(arr)
        labels = clustering.labels_
        clusters = {}
        for p, lbl in zip(points, labels):
            if lbl == -1:
                continue
            clusters.setdefault(lbl, []).append(p)
        return list(clusters.values())

    def _fit_line(self, points: List[Tuple[float, float]]) -> Tuple[np.ndarray, np.ndarray]:
        arr = np.array(points)
        mean_pt = arr.mean(axis=0)
        centered = arr - mean_pt
        _, eigvecs = np.linalg.eigh(centered.T @ centered)
        direction = eigvecs[:, -1]
        return mean_pt, direction

    def _extend_line_to_frame_edges(self, point: np.ndarray, direction: np.ndarray,
                                    frame_w: int, frame_h: int) -> List[Tuple[int, int]]:
        direction = direction / np.linalg.norm(direction)
        ts = []
        if abs(direction[0]) > 1e-10:
            ts += [-point[0]/direction[0], (frame_w - point[0])/direction[0]]
        if abs(direction[1]) > 1e-10:
            ts += [-point[1]/direction[1], (frame_h - point[1])/direction[1]]
        
        intersections = []
        for t in ts:
            x, y = point + t*direction
            if 0 <= x <= frame_w and 0 <= y <= frame_h:
                intersections.append((int(x), int(y)))
        if len(intersections) >= 2:
            return sorted(intersections)[:2]
        return []

    # ---------------------------------------------------------------------
    # (3) SEQUENCE -> "LEFT vs RIGHT" LOGIC
    #
    # We'll define short names for YOLO classes: 
    #   "wall" => "wall",
    #   "5-meter-mark" => "5m",
    #   "15-meter-mark" => "15m"
    # Then we match them to the enumerated sequences you gave.
    # ---------------------------------------------------------------------
    
    def _short_name(self, cls_name: str) -> str:
        if cls_name == "15-meter-mark":
            return "15m"
        elif cls_name == "5-meter-mark":
            return "5m"
        elif cls_name == "wall":
            return "wall"
        else:
            return cls_name  # fallback

    def _build_pattern_map(self):
        """
        Returns a dictionary:
           pattern_map[("wall","5m")] = ("left wall","left 5m")
           pattern_map[("wall","5m","15m")] = ("left wall","left 5m","left 15m")
           ...
        covering all enumerations you gave.
        """
        pattern_map = {}

        # 1) wall-5m => left wall, left 5m
        pattern_map[("wall","5m")] = ("left wall", "left 5m")

        # 2) wall-5m-15m => left wall, left 5m, left 15
        pattern_map[("wall","5m","15m")] = ("left wall", "left 5m", "left 15m")

        # 3) wall-5m-15m-15m => left wall, left 5, left 15, right 15
        pattern_map[("wall","5m","15m","15m")] = ("left wall","left 5m","left 15m","right 15m")

        # 4) wall-5m-15m-15m-5m => left wall, left 5, left 15, right 15, right 5
        pattern_map[("wall","5m","15m","15m","5m")] = (
            "left wall","left 5m","left 15m","right 15m","right 5m"
        )

        # 5) wall-5m-15m-15m-5m-wall => left wall, left 5, left 15, right 15, right 5, right wall
        pattern_map[("wall","5m","15m","15m","5m","wall")] = (
            "left wall","left 5m","left 15m","right 15m","right 5m","right wall"
        )

        # 6) 5m-15m-15m-5m-wall => left 5, left 15, right 15, right 5, right wall
        pattern_map[("5m","15m","15m","5m","wall")] = (
            "left 5m","left 15m","right 15m","right 5m","right wall"
        )

        # 7) 15m-15m-5m-wall => left 15, right 15, right 5, right wall
        pattern_map[("15m","15m","5m","wall")] = (
            "left 15m","right 15m","right 5m","right wall"
        )

        # 8) 15m-5m-wall => right 15, right 5, right wall
        pattern_map[("15m","5m","wall")] = (
            "right 15m","right 5m","right wall"
        )

        # 9) 5m-wall => right 5, right wall
        pattern_map[("5m","wall")] = (
            "right 5m","right wall"
        )

        return pattern_map

    def interpret_sequence(self, classes_in_x_order: List[str]) -> List[str]:
        """
        Match the short-class sequence (like ["wall","5m","15m","15m"]) 
        to one of your enumerations. If no match, fallback with "???"
        """
        pattern_map = self._build_pattern_map()
        key_tuple = tuple(classes_in_x_order)
        
        if key_tuple in pattern_map:
            return list(pattern_map[key_tuple])  # return the side-labeled strings
        else:
            # fallback => label each with ??? 
            # plus original short name, e.g. "??? wall", "??? 5m", ...
            return [f"??? {cl}" for cl in classes_in_x_order]

    def get_side_labels(self, tracked_elements: Dict[str, Dict[int, Tuple[float, float]]],
                        frame_width: float) -> List[str]:
        """
        1) Gather all detections as (short_class, x), sorted left->right.
        2) Extract just the short_class sequence.
        3) interpret_sequence(...) to get side-labeled results.
        4) Return side-labeled results in the same left->right order.
        """
        # 1) Flatten + sort by x
        all_dets = []  # list of (x, short_class)
        for cls_name, track_dict in tracked_elements.items():
            for (cx, cy) in track_dict.values():
                all_dets.append((cx, self._short_name(cls_name)))
        
        if not all_dets:
            return []
        
        all_dets.sort(key=lambda x: x[0])  # sort by x ascending (left->right)

        # 2) Extract just the short_class in x order
        seq = [d[1] for d in all_dets]

        # 3) interpret_sequence => side-labeled (e.g. ["left wall","left 5m","left 15m","right 15m"])
        side_seq = self.interpret_sequence(seq)

        # 4) Return those side-labeled results in the same order
        #    We'll have the same length as seq
        return side_seq

    # ---------------------------------------------------------------------
    # (4) Final Drawing
    # ---------------------------------------------------------------------

    def visualize_side_sequence(self, side_seq: List[str]) -> np.ndarray:
        """
        Create a 600x80 mini-map with up to 6 lines from left to right.
        Each item in side_seq is something like "left wall" or "right 5m".
        We'll parse the side ("left"/"right") and the base class ("wall","5m","15m").
        
        Then we draw a vertical line for each item, colored by the base class,
        labeled with e.g. "wall(L)" or "5m(R)".
        """
        # 1) Create a white image
        width, height = 600, 80
        mini = np.ones((height, width, 3), dtype=np.uint8) * 255

        n = len(side_seq)  # how many lines we need to draw
        if n == 0:
            return mini  # blank if no lines

        # 2) We'll define positions along the x-axis for each line, spaced evenly
        #    For example, if n=4, we have positions at x=0, 1/3, 2/3, 1.0 of the width.
        #    We'll do [0..n-1], each mapped to [0..1].
        step_fraction = 1.0 / max(n-1, 1)  # avoid division by zero
        x_positions = [(int(step_fraction * i * (width - 1))) for i in range(n)]
        
        # 3) For each line in side_seq, parse the base class and side
        #    side_seq item is "left wall" or "right 15m", etc.
        for i, label in enumerate(side_seq):
            # label might be "left wall"
            parts = label.split()  # => ["left", "wall"]
            if len(parts) == 2:
                side_str, base_cls = parts
            else:
                # fallback if it doesn't match the pattern (like ??? 5m)
                side_str = "?"
                base_cls = label  # entire string

            # 4) YOLO color for "wall","5m","15m"
            if base_cls.endswith("wall"):
                color = (0, 0, 255)   # Red
                short_cls = "wall"
            elif base_cls.endswith("5m"):
                color = (0, 255, 0)   # Green
                short_cls = "5m"
            elif base_cls.endswith("15m"):
                color = (255, 0, 0)   # Blue
                short_cls = "15m"
            else:
                color = (150,150,150) # Gray fallback
                short_cls = base_cls

            # 5) The text we place might be "wall(L)" or "5m(R)"
            side_char = "(L)" if "left" in side_str else "(R)" if "right" in side_str else ""
            final_label = f"{short_cls}{side_char}"

            # 6) Draw a vertical line at x_positions[i]
            x = x_positions[i]
            cv2.line(mini, (x, 0), (x, height - 1), color, 2)
            
            # 7) Put the text near the middle
            #    We'll place it a bit below mid-height
            text_y = height // 2
            cv2.putText(mini, final_label, (x + 2, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return mini

    
    def draw_tracked_elements(self, frame, tracked_elements: Dict[str, Dict[int, Tuple[float, float]]]) -> np.ndarray:
        output_frame = frame.copy()
        frame_h, frame_w = output_frame.shape[:2]

        # 1) Draw DBSCAN lines as usual
        #    (unchanged code from your snippet)
        colors = {
            'wall': (0, 0, 255),  
            '5-meter-mark': (0, 255, 0),
            '15-meter-mark': (255, 0, 0)
        }
        for class_name, elements in tracked_elements.items():
            color = colors[class_name]
            points = list(elements.values())
            clusters = self._cluster_points(points)
            for cluster in clusters:
                if len(cluster) >= 2:
                    mean_pt, direction = self._fit_line(cluster)
                    line_pts = self._extend_line_to_frame_edges(mean_pt, direction, frame_w, frame_h)
                    if len(line_pts) == 2:
                        cv2.line(output_frame, line_pts[0], line_pts[1], color, 2)
                # Draw points
                for pt in cluster:
                    x, y = map(int, pt)
                    cv2.circle(output_frame, (x,y), 5, color, -1)
                    # find track ID
                    tid = next(k for k,v in elements.items() if v[0] == pt[0] and v[1] == pt[1])
                    cv2.putText(output_frame, f"{class_name}:{tid}", (x+10, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 2) Build the "side_seq" from your enumerations
        side_seq = self.get_side_labels(tracked_elements, frame_w)
        # e.g. ["left wall","left 5m","right 15m"] or whatever your enumerations produce

        # 3) Create the mini‐map image
        mini_map = self.visualize_side_sequence(side_seq)

        # 4) Overlay it top-left
        mh, mw = mini_map.shape[:2]
        if mh < frame_h and mw < frame_w:
            output_frame[10:10+mh, 10:10+mw] = mini_map

        return output_frame


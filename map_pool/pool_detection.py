from ultralytics import YOLO
import numpy as np
from typing import List, Dict, Tuple
import cv2
from sklearn.cluster import DBSCAN

class PoolDetector:
    def __init__(self, model_path: str = 'models/pool_keypoints_detection.pt', confidence_threshold: float = 0.4):
        """
        Initialize the PoolDetector with a YOLO model for detecting pool elements.
        
        Args:
            model_path: Path to the YOLO model weights
            confidence_threshold: Confidence threshold for detections
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.class_names = ['15-meter-mark', '5-meter-mark', 'wall']
        self.tracked_elements = {class_name: {} for class_name in self.class_names}  # Track IDs for each class
        self.next_id = {class_name: 0 for class_name in self.class_names}  # Next available ID for each class
        self.max_distance = 50  # Maximum pixel distance for tracking association
        self.POOL_LAYOUT = [
            ('wall', 'left'),             # index 0
            ('5-meter-mark', 'left'),     # index 1
            ('15-meter-mark', 'left'),    # index 2
            ('15-meter-mark', 'right'),   # index 3
            ('5-meter-mark', 'right'),    # index 4
            ('wall', 'right'),            # index 5
        ]
        
    def _assign_tracks(self, current_detections: Dict[str, List[Tuple[float, float]]],
                    previous_elements: Dict[int, Tuple[float, float]]) -> Dict[int, Tuple[float, float]]:
        """
        Assign track IDs to current detections based on proximity to previous detections.
        """
        if not previous_elements:
            return {}
            
        # Calculate distances between current and previous detections
        assignments = {}
        used_prev_ids = set()
        
        for curr_pos in current_detections:
            min_dist = float('inf')
            best_id = None
            
            for prev_id, prev_pos in previous_elements.items():
                if prev_id in used_prev_ids:
                    continue
                    
                dist = np.sqrt((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2)
                if dist < min_dist and dist < self.max_distance:
                    min_dist = dist
                    best_id = prev_id
            
            if best_id is not None:
                assignments[best_id] = curr_pos
                used_prev_ids.add(best_id)
                
        return assignments
        
    def detect_and_track(self, frame) -> Dict[str, Dict[int, Tuple[float, float]]]:
        """
        Detect pool elements and maintain their tracking IDs.
        
        Args:
            frame: Input frame/image
            
        Returns:
            Dictionary containing tracked elements for each class with their IDs
        """
        # Get new detections
        results = self.model(frame, conf=self.confidence_threshold)[0]
        current_detections = {class_name: [] for class_name in self.class_names}
        
        for box in results.boxes:
            if box.conf.item() >= self.confidence_threshold:
                class_id = int(box.cls.item())
                class_name = self.class_names[class_id]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                current_detections[class_name].append((center_x, center_y))
        
        # Update tracks for each class
        new_tracked_elements = {class_name: {} for class_name in self.class_names}
        
        for class_name in self.class_names:
            # Assign existing tracks
            assignments = self._assign_tracks(current_detections[class_name], 
                                           self.tracked_elements[class_name])
            
            # Add assignments to new tracks
            new_tracked_elements[class_name].update(assignments)
            
            # Create new tracks for unassigned detections
            assigned_positions = set(assignments.values())
            for pos in current_detections[class_name]:
                if pos not in assigned_positions:
                    new_tracked_elements[class_name][self.next_id[class_name]] = pos
                    self.next_id[class_name] += 1
        
        self.tracked_elements = new_tracked_elements
        return self.tracked_elements
    
    def process_video(self, frames: List[np.ndarray]) -> List[Dict[str, Dict[int, Tuple[float, float]]]]:
        """
        Process a list of video frames and track pool elements across frames.
        
        Args:
            frames: List of input frames
            
        Returns:
            List of dictionaries containing tracked elements for each frame
        """
        return [self.detect_and_track(frame) for frame in frames]
    
    def _cluster_points(self, points: List[Tuple[float, float]], eps: float = 100) -> List[List[Tuple[float, float]]]:
        """
        Cluster points into groups that likely belong to the same line.
        Uses DBSCAN for clustering to handle noise and parallel lines.
        """
        if not points:
            return []
            
        # Convert points to numpy array
        points_array = np.array(points)
        
        # Perform clustering
        clustering = DBSCAN(eps=eps, min_samples=2).fit(points_array)
        labels = clustering.labels_
        
        # Group points by cluster
        clusters = {}
        for point, label in zip(points, labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(point)
            
        # Return only valid clusters (ignore noise points with label -1)
        return [cluster for label, cluster in clusters.items() if label != -1]
        
    def _fit_line(self, points: List[Tuple[float, float]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit a line to a set of points using robust linear regression.
        Returns line parameters in the form of a point and direction vector.
        """
        points_array = np.array(points)
        
        # Calculate mean point (which line will pass through)
        mean_point = np.mean(points_array, axis=0)
        
        # Perform PCA to find line direction
        centered_points = points_array - mean_point
        _, eigenvectors = np.linalg.eigh(centered_points.T @ centered_points)
        direction = eigenvectors[:, -1]  # Use the principal eigenvector
        
        return mean_point, direction
        
    def _extend_line_to_frame_edges(self, point: np.ndarray, direction: np.ndarray, 
                                  frame_width: int, frame_height: int) -> List[Tuple[int, int]]:
        """
        Extend a line to the edges of the frame.
        Returns two points that define the line segment.
        """
        # Normalize direction vector
        direction = direction / np.linalg.norm(direction)
        
        # Parametric line equation: point + t * direction
        # Solve for t at frame boundaries
        ts = []
        
        # Intersect with x = 0 and x = width
        if abs(direction[0]) > 1e-10:  # Avoid division by zero
            ts.extend([
                -point[0] / direction[0],  # x = 0
                (frame_width - point[0]) / direction[0]  # x = width
            ])
            
        # Intersect with y = 0 and y = height
        if abs(direction[1]) > 1e-10:  # Avoid division by zero
            ts.extend([
                -point[1] / direction[1],  # y = 0
                (frame_height - point[1]) / direction[1]  # y = height
            ])
            
        # Calculate intersection points
        intersections = []
        for t in ts:
            x, y = point + t * direction
            if 0 <= x <= frame_width and 0 <= y <= frame_height:
                intersections.append((int(x), int(y)))
                
        # Return the two most distant points
        if len(intersections) >= 2:
            return sorted(intersections)[:2]  # Sort to ensure consistent order
        return []
    
    def match_sequence_to_layout(self, detected_sequence):
        """
        Given a list of detected classes in left-to-right order, e.g.:
        ['15-meter-mark', '5-meter-mark', 'wall']
        return a list of (class_name, side_label) that best matches
        the canonical POOL_LAYOUT.
        
        For 15m marks:
        - If we see two 15m marks, keep the sequence-based logic (first=left, second=right)
        - If we see one 15m mark and right-side markers (5m-right, wall-right), 
          then the 15m mark is likely right-side too
        """
        # For convenience, separate the layout into just the class portion:
        layout_classes = [item[0] for item in self.POOL_LAYOUT]
        
        # First pass: check if we have right-side context
        has_right_context = False
        fifteen_count = detected_sequence.count('15-meter-mark')
        
        if fifteen_count == 1:
            # Look for right-side markers after the 15m mark
            fifteen_idx = detected_sequence.index('15-meter-mark')
            right_side_sequence = detected_sequence[fifteen_idx:]
            
            # If we see either 5m mark OR wall after 15m mark, it's likely the right side
            if '5-meter-mark' in right_side_sequence or 'wall' in right_side_sequence:
                has_right_context = True
        
        matched = []
        i_layout = 0
        i_det = 0
        
        # Modified matching logic
        while i_det < len(detected_sequence) and i_layout < len(layout_classes):
            if detected_sequence[i_det] == layout_classes[i_layout]:
                # Special case for single 15m mark with right context
                if detected_sequence[i_det] == '15-meter-mark' and fifteen_count == 1 and has_right_context:
                    # Skip to the right 15m mark in the layout
                    while i_layout < len(layout_classes) and (
                        layout_classes[i_layout] != '15-meter-mark' or 
                        self.POOL_LAYOUT[i_layout][1] != 'right'
                    ):
                        i_layout += 1
                    if i_layout < len(layout_classes):
                        matched.append((detected_sequence[i_det], 'right'))
                else:
                    # Normal sequence-based matching
                    matched.append((detected_sequence[i_det], self.POOL_LAYOUT[i_layout][1]))
                i_det += 1
                i_layout += 1
            else:
                # No match, try to move forward in the layout to find a matching class
                i_layout += 1
        
        return matched

    def get_detected_elements(self, tracked_elements: Dict[str, Dict[int, Tuple[float, float]]]) -> List[Tuple[str, str, int]]:
        """
        Convert tracked elements into a format suitable for minimap drawing.
        
        Args:
            tracked_elements: Dictionary containing tracked elements
            
        Returns:
            List of tuples (class_name, side, position_in_meters)
        """
        # Process clusters and get annotated elements
        all_points = []
        for class_name, elements_dict in tracked_elements.items():
            for track_id, (x, y) in elements_dict.items():
                all_points.append((x, y, class_name))
        
        clusters_by_class = {}
        for class_name in self.class_names:
            class_points = [(x, y) for (x, y, c) in all_points if c == class_name]
            clusters = self._cluster_points(class_points)
            clusters_by_class[class_name] = clusters
        
        combined_clusters = []
        for class_name, cluster_list in clusters_by_class.items():
            for cluster_points in cluster_list:
                xs = [p[0] for p in cluster_points]
                avg_x = sum(xs)/len(xs) if len(xs)>0 else 0
                combined_clusters.append((avg_x, class_name, cluster_points))
        
        combined_clusters.sort(key=lambda c: c[0])
        detected_sequence = [cls_name for (_, cls_name, _) in combined_clusters]
        matched = self.match_sequence_to_layout(detected_sequence)
        
        # Convert matched elements to position in meters
        detected_elements = []
        for cls_name, side in matched:
            if cls_name == 'wall':
                pos = 0 if side == 'left' else 25
            elif cls_name == '5-meter-mark':
                pos = 5 if side == 'left' else 20
            elif cls_name == '15-meter-mark':
                pos = 10 if side == 'left' else 15
            detected_elements.append((cls_name, side, pos))
            
        return detected_elements

    def draw_tracked_elements(self, frame, tracked_elements: Dict[str, Dict[int, Tuple[float, float]]]) -> np.ndarray:
        """
        Draw tracked elements and cluster them into lines.
        """
        colors = {
            'wall': (0, 0, 255),         # Red
            '5-meter-mark': (0, 255, 0), # Green
            '15-meter-mark': (255, 0, 0) # Blue
        }
        
        output_frame = frame.copy()
        frame_height, frame_width = output_frame.shape[:2]

        # Process clusters and get annotated elements
        all_points = []
        for class_name, elements_dict in tracked_elements.items():
            for track_id, (x, y) in elements_dict.items():
                all_points.append((x, y, class_name))
        
        clusters_by_class = {}
        for class_name in self.class_names:
            class_points = [(x, y) for (x, y, c) in all_points if c == class_name]
            clusters = self._cluster_points(class_points)
            clusters_by_class[class_name] = clusters
        
        combined_clusters = []
        for class_name, cluster_list in clusters_by_class.items():
            for cluster_points in cluster_list:
                xs = [p[0] for p in cluster_points]
                avg_x = sum(xs)/len(xs) if len(xs)>0 else 0
                combined_clusters.append((avg_x, class_name, cluster_points))
        
        combined_clusters.sort(key=lambda c: c[0])
        detected_sequence = [cls_name for (_, cls_name, _) in combined_clusters]
        matched = self.match_sequence_to_layout(detected_sequence)
        
        # Draw each cluster
        for (cls_name, side_label, cluster_pts) in zip(
            [m[0] for m in matched], 
            [m[1] for m in matched], 
            [c[2] for c in combined_clusters[:len(matched)]]
        ):
            color = colors[cls_name]
            
            # Fit line if >= 2 points
            if len(cluster_pts) >= 2:
                mean_point, direction = self._fit_line(cluster_pts)
                line_points = self._extend_line_to_frame_edges(mean_point, direction, frame_width, frame_height)
                if len(line_points) == 2:
                    cv2.line(output_frame, line_points[0], line_points[1], color, 2)

            # Draw each point in cluster
            for (px, py) in cluster_pts:
                cv2.circle(output_frame, (int(px), int(py)), 5, color, -1)
                
                label_text = cls_name
                if side_label is not None:
                    label_text += f" ({side_label})"
                cv2.putText(
                    output_frame, 
                    label_text, 
                    (int(px) + 10, int(py) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    color, 
                    2
                )
        
        return output_frame

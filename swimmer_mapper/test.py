import numpy as np
from typing import List, Dict, Tuple, Optional

class SwimmerMapper:
    def __init__(self, frame_width: int = 1920, frame_height: int = 1080, 
                 minimap_width: int = 200, minimap_height: int = 100):
        """
        Initialize the SwimmerMapper which transforms swimmer positions from image coordinates
        to pool space coordinates (in meters).
        
        Args:
            frame_width: Width of the input video frame
            frame_height: Height of the input video frame
            minimap_width: Width of the minimap
            minimap_height: Height of the minimap
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.minimap_width = minimap_width
        self.minimap_height = minimap_height
        
        # Pool dimensions
        self.pool_length = 25  # Standard pool length in meters
        self.pool_width = 8    # Standard number of lanes
        
        # Define the pool layout
        self.POOL_LAYOUT = [
            ('wall', 'left'),             # index 0
            ('5-meter-mark', 'left'),     # index 1
            ('15-meter-mark', 'left'),    # index 2
            ('15-meter-mark', 'right'),   # index 3
            ('5-meter-mark', 'right'),    # index 4
            ('wall', 'right'),            # index 5
        ]
        
        # Will store the detected pool elements for coordinate transformation
        self.pool_elements: List[Tuple[str, str, int]] = []
        
        # Calculate lane heights
        self.lane_height = self.minimap_height / self.pool_width  # Height of each lane in minimap
        
        # Store detected lines with their side information
        self.detected_lines: Dict[Tuple[str, str], List[Tuple[float, float]]] = {}
        
    def update_frame_dimensions(self, frame_width: int, frame_height: int) -> None:
        """
        Update the frame dimensions if they change.
        
        Args:
            frame_width: New frame width
            frame_height: New frame height
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        
    def update_minimap_dimensions(self, minimap_width: int, minimap_height: int) -> None:
        """
        Update the minimap dimensions if they change.
        
        Args:
            minimap_width: New minimap width
            minimap_height: New minimap height
        """
        self.minimap_width = minimap_width
        self.minimap_height = minimap_height
        self.lane_height = self.minimap_height / self.pool_width
        
    def update_pool_elements(self, detected_elements: List[Tuple[str, str, int]]) -> None:
        """
        Update the pool elements used for coordinate transformation.
        
        Args:
            detected_elements: List of tuples (class_name, side, position_in_meters)
        """
        self.pool_elements = detected_elements
        
    def update_detected_lines(self, lines: Dict[str, Dict[int, Tuple[float, float]]], matched_elements: List[Tuple[str, str, int]]) -> None:
        """
        Update the detected lines for distance calculations.
        
        Args:
            lines: Dictionary mapping line types to dict of track_id -> points
            matched_elements: List of tuples (class_name, side, position) from pool detector
        """
        self.detected_lines = {}
        
        # Process each line type
        for class_name, elements_dict in lines.items():
            # Find matching elements for this class
            matching_elements = [(elem[0], elem[1]) for elem in matched_elements 
                               if elem[0] == class_name]
            
            if len(matching_elements) == 2:  # If we have both left and right
                # Collect all points for this line type
                points = list(elements_dict.values())
                if len(points) >= 2:
                    # Sort points by x coordinate
                    sorted_points = sorted(points, key=lambda p: p[0])
                    
                    # Left points are those with smaller x coordinates
                    mid_idx = len(sorted_points) // 2
                    left_points = sorted_points[:mid_idx]
                    right_points = sorted_points[mid_idx:]
                    
                    # Add to detected lines
                    self.detected_lines[(class_name, 'left')] = left_points
                    self.detected_lines[(class_name, 'right')] = right_points
                    
            elif len(matching_elements) == 1:  # If we only have one side
                # Use the side from matched_elements
                side = matching_elements[0][1]
                self.detected_lines[(class_name, side)] = list(elements_dict.values())

    def find_line_x_at_y(self, line_points: List[Tuple[float, float]], y: float) -> Optional[float]:
        """
        Find the X coordinate where a line intersects with a given Y coordinate.
        Uses linear interpolation between points.
        
        Args:
            line_points: List of points (x,y) that form the line
            y: Y coordinate to find intersection at
            
        Returns:
            X coordinate of intersection, or None if no intersection found
        """
        if not line_points or len(line_points) < 2:
            return None
            
        # Sort points by y coordinate
        sorted_points = sorted(line_points, key=lambda p: p[1])
        
        # Find the two points that bracket our y coordinate
        for i in range(len(sorted_points) - 1):
            y1 = sorted_points[i][1]
            y2 = sorted_points[i + 1][1]
            
            if y1 <= y <= y2 or y2 <= y <= y1:
                x1 = sorted_points[i][0]
                x2 = sorted_points[i + 1][0]
                
                # Linear interpolation
                if y1 == y2:
                    return x1  # Horizontal line segment
                    
                t = (y - y1) / (y2 - y1)
                x = x1 + t * (x2 - x1)
                return x
                
        return None
        
    def calculate_line_to_line_distances(self, y_level: float) -> Dict[Tuple[Tuple[str, str], Tuple[str, str]], float]:
        """
        Calculate distances between pairs of lines at a specific Y level.
        
        Args:
            y_level: Y coordinate to measure distances at
            
        Returns:
            Dictionary mapping (line1_tuple, line2_tuple) to distance in pixels
            where line_tuple is (line_type, side)
        """
        # First get all line X positions at this Y level
        line_positions = {}
        for line_type, side in self.POOL_LAYOUT:
            key = (line_type, side)
            if key in self.detected_lines and self.detected_lines[key]:
                x_pos = self.find_line_x_at_y(self.detected_lines[key], y_level)
                if x_pos is not None:
                    line_positions[key] = x_pos
        
        # Calculate distances between each pair of detected lines
        distances = {}
        line_keys = list(line_positions.keys())
        for i in range(len(line_keys)):
            for j in range(i + 1, len(line_keys)):
                key1 = line_keys[i]
                key2 = line_keys[j]
                distance = abs(line_positions[key1] - line_positions[key2])
                distances[(key1, key2)] = distance
                
        return distances

    def calculate_pixel_to_meter_ratio(self, y_level: float) -> Optional[float]:
        """
        Calculate the pixel-to-meter ratio at a specific Y level using known 5m distances.
        
        Args:
            y_level: Y coordinate to calculate ratio at
            
        Returns:
            Pixels per meter ratio, or None if can't be calculated
        """
        # Get X positions of all lines at this Y level
        line_positions = {}
        for line_type, side in self.POOL_LAYOUT:
            key = (line_type, side)
            if key in self.detected_lines and self.detected_lines[key]:
                x_pos = self.find_line_x_at_y(self.detected_lines[key], y_level)
                if x_pos is not None:
                    line_positions[key] = x_pos
        
        # Calculate ratios for each sequential 5m segment
        ratios = []
        
        # Known sequential pairs and their distances
        SEQUENTIAL_PAIRS = [
            (('wall', 'left'), ('5-meter-mark', 'left'), 5),
            (('5-meter-mark', 'left'), ('15-meter-mark', 'left'), 5),
            (('15-meter-mark', 'left'), ('15-meter-mark', 'right'), 5),
            (('15-meter-mark', 'right'), ('5-meter-mark', 'right'), 5),
            (('5-meter-mark', 'right'), ('wall', 'right'), 5),
        ]
        
        for line1, line2, meters in SEQUENTIAL_PAIRS:
            if line1 in line_positions and line2 in line_positions:
                pixels = abs(line_positions[line1] - line_positions[line2])
                ratio = pixels / meters  # pixels per meter
                ratios.append(ratio)
        
        if ratios:
            # Use median ratio to be robust against outliers
            return np.median(ratios)
        return None

    def calculate_distances(self, swimmer_pos: Tuple[float, float]) -> Dict[str, Dict]:
        """
        Calculate distances from a swimmer to each detected line at the swimmer's Y level.
        Will calculate pixel-to-meter ratio even when sequential lines are not visible,
        using any available reference points.
        
        Args:
            swimmer_pos: Tuple of (x,y) coordinates of the swimmer
            
        Returns:
            Dictionary with keys:
                'to_swimmer': Dict mapping (line_type, side) to distances (pixels)
                'to_swimmer_meters': Dict mapping (line_type, side) to distances (meters)
                'pixels_per_meter': Pixel to meter ratio if calculable
        """
        swimmer_x, swimmer_y = swimmer_pos
        
        # Get line positions at swimmer's Y level
        line_positions = {}
        for line_type, side in self.POOL_LAYOUT:
            key = (line_type, side)
            if key in self.detected_lines and self.detected_lines[key]:
                line_x = self.find_line_x_at_y(self.detected_lines[key], swimmer_y)
                if line_x is not None:
                    line_positions[key] = line_x
        
        # Calculate pixel distances to swimmer
        distances_to_swimmer = {}
        for key, line_x in line_positions.items():
            distances_to_swimmer[key] = abs(line_x - swimmer_x)
        
        # Try to calculate pixel to meter ratio using sequential pairs first
        ratios = []
        SEQUENTIAL_PAIRS = [
            (('wall', 'left'), ('5-meter-mark', 'left'), 5),
            (('5-meter-mark', 'left'), ('15-meter-mark', 'left'), 5),
            (('15-meter-mark', 'left'), ('15-meter-mark', 'right'), 5),
            (('15-meter-mark', 'right'), ('5-meter-mark', 'right'), 5),
            (('5-meter-mark', 'right'), ('wall', 'right'), 5),
        ]
        
        # First try sequential pairs
        for line1, line2, meters in SEQUENTIAL_PAIRS:
            if line1 in line_positions and line2 in line_positions:
                pixels = abs(line_positions[line1] - line_positions[line2])
                ratio = pixels / meters  # pixels per meter
                ratios.append(ratio)
        
        # If no sequential pairs found, try any pair with known distance
        if not ratios:
            # Known distances between any pair of reference points
            KNOWN_DISTANCES = {
                (('wall', 'left'), ('5-meter-mark', 'left')): 5,
                (('wall', 'left'), ('15-meter-mark', 'left')): 10,
                (('wall', 'left'), ('15-meter-mark', 'right')): 15,
                (('wall', 'left'), ('5-meter-mark', 'right')): 20,
                (('wall', 'left'), ('wall', 'right')): 25,
                (('5-meter-mark', 'left'), ('15-meter-mark', 'left')): 5,
                (('5-meter-mark', 'left'), ('15-meter-mark', 'right')): 10,
                (('5-meter-mark', 'left'), ('5-meter-mark', 'right')): 15,
                (('5-meter-mark', 'left'), ('wall', 'right')): 20,
                (('15-meter-mark', 'left'), ('15-meter-mark', 'right')): 5,
                (('15-meter-mark', 'left'), ('5-meter-mark', 'right')): 10,
                (('15-meter-mark', 'left'), ('wall', 'right')): 15,
                (('15-meter-mark', 'right'), ('5-meter-mark', 'right')): 5,
                (('15-meter-mark', 'right'), ('wall', 'right')): 10,
                (('5-meter-mark', 'right'), ('wall', 'right')): 5,
            }
            
            # Try all possible pairs
            for (line1, line2), meters in KNOWN_DISTANCES.items():
                if line1 in line_positions and line2 in line_positions:
                    pixels = abs(line_positions[line1] - line_positions[line2])
                    ratio = pixels / meters
                    ratios.append(ratio)
        
        pixels_per_meter = np.median(ratios) if ratios else None
        
        # Convert distances to meters if ratio is available
        distances_to_swimmer_meters = {}
        if pixels_per_meter is not None:
            for key, distance in distances_to_swimmer.items():
                distances_to_swimmer_meters[key] = distance / pixels_per_meter
        
        return {
            'to_swimmer': distances_to_swimmer,
            'to_swimmer_meters': distances_to_swimmer_meters,
            'pixels_per_meter': pixels_per_meter
        }

    def format_distance_text(self, line_type: str, side: str, 
                           distance_px: Optional[float], 
                           distance_m: Optional[float]) -> str:
        """
        Format the distance text to show both pixels and meters.
        """
        # Shorten the line type names
        type_map = {
            'wall': 'W',
            '5-meter-mark': '5m',
            '15-meter-mark': '15m'
        }
        short_name = type_map.get(line_type, line_type)
        
        if distance_px is None:
            return f"{short_name}({side[0]}): ns"
        elif distance_m is not None:
            return f"{short_name}({side[0]}): {distance_m:.1f}m"
        else:
            return f"{short_name}({side[0]}): {distance_px:.0f}px"

    def format_line_distance_text(self, line1: Tuple[str, str], line2: Tuple[str, str], 
                                distance_px: float, distance_m: Optional[float] = None) -> str:
        """
        Format the line-to-line distance text to show both pixels and meters.
        """
        # Shorten the line type names
        type_map = {
            'wall': 'W',
            '5-meter-mark': '5m',
            '15-meter-mark': '15m'
        }
        name1 = type_map.get(line1[0], line1[0])
        name2 = type_map.get(line2[0], line2[0])
        
        if distance_m is not None:
            return f"{name1}({line1[1][0]}) → {name2}({line2[1][0]}): {distance_m:.1f}m"
        return f"{name1}({line1[1][0]}) → {name2}({line2[1][0]}): {distance_px:.0f}px"

    def format_scale_text(self, pixels_per_meter: float) -> str:
        """
        Format the scale text.
        """
        return f"Scale: {pixels_per_meter:.0f} px/m"

    def get_all_swimmer_distances(self, 
                                swimmer_positions: Dict[int, Tuple[float, float]]) -> Dict[int, Dict[str, Dict]]:
        """
        Calculate all distances for each swimmer.
        
        Args:
            swimmer_positions: Dictionary mapping track_ids to positions
            
        Returns:
            Dictionary mapping track_ids to their distance measurements
        """
        all_distances = {}
        for track_id, pos in swimmer_positions.items():
            all_distances[track_id] = self.calculate_distances(pos)
        return all_distances
        
    def get_lane_y_position(self, lane_number: int) -> float:
        """
        Get the y-coordinate in minimap for a given lane number.
        
        Args:
            lane_number: Lane number (1-8)
            
        Returns:
            Y coordinate in minimap
        """
        # Convert lane number to 0-based index
        lane_idx = lane_number - 1
        # Calculate center of lane
        return lane_idx * self.lane_height + (self.lane_height / 2)
        
    def map_swimmers(self, swimmer_positions: Dict[int, Tuple[float, float]], 
                    distances_dict: Dict[int, Dict]) -> Dict[int, Tuple[float, float]]:
        """
        Transform swimmer positions to minimap coordinates using actual measured distances.
        Uses any available distance measurements to reference points, even when sequential
        points are not visible.
        
        The pool is 25m long with reference points at:
        - Left wall: 0m
        - Left 5m mark: 5m
        - Left 15m mark: 10m (labeled as 15m mark)
        - Right 15m mark: 15m
        - Right 5m mark: 20m
        - Right wall: 25m
        """
        if not swimmer_positions or not distances_dict:
            return {}
        
        # Sort swimmers by Y position for lane assignment
        sorted_swimmers = sorted(
            swimmer_positions.items(),
            key=lambda x: x[1][1],
            reverse=False
        )
        
        # Reference point positions from left wall
        ref_positions = {
            ('wall', 'left'): 0.0,
            ('5-meter-mark', 'left'): 5.0,
            ('15-meter-mark', 'left'): 10.0,  # Actually 10m mark
            ('15-meter-mark', 'right'): 15.0,
            ('5-meter-mark', 'right'): 20.0,
            ('wall', 'right'): 25.0
        }
        
        mapped_positions = {}
        for i, (track_id, pos) in enumerate(sorted_swimmers):
            swimmer_x, swimmer_y = pos
            
            # Assign lane based on Y position
            lane_number = min(i + 1, self.pool_width)
            minimap_y = self.get_lane_y_position(lane_number)
            
            if track_id not in distances_dict:
                continue
            
            track_distances = distances_dict[track_id]
            distances_meters = track_distances['to_swimmer_meters']
            
            if not distances_meters:
                continue
            
            # Get valid distances and their x-coordinates
            valid_distances = {}
            for point, dist in distances_meters.items():
                if dist is not None and point in self.detected_lines:
                    line_x = self.find_line_x_at_y(self.detected_lines[point], swimmer_y)
                    if line_x is not None:
                        valid_distances[point] = (dist, line_x)
            
            if not valid_distances:
                continue
            
            # Calculate positions from all valid reference points
            positions = []
            for point, (distance, ref_x) in valid_distances.items():
                # Determine if swimmer is to the left or right of the reference point
                is_to_right = swimmer_x > ref_x
                
                # Calculate position from this reference point
                base_position = ref_positions[point]
                position = base_position + (distance if is_to_right else -distance)
                
                # Only use reasonable positions
                if 0.0 <= position <= 25.0:
                    positions.append(position)
            
            if not positions:
                continue
            
            # Use median of all valid positions
            absolute_position = np.median(positions)
            
            # Convert absolute position (0-25m) to minimap x coordinate
            minimap_x = (absolute_position / 25.0) * self.minimap_width
            # Ensure x coordinate is within bounds
            minimap_x = max(0, min(minimap_x, self.minimap_width))
            mapped_positions[track_id] = (minimap_x, minimap_y)
        
        return mapped_positions

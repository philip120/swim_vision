import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional

class MinimapDrawer:
    def __init__(self, width: int = 200, height: int = 100):
        self.width = width
        self.height = height
        self.POOL_LAYOUT = [
            ('wall', 'left'),             # 0m
            ('5-meter-mark', 'left'),     # 5m
            ('15-meter-mark', 'left'),    # 15m
            ('15-meter-mark', 'right'),   # 10m
            ('5-meter-mark', 'right'),    # 20m
            ('wall', 'right'),            # 25m
        ]

    def draw_minimap(self, 
                    detected_elements: List[Tuple[str, str, int]], 
                    swimmer_positions: Optional[Dict[int, Tuple[float, float]]] = None) -> np.ndarray:
        """
        Draw a minimap of the pool showing lanes, distance markers and swimmers.
        
        Args:
            detected_elements: List of tuples (class_name, side, position_in_meters)
            swimmer_positions: Optional dictionary of swimmer positions {track_id: (x,y)}
            
        Returns:
            numpy.ndarray: The minimap image
        """
        # Create a white background
        minimap = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255
        
        # Define colors
        BLUE = (255, 150, 0)  # Light blue for water
        BLACK = (0, 0, 0)     # Black for lines
        RED = (0, 0, 255)     # Red for wall
        GREEN = (0, 255, 0)   # Green for 5m
        BLUE_HIGHLIGHT = (255, 0, 0)  # Blue for 15m
        
        # Fill the pool area with light blue
        cv2.rectangle(minimap, (0, 0), (self.width, self.height), BLUE, -1)
        
        # Draw 8 lanes
        lane_height = self.height // 8
        for i in range(9):  # 9 lines to create 8 lanes
            y = i * lane_height
            cv2.line(minimap, (0, y), (self.width, y), BLACK, 1)
            
            # Add lane numbers
            if i < 8:
                cv2.putText(minimap, f"{i+1}", 
                           (5, y + lane_height//2 + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, BLACK, 1)
        
        # Draw distance markers
        distances = [0, 5, 10, 15, 20, 25]  # meters
        marker_positions = [(pos * self.width) // 25 for pos in distances]
        
        for pos, dist in zip(marker_positions, distances):
            # Draw vertical line
            cv2.line(minimap, (pos, 0), (pos, self.height), BLACK, 1)
            # Add distance label
            cv2.putText(minimap, f"{dist}m", 
                       (pos - 10, self.height - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, BLACK, 1)

        # Highlight detected elements
        if detected_elements:
            for cls_name, side, pos in detected_elements:
                # Calculate x position
                x_pos = (pos * self.width) // 25
                
                # Choose color based on class
                if cls_name == 'wall':
                    color = RED
                elif cls_name == '5-meter-mark':
                    color = GREEN
                else:  # 15-meter-mark
                    color = BLUE_HIGHLIGHT
                
                # Draw thicker highlighted line
                cv2.line(minimap, (x_pos, 0), (x_pos, self.height), color, 3)
                
                # Add label at the top
                label = f"{cls_name.split('-')[0]}({side[0].upper()})"
                cv2.putText(minimap, label, 
                           (x_pos - 15, 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)

        # Draw swimmers if positions provided
        if swimmer_positions:
            for track_id, (x, y) in swimmer_positions.items():
                x, y = int(x), int(y)
                # Draw swimmer dot
                cv2.circle(minimap, (x, y), 3, RED, -1)
                # Draw track ID
                cv2.putText(minimap, str(track_id), 
                           (x + 5, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, RED, 1)

        return minimap

    def add_minimap_to_frame(self, frame: np.ndarray, minimap: np.ndarray) -> np.ndarray:
        """
        Add minimap to the top-right corner of the frame
        
        Args:
            frame: The main video frame
            minimap: The minimap image to add
            
        Returns:
            numpy.ndarray: Frame with minimap added
        """
        frame = frame.copy()
        mh, mw = minimap.shape[:2]
        frame_height, frame_width = frame.shape[:2]
        frame[10:10+mh, frame_width-mw-10:frame_width-10] = minimap
        return frame 
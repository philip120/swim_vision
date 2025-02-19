from ultralytics import YOLO
from utils import read_video_in_chunks, save_video_incremental
from trackers import Tracker
from map_pool import PoolDetector
from minimap import MinimapDrawer
from swimmer_mapper import SwimmerMapper
import logging
import psutil
import gc
import os
import cv2

logging.basicConfig(level=logging.INFO)

def process_batch(frames_batch, tracker, pool_detector, minimap_drawer, swimmer_mapper, output_path, batch_index):
    if len(frames_batch) == 0:
        return []
        
    # Detect and track pool elements in the batch
    pool_tracks = pool_detector.process_video(frames_batch)
    
    # Get swimmer tracks
    swimmer_tracks = tracker.get_object_tracks(frames_batch)
    
    # Create annotated frames
    annotated_frames = []
    for i, frame in enumerate(frames_batch):
        # Start with a clean frame
        annotated_frame = frame.copy()
        
        # Get detected elements and their positions for minimap
        detected_elements = pool_detector.get_detected_elements(pool_tracks[i])
        
        # Pass the tracked elements directly to swimmer mapper
        swimmer_mapper.update_detected_lines(pool_tracks[i], detected_elements)
        
        # Draw pool elements with tracking IDs
        annotated_frame = pool_detector.draw_tracked_elements(annotated_frame, pool_tracks[i])
        
        # Draw swimmer tracking on top
        annotated_frame = tracker.draw_annotations([annotated_frame], {"swimmers": [swimmer_tracks["swimmers"][i]]})[0]
        
        # Extract just the positions for swimmers
        swimmer_positions = {
            track_id: swimmer_data["position"] 
            for track_id, swimmer_data in swimmer_tracks["swimmers"][i].items()
        }
        
        # Calculate distances once for all swimmers
        distances = swimmer_mapper.get_all_swimmer_distances(swimmer_positions)
        
        # Map swimmer positions to minimap coordinates using the same distances
        minimap_positions = swimmer_mapper.map_swimmers(swimmer_positions, distances)
        
        # Draw minimap first
        minimap = minimap_drawer.draw_minimap(detected_elements, minimap_positions)
        annotated_frame = minimap_drawer.add_minimap_to_frame(annotated_frame, minimap)
        
        # Draw distances on frame
        for track_id, track_distances in distances.items():
            pos = swimmer_positions[track_id]
            y_offset = 30  # Start text 30 pixels above swimmer
            
            # Display scale first if available
            if track_distances['pixels_per_meter'] is not None:
                ratio_text = swimmer_mapper.format_scale_text(track_distances['pixels_per_meter'])
                #cv2.putText(annotated_frame, ratio_text,
                           #(int(pos[0]) - 50, int(pos[1]) - y_offset),
                           #cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                y_offset += 20
            
            # Display distances from swimmer to lines
            # First left side (cyan)
            for line_type, side in swimmer_mapper.POOL_LAYOUT[:3]:  # First 3 are left side
                distance_px = track_distances['to_swimmer'].get((line_type, side))
                distance_m = track_distances['to_swimmer_meters'].get((line_type, side))
                text = swimmer_mapper.format_distance_text(line_type, side, distance_px, distance_m)
                #cv2.putText(annotated_frame, text, 
                           #(int(pos[0]) - 50, int(pos[1]) - y_offset),
                           #cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)  # Cyan
                y_offset += 15  # Reduced spacing
            
            # Then right side (yellow)
            for line_type, side in swimmer_mapper.POOL_LAYOUT[3:]:  # Last 3 are right side
                distance_px = track_distances['to_swimmer'].get((line_type, side))
                distance_m = track_distances['to_swimmer_meters'].get((line_type, side))
                text = swimmer_mapper.format_distance_text(line_type, side, distance_px, distance_m)
                #cv2.putText(annotated_frame, text, 
                           #(int(pos[0]) - 50, int(pos[1]) - y_offset),
                           #cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)  # Yellow
                y_offset += 15  # Reduced spacing
        
        annotated_frames.append(annotated_frame)
    
    save_video_incremental(annotated_frames, output_path, batch_index)
    return annotated_frames

def main():
    video_path = '25m.mp4'
    output_path = 'output_videos/output.mp4'
    detection_model_path = 'models/best (17).pt'  # swimmer detection
    classification_model_path = 'models/final_model_weights (14).pth'  # swimmer classification
    pool_model_path = 'models/pool_keypoints_detection.pt'  # pool detection
    
    # Initialize detectors and minimap drawer
    pool_detector = PoolDetector(pool_model_path)
    tracker = Tracker(detection_model_path, classification_model_path)
    minimap_drawer = MinimapDrawer()
    swimmer_mapper = SwimmerMapper()  # Initialize the swimmer mapper
    
    batch_size = 500

    # Remove previous output file if exists
    if os.path.exists(output_path):
        os.remove(output_path)

    batch_index = 0
    for frames_batch in read_video_in_chunks(video_path, batch_size):
        process_batch(frames_batch, tracker, pool_detector, minimap_drawer, swimmer_mapper, output_path, batch_index)
        batch_index += 1
        
        # Log memory usage
        memory_info = psutil.virtual_memory()
        logging.info(f"Memory usage: {memory_info.percent}% used, {memory_info.available / (1024 * 1024):.2f} MB available")
        logging.info(f"Processed batch {batch_index} with {len(frames_batch)} frames.")
        
        # Force garbage collection to free memory
        gc.collect()

if __name__ == '__main__':
    main()

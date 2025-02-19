import os
import cv2
import supervision as sv
from ultralytics import YOLO
from utils import get_center_of_bbox
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models

class Tracker:
    def __init__(self, detection_model_path, classification_model_path):  # removed keypoint_model_path
        self.model = YOLO(detection_model_path)
        self.class_names = {0: "freestyle",1: "start", 2: "underwater"} #freestyle,start,underwater
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classification_model = self.load_classification_model(classification_model_path).to(self.device)
        self.tracker = sv.ByteTrack(track_buffer=120,
                                    track_thresh=0.40,
                                    match_thresh=.95,
                                    frame_rate=30)
        self.transform = transforms.Compose([
            transforms.Resize((232, 80)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.activity_segments = []
        self.speed_history = {}  # Dictionary to store speed history for each track_id

    def load_classification_model(self, model_path):
        class MobileNetV2(nn.Module):
            def __init__(self, num_classes):
                super(MobileNetV2, self).__init__()
                self.model = models.mobilenet_v2(pretrained=False)
                self.model.classifier[1] = nn.Linear(self.model.last_channel, num_classes)  # Adjust according to the new model's output features

            def forward(self, x):
                return self.model(x)

        model = MobileNetV2(num_classes=len(self.class_names))
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model

    def load_image(self, image_path, transform):
        image = Image.open(image_path).convert('RGB')
        image = transform(image)
        image = image.unsqueeze(0)  # Add batch dimension
        return image

    def predict(self, model, image_path, transform, device):
        model.eval()  # Set model to evaluation mode
        image = self.load_image(image_path, transform).to(device)
        with torch.no_grad():
            output = model(image)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        return predicted.cpu().item(), confidence.cpu().item()

    def display_prediction(self, image_path, prediction, confidence, track_dir):
        image = Image.open(image_path)
        predicted_class = self.class_names[prediction]
        plt.imshow(image)
        plt.title(f'Prediction: {predicted_class} with confidence {confidence:.2f}')
        plt.axis('off')
        prediction_output_path = os.path.join(track_dir, f"prediction_{os.path.basename(image_path)}")
        plt.savefig(prediction_output_path)
        plt.close

    def detect_frame(self, frame):
        results = self.model(frame)[0]
        return results

    def detect_frames(self, frames):
        swimmer_detections = []

        for frame in frames:
            detection = self.detect_frame(frame)
            swimmer_detections.append(detection)
        
        return swimmer_detections

    def classify_image(self, image):
        image_path = "temp_image.png"
        cv2.imwrite(image_path, image)
        prediction, confidence = self.predict(self.classification_model, image_path, self.transform, self.device)
        os.remove(image_path)
        return prediction, confidence

    def get_object_tracks(self, frames):
        detections = self.detect_frames(frames)

        tracks = {
            "swimmers": []
        }

        for frame_num, detection in enumerate(detections):
            cls_names = self.model.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            detection_supervision = sv.Detections.from_ultralytics(detection)
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["swimmers"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]
                detection_confidence = frame_detection[2]  # Confidence score of the detection

                if cls_id == cls_names_inv.get('swimmer', None):
                    center_position = get_center_of_bbox(bbox)

                    tracks["swimmers"][frame_num][track_id] = {
                        "bbox": bbox,
                        "position": center_position,
                        "detection_confidence": detection_confidence
                    }

        return tracks


    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x, y],
            [x - 10, y - 20],
            [x + 10, y - 20],
        ])

        overlay = frame.copy()
        cv2.drawContours(overlay, [triangle_points], 0, color, cv2.FILLED)
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        shadow_offset = 2
        shadow_points = np.array([
            [x + shadow_offset, y + shadow_offset],
            [x - 10 + shadow_offset, y - 20 + shadow_offset],
            [x + 10 + shadow_offset, y - 20 + shadow_offset],
        ])
        cv2.drawContours(frame, [shadow_points], 0, (50, 50, 50), cv2.FILLED)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.drawContours(frame, [shadow_points], 0, (0, 0, 0), 2)

        return frame

    def calculate_average_speed(self, track_id, current_speed):
        """Calculate moving average speed over last 10 frames"""
        if track_id not in self.speed_history:
            self.speed_history[track_id] = []
            
        self.speed_history[track_id].append(current_speed)
        
        # Keep only last 10 speeds
        if len(self.speed_history[track_id]) > 10:
            self.speed_history[track_id].pop(0)
            
        # Calculate average speed
        return np.mean(self.speed_history[track_id])
    
    def draw_annotations(self, video_frames, tracks):
        output_video_frames = []
        previous_positions = {}
        
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            swimmer_dict = tracks["swimmers"][frame_num]

            for track_id, swimmer in swimmer_dict.items():
                bbox = swimmer["bbox"]
                frame = self.draw_triangle(frame, bbox, (0, 255, 0))
                video_pos = get_center_of_bbox(bbox)
                
                # Calculate instantaneous speed if we have previous position
                current_speed = None
                avg_speed = None
                if track_id in previous_positions:
                    prev_pos = previous_positions[track_id]
                    current_speed = self.calculate_speed(prev_pos, video_pos)
                    # Calculate average speed
                    if current_speed is not None:
                        avg_speed = self.calculate_average_speed(track_id, current_speed)
                
                # Store current position for next frame
                previous_positions[track_id] = video_pos

                # Classify the cropped image
                x1, y1, x2, y2 = map(int, bbox)
                bbox_image = frame[y1:y2, x1:x2]
                classification, classification_confidence = self.classify_image(bbox_image)
                class_name = self.class_names[classification]

                # Draw on video frame
                label = f"ID:{track_id} - {class_name}"
                if avg_speed is not None:
                    label += f" - {avg_speed:.1f}m/s"
                cv2.putText(frame, label, 
                           (video_pos[0] - 10, video_pos[1] - 40), 
                           cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)

            output_video_frames.append(frame)

        return output_video_frames

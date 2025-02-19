import cv2
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import pickle
import pandas as plt
import matplotlib.pyplot as plt

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def read_video_in_chunks(video_path, batch_size):
    cap = cv2.VideoCapture(video_path)
    while True:
        frames = []
        for _ in range(batch_size):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        if not frames:
            break
        yield frames
    cap.release()

def save_video_incremental(output_video_frames, output_video_path, batch_index):
    if not output_video_frames:
        print("No frames to save for batch index", batch_index)
        return

    print(f"Saving {len(output_video_frames)} frames to {output_video_path} for batch index {batch_index}")
    
    if batch_index == 0:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, 30, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
        for frame in output_video_frames:
            out.write(frame)
        out.release()
    else:
        # Append to the existing video
        temp_video_path = 'temp.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        temp_out = cv2.VideoWriter(temp_video_path, fourcc, 30, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
        for frame in output_video_frames:
            temp_out.write(frame)
        temp_out.release()

        # Concatenate videos
        original_cap = cv2.VideoCapture(output_video_path)
        temp_cap = cv2.VideoCapture(temp_video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('concatenated.mp4', fourcc, 30, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))

        while original_cap.isOpened():
            ret, frame = original_cap.read()
            if not ret:
                break
            out.write(frame)
        original_cap.release()

        while temp_cap.isOpened():
            ret, frame = temp_cap.read()
            if not ret:
                break
            out.write(frame)
        temp_cap.release()
        out.release()

        # Replace original video with concatenated video
        os.remove(output_video_path)
        os.rename('concatenated.mp4', output_video_path)
        os.remove(temp_video_path)
    
    print(f"Saved batch index {batch_index} successfully.")


def save_video(output_video_frames, output_video_path):
    if not output_video_frames:
        print("No frames to save.")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 30, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    for frame in output_video_frames:
        out.write(frame)
    out.release()

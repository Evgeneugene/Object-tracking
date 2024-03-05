from ultralytics import YOLO
import cv2
import math
from collections import defaultdict
import numpy as np
import time
from tqdm import tqdm
import sys
from collections import defaultdict, deque


def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : tuple
        (x1, y1, x2, y2) of the first bounding box.
    bb2 : tuple
        (x1, y1, x2, y2) of the second bounding box.
    """
    assert bb1[0] < bb1[2]
    assert bb1[1] < bb1[3]
    assert bb2[0] < bb2[2]
    assert bb2[1] < bb2[3]

    # Determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The area of intersection rectangle
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # The area of both bounding boxes
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    # The intersection over union (IoU)
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    return iou


class SimpleTracker:
    def __init__(self, max_lost=5, iou_threshold=0.4):
        self.max_lost = max_lost
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.next_id = 1

    def update(self, detections):
        for track in self.tracks:
            track['lost'] += 1

        for det in detections:
            (x1, y1, x2, y2, conf, cls_id) = det

            best_track = None
            max_iou = self.iou_threshold
            for track in self.tracks:
                track_bb = (track['x1'], track['y1'], track['x2'], track['y2'])
                iou = get_iou(track_bb, (x1, y1, x2, y2))
                if iou > max_iou:
                    max_iou = iou
                    best_track = track

            if best_track is not None:
                best_track['x1'] = x1
                best_track['y1'] = y1
                best_track['x2'] = x2
                best_track['y2'] = y2
                best_track['lost'] = 0
                best_track['conf'] = conf
                best_track['cls_id'] = cls_id
            elif conf > 0.5:
                self.tracks.append({'id': self.next_id, 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'lost': 0,
                                    'conf': conf, 'cls_id': cls_id})
                self.next_id += 1

        self.tracks = [track for track in self.tracks if track['lost'] <= self.max_lost]

        return self.tracks


def process_batch(model, frames, tracker, class_names, track_histories):
    batch_detections = model(frames, stream=True, classes=list(class_names.keys()), verbose=False)

    for i, results in enumerate(batch_detections):
        # print(i, results)
        img = frames[i]
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy
            data = self.boxes.data.cpu().tolist()
            # x1, y1, x2, y2, conf, cls_id = map(int, box[:4]) + [box[4], int(box[5])]
            # detections.append((x1, y1, x2, y2, conf, cls_id))
    #     tracks = tracker.update(detections)
    #     print("P2")
    #     # Draw the tracks
    #     for track in tracks:
    #         x1, y1, x2, y2 = track['x1'], track['y1'], track['x2'], track['y2']
    #         obj_id = track['id']
    #         cls_id = track['cls_id']
    #         conf = track['conf']
    #
    #         # Calculate the center coordinates of the bounding box
    #         cx = (x1 + x2) // 2
    #         cy = (y1 + y2) // 2
    #
    #         # Update history of track
    #         track_histories[obj_id].append({'cx': cx, 'cy': cy, 'current_frames': current_frames})
    #
    #         # Draw a circle at the center and display the object ID
    #         cv2.circle(img, (cx, cy), 5, (0, 255, 0), -1)
    #
    #         # Calculate dynamic font size based on bounding box size
    #         box_width = x2 - x1
    #         box_height = y2 - y1
    #         avg_box_size = (box_width + box_height) / 2
    #         font_scale = max(0.4, avg_box_size / 250)  # Adjust the denominator to control the scaling factor
    #
    #         # Display object ID with dynamic font size
    #         cv2.putText(img, f"ID: {obj_id}", (cx - 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255),
    #                     2)
    #
    #         # Draw bounding box
    #         cv2.rectangle(img, (x1, y1), (x2, y2), (0, 140, 255), 3)
    #
    #         # Get confidence and class name
    #         confidence_rounded = math.ceil(track['conf'] * 100) / 100
    #         class_name = class_names[cls_id]
    #
    #         font_thickness = max(1, int(font_scale * 2))
    #         text = f"{class_name}, {confidence_rounded}"
    #         # Get text size
    #         (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    #
    #         # Set background rectangle coordinates
    #         background_tl = x1, y1 - text_height - 3
    #         background_br = x1 + text_width, y1
    #
    #         # Draw filled rectangle for text background
    #         cv2.rectangle(img, background_tl, background_br, (0, 140, 255), -1)  # Orange background
    #
    #         # Draw text
    #         text_position = x1, y1 - 2
    #         cv2.putText(img, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
    #     print("P3")
    #     # Draw tracking line
    #     for obj_id in track_histories:
    #         if len(track_histories[obj_id]) > 0 and current_frames - track_histories[obj_id][0]['current_frames'] >= 50:
    #             track_histories[obj_id].popleft()
    #
    #         for i in range(1, len(track_histories[obj_id])):
    #             start_point = (track_histories[obj_id][i - 1]['cx'], track_histories[obj_id][i - 1]['cy'])
    #             end_point = (track_histories[obj_id][i]['cx'], track_histories[obj_id][i]['cy'])
    #             cv2.line(img, start_point, end_point, (0, 255, 0), 2)
    #     print(frames)
    #     out.write(img)


if __name__ == "__main__":
    args = sys.argv[1:]
    output_filename = "output_videos/" + args[0]
    # Define your class names based on IDs
    class_names = {
        2: 'car',
        3: 'motorcycle',
        5: 'bus',
        7: 'truck'
    }

    # Open the video file
    video_path = 'input_videos/min_trim.mkv'
    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_filename, fourcc, frame_rate, size)

    # Load model
    model = YOLO("models/yolov8n.pt")
    tracker = SimpleTracker()
    progress_bar = tqdm(total=total_frames, desc='Processing Frames', unit='frames')
    track_histories = defaultdict(lambda: deque(maxlen=50))

    batch_size = 4  # Define your batch size here
    frames_batch = []
    current_frames = 0
    while True:
        success, img = cap.read()
        if not success and len(frames_batch) == 0:
            break  # Exit if the video ends and no frames are left in the batch
        if success:
            frames_batch.append(img)
            if len(frames_batch) < batch_size and success:
                continue  # Collect enough frames for a batch

        # Process the batch
        process_batch(model, frames_batch, tracker, class_names, track_histories)
        print("P3")
        current_frames += len(frames_batch)
        print(current_frames)
        progress_bar.update(len(frames_batch))
        frames_batch = []  # Clear the batch

    cap.release()
    cv2.destroyAllWindows()
    progress_bar.close()

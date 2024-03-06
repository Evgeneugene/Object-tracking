import cv2
import math


def draw_results(img, tracks, track_histories, current_frames, class_names):
    """
    Draws tracking lines and bounding boxes with labels on the image for each tracked object.

    Parameters:
    - img (numpy.ndarray): The image on which to draw the tracking information.
    - tracks (list): A list of dictionaries, each representing a tracked object with keys for bounding box coordinates ('x1', 'y1', 'x2', 'y2'), object ID ('id'), class ID ('cls_id'), and confidence score ('conf').
    - track_histories (defaultdict): A defaultdict containing the history of each track. Each key is an object ID, and the value is a deque of dictionaries, each with keys 'cx', 'cy', and 'current_frames', representing the center point of the object's bounding box at a given frame.
    - current_frames (int): The current frame number in the video sequence being processed. (for removing trails that are over 50 frames old)
    - class_names (dict): A dictionary mapping class IDs  class names.

    The function draws a line for the motion trail of each tracked object, based on the history stored in 'track_histories'. It also draws bounding boxes around each tracked object with their IDs and class names and adds a filled rectangle behind the class name and confidence score for better readability.
    """
    # Draw tracking line
    for obj_id in track_histories:
        if len(track_histories[obj_id]) > 0 and current_frames - track_histories[obj_id][0]['current_frames'] >= 50:
            track_histories[obj_id].popleft()

        for i in range(1, len(track_histories[obj_id])):
            start_point = (
                track_histories[obj_id][i - 1]['cx'], track_histories[obj_id][i - 1]['cy'])
            end_point = (track_histories[obj_id][i]
                         ['cx'], track_histories[obj_id][i]['cy'])
            cv2.line(img, start_point, end_point, (0, 255, 0), 2)

    # Draw the tracks
    for track in tracks:
        x1, y1, x2, y2 = int(track['x1']), int(
            track['y1']), int(track['x2']), int(track['y2'])
        obj_id = int(track['id'])
        cls_id = int(track['cls_id'])
        confidence_rounded = math.ceil(track['conf'] * 100) / 100

        class_name = class_names[cls_id]

        # Calculate the center coordinates of the bounding box
        cx = int((x1 + x2) // 2)
        cy = int((y1 + y2) // 2)

        # Update history of track
        track_histories[obj_id].append(
            {'cx': cx, 'cy': cy, 'current_frames': current_frames})

        # Draw a circle at the center and display the object ID
        cv2.circle(img, (cx, cy), 5, (0, 255, 0), -1)

        # Calculate dynamic font size based on bounding box size
        box_width = x2 - x1
        box_height = y2 - y1
        avg_box_size = (box_width + box_height) / 2
        # Adjust the denominator to control the scaling factor
        font_scale = max(0.4, avg_box_size / 250)

        # Display object ID with dynamic font size
        cv2.putText(img, f"ID: {obj_id}", (cx - 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255),
                    2)

        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 140, 255), 3)

        # Set text options
        font_thickness = max(1, int(font_scale * 2))
        text = f"{class_name}, {confidence_rounded}"

        # Get text size
        (text_width, text_height), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

        # Set background rectangle coordinates
        background_tl = x1, y1 - text_height - 3
        background_br = x1 + text_width, y1

        # Draw filled rectangle for text background
        cv2.rectangle(img, background_tl, background_br,
                      (0, 140, 255), -1)  # Orange background

        # Draw text
        text_position = x1, y1 - 2
        cv2.putText(img, text, text_position, cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), font_thickness)

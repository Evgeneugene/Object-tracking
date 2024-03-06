
def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters:
    - bb1 (tuple): (x1, y1, x2, y2) of the first bounding box.
    - bb2 (tuple): (x1, y1, x2, y2) of the second bounding box.

    Returns:
    - iou (float): The intersection over union 
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
    """
    A simple object tracking class that tracks objects based on their bounding box coordinates
    and updates their positions with new detections.

    Attributes:
        max_lost (int): The maximum number of frames an object can be lost (not detected) before it's removed from tracking.
        iou_threshold (float): The minimum Intersection Over Union (IoU) value to consider two bounding boxes as the same object.
        tracks (list): A list of dictionaries, each representing a tracked object.
        next_id (int): The next available unique ID for a new object to be tracked.
        class_names (dict): A dictionary mapping class IDs to their corresponding class names.
    """

    def __init__(self, class_names, max_lost=5, iou_threshold=0.35):
        self.max_lost = max_lost
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.next_id = 1
        self.class_names = class_names


    def update(self, detections):
        """
        Updates the tracker with new object detections, modifying the tracked objects list
        based on detection data.

        Parameters:
        - detections (list): A list of detections (x1, y1, x2, y2, conf, cls_id)

        Returns:
            list: The updated list of tracked objects after processing the new detections.
        """

        for track in self.tracks:
            track['lost'] += 1

        for det in detections:
            (x1, y1, x2, y2, conf, cls_id) = det

            if int(cls_id) not in self.class_names:
                continue

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

        self.tracks = [
            track for track in self.tracks if track['lost'] <= self.max_lost]

        return self.tracks

import cv2
from tqdm import tqdm
import sys
from collections import defaultdict, deque
from pathlib import Path
import openvino as ov
import image_processing
from tracking import SimpleTracker
from draw import draw_results
import time


def init_model(DET_MODEL_NAME, models_dir):
    """
    Initializes and compiles the object detection model.

    Parameters:
    - DET_MODEL_NAME (str): The name of the detection model to be initialized. (Already converted to OpenVino IR)
    - models_dir (str or Path): The directory where the model files are stored.

    Returns:
    - det_compiled_model: An OpenVINO compiled model ready for inference.
    """

    models_dir = Path(models_dir)

    # object detection model
    det_model_path = models_dir / \
        f"{DET_MODEL_NAME}_openvino_model/{DET_MODEL_NAME}.xml"

    core = ov.Core()

    device = "CPU"

    core = ov.Core()

    det_ov_model = core.read_model(det_model_path)

    det_compiled_model = core.compile_model(det_ov_model, device)

    return det_compiled_model


def main(args):
    """
    Main function to process the video file and perform object tracking.

    Parameters:
    - args (list): Command-line arguments passed to the script. 
                   The first argument is expected to be the input video filename.

    This function processes the input video frame by frame, performs object
    detection, updates the tracker, and draws the tracking results on each frame.
    The processed frames are saved to an output video file.
    """
    # Turns input_videos/filename.extension ==> output_videos/filename_detections.extension
    input_filename = args[0]
    input_filename_no_ext = ".".join(input_filename.split(".")[:-1])
    file_extension = input_filename.split(".")[-1]
    output_filename = "/app/output_videos/" + \
        input_filename_no_ext + "_detections." + file_extension

    # Define your class names based on IDs
    class_names = {
        2: 'car',
        3: 'motorcycle',
        5: 'bus',
        7: 'truck'
    }

    # Open the video file
    video_path = '/app/input_videos/' + input_filename
    cap = cv2.VideoCapture(video_path)

    # Options for output video writing
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_filename, fourcc, frame_rate, size)

    # Load model
    model = init_model("yolov8n", "/app/models")

    # Initialize the simple tracker
    tracker = SimpleTracker(class_names)

    progress_bar = tqdm(total=total_frames,
                        desc='Processing Frames', unit='frames')

    # Store up to 50 points for each track
    track_histories = defaultdict(lambda: deque(maxlen=50))

    current_frames = 0
    total_time = 0
    try:
        while True:
            success, img = cap.read()
            if not success:
                break  # Exit if the video ends or there are no frames left

            start_time = time.time()
            current_frames += 1

            # Make predictioins
            preprocessed_image = image_processing.preprocess_image(img)
            input_tensor = image_processing.image_to_tensor(preprocessed_image)
            result = model(input_tensor)
            boxes = result[model.output(0)]
            input_hw = input_tensor.shape[2:]
            detections = image_processing.postprocess(
                pred_boxes=boxes, input_hw=input_hw, orig_img=img)[0]['det']

            # Update the tracker with the new detections
            tracks = tracker.update(detections)

            draw_results(img, tracks, track_histories,
                         current_frames, class_names)

            progress_bar.update(1)
            out.write(img)

            end_time = time.time()  # Record the end time after processing the frame
            # Calculate the processing time for the frame
            processing_time = end_time - start_time
            total_time += processing_time

    except KeyboardInterrupt:
        print("Keyboard Interrupt detected. Exiting gracefully.")
    finally:
        cap.release()
        progress_bar.close()
        if current_frames > 0:
            average_fps = current_frames / total_time
            print(f"Input FPS: {frame_rate}")
            print(f"Average FPS: {average_fps:.2f}")
            print(f"Average speed: {average_fps / frame_rate:.2f} of real time")
        else:
            print("No frames processed")
        


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)

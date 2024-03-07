# EV CV Test Task – Simple Tracker using YOLOv8n

I have read articles about comparison of different pre-trained models for object detection and decided to choose YOLOv8n as a model being both accurate and fast enough for real-time perfomance.

## Installation
```bash
git clone git@github.com:Evgeneugene/Object-tracking.git
cd Object-tracking
docker build -t get_tracks . 
```
## Usage
You can put any video in `input_videos` folder and pass it to docker container
```bash
docker run -v $PWD:/app -it get_tracks min_trim.mkv 
``` 
The resulted video with visualized tracks will be generated in `output_videos` folder.

## Approach description

This approach leverages OpenVino's efficiency for fast inference on CPU and Non-maximum suppression for accurate and reliable object tracking in video streams.

1. **Model Initialization**: The system initializes the object detection model by loading a pre-trained model specified by `DET_MODEL_NAME`. This model, converted to the OpenVino Intermediate Representation (IR), allows efficient execution on various hardware.

2. **Video Processing**: The system processes the input video frame by frame, performing the following steps for each frame:
    - **Preprocessing**: The frame is resized and padded to meet the detection model's input requirements without distorting the aspect ratio. The `letterbox` function ensures the new image has the correct dimensions and is padded with a specific color.
    - **Detection**: The preprocessed frame is passed through the object detection model, which outputs bounding boxes identifying detected objects' locations. These boxes come with confidence scores and class IDs.
    - **Postprocessing**: The raw detections are refined using Non-Maximum Suppression (NMS), essential for reducing redundancy among detected bounding boxes by eliminating overlapping boxes based on their confidence scores and Intersection over Union (IoU).
    - **Tracking**: A simple tracking algorithm matches detected objects across frames based on their bounding box coordinates, using IoU for comparison. New detections become new tracks, and tracks not updated for a specified number of frames are removed.

3. **Output Generation**: The tracking results, including the bounding boxes and class IDs of tracked objects, are drawn on the frames. These modified frames are compiled into an output video, visually representing the tracking process.

## Perfomance

1. **Speed** The program writes the output video with visualized tracks on average 10fps, whereas the input video framerate is 25fps, so there is a little overhead, unfortunately. As a simple solution, we can skip some frames.
2. **Accuracy** In general, the tracks are qiute accurate, but sometimes some perturbations arise. It works much more accurately with a bit larger model as YOLOv8s or YOLOv8s, but the speed suffers.

## Demo

![Preview](img_demo/photo.png)

![Video link](https://youtu.be/zjbV-xx7T8g)
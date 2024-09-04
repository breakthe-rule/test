from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("best.pt")

# Open the video file
video_path = "12sec.mp4"
cap = cv2.VideoCapture(video_path)

# Resize dimensions
resize_width, resize_height = 416, 640

# Store the track history
track_history = defaultdict(lambda: [])

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Resize the frame
        original_height, original_width = frame.shape[:2]
        frame_resized = cv2.resize(frame, (resize_width, resize_height))

        # Run YOLOv8 tracking on the resized frame
        results = model.track(frame_resized, persist=True)
        
        if results[0] is not None and results[0].boxes.id is not None:
            # Get the boxes and track IDs
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            # Visualize the results on the resized frame
            annotated_frame = results[0].plot()

            # Adjust bounding boxes and tracking lines for the original frame size
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                # Convert box coordinates from resized frame to original size
                x1 = int((x - w / 2) * original_width / resize_width)
                y1 = int((y - h / 2) * original_height / resize_height)
                x2 = int((x + w / 2) * original_width / resize_width)
                y2 = int((y + h / 2) * original_height / resize_height)

                track = track_history[track_id]
                track.append((float(x1), float(y1)))  # x, y top-left point
                if len(track) > 30:  # retain 90 tracks for 90 frames
                    track.pop(0)

                # Draw the tracking lines
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

                # Draw the bounding box on the original frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Put class label (assumed to be available in results)
                label = f'ID: {track_id}'
                cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Display the frame with tracking and bounding boxes
            cv2.imshow("YOLOv8 Tracking", frame)

        else:
            # If no results or boxes, just display the resized frame
            cv2.imshow("YOLOv8 Tracking", frame_resized)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()

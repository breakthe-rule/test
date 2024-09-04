import cv2
import pandas as pd
from ultralytics import YOLO

# Load the model
model = YOLO('best.pt')

# Open the video file
cap = cv2.VideoCapture('12sec.mp4')

# Ensure video opened successfully
if not cap.isOpened():
    print("Error opening video file")
    exit()

# Load class names (replace with actual class names for your model)
class_list = ['shuttle']  # Update this with your actual class names

# Resize dimensions to match model input size
resize_width, resize_height = 640, 640

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Print frame size for debugging
    print(f"Original frame size: {frame.shape[1]}x{frame.shape[0]}")
    
    # Resize frame to match model input size
    frame = cv2.resize(frame, (resize_width, resize_height))
    
    # Perform object detection
    results = model.predict(frame)
    a = results[0].boxes.data
    a = a.detach().cpu().numpy()
    
    # Convert detections to DataFrame
    px = pd.DataFrame(a, columns=['x1', 'y1', 'x2', 'y2', 'confidence', 'class_id']).astype("float")

    # Convert bounding box coordinates from resized image to original size
    original_height, original_width = frame.shape[:2]
    x_scale = original_width / resize_width
    y_scale = original_height / resize_height

    # Loop through detections and draw bounding boxes
    for index, row in px.iterrows():
        x1, y1, x2, y2, confidence, class_id = row
        class_id = int(class_id)
        class_name = class_list[class_id]
        
        if 'shuttle' in class_name:
            # Convert bounding box coordinates to original frame size
            x1 = int(x1 * x_scale)
            y1 = int(y1 * y_scale)
            x2 = int(x2 * x_scale)
            y2 = int(y2 * y_scale)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Put class label
            label = f'{class_name} {confidence:.2f}'
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display the frame with bounding boxes
    cv2.imshow('Detection', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

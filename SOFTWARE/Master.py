import cv2
import torch
from PIL import Image
import numpy as np

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/exp2/weights/best.pt')  # Adjust path as needed

# Initialize the video capture object
cap = cv2.VideoCapture(0)  # 0 for the default camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

class HomogeneousBgDetector:
    def __init__(self):
        pass

    def detect_objects(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 19, 5)
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        objects_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 2000:
                objects_contours.append(cnt)

        return objects_contours

# Initialize the Aruco detector
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
parameters = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# Initialize the object detector
detector_obj = HomogeneousBgDetector()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break

    # Shadow Removal Step
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    _, _, v = cv2.split(hsv)
    shadow_thresh = cv2.threshold(v, 50, 255, cv2.THRESH_BINARY)[1]  # Threshold value might need tuning
    kernel = np.ones((5,5), np.uint8)
    shadow_mask = cv2.morphologyEx(shadow_thresh, cv2.MORPH_DILATE, kernel)

    # Use shadow mask to update the frame
    frame[shadow_mask == 0] = [0, 0, 0]

    # Convert the frame to RGB (from BGR) and then to PIL Image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    
    # Perform inference
    results = model(pil_img, size=640)  # Pass the PIL image and specify the input size

    # Convert results to pandas DataFrame
    results_df = results.pandas().xyxy[0]

    # Display the most likely object on the frame
    if not results_df.empty:
        # Get the row with the highest confidence
        most_likely = results_df.loc[results_df['confidence'].idxmax()]
        confidence, class_name = most_likely['confidence'], most_likely['name']
        label = f"{class_name} {confidence:.2f}"
        
        # Display the label on the frame
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, lineType=cv2.LINE_AA)

        # Detect Aruco markers
        corners, _, _ = aruco_detector.detectMarkers(frame)
        if corners:
            int_corners = np.intp(corners)

            # Draw crosshair on the frame
            height, width = frame.shape[:2]
            crosshair_img = cv2.line(frame.copy(), (width // 2, 0), (width // 2, height), (0, 0, 255), 2)
            crosshair_img = cv2.line(crosshair_img, (0, height // 2), (width, height // 2), (0, 0, 255), 2)

            # Calculate Aruco perimeter and pixel to cm ratio
            aruco_perimeter = cv2.arcLength(corners[0], True)
            pixel_cm_ratio = aruco_perimeter / 200.00

            # Detect objects
            contours = detector_obj.detect_objects(frame)

            for cnt in contours:
                rect = cv2.minAreaRect(cnt)
                (x, y), (w, h), angle = rect
                box = cv2.boxPoints(rect)
                box = np.int0(box)

                dist_to_center = np.sqrt((x - width // 2) ** 2 + (y - height // 2) ** 2)
                if dist_to_center <= max(w, h) / 2:
                    if class_name == "Water Bottle Cap (circle shape)":
                        (x, y), radius = cv2.minEnclosingCircle(cnt)
                        center = (int(x), int(y))
                        radius = int(radius)
                        object_diameter = 2 * radius / pixel_cm_ratio
                        cv2.putText(crosshair_img, "Diameter {} mm".format(round(object_diameter, 0)), (center[0] - 100, center[1]), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)
                    elif class_name == "Steel Plate (square shape)":
                        object_length = w / pixel_cm_ratio
                        cv2.putText(crosshair_img, "Length {} mm".format(round(object_length, 0)), (int(x - 100), int(y)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)
                    elif class_name == "External Hard Drive (rectangular shape)":
                        object_length = w / pixel_cm_ratio
                        object_width = h / pixel_cm_ratio
                        cv2.putText(crosshair_img, "Length {} mm".format(round(object_length, 0)), (int(x - 100), int(y - 20)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)
                        cv2.putText(crosshair_img, "Width {} mm".format(round(object_width, 0)), (int(x - 100), int(y + 20)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)

        else:
            crosshair_img = frame

    else:
        crosshair_img = frame

    # Display the resulting frame
    cv2.imshow('Master - YOLOv5 and Object Measurement', crosshair_img)

    # Break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()

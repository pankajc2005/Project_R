from ultralytics import YOLO 
import cv2
import math
import pygame

pygame.mixer.init()
siren_sound = pygame.mixer.Sound('sirenn.mp3')

# For Running real-time from webcam
cap = cv2.VideoCapture(0)

# Uncomment below For Running real-time from Video

# video_path = 'demo.mp4'  # Replace with video file path
# cap = cv2.VideoCapture(video_path)  # 0 corresponds to the default webcam

# YOLO Train model (replace 'best.pt' with actual trained model path)
model = YOLO('best.pt')

# Classes Name that was in dataset 

classnames = [
    'Local_train', 'Train', 'Train_Side_View','Vande_bharat'
]

frame_counter = 0  

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 480))
    
    # Perform inference with YOLOv8
    
    result = model(frame, stream=True)

    # Process bounding boxes and display results
    
    for info in result:
        boxes = info.boxes
        for box in boxes:
            confidence = box.conf[0]
            confidence = math.ceil(confidence * 100)
            class_index = int(box.cls[0])
            class_name = classnames[class_index]

            #Increase confidence for better accuracy

            if confidence > 75 and class_name in classnames:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Display bounding box and class label
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                cv2.putText(frame, f'{class_name} {confidence}%', (x1 + 8, y1 + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Play siren sound if detected Train is in the specified class list
                
                if class_name in ['Local_train', 'Train', 'Train_Side_View','Vande_bharat']:
                    siren_sound.play()
                    siren_duration = 1000
                    print(f"ALERT: {class_name} detected! Siren is playing.")
                    pygame.time.wait(int(siren_sound.get_length() * 1000)) 

                    # Save the frame where Train is detected
                    
                    frame_counter += 1
                    frame_filename = f"detected_frame_{frame_counter}.jpg"  # Unique filename
                    cv2.imwrite(frame_filename, frame)  # Save the frame as an image
                    print(f"Frame saved as {frame_filename}")

    # Display the frame with detected Train
    
    cv2.imshow('Train Detection', frame)

    # Press 'Esc' key to exit the loop
    
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the webcam and close the window

cap.release()
cv2.destroyAllWindows()

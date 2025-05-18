import cv2
import mediapipe as mp
import math
import pyautogui
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize video capture
cap = cv2.VideoCapture(0)

# Variables for volume control
min_distance = 20  # Minimum distance between fingers (pixels)
max_distance = 200  # Maximum distance for volume mapping
last_volume_change = 0
volume_change_delay = 0.2  # Delay between volume changes (seconds)

def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def set_volume(distance):
    # Map distance to volume (0-100)
    volume = ((distance - min_distance) / (max_distance - min_distance)) * 100
    volume = max(0, min(100, volume))  # Clamp between 0 and 100
    
    # Use pyautogui to control system volume
    # This simulates pressing volume up/down keys
    current_volume = 50  # Placeholder: actual volume reading not implemented
    volume_diff = volume - current_volume
    
    if abs(volume_diff) > 5:  # Only change if significant difference
        if volume_diff > 0:
            pyautogui.press('volumeup')
        else:
            pyautogui.press('volumedown')
            
    return volume

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Flip frame for mirror effect
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame for hand detection
        results = hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            for hand_lm in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)
                
                # Get thumb tip (landmark 4) and index finger tip (landmark 8)
                h, w, _ = frame.shape
                thumb_tip = hand_lm.landmark[4]
                index_tip = hand_lm.landmark[8]
                
                # Convert normalized coordinates to pixel values
                thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
                index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)
                
                # Calculate distance between thumb and index finger
                distance = calculate_distance(thumb_x, thumb_y, index_x, index_y)
                
                # Draw line between thumb and index finger
                cv2.line(frame, (thumb_x, thumb_y), (index_x, index_y), (0, 255, 0), 2)
                
                # Update volume if enough time has passed
                current_time = time.time()
                if current_time - last_volume_change > volume_change_delay:
                    volume = set_volume(distance)
                    last_volume_change = current_time
                    
                    # Display volume level
                    cv2.putText(frame, f'Volume: {int(volume)}%', 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                              1, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow('Hand Gesture Volume Control', frame)
        
        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Program terminated by user")

finally:
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    hands.close()

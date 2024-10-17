import cv2
import mediapipe as mp
import pyautogui
import math

# Disable PyAutoGUI fail-safe
pyautogui.FAILSAFE = False

# Initialize Mediapipe for face mesh detection
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Capture webcam video
cap = cv2.VideoCapture(0)

# Screen size to calculate mouse movement proportion
screen_width, screen_height = pyautogui.size()

# Define the key landmarks for detecting face direction
NOSE_TIP = 1  # Nose tip landmark index in Mediapipe
CHIN = 152    # Chin landmark index in Mediapipe

# Eye landmarks
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Define a speed factor for scaling mouse speed
SPEED_FACTOR = 300  # Adjust this value to control sensitivity
BLINK_THRESHOLD = 0.33  # Threshold for detecting blink

# Initialize variables for blink detection
blink_counter = 0
blink_frames_threshold = 6  # Number of consecutive frames to consider a blink

# Function to calculate the distance between two points
def calculate_distance(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)

# Function to calculate the Eye Aspect Ratio (EAR) for blink detection
def calculate_ear(landmarks, eye_landmarks):
    # Vertical distances
    vertical_1 = calculate_distance(landmarks[eye_landmarks[1]], landmarks[eye_landmarks[5]])
    vertical_2 = calculate_distance(landmarks[eye_landmarks[2]], landmarks[eye_landmarks[4]])
    
    # Horizontal distance
    horizontal = calculate_distance(landmarks[eye_landmarks[0]], landmarks[eye_landmarks[3]])
    
    # EAR formula
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear

def get_face_direction(landmarks):
    # Get the x, y coordinates of the nose tip
    nose_x = landmarks[NOSE_TIP].x
    nose_y = landmarks[NOSE_TIP].y
    
    # Print the nose X and Y values for debugging
    # print(f"Nose X: {nose_x:.4f}, Nose Y: {nose_y:.4f}")  # Rounded for readability

    # Return the nose position
    return nose_x, nose_y

def calculate_speed(threshold, current_pos):
    # Calculate the distance from the threshold and scale the speed
    distance = abs(current_pos - threshold)
    return distance * pow(SPEED_FACTOR, 1.1)



while True:
    success, image = cap.read()
    if not success:
        break

    # Convert the image color to RGB (for Mediapipe)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(image_rgb)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            # Draw face landmarks (optional, for visualization)
            mp_drawing.draw_landmarks(
                image,
                face_landmarks,
                mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
            )

            # Get face movement direction based on nose position
            nose_x, nose_y = get_face_direction(face_landmarks.landmark)

            # Calculate the dynamic speed based on the distance from the threshold (0.5 for both x and y)
            if nose_x < 0.50:
                move_speed_x = calculate_speed(0.51, nose_x)
                pyautogui.move(move_speed_x, 0)  # Look left
            elif nose_x > 0.55:
                move_speed_x = calculate_speed(0.54, nose_x)
                pyautogui.move(-move_speed_x, 0)   # Look right
            
            if nose_y < 0.63:
                move_speed_y = calculate_speed(0.63, nose_y)
                pyautogui.move(0, -move_speed_y)  # Look up
            elif nose_y > 0.7:
                move_speed_y = calculate_speed(0.7, nose_y)
                pyautogui.move(0, move_speed_y)   # Look down

            # Calculate EAR for both eyes
            left_ear = calculate_ear(face_landmarks.landmark, LEFT_EYE)
            right_ear = calculate_ear(face_landmarks.landmark, RIGHT_EYE)
            #avg_ear = (left_ear + right_ear) / 2.0

            print(f"Left EAR: {left_ear:.4f} | Right EAR: {right_ear:.4f}")

            # Blink detection
            if left_ear < BLINK_THRESHOLD and right_ear > BLINK_THRESHOLD and (abs(left_ear - right_ear) > 0.06):
                blink_counter += 1
            else:
                if blink_counter >= blink_frames_threshold:
                    print("Blink detected, performing click!")
                    pyautogui.click()  # Perform a left-click on blink
                blink_counter = 0  # Reset blink counter

    # Show the video feed with landmarks
    cv2.imshow('Face Tracking', image)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and destroy windows
cap.release()
cv2.destroyAllWindows()

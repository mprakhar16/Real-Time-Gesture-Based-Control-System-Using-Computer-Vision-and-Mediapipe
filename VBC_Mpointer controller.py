# Import required libraries
import cv2                                                   # OpenCV for computer vision tasks
import mediapipe as mp                                       # Mediapipe for hand tracking
import math                                                  # Math for distance calculation
import screen_brightness_control as sbc                      # For controlling screen brightness
import numpy as np                                           # Numpy for numerical operations
import pyautogui                                             # To simulate mouse movement and clicks
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL

# Initialize Audio device (PyCaw)
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)

# Get the system volume range
minVol, maxVol, _ = volume.GetVolumeRange()

# Initialize variables
volBar = 400
volPer = 0
BrightBar = 400
BrightPer = 0
click_state = False  # Variable to track click state for the left hand

# Initialize MediaPipe Hands module
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=2,                       # Allow detection of both hands
                      min_detection_confidence=0.7,          # Minimum confidence for detecting a hand
                      min_tracking_confidence=0.7)           # Minimum confidence for tracking a hand
mpDraw = mp.solutions.drawing_utils                          # Utility for drawing hand landmarks

# Set the webcam settings (resolution)
wCam, hCam = 1280, 720                                       # Width and height of the webcam feed
cap = cv2.VideoCapture(0)                                    # Initialize the webcam (0 is the default camera)
cap.set(3, wCam)                                             # Set the width of the webcam feed
cap.set(4, hCam)                                             # Set the height of the webcam feed

# Start the loop to process the webcam feed
while True:
    success, img = cap.read()                                # Capture a frame from the webcam
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)            # Convert the captured image to RGB
    results = hands.process(imgRGB)                          # Process the image with the hand tracking solution

    # Check if any hand landmarks are detected
    if results.multi_hand_landmarks and results.multi_handedness:
        # Loop through all detected hands
        for i, handLms in enumerate(results.multi_hand_landmarks):
            # Draw landmarks of the hand on the image
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            # Determine if the current hand is left or right
            hand_label = results.multi_handedness[i].classification[0].label  # 'Left' or 'Right'

            # Create a list to store the positions of all landmarks
            lmList = []
            for id, lm in enumerate(handLms.landmark):                  # Loop through all hand landmarks
                h, w, c = img.shape                                     # Get the height, width, and channels of the image
                cx, cy = int(lm.x * w), int(lm.y * h)                   # Convert normalized coordinates to pixel values
                lmList.append((id, cx, cy))                             # Add the landmark ID and coordinates to the list

            # If the hand is right, control volume and brightness
            if hand_label == "Right" and len(lmList) >= 8:
                x1, y1 = lmList[4][1], lmList[4][2]  # Thumb tip
                x2, y2 = lmList[16][1], lmList[16][2]  # Middle finger tip
                x3, y3 = lmList[8][1], lmList[8][2]  # Index finger tip
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # Midpoint between thumb and middle finger
                cx1, cy1 = (x1 + x3) // 2, (y1 + y3) // 2  # Midpoint between thumb and index finger

                # Draw circles and lines for visual feedback
                cv2.circle(img, (x1, y1), 8, (255, 0, 255), cv2.FILLED)
                cv2.circle(img, (x2, y2), 8, (255, 0, 255), cv2.FILLED)
                cv2.circle(img, (x3, y3), 8, (255, 0, 255), cv2.FILLED)
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cv2.line(img, (x1, y1), (x3, y3), (255, 0, 255), 3)

                # Calculate distances for volume and brightness control
                length = math.hypot(x2 - x1, y2 - y1)
                length1 = math.hypot(x3 - x1, y3 - y1)

                # Change midpoint color to blue if distance is 0, otherwise green
                vol_midpoint_color = (255, 0, 0) if length <3  else (0, 255, 0)
                bright_midpoint_color = (255, 0, 0) if length1 <3 else (0, 255, 0)

                # Draw midpoints
                cv2.circle(img, (cx, cy), 8, vol_midpoint_color, cv2.FILLED)
                cv2.circle(img, (cx1, cy1), 8, bright_midpoint_color, cv2.FILLED)

                # Volume control
                if length:
                    volValue = int(np.interp(length, [20, 150], [0, 100]))
                    volBar = int(np.interp(length, [20, 150], [400, 140]))
                    volPer = int(np.interp(length, [20, 150], [0, 100]))
                    volScalar = np.interp(volPer, [0, 100], [0.0, 1.0])
                    volume.SetMasterVolumeLevelScalar(volScalar, None)

                # Brightness control
                if length1:
                    brightness = int(np.interp(length1, [20, 150], [0, 100]))
                    sbc.set_brightness(brightness)
                    BrightBar = int(np.interp(length1, [20, 150], [400, 140]))
                    BrightPer = int(np.interp(length1, [20, 150], [0, 100]))

                # Display volume and brightness bars
                cv2.rectangle(img, (50, 140), (85, 400), (0, 0, 255), 3)
                cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 0, 255), cv2.FILLED)
                cv2.putText(img, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(img, "Volume", (40, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

                cv2.rectangle(img, (200, 140), (235, 400), (0, 255, 0), 3)
                cv2.rectangle(img, (200, int(BrightBar)), (235, 400), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, f'{int(BrightPer)} %', (190, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(img, "Brightness", (180, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

            # If the hand is left, control the mouse pointer
            if hand_label == "Left" and len(lmList) >= 8:
                lx1, ly1 = lmList[4][1], lmList[4][2]  # Thumb tip
                lx2, ly2 = lmList[8][1], lmList[8][2]  # Index finger tip
                lcx, lcy = (lx1 + lx2) // 2, (ly1 + ly2) // 2  # Midpoint between thumb and index finger

                # Map index finger coordinates to screen dimensions
                screen_width, screen_height = pyautogui.size()
                
                # Invert horizontal movement
                inverted_lx2 = wCam - lx2                                           # Flip the x-coordinate
                
                mouse_x = np.interp(inverted_lx2, [0, wCam], [0, screen_width])
                mouse_y = np.interp(ly2, [0, hCam], [0, screen_height])

                # Move the mouse pointer
                pyautogui.moveTo(mouse_x, mouse_y, duration=0.01)

                # Change midpoint color to blue if distance is 0, otherwise green
                click_distance = math.hypot(lx2 - lx1, ly2 - ly1)
                if click_distance < 5:
                    click_midpoint_color = (255, 0, 0)
                else:
                    click_midpoint_color = (0, 255, 0)
                    
                    
                # Draw line and midpoint
                cv2.line(img, (lx1, ly1), (lx2, ly2), (0, 255, 0), 3)             # Draw a line between thumb and index finger
                cv2.circle(img, (lcx, lcy), 8, click_midpoint_color, cv2.FILLED)


                # Detect click gesture
                if click_distance < 20:  # Threshold for click
                    if not click_state:
                        pyautogui.click()
                        click_state = True
                else:
                    click_state = False

    # Display the updated frame
    cv2.imshow("Volume, Brightness, and Mouse Control", img)

    # Exit on pressing 'ESC'
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

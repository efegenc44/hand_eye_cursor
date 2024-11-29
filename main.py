from operator import le
import cv2
import pyautogui
import mediapipe as mp

from enum import Enum

LEFT_PUPIL = 468
RIGTH_PUPIL = 473

ENTER_KEY = 13

class State(Enum):
    Config = 0
    Cursor = 1

pyautogui.FAILSAFE = False

face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_width, screen_height = pyautogui.size()

cam = cv2.VideoCapture(0)

right = None
left = None
up = None
down = None

def main():
    global right, left, up, down

    done = False
    state = State.Config
    current = "right"
    while not done:
        _success, frame = cam.read()
        frame = cv2.flip(frame, 1) 
        frame_height, frame_width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
        
        output = face_mesh.process(rgb_frame)
        
        pos = None
        # landmarks are normalised
        if landmarks := output.multi_face_landmarks:
            left_pupil = landmarks[0].landmark[LEFT_PUPIL]
            right_pupil = landmarks[0].landmark[RIGTH_PUPIL]

            x = int((left_pupil.x + right_pupil.x) * frame_width) // 2 
            y = int((left_pupil.y + right_pupil.y) * frame_height) // 2 

            pos = (x, y)
            cv2.circle(frame, pos, 3, (0, 255, 0))

        match state:
            case State.Config:
                match current:
                    case "right":
                        cv2.putText(frame, "En Sag Orta", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))

                        if cv2.waitKey(1) == ENTER_KEY:
                            right = pos

                        if right:
                            current = "left"

                    case "left":
                        cv2.putText(frame, "En Sol Orta", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))

                        if cv2.waitKey(1) == ENTER_KEY:
                            left = pos

                        if left:
                            current = "down"

                    case "down":
                        cv2.putText(frame, "En Asagi Orta", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))

                        if cv2.waitKey(1) == ENTER_KEY:
                            down = pos

                        if down:
                            current = "up"

                    case "up":
                        cv2.putText(frame, "En Yukari Orta", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))

                        if cv2.waitKey(1) == ENTER_KEY:
                            up = pos
            
                        if up:
                            current = "done"
            
                    case "done":
                        print("right", right)
                        print("left", left)
                        print("down", down)
                        print("up", up)
                        state = State.Cursor

            case State.Cursor:
                move_cursor(x, y)
                if cv2.waitKey(1) == ord("q"):
                    break

        cv2.imshow("Hand Eye Cursor", frame)

def move_cursor(x, y):
    y = (down[1] - y) / (down[1] - up[1]) * screen_height
    x = ((x - left[0]) / (right[0] - left[0])) * screen_width

    y = clamp(y, 0, screen_height) 
    x = clamp(x, 0, screen_width) 

    pyautogui.moveTo(x, screen_height - y, 0.1)

def clamp(value, minimum, maximum):
    if value < minimum:
        return minimum

    if value > maximum:
        return maximum
    
    return value

if __name__ == "__main__":
    main()
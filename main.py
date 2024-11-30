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

class Config:
    def __init__(self):
        self.right = None
        self.left = None
        self.down = None
        self.up = None

pyautogui.FAILSAFE = False

face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_width, screen_height = pyautogui.size()

cam = cv2.VideoCapture(0)

def main():
    config = Config()
    state = State.Config
    current = "right"
    while True:
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
                            config.right = pos

                        if config.right:
                            current = "left"

                    case "left":
                        cv2.putText(frame, "En Sol Orta", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))

                        if cv2.waitKey(1) == ENTER_KEY:
                            config.left = pos

                        if config.left:
                            current = "down"

                    case "down":
                        cv2.putText(frame, "En Asagi Orta", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))

                        if cv2.waitKey(1) == ENTER_KEY:
                            config.down = pos

                        if config.down:
                            current = "up"

                    case "up":
                        cv2.putText(frame, "En Yukari Orta", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))

                        if cv2.waitKey(1) == ENTER_KEY:
                            config.up = pos
            
                        if config.up:
                            current = "done"
            
                    case "done":
                        print("right", config.right)
                        print("left", config.left)
                        print("down", config.down)
                        print("up", config.up)
                        state = State.Cursor

            case State.Cursor:
                move_cursor(x, y, config)
                if cv2.waitKey(1) == ord("q"):
                    break

        cv2.imshow("Hand Eye Cursor", frame)

def move_cursor(x, y, config):
    y = (config.down[1] - y) / (config.down[1] - config.up[1]) * screen_height
    x = ((x - config.left[0]) / (config.right[0] - config.left[0])) * screen_width

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
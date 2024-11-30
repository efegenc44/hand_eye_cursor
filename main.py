import cv2
import pyautogui
import mediapipe as mp

from enum import Enum, IntEnum

def main():
    cam = cv2.VideoCapture(0)
    cursor = HandEyeCursor()

    while True:
        _, frame = cam.read()
        frame = cv2.flip(frame, 1)

        cursor.update(cursor.eye_position(frame), frame)

        if cv2.waitKey(1) == ord("q"):
            break

        cv2.imshow("Hand Eye Cursor", frame)

class HandEyeCursor:
    ENTER_KEY = 13
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    LEFT_PUPIL = 468
    RIGTH_PUPIL = 473

    class State(Enum):
        Config = 0
        Cursor = 1

        def __str__(self):
            match self:
                case self.Cursor: return "Cursor"
                case self.Config: return "Config"

    class Config(IntEnum):
        Right = 0
        Left = 1
        Down = 2
        Up = 3

        def __str__(self):
            match self:
                case self.Right: return "En Sag Orta"
                case self.Left: return "En Sol Orta"
                case self.Down: return "En Asagi Orta"
                case self.Up: return "En Yukari Orta"

    def __init__(self):
        # pyautogui by default quits when cursor goes one of the corners
        # to not let softlock yourself, but we can use 'q' to quit
        pyautogui.FAILSAFE = False

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
        self.screen_width, self.screen_height = pyautogui.size()
        self.current_state = self.State.Config
        self.current_config = self.Config.Right
        self.config = [None, None, None, None]

    def eye_position(self, frame):
        frame_height, frame_width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output = self.face_mesh.process(rgb_frame)

        eye = None
        # landmarks are normalised (means between [0, 1])
        if landmarks := output.multi_face_landmarks:
            left_pupil = landmarks[0].landmark[self.LEFT_PUPIL]
            right_pupil = landmarks[0].landmark[self.RIGTH_PUPIL]

            x = int((left_pupil.x + right_pupil.x) * frame_width) // 2
            y = int((left_pupil.y + right_pupil.y) * frame_height) // 2

            eye = (x, y)
            cv2.circle(frame, eye, 3, (0, 255, 0))

        return eye

    def update(self, eye, frame):
        text = f"State: {self.current_state}"
        size = cv2.getTextSize(text, self.FONT, 0.75, 1)[0]
        cv2.putText(frame, text, (0, frame.shape[0] - size[1]),
                    self.FONT, 0.75, (0, 0, 255), 2)

        text = "Cikmak icin 'q'"
        size = cv2.getTextSize(text, self.FONT, 0.75, 1)[0]
        cv2.putText(frame, text, (frame.shape[1] - size[0], frame.shape[0] - size[1]),
                    self.FONT, 0.75, (0, 0, 255), 2)

        match self.current_state:
            case self.State.Config:
                self.configuration(eye, frame)
            case self.State.Cursor:
                self.cursor(eye, frame)

    def cursor(self, eye, frame):
        text = f"Eye (in frame): {eye}"
        cv2.putText(frame, text, (30, 30), self.FONT, 1, (255, 0, 0), 2)

        screen_pos = self.eye_to_screen_pos(eye)
        text = f"Cursor (in screen): {screen_pos}"
        cv2.putText(frame, text, (30, 60), self.FONT, 1, (255, 0, 0), 2)

        # 0.1 is for smoothing the movement a bit
        pyautogui.moveTo(*screen_pos, 0.1)

    def configuration(self, eye, frame):
        text = f"Ekranin {self.current_config}sina bakip"
        cv2.putText(frame, text, (30, 30), self.FONT, 1, (255, 0, 0), 2)

        text = f"'Enter' tusuna basiniz."
        cv2.putText(frame, text, (30, 60), self.FONT, 1, (255, 0, 0), 2)

        text = f"Right: {self.config[self.Config.Right]}"
        cv2.putText(frame, text, (30, 120), self.FONT, 0.75, (255, 0, 0), 2)

        text = f"Left: {self.config[self.Config.Left]}"
        cv2.putText(frame, text, (30, 150), self.FONT, 0.75, (255, 0, 0), 2)

        text = f"Down: {self.config[self.Config.Down]}"
        cv2.putText(frame, text, (30, 180), self.FONT, 0.75, (255, 0, 0), 2)

        text = f"Up: {self.config[self.Config.Up]}"
        cv2.putText(frame, text, (30, 210), self.FONT, 0.75, (255, 0, 0), 2)

        if cv2.waitKey(1) == self.ENTER_KEY:
            self.register_config(eye)

    def register_config(self, value):
        if value is not None:
            self.config[self.current_config] = value
            self.current_config += 1

            print(f"value: {type(value)}; new config: {self.current_config}")

            if self.current_config > self.Config.Up:
                self.current_state = self.State.Cursor

    def eye_to_screen_pos(self, eye):
        [right, left, down, up] = self.config

        x = (eye[0] - left[0]) / (right[0] - left[0])
        y = (down[1] - eye[1]) / (down[1] - up[1])

        x = clamp(x * self.screen_width, 0, self.screen_width)
        y = clamp(y * self.screen_height, 0, self.screen_height)

        # (y = self.screen_height - y) beacuse we have flipped the frame
        return (int(x), int(self.screen_height - y))

def clamp(value, minimum, maximum):
    if value < minimum:
        return minimum

    if value > maximum:
        return maximum

    return value

if __name__ == "__main__":
    main()
import cv2
import pyautogui
import mediapipe as mp

from enum import Enum, IntEnum
import time

from utils import clamp, get_angle, get_distance

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

    def __init__(self, debug=False, reset_interval_seconds=1, dragging_threshold=0.6, double_click_threshold=0.3):
        # pyautogui by default quits when cursor goes one of the corners
        # to not let softlock yourself, but we can use 'q' to quit
        pyautogui.FAILSAFE = False

        self.debug = debug
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
        self.screen_width, self.screen_height = pyautogui.size()
        self.current_state = self.State.Config
        self.current_config = self.Config.Right
        self.config = [None, None, None, None]

        self.left_click_triggered = False
        self.right_click_triggered = False
        self.last_click_time = time.time()
        self.drag_start_time = 0
        self.dragging = False
        self.dragging_threshold = dragging_threshold
        self.double_click_threshold = double_click_threshold
        self.reset_interval_seconds = reset_interval_seconds

        self.draw = mp.solutions.drawing_utils
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode = False,
            model_complexity = 1,
            min_detection_confidence = 0.7,
            min_tracking_confidence = 0.7,
            max_num_hands = 1
        )



    def eye_position(self, frame):
        frame_height, frame_width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output = self.face_mesh.process(rgb_frame)

        eye = None
        # landmarks are normalised (means between [0, 1])
        if landmarks := output.multi_face_landmarks:
            left_pupil = landmarks[0].landmark[self.LEFT_PUPIL]
            right_pupil = landmarks[0].landmark[self.RIGTH_PUPIL]

            x = round((left_pupil.x + right_pupil.x) * frame_width / 2)
            y = round((left_pupil.y + right_pupil.y) * frame_height / 2)

            eye = (x, y)
            if self.debug:
                cv2.circle(frame, eye, 3, (0, 255, 0))

        return eye

    def update(self, frame):
        eye = self.eye_position(frame)
        self.process_hands(frame)

        if self.debug:
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
        screen_pos = self.eye_to_screen_pos(eye)

        if self.debug:
            text = f"Eye (in frame): {eye if screen_pos else "No eyes in frame"}"
            cv2.putText(frame, text, (30, 30), self.FONT, 1, (255, 0, 0), 2)

            text = f"Cursor (in screen): {screen_pos if screen_pos else "No eyes in frame"}"
            cv2.putText(frame, text, (30, 60), self.FONT, 1, (255, 0, 0), 2)

        if screen_pos:
            pyautogui.moveTo(*screen_pos)

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

            if self.debug:
                print(f"value: {value}; new config: {self.current_config}")

            if self.current_config > self.Config.Up:
                self.current_state = self.State.Cursor

    def eye_to_screen_pos(self, eye):
        if not eye:
            return None

        [right, left, down, up] = self.config

        x = (eye[0] - left[0]) / (right[0] - left[0])
        y = (down[1] - eye[1]) / (down[1] - up[1])

        x = clamp(x * self.screen_width, 0, self.screen_width)
        y = clamp(y * self.screen_height, 0, self.screen_height)

        # (y = self.screen_height - y) because y increases downwards
        return (int(x), int(self.screen_height - y))

    #methods below are for hand tracking
    def process_hands(self, frame):
        self.reset_click_flags()

        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed = self.hands.process(frameRGB)

        if processed.multi_hand_landmarks:
            hand_landmarks = processed.multi_hand_landmarks[0]
            self.draw.draw_landmarks(frame, hand_landmarks, self.mpHands.HAND_CONNECTIONS)

            count_of_landmarks = len(hand_landmarks.landmark)
            if count_of_landmarks >= 21 and (self.debug or self.current_state == self.State.Cursor):
                self.detect_gestures(frame, processed)

    def detect_gestures(self, frame, processed):
        index_tip = self.find_tip(processed, self.mpHands.HandLandmark.INDEX_FINGER_TIP)
        pinky_tip = self.find_tip(processed, self.mpHands.HandLandmark.PINKY_TIP)
        thumb_tip = self.find_tip(processed, self.mpHands.HandLandmark.THUMB_TIP)

        if self.is_left_click(index_tip, thumb_tip) and not self.dragging and not self.left_click_triggered:
            self.drag_start_time = time.time()
            self.left_click_triggered = True

        if self.is_left_click(index_tip, thumb_tip) and self.left_click_triggered:
            if time.time() - self.drag_start_time > self.dragging_threshold:
                pyautogui.mouseDown(button="left")
                self.dragging = True
                self.left_click_triggered = False

        if not self.is_left_click(index_tip, thumb_tip) and self.left_click_triggered:
            if time.time() - self.drag_start_time > self.double_click_threshold:
                pyautogui.mouseDown(button="left")
                pyautogui.mouseUp(button="left")
                pyautogui.mouseDown(button="left")
                pyautogui.mouseUp(button="left")
                cv2.putText(frame, "Double Click", (30, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                pyautogui.mouseDown(button="left")
                pyautogui.mouseUp(button="left")
                cv2.putText(frame, "Left Click", (30, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            self.left_click_triggered = False

        if self.dragging and not self.is_left_click(index_tip, thumb_tip):
            pyautogui.mouseUp(button="left")
            self.dragging = False

        if self.dragging:
            cv2.putText(frame, "Dragging", (30, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if self.is_right_click(pinky_tip, thumb_tip) and not self.right_click_triggered:
            pyautogui.mouseDown(button="right")
            pyautogui.mouseUp(button="right")
            cv2.putText(frame, "Right Click", (30, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            self.right_click_triggered = True


    def find_tip(self, processed, landmark):
        if processed.multi_hand_landmarks:
            hand_landmarks = processed.multi_hand_landmarks[0]
            return hand_landmarks.landmark[landmark]

        return None

    def is_left_click(self, index_tip, thumb_tip):
        return get_distance(index_tip, thumb_tip) < 50

    def is_right_click(self, pinky_tip, thumb_tip):
        return get_distance(pinky_tip, thumb_tip) < 100

    def reset_click_flags(self):
        current_time = time.time()
        if current_time - self.last_click_time >= self.reset_interval_seconds:
            self.right_click_triggered = False
            self.last_click_time = current_time

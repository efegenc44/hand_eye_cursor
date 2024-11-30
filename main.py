import cv2

from handeyecursor import HandEyeCursor

def main():
    cam = cv2.VideoCapture(0)
    cursor = HandEyeCursor(debug=True)

    while True:
        _, frame = cam.read()
        frame = cv2.flip(frame, 1)

        cursor.update(cursor.eye_position(frame), frame)

        if cv2.waitKey(1) == ord("q"):
            break

        cv2.imshow("Hand Eye Cursor", frame)

if __name__ == "__main__":
    main()
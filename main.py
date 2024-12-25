import cv2

from handyecursor import HandyeCursor

def main():
    cam = cv2.VideoCapture(0)
    cursor = HandyeCursor(debug=True)

    while True:
        _, frame = cam.read()
        frame = cv2.flip(frame, 1)

        cursor.update(frame)

        if cv2.waitKey(1) == ord("q"):
            break

        cv2.imshow("Handye Cursor", frame)

if __name__ == "__main__":
    main()
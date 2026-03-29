import cv2
from modules.hand_tracking import HandDetector

def main():
    #Initialize the camera and the object
    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    while True:
        work, frame = cap.read()

        #Turn the frame and get the frame returned from the function
        frame = cv2.flip(frame, 1)
        frame = detector.FindHands(frame)

        #Count fingers
        fingers = detector.FingersUp()
        total_fingers = fingers.count(1)

        #Show the number of finers up with a contour
        cv2.putText(frame, f"Fingers: {total_fingers}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 6)
        cv2.putText(frame, f"Fingers: {total_fingers}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)

        cv2.imshow("Fingers", frame)
        if cv2.waitKey(1) == 27: break
    
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__": main()
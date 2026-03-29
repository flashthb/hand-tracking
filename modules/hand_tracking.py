import cv2
import mediapipe as mp

class HandDetector:
    def __init__(self, mode = False, max_hands = 2, detection = 0.7, track = 0.7):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(mode, max_hands, 1, detection, track)
        self.mp_draw = mp.solutions.drawing_utils
        self.ids = [4, 8, 12, 16, 20]
        self.result = None
        self.hand_type = None

    
    def FindHands(self, frame, draw = True):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.result = self.hands.process(rgb)

        #Which hand
        self.hand_type = []
        if self.result.multi_handedness:
            for hand in self.result.multi_handedness:
                #"Left" or "Right"
                self.hand_type.append(hand.classification[0].label)

        #Do the lines on the hand
        if self.result.multi_hand_landmarks and draw:
            for hand_lms in self.result.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hand_lms, self.mp_hands.HAND_CONNECTIONS)

        return frame
    
    
    def FingersUp(self):
        fingers = []

        #Compare the position of thee  fingers and append 1 if it's extended
        if self.result.multi_hand_landmarks:
            for i, hand in enumerate(self.result.multi_hand_landmarks):
                #Special logic for the thumb with X and the hand
                if self.hand_type[i] == "Right":
                    if hand.landmark[self.ids[0]].x < hand.landmark[self.ids[0] - 1].x:
                        fingers.append(1)
                    else:
                        fingers.append(0)
                else:
                    if hand.landmark[self.ids[0]].x > hand.landmark[self.ids[0] - 1].x:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                #If the finger is up or not
                for id in range(1, 5):
                    if hand.landmark[self.ids[id]].y < hand.landmark[self.ids[id] - 2].y:
                        fingers.append(1)
                    else:
                        fingers.append(0)
        
        return fingers
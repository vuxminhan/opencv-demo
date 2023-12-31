import cv2
import mediapipe as mp
import time
import math
import numpy as np

class poseDetector():
            #    static_image_mode=False,
            #    model_complexity=1,
            #    smooth_landmarks=True,
            #    enable_segmentation=False,
            #    smooth_segmentation=True,
            #    min_detection_confidence=0.5,
            #    min_tracking_confidence=0.5):
            
    def __init__(self,  static_image_mode=False, model_complexity=1, smooth_landmarks=True,enable_segmentation=False,smooth_segmentation=True,min_detection_confidence=0.5,min_tracking_confidence=0.5):
        
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose()
        self.angle_threshold = 60 # Threshold for the angle to consider as successful curl
        self.lower_threshold = 160 # Threshold for the angle to consider as successful curl

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList

    def findAngle(self, img, p1, p2, p3, draw=True):

        # Get the landmarks
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        # Calculate the Angle
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                             math.atan2(y1 - y2, x1 - x2))
        if angle > 180.0:
            angle = 360 - angle
        if angle < 0 :
            angle = -angle
                
        

        # print(angle)

        # Draw
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
            cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        return angle
    
    def provide_feedback(self, img, landmarks, angle, armpit_angle):
        feedback_text = ""

        # Detect relying on momentum
        if angle > self.angle_threshold and armpit_angle < self.lower_threshold:
            feedback_text += "Avoid relying on momentum. Control the movement."

        # Detect rushing the reps
        if angle < self.angle_threshold and armpit_angle > self.angle_threshold:
            feedback_text += "Slow down and maintain a controlled pace."

        # Detect partial range of motion
        if angle > self.angle_threshold and armpit_angle > self.angle_threshold:
            feedback_text += "Ensure full range of motion by extending fully."

        # Detect moving elbows
        if landmarks:
            left_shoulder = landmarks[11][1:]
            right_shoulder = landmarks[12][1:]
            left_elbow = landmarks[13][1:]
            right_elbow = landmarks[14][1:]

            if abs(left_shoulder[1] - left_elbow[1]) > 30 or abs(right_shoulder[1] - right_elbow[1]) > 30:
                feedback_text += "Keep your elbows stable and close to your body."

        cv2.putText(img, feedback_text, (70, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

# Modify your main lo
    

def main():
    cap = cv2.VideoCapture('PoseVideos/1.mp4')
    pTime = 0
    detector = poseDetector()
    counter = 0
    up = True

    while True:
        success, img = cap.read()
        landmarks = detector.findPose(img)
        lmList = detector.findPosition(landmarks, draw=False)
        
        #counter
        current_angle = detector.findAngle(landmarks, 11, 13, 15)
        armpit_angle = detector.findAngle(landmarks, 23, 11, 13)
        
        if len(lmList) != 0:
            cv2.circle(landmarks, (lmList[14][1], lmList[14][2]), 15, (0, 0, 255), cv2.FILLED)
            
            # Provide feedback
            detector.provide_feedback(img, lmList, current_angle, armpit_angle)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime


        if current_angle < detector.angle_threshold and up:
            counter += 1
            up = False
            state = 2
        elif current_angle > detector.lower_threshold and not up:
            up = True
            state = 0
        else:
            state = 1

        # Display the counter value on the image
        cv2.putText(landmarks, str(counter), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (175, 175, 0), 3)
        cv2.imshow("Image", landmarks)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
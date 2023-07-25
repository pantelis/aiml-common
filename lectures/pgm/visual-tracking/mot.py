# %% [markdown]
# # Multiple Object Tracking
# - Andre Hu
# - Professor Pantelis
# - Introduction to Artificial Intelligence
# - 4/16/2023

# %%
# !mkdir files
# !rm -rf files/*

# # Downloading assignment videos
# !wget -P files/videos https://raw.githubusercontent.com/sseshadr/auvsi-cv-all/master/objectTracking/examples/ball.mp4
# !wget -P files/videos https://github.com/sseshadr/auvsi-cv-all/raw/master/objectTracking/examples/multiObject.avi

# !pip install numpy
# !pip install filterpy
# !pip install opencv-python
# !pip install -r https://raw.githubusercontent.com/ultralytics/yolov5/master/requirements.txt

# %%
import cv2
import torch
import numpy as np
from scipy.linalg import block_diag
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

# %% [markdown]
# ## Task 2: Object Detector

# %%
# Loading YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Filter out classes, only want to detect sports ball
model.classes = [32]

# %%
class Ball:
    def __init__(self, x, y):
        self.actual_points = []
        self.predicted_points = []
        self.filter = self.createFilter(x, y)

    def createFilter(self, x, y):
        """
        Based off of https://share.cocalc.com/share/7557a5ac1c870f1ec8f01271959b16b49df9d087/Kalman-and-Bayesian-Filters-in-Python/08-Designing-Kalman-Filters.ipynb?viewer=share
        """
        kf = KalmanFilter(dim_x=4, dim_z=2)
        dt = 1.0

        kf.F = np.array([[1, dt, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, dt],
                         [0, 0, 0, 1]])
        
        q = Q_discrete_white_noise(dim=2, dt=dt, var=0.05)
        kf.Q = block_diag(q, q)

        kf.H = np.array([[1, 0, 0, 0],
                         [0, 0, 1, 0]])
        
        kf.R = np.array([[0.5, 0],
                         [0, 0.5]])

        kf.P = np.eye(4)
        kf.x = np.array([[x, 0, y, 0]]).T

        return kf
    
    def calculateDistance(self, x, y):
        """
        Calculates the distance between the current position and the predicted position
        """
        return np.sqrt((x - self.filter.x[0]) ** 2 + (y - self.filter.x[2]) ** 2)

    def predict(self):
        """
        Have the filter predict the next point
        """
        self.filter.predict()
        predicted_x = int(self.filter.x[0])
        predicted_y = int(self.filter.x[2])

        self.predicted_points.append((predicted_x, predicted_y))

    def update(self, x, y):
        """
        Update the filter with a new point
        """
        self.actual_points.append((x, y))
        self.filter.update(np.array([[x, y]]))

    def addPointsToFrame(self, frame):
        """
        Draw the actual and predicted points on the frame
        """
        for i in self.actual_points:
            cv2.circle(frame, i, 1, (0, 255, 0), 2) # green
        for i in self.predicted_points:
            cv2.circle(frame, i, 1, (204, 51, 255), 2) # purple


def getNumberOfBalls(filename):
    """
    Retrieves the maximum number of balls in a video
    """
    cap = cv2.VideoCapture(filename)
    t = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            results = model(frame)
            t.append(len(results.xyxy[0]))
        else:
            break
    return max(t)


def calculateCenter(box):
    """
    Calculates the center of a bounding box
    """
    return int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)


# %%
def ballDetector(filename):
    # Keep track of all the balls in the video
    balls = [] 

    # Get the number of balls in the video
    numberOfBalls = getNumberOfBalls(filename)

    cap = cv2.VideoCapture(filename)

    # The code below renders the video and applies the YOLOv5 model
    # loaded previously to detect objects in the video
    
    # We need to set resolutions.
    # so, convert them from float to integer.
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    
    size = (frame_width, frame_height)
   
   
    result = cv2.VideoWriter('tracked.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, size)
    
    while(cap.isOpened()):
        # Read a frame from the video
        ret, frame = cap.read()

        if ret == True:
            # Apply the YOLOv5 model to the frame
            results = model(frame)
            frame = results.render()[0]

            # Find first frame where all the balls are detected
            if len(balls) == 0 and len(results.xyxy[0]) == numberOfBalls:
                for detection in results.xyxy[0]:
                    cx, cy = calculateCenter(detection)
                    balls.append(Ball(cx, cy))

            if len(balls) == numberOfBalls:

                # Update the filters for the balls
                for detection in results.xyxy[0]:
                    cx, cy = calculateCenter(detection)
                    
                    distances = [b.calculateDistance(cx, cy) for b in balls]
                    indexOfMin = distances.index(min(distances))

                    ball = balls[indexOfMin]
                    ball.update(cx, cy)

            for b in balls:
                b.predict()
                b.addPointsToFrame(frame)

                    
            # Render the results
            cv2.imshow('Object Detection', frame)
            result.write(frame)
            
            # Used for exiting the video
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # Release the video and close all windows
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    

# Open the videos
ballDetector("files/videos/ball.mp4")
ballDetector("files/videos/multiObject.avi")



# %%

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FileVideoStream
from imutils.video import FPS
import argparse
import imutils
import numpy as np
import os
import time
import cv2

# import the necessary packages
from threading import Thread
import sys

# initialize a dictionary that maps strings to their corresponding
# OpenCV object tracker implementations
OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.TrackerCSRT_create,
    "kcf": cv2.TrackerKCF_create,
    # "boosting": cv2.TrackerBoosting_create,
    # "mil": cv2.TrackerMIL_create,
    # "tld": cv2.TrackerTLD_create,
    # "medianflow": cv2.TrackerMedianFlow_create,
    # "mosse": cv2.TrackerMOSSE_create
}

# initialize the bounding box coordinates of the object we are going
# to track
initBB = None

# initialize the FPS throughput estimator
fps = None

foreground_dir = 'padding/foregroud'
optical_flow_dir = 'padding/optical_flow'

def create_output_dir():
    # Checking output directory
    if not os.path.isdir(foreground_dir):
        os.makedirs(os.path.join(os.getcwd(), foreground_dir))
    if not os.path.isdir(optical_flow_dir):
        os.makedirs(os.path.join(os.getcwd(), optical_flow_dir))

def load_vdo(args):
    # extract the OpenCV version info
    (major, minor) = cv2.__version__.split(".")[:2]
    # if we are using OpenCV 3.2 OR BEFORE, we can use a special factory
    # function to create our object tracker
    if int(major) == 3 and int(minor) < 3:
        tracker = cv2.Tracker_create(args["tracker"].upper())
    # otherwise, for OpenCV 3.3 OR NEWER, we need to explicity call the
    # approrpiate object tracker constructor:
    else:

        # grab the appropriate object tracker using our dictionary of
        # OpenCV object tracker objects
        tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()


    # if a video path was not supplied, grab the reference to the web cam
    if not args.get("video", False):
        print("[INFO] starting video stream...")
        vs = VideoStream(src=0).start()
        time.sleep(1.0)
    # otherwise, grab a reference to the video file
    else:
        vs = cv2.VideoCapture(args["video"])

    return vs, tracker

def tracking(fvs, tracker, width=1024, fg_dir=foreground_dir, of_dir=optical_flow_dir):

    global initBB

    # ret = a boolean return value from 
    # getting the frame, first_frame = the 
    # first frame in the entire video sequence 
    first_frame = fvs.read() 
    # first_frame = first_frame[1] if args.get("video", False) else first_frame
    
    # resize the frame (so we can process it faster) and grab the
    # frame dimensions
    first_frame = imutils.resize(first_frame, width=width)

    # Converts frame to grayscale because we 
    # only need the luminance channel for 
    # detecting edges - less computationally 
    # expensive 
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    # Creates an image filled with zero 
    # intensities with the same dimensions 
    # as the frame 
    mask = np.zeros_like(first_frame) 

    # Sets image saturation to maximum 
    mask[..., 1] = 255

    # loop over frames from the video stream
    while fvs.more():
        # grab the current frame, then handle if we are using a
        # # VideoStream or VideoCapture object
        # ret, frame = fvs.read()
        frame = fvs.read()
        
        # check to see if we have reached the end of the stream
        if frame is None:
            break
        # resize the frame (so we can process it faster) and grab the
        # frame dimensions
        frame = imutils.resize(frame, width=width)
        (H, W) = frame.shape[:2]
        
        # Converts each frame to grayscale - we previously 
        # only converted the first frame to grayscale 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

        # Calculates dense optical flow by Farneback method 
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, 
                                        None, 
                                        0.5, 3, 15, 3, 5, 1.2, 0) 


        # Computes the magnitude and angle of the 2D vectors 
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1]) 
        
        # Sets image hue according to the optical flow 
        # direction 
        mask[..., 0] = angle * 180 / np.pi / 2
        
        # Sets image value according to the optical flow 
        # magnitude (normalized) 
        mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX) 
        
        # Converts HSV to RGB (BGR) color representation 
        rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR) 
        
        # Opens a new window and displays the output frame 
        cv2.imshow("dense optical flow", rgb) 
        
        # Updates previous frame 
        prev_gray = gray 

        # check to see if we are currently tracking an object
        if initBB is not None:
            # grab the new bounding box coordinates of the object
            (success, box) = tracker.update(frame)
            # check to see if the tracking was a success
            if success:
                (x, y, w, h) = [int(v) for v in box]
                
                frame_ori = frame.copy()
                cv2.rectangle(frame, (x, y), (x + w, y + h),
                    (0, 255, 0), 2)

                # Transparency factor.
                alpha = 0

                # Following line overlays transparent rectangle over the image
                transparent_fg_border = cv2.addWeighted(frame, alpha, frame_ori, 1 - alpha, 0)

                # # Following line overlays transparent rectangle over the image
                # transparent_of_border = cv2.addWeighted(rgb, alpha, frame_ori, 1 - alpha, 0)

                current_time = time.time()

                img_fg_crop = transparent_fg_border[int(box[1]):int(box[1]+box[3]), int(box[0]):int(box[0]+box[2])]
                img_of_crop = rgb[int(box[1]):int(box[1]+box[3]), int(box[0]):int(box[0]+box[2])]

                cv2.imwrite('{}/{}.png'.format(fg_dir, current_time), img_fg_crop)
                cv2.imwrite('{}/{}.png'.format(of_dir, current_time), img_of_crop)
                
            # update the FPS counter
            fps.update()
            fps.stop()
            # initialize the set of information we'll be displaying on
            # the frame
            info = [
                ("Tracker", args["tracker"]),
                ("Success", "Yes" if success else "No"),
                ("FPS", "{:.2f}".format(fps.fps())),
                ("Queue Size", "{}".format(fvs.Q.qsize()))
            ]
            # loop over the info tuples and draw them on our frame
            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # show the output frame
        cv2.imshow("Frame", frame)
        
        key = cv2.waitKey(1) & 0xFF
        # if the 's' key is selected, we are going to "select" a bounding
        # box to track
        if key == ord("s"):
            # select the bounding box of the object we want to track (make
            # sure you press ENTER or SPACE after selecting the ROI)
            initBB = cv2.selectROI("Frame", frame, fromCenter=False,
                showCrosshair=True)
            # start OpenCV object tracker using the supplied bounding box
            # coordinates, then start the FPS throughput estimator as well
            tracker.init(frame, initBB)
            fps = FPS().start()
        # if the `q` key was pressed, break from the loop
        elif key == ord("q"):
            break
        
    # if we are using a webcam, release the pointer
    if not args.get("video", False):
        fvs.stop()
    # otherwise, release the file pointer
    else:
        fvs.release()
    # close all windows
    cv2.destroyAllWindows()

# import the Queue class from Python 3
if sys.version_info >= (3, 0):
	from queue import Queue
# otherwise, import the Queue class for Python 2.7
else:
	from Queue import Queue

# -------------------------------------------------------------------
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str,
    help="path to input video file")
ap.add_argument("-t", "--tracker", type=str, default="kcf",
    help="OpenCV object tracker type")
args = vars(ap.parse_args())

# start the file video stream thread and allow the buffer to
# start to fill
print("[INFO] starting video file thread...")
fvs = FileVideoStream(args["video"], queue_size=4096).start()
time.sleep(1.0)

# Load video stream
vs, tracker = load_vdo(args)
# Prepare output folder
create_output_dir()
# Tracking loop
tracking(fvs, tracker, width=512)
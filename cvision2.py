import cv2
from cv2 import aruco
import os
import numpy as np
import matplotlib.pyplot as plt
import IPython
from IPython.display import display, clear_output
import time
import math

############################################################################################################################################

# Find the USB camera connected to the PC
def find_camera_index(i):
    cap = cv2.Videocapture(i)
    if cap.isOpened():
        print(f"Camera index {i} found.")
        cap.release()
        return(i)
    else:
        print(f"No camera found at index {i}.")
        return find_camera_index(i+1)

############################################################################################################################################

#
def show_frame(frame):
    _, img_encoded = cv2.imencode('.png', frame)
    IPython.display.display(IPython.display.Image(data=img_encoded.tobytes()))

############################################################################################################################################

#
def vector_to_angle(vector):
    if np.size(vector) == 2:
        angle = -math.atan2(vector[1], vector[0])
        return angle
    else:
        return None

############################################################################################################################################

############################################################################################################################################
# Color recognition
############################################################################################################################################

############################################################################################################################################

# Given an image, calculates the mean and the standard deviation of the color 
# of all the pixels and return the upper and lower bound colors.
def upper_lower_color(color_sample, c):
    mean, stddev = cv2.meanStdDev(color_sample)
    lower_color = mean - c*stddev
    upper_color = mean + c*stddev
    return lower_color, upper_color

############################################################################################################################################

# Given a frame, give the contour of all pixels in the image of a particular color.
def find_contours(frame, color, c):
    lower_b, upper_b = upper_lower_color(color, c)
    # Find contours in the binary mask
    mask = cv2.inRange(frame, lower_b, upper_b)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

############################################################################################################################################

############################################################################################################################################
# ArUco marker recognition
############################################################################################################################################

############################################################################################################################################
# Given an array of pixel coordinates (for example the 4 corners of an ArUco marker),
# return the centroid.
def get_center(corners):
    # Get the pixel coordinates of the corners of the marker
    marker_corners = np.int32(corners)
    # Calculate the centroid of the marker
    centroid = np.round(np.mean(marker_corners, axis=0, dtype=np.int32)).astype(int)
    return centroid

############################################################################################################################################
# Create an ArucoDetector given a specific dictionary
def create_detector(dictionary):
    # Create a detector
    parameters = cv2.aruco.DetectorParameters()
    parameters.adaptiveThreshWinSizeMin = 10
    parameters.adaptiveThreshWinSizeMax = 100
    parameters.adaptiveThreshWinSizeStep = 10
    """ parameters.minMarkerPerimeterRate = 0.03
    parameters.maxMarkerPerimeterRate = 4.0
    parameters.polygonalApproxAccuracyRate = 0.05
    parameters.minCornerDistanceRate = 0.05
    parameters.minDistanceToBorder = 3
    parameters.minMarkerDistanceRate = 0.05
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    parameters.cornerRefinementWinSize = 5
    parameters.cornerRefinementMaxIterations = 30
    parameters.cornerRefinementMinAccuracy = 0.1
    parameters.markerBorderBits = 1
    parameters.perspectiveRemovePixelPerCell = 4
    parameters.perspectiveRemoveIgnoredMarginPerCell = 0.13
    parameters.maxErroneousBitsInBorderRate = 0.04
    parameters.minOtsuStdDev = 60.0
    parameters.errorCorrectionRate = 0.6 """
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    return detector

############################################################################################################################################
# Detect given objects on a given frame with a given detector. 
# Update the image positions and corners of the objects. 
def detect_objects(frame, detector, objects):
    corners = np.array([[]])
    corners, ids, _ = detector.detectMarkers(frame)
    for o in range(np.size(objects,0)):
        centers = []
        c = []
        if np.size(corners,0) > 0:
            for i in range(np.size(ids,0)):
                if objects[o].id == ids[i][0]:
                    centers.append(get_center(corners[i][0]))
                    c.append(corners[i][0])
        objects[o].img_pos = np.array(centers)
        objects[o].corners = np.array(c)
    return

############################################################################################################################################

############################################################################################################################################
# Classes
############################################################################################################################################

############################################################################################################################################
# Class Obj :
# Defining all the objects to be recognized with the camera. 
# Attributes :  id - the id of the ArUco marker assigned to this object, 
#               img_pos - position in pixels on the last image analysed 
#                           (array of empty array of if none detected, array of arrays if several detected)
#               corners - position of the 4 corners of the ArUco marker 
#                           (empty if none detected, array of arrays if several detected)
class Obj:
    def __init__(self, id, img_pos = np.array([[]])):
        self.id = id
        self.img_pos = img_pos
        self.corners = ()
    
    # Calculate the direction of the ArUco marker from the position of the 4 corners.
    # Return array of array, with vectors of direction of all detected objects.
    def get_img_dir(self):
        img_dir = []
        for p in range(np.size(self.corners,0)):
            if np.size(self.corners[p]) > 0:
                c1 = self.corners[p][0]
                c2 = self.corners[p][1]
                c3 = self.corners[p][2]
                c4 = self.corners[p][3]
                front = np.mean([c1,c2], 0)
                back = np.mean([c3,c4], 0)
                img_dir.append(front-back)
        return np.array(img_dir)
    
    # Draw the centroids of the objects on a frame.
    def draw(self, frame, color):
        for p in range(np.size(self.img_pos,0)):
            if np.size(self.img_pos[p]) > 0:
                cv2.circle(frame, self.img_pos[p], 5, color, -1)
        return
        
############################################################################################################################################
# Class Map :
# Define a map given three references objects and their respective distances. 
# This map can be used to localize other objects given their image position of the actual view.
# Before use, check that references are detected !
class Map:
    def __init__(self, ref1, ref2, ref3, distance_r1r2, distance_r3r2):
        self.ref1 = ref1
        self.ref1_pos = np.array([1, 0])
        self.ref2 = ref2
        self.ref2_pos = np.array([0, 0])
        self.ref3 = ref3
        self.ref3_pos = np.array([0, 1])
        self.origin = np.array([0, 0])
        self.distance_r1r2 = distance_r1r2  #[mm]
        self.distance_r3r2 = distance_r3r2  #[mm]
        self.e1 = (self.ref1_pos - self.ref2_pos)/np.linalg.norm(self.ref1_pos - self.ref2_pos)
        self.e2 = (self.ref3_pos - self.ref2_pos)/np.linalg.norm(self.ref3_pos - self.ref2_pos)
        self.scale = np.array([distance_r1r2/np.linalg.norm(self.ref1_pos - self.ref2_pos), distance_r3r2/np.linalg.norm(self.ref3_pos - self.ref2_pos)])
    
    # Check that all the three references needed to use the map are detected.
    def references_detected(self):
        if (np.size(self.ref1.img_pos,0) > 0) and (np.size(self.ref2.img_pos,0) > 0) and (np.size(self.ref3.img_pos,0) > 0):
            return True
        else:
            return False
    
    # Convert image coordinates (in pixels) to map coordinates
    def loc_img2map(self, img_coordinates):
        if np.size(img_coordinates) > 1:
            r1_img_pos = self.ref1.img_pos[0]
            r2_img_pos = self.ref2.img_pos[0]
            r3_img_pos = self.ref3.img_pos[0]
            e1_img = r1_img_pos - r2_img_pos
            e2_img = r3_img_pos - r2_img_pos
            basis = np.array([e1_img,e2_img])
            # Check if the basis vectors form a linearly independent set
            if np.linalg.matrix_rank(basis) != 2:
                return np.array([])
            else:
                map_coordinates = np.linalg.solve(np.transpose(basis), img_coordinates-r2_img_pos)
                map_coordinates = np.multiply(map_coordinates, self.scale)
                return map_coordinates
        else:
            return np.array([])
    
    # Convert a vector in image coordinates (in pixels) to map coordinates
    def vec_img2map(self, img_coordinates):
        if np.size(img_coordinates) > 1:
            r1_img_pos = self.ref1.img_pos[0]
            r2_img_pos = self.ref2.img_pos[0]
            r3_img_pos = self.ref3.img_pos[0]
            e1_img = r1_img_pos - r2_img_pos
            e2_img = r3_img_pos - r2_img_pos
            basis = np.array([e1_img,e2_img])
            # Check if the basis vectors form a linearly independent set
            if np.linalg.matrix_rank(basis) != 2:
                return np.array([])
            else:
                map_coordinates = np.linalg.solve(np.transpose(basis), img_coordinates)
                map_coordinates = np.multiply(map_coordinates, self.scale)
                return map_coordinates
        else:
            return np.array([])
    
    # Convert map coordinates to image coordinates (in pixels)
    def loc_map2img(self, map_coordinates):
        if np.size(map_coordinates) > 1:
            map_coordinates = np.divide(map_coordinates, self.scale)
            r1_img_pos = self.ref1.img_pos[0]
            r2_img_pos = self.ref2.img_pos[0]
            r3_img_pos = self.ref3.img_pos[0]
            e1_img = r1_img_pos - r2_img_pos
            e2_img = r3_img_pos - r2_img_pos
            img_coordinates = np.round(r2_img_pos + map_coordinates[0]*e1_img + map_coordinates[1]*e2_img).astype(int)
            return img_coordinates
        else:
            return np.array([])
    
    # 
    def localize(self, obj):
        img_position = obj.img_pos
        map_position = []
        for o in range(np.size(img_position,0)):
            map_position.append(self.loc_img2map(img_position[o]))
        return np.array(map_position)
    
    # Give the map in a matrix format with the given objects represented as numbers
    def matrix(self, frame, robot, goal):
        # Matrix cell real size
        square_size = 10 #[mm]
        margins = 0
        # Position of the origin of the map in the matrix
        origin = np.array([margins, margins])
        matrix = np.zeros([2*margins+self.distance_r1r2//square_size,2*margins+self.distance_r3r2//square_size])
        # Write the obstacles positions in the matrix with a 1
        color_obstacle = cv2.imread('obstacle.jpg')
        lower_b, upper_b = upper_lower_color(color_obstacle, 1.5)
        # Find contours in the binary mask
        mask = cv2.inRange(frame, lower_b, upper_b)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Create a matrix to store contour points
        contour_matrix = np.multiply(np.ones_like(frame),255)
        # Iterate through the contours and draw them on the matrix
        for contour in contours:
            area = cv2.contourArea(contour)
            # Adjust the area threshold based on your specific case
            if area > 50:
                # Draw the contour on the matrix
                cv2.drawContours(contour_matrix, [contour], -1, 0, thickness=cv2.FILLED)
        show_frame(contour_matrix)
        max_y, max_x, _ = contour_matrix.shape
        for x in range(2*margins+self.distance_r1r2//square_size):
            for y in range(2*margins+self.distance_r3r2//square_size):
                loc_img = self.loc_map2img([square_size*x,square_size*y])
                if max_x > loc_img[0] and max_y > loc_img[1]:
                    if contour_matrix[loc_img[1], loc_img[0]].all() < 1:
                        matrix[x,y] = 1
        # Write the robot position in the matrix with a 2
        map_pos = self.localize(robot)
        if np.size(map_pos,0) > 0:
            if not(np.isnan(map_pos[0]).any()):
                mat_pos = origin + np.round(np.divide(map_pos[0],square_size)).astype(int)
                if mat_pos[0] > 0 and mat_pos[1] > 0 and mat_pos[0] < np.size(matrix,0) and mat_pos[1] < np.size(matrix,1):
                    matrix[mat_pos[0],mat_pos[1]] = 2
        # Write the goal position in the matrix with a 3
        map_pos = self.localize(goal)
        if np.size(map_pos,0) > 0:
            if not(np.isnan(map_pos[0]).any()):
                mat_pos = origin + np.round(np.divide(map_pos[0],square_size)).astype(int)
                if mat_pos[0] > 0 and mat_pos[1] > 0 and mat_pos[0] < np.size(matrix,0) and mat_pos[1] < np.size(matrix,1):
                    matrix[mat_pos[0],mat_pos[1]] = 3

        return matrix

############################################################################################################################################


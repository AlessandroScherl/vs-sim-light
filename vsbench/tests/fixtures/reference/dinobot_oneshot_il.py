"""
In this script, we demonstrate how to use DINOBot to do one-shot imitation learning.
You first need to install the following repo and its requirements: https://github.com/ShirAmir/dino-vit-features.
You can then run this file inside that repo.

There are a few setup-dependent functions you need to implement, like getting an RGBD observation from the camera
or moving the robot, that you will find on top of this file.
"""
import torch
import numpy as np 
import matplotlib.pyplot as plt 
from torchvision import transforms,utils
from PIL import Image
import torchvision.transforms as T
import warnings 
import glob
import time
warnings.filterwarnings("ignore")

#Install this DINO repo to extract correspondences: https://github.com/ShirAmir/dino-vit-features
from correspondences import find_correspondences, draw_correspondences


#Hyperparameters for DINO correspondences extraction
num_pairs = 8 
load_size = 224 
layer = 9 
facet = 'key' 
bin=True 
thresh=0.05 
model_type='dino_vits8' 
stride=4 

#Deployment hyperparameters    
ERR_THRESHOLD = 50 #A generic error between the two sets of points


#Here are the functions you need to create based on your setup.
def camera_get_rgbd():
    """
    Outputs a tuple (rgb, depth) taken from a wrist camera.
    The two observations should have the same dimension.
    """
    raise NotImplementedError
    
def project_to_3d(points, depth, intrinsics):
    """
    Inputs: points: list of [x,y] pixel coordinates, 
            depth (H,W,1) observations from camera.
            intrinsics: intrinsics of the camera, used to 
            project pixels to 3D space.
    Outputs: point_with_depth: list of [x,y,z] coordinates.
    
    Projects the selected pixels to 3D space using intrinsics and
    depth value. Based on your setup the implementation may vary,
    but here you can find a simple example or the explicit formula:
    https://www.open3d.org/docs/0.6.0/python_api/open3d.geometry.create_point_cloud_from_rgbd_image.html.
    """
    
    raise NotImplementedError 
    
    
def robot_move(t_meters,R):
    """
    Inputs: t_meters: (x,y,z) translation in end-effector frame
            R: (3x3) array - rotation matrix in end-effector frame
    
    Moves and rotates the robot according to the input translation and rotation.
    """
    raise NotImplementedError
    
def record_demo()
    """
    Record a demonstration by moving the end-effector, and stores velocities
    that can then be replayed by the "replay_demo" function.
    """
    raise NotImplementedError
    
def replay_demo(demo)
    """
    Inputs: demo: list of velocities that can then be executed by the end-effector.
    Replays a demonstration by moving the end-effector given recorded velocities.
    """
    raise NotImplementedError
    

def find_transformation(X, Y):
    """
    Inputs: X, Y: lists of 3D points
    Outputs: R - 3x3 rotation matrix, t - 3-dim translation array.
    Find transformation given two sets of correspondences between 3D points.
    """
    # Calculate centroids
    cX = np.mean(X, axis=0)
    cY = np.mean(Y, axis=0)
    # Subtract centroids to obtain centered sets of points
    Xc = X - cX
    Yc = Y - cY
    # Calculate covariance matrix
    C = np.dot(Xc.T, Yc)
    # Compute SVD
    U, S, Vt = np.linalg.svd(C)
    # Determine rotation matrix
    R = np.dot(Vt.T, U.T)
    # Determine translation vector
    t = cY - np.dot(R, cX)
    return R, t

def compute_error(points1, points2):
    return np.linalg.norm(np.array(points1) - np.array(points2))

if __name__ == “__main__”:
 
    # RECORD DEMO:
    # Move the end-effector to the bottleneck pose and store observation.
    #Get rgbd from wrist camera.
    rgb_bn, depth_bn = camera_get_rgbd()

    #Record demonstration.
    demo_vels = record_demo()

    # TEST TIME DEPLOYMENT
    # Move/change the object and move the end-effector to the home (or a random) pose.
    while 1:
        error = 100000
        while error > ERR_THRESHOLD:
            #Collect observations at the current pose.
            rgb_live, depth_live = camera_get_rgbd()

            #Compute pixel correspondences between new observation and bottleneck observation.
            with torch.no_grad():
                
            #This function from an external library takes image paths as input. Therefore, store the paths of the
            #observations and then pass those
                        points1, points2, image1_pil, image2_pil = find_correspondences(rgb_bn, rgb_live, num_pairs, load_size, layer,
                                                                                       facet, bin, thresh, model_type, stride)
                                                                                       
            #Given the pixel coordinates of the correspondences, and their depth values,
            #project the points to 3D space.
            points1 = project_to_3d(points1, depth_bn, intrinsics)
            points2 = project_to_3d(points2, depth_live, intrinsics)

            #Find rigid translation and rotation that aligns the points by minimising error, using SVD.
            R, t = find_transformation(points1, points2)

            #Move robot
            robot.move(t_meters,R)
            error = compute_error(points1, points2)

        #Once error is small enough, replay demo.
        replay_demo(demo_vels)


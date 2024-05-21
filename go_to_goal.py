#!/usr/bin/env python3

import cv2
import cozmo
from cozmo.util import Pose
import cozmo.camera
from cozmo.anim import Triggers
import numpy as np
from numpy.linalg import inv
import threading
import time

from ar_markers.hamming.detect import detect_markers
from grid import CozGrid
from gui import GUIWindow
from particle import Particle
from setting import *
from particle_filter import *
from utils import *

# camera params
camK = np.matrix([[295, 0, 160], [0, 295, 120], [0, 0, 1]], dtype='float32')

#marker size in inches
marker_size = 3.5

# tmp cache
last_pose = cozmo.util.Pose(0,0,0,angle_z=cozmo.util.Angle(degrees=0))

# goal location for the robot to drive to, (x, y, theta)
goal = (6,10,0)

# map
Map_filename = "map_arena.json"

async def image_processing(robot: cozmo.robot.Robot):

    global camK, marker_size

    event = await robot.world.wait_for(cozmo.camera.EvtNewRawCameraImage, timeout=30)

    # convert camera image to opencv format
    opencv_image = np.asarray(event.image)
    
    # detect markers
    markers = detect_markers(opencv_image, marker_size, camK)
    
    # show markers
    for marker in markers:
        try:
            marker.highlite_marker(opencv_image, draw_frame=True, camK=camK)
        except:
            pass
        #print("ID =", marker.id);
        #print(marker.contours);
    #cv2.imshow("Markers", opencv_image)

    return markers

#calculate marker pose
def cvt_2Dmarker_measurements(ar_markers):
    
    marker2d_list = []
    
    for m in ar_markers:
        R_1_2, J = cv2.Rodrigues(m.rvec)
        R_1_1p = np.matrix([[0,0,1], [0,-1,0], [1,0,0]])
        R_2_2p = np.matrix([[0,-1,0], [0,0,-1], [1,0,0]])
        R_2p_1p = np.matmul(np.matmul(inv(R_2_2p), inv(R_1_2)), R_1_1p)
        yaw = -math.atan2(R_2p_1p[2,0], R_2p_1p[0,0])
        
        x, y = m.tvec[2][0] + 0.5, -m.tvec[0][0]
        
        # remove any duplate markers
        dup_thresh = 2.0
        find_dup = False
        for m2d in marker2d_list:
            if grid_distance(m2d[0], m2d[1], x, y) < dup_thresh:
                find_dup = True
                break
        if not find_dup:
            marker2d_list.append((x,y,math.degrees(yaw)))

    return marker2d_list

#compute robot odometry based on past and current pose
def compute_odometry(curr_pose: Pose, cvt_inch=True):
    global last_pose
    last_x, last_y, last_h = last_pose.position.x, last_pose.position.y, \
        last_pose.rotation.angle_z.degrees
    curr_x, curr_y, curr_h = curr_pose.position.x, curr_pose.position.y, \
        curr_pose.rotation.angle_z.degrees

    if cvt_inch:
        last_x, last_y = last_x / 25.6, last_y / 25.6
        curr_x, curr_y = curr_x / 25.6, curr_y / 25.6

    return [[last_x, last_y, last_h],[curr_x, curr_y, curr_h]]

#particle filter functionality
class ParticleFilter:

    def __init__(self, grid: CozGrid):
        self.particles = Particle.create_random(PARTICLE_COUNT, grid)
        self.grid = grid

    def update(self, odom, r_marker_list):

        # ---------- Motion model update ----------
        self.particles = motion_update(self.particles, odom)

        # ---------- Sensor (markers) model update ----------
        self.particles = measurement_update(self.particles, r_marker_list, self.grid)

        # ---------- Show current state ----------
        # Try to find current best estimate for display
        m_x, m_y, m_h, m_confident = compute_mean_pose(self.particles)
        return (m_x, m_y, m_h, m_confident)
    
    def is_converged(self):
        max_distance = 0
        total = len(self.particles)
        for i in range(total):
            if i < total-1:
                x1, y1 = self.particles[i].xy
                x2, y2 = self.particles[i+1].xy
                distance = math.sqrt((x2-x1)**2 + (y2-y1)**2)
                if max_distance < distance:
                    max_distance = distance
        return max_distance < 2.0

IN2MM = 25.4

async def run(robot: cozmo.robot.Robot):
    global last_pose
    global grid, gui

    # start streaming
    robot.camera.image_stream_enabled = True
    await robot.set_lift_height(0).wait_for_completed()

    #start particle filter
    pf = ParticleFilter(grid)

    # Obtain odometry information
    current_odom = compute_odometry(last_pose)

    # Obtain list of currently seen markers and their poses
    marker_list = await image_processing(robot)

    # Update the particle filter using the obtained information
    pf.update(current_odom, marker_list)

    pf_converge = False
    goal_reached = False

    while True:
        pf_converge = pf.is_converged()

        if robot.is_picked_up:
            print("Kidnapped")
            await robot.play_anim_trigger(Triggers.KnockOverFailure, in_parallel=True).wait_for_completed()
            await robot.say_text("Kidnapped", duration_scalar=.4, in_parallel=True).wait_for_completed() 
            pf = ParticleFilter(grid)
            time.sleep(5)
            continue

        if not pf_converge:
            await robot.set_head_angle(cozmo.util.degrees(10)).wait_for_completed()
            currentPose = robot.pose
            odom = compute_odometry(currentPose)
            markers = await image_processing(robot)
            measurement = cvt_2Dmarker_measurements(markers)
            est = pf.update(odom, measurement) 
            gui.show_particles(pf.particles)
            gui.show_mean(est[0], est[1], est[2], est[3])
            gui.updated.set()
            last_pose = currentPose

            if len(markers) > 0 and measurement[0][0] > 2.0:
                await robot.drive_straight(cozmo.util.distance_mm(40), cozmo.util.speed_mmps(40)).wait_for_completed()
            else:
                await robot.turn_in_place(cozmo.util.degrees(-30)).wait_for_completed()

        elif pf_converge:
            print("Converged")
            mX, mY, mH, mC = compute_mean_pose(pf.particles)
            diff_y = goal[1] - mY
            diff_x = goal[0] - mX
            arc = math.degrees(math.atan2(diff_y, diff_x))
            angle_to_turn = diff_heading_deg(arc, mH) 
            dist = math.sqrt(diff_y**2 + diff_x**2)

            await robot.turn_in_place(cozmo.util.degrees(angle_to_turn)).wait_for_completed()
            if robot.is_picked_up:
                # Reset localization if the robot is picked up
                pf = ParticleFilter(grid)
                break
            await robot.drive_straight(cozmo.util.distance_mm(dist*IN2MM), cozmo.util.speed_mmps(30)).wait_for_completed()

            if not robot.is_picked_up:
                await robot.turn_in_place(cozmo.util.degrees(-arc)).wait_for_completed()
                await robot.play_anim_trigger(Triggers.AcknowledgeObject).wait_for_completed()
                goal_reached = True
            else:
                pf = ParticleFilter(grid)

            time.sleep(1)
            await robot.drive_straight(cozmo.util.distance_mm(0), cozmo.util.speed_mmps(30)).wait_for_completed()
            await robot.play_anim("anim_pyramid_success_01").wait_for_completed()
            
            if goal_reached:
                print("Goal Reached!")
            break


############################################################################


class CozmoThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self, daemon=True)

    def run(self):
        cozmo.run_program(run, use_viewer=False)

if __name__ == '__main__':

    # cozmo thread
    cozmo_thread = CozmoThread()
    cozmo_thread.start()

    # init
    grid = CozGrid(Map_filename)
    gui = GUIWindow(grid)
    gui.start()
 
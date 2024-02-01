#This file is to the motion control of the robot when he's in the global path
import numpy as np
import math
from tdmclient import ClientAsync, aw
           
############################################################################################################################################

def motors(left, right):
    return {
        "motor.left.target": [left],
        "motor.right.target": [right],
    }

############################################################################################################################################

def stop_motors(node):
  
    aw(node.set_variables(motors(0,0)))
    
############################################################################################################################################

def set_motors(left,right,node):
    
    aw(node.set_variables(motors(left,right)))

############################################################################################################################################

def memory_speed(vr,vl):
    
    speed = [0, 0]
    speed[0] = vr
    speed[1] = vl
    
    return speed

############################################################################################################################################

def return_memory_speed(memory_speed):
    
    v_right = int(memory_speed[0])
    v_left = int(memory_speed[1])
    speed_v = [v_right, v_left] 
    
    return speed_v

############################################################################################################################################

def control_law(state_estimate_k, x_goal, y_goal, speed0, speedGain):
     
    orient_goal = math.atan2(y_goal - state_estimate_k[1], x_goal - state_estimate_k[0])
    delta_angle = orient_goal - state_estimate_k[2]

    if abs(delta_angle) > 0.8:
        vr = int(speedGain * delta_angle)
        vl = int(-speedGain * delta_angle)
    else:
        vr = int(speed0 + speedGain * delta_angle)
        vl = int(speed0 - speedGain * delta_angle)
        
    return vr, vl

############################################################################################################################################
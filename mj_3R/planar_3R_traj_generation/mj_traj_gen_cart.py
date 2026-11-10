import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os
import utility as ram
from forward_kinematics import forward_kinematics
from inverse_kinematics import inverse_kinematics
from scipy.optimize import fsolve
from scipy.interpolate import CubicSpline
from quintic_interp import quintic_interp

import matplotlib.pyplot as plt


xml_path = 'planar_3R.xml' #xml file (assumes this is in the same folder as this file)
simend = 5 #simulation time
print_camera_config = 0 #set to 1 to print camera config
                        #this is useful for initializing view of the model)

# For callback functions
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0



def init_controller(model,data):
    #initialize the controller here. This function is called once, in the beginning
    pass

def controller(model, data):
    #put the controller here. This function is called inside the simulation.
    pass

def keyboard(window, key, scancode, act, mods):
    if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)

def mouse_button(window, button, act, mods):
    # update button state
    global button_left
    global button_middle
    global button_right

    button_left = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
    button_middle = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
    button_right = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)

    # update mouse position
    glfw.get_cursor_pos(window)

def mouse_move(window, xpos, ypos):
    # compute mouse displacement, save
    global lastx
    global lasty
    global button_left
    global button_middle
    global button_right

    dx = xpos - lastx
    dy = ypos - lasty
    lastx = xpos
    lasty = ypos

    # no buttons down: nothing to do
    if (not button_left) and (not button_middle) and (not button_right):
        return

    # get current window size
    width, height = glfw.get_window_size(window)

    # get shift key state
    PRESS_LEFT_SHIFT = glfw.get_key(
        window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
    PRESS_RIGHT_SHIFT = glfw.get_key(
        window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
    mod_shift = (PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT)

    # determine action based on mouse button
    if button_right:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_MOVE_H
        else:
            action = mj.mjtMouse.mjMOUSE_MOVE_V
    elif button_left:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_ROTATE_H
        else:
            action = mj.mjtMouse.mjMOUSE_ROTATE_V
    else:
        action = mj.mjtMouse.mjMOUSE_ZOOM

    mj.mjv_moveCamera(model, action, dx/height,
                      dy/height, scene, cam)

def scroll(window, xoffset, yoffset):
    action = mj.mjtMouse.mjMOUSE_ZOOM
    mj.mjv_moveCamera(model, action, 0.0, -0.05 *
                      yoffset, scene, cam)

#get the full path
dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname + "/" + xml_path)
xml_path = abspath

# MuJoCo data structures
model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
data = mj.MjData(model)                # MuJoCo data
cam = mj.MjvCamera()                        # Abstract camera
opt = mj.MjvOption()                        # visualization options

# Init GLFW, create window, make OpenGL context current, request v-sync
glfw.init()
window = glfw.create_window(1200, 900, "Demo", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

# initialize visualization data structures
mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

# install GLFW mouse and keyboard callbacks
glfw.set_key_callback(window, keyboard)
glfw.set_cursor_pos_callback(window, mouse_move)
glfw.set_mouse_button_callback(window, mouse_button)
glfw.set_scroll_callback(window, scroll)

# Example on how to set camera configuration
# cam.azimuth = 90
# cam.elevation = -45
# cam.distance = 2
# cam.lookat = np.array([0.0, 0.0, 0])
cam.azimuth = 89.8231032550546 ; cam.elevation = -88.84857804659835 ; cam.distance =  6.025299868401033
cam.lookat =np.array([ 0.0 , 0.0 , 0.0 ])

key_id = model.key("home").id
key_qpos = model.key_qpos[key_id]  #Access qpos
key_ctrl = model.key_ctrl[key_id] #Access ctrl

#7) mocap id
#get the mocap id of reference
mocap_id = model.body("reference").mocapid[0]

# #mocap_pos = data.mocap_pos[mocap_id]

#print(model.body("via1").pos)
#print(model.body("via1").quat)


x0_ref = model.body("via0").pos[0]
y0_ref = model.body("via0").pos[1]

x1_ref = model.body("via1").pos[0]
y1_ref = model.body("via1").pos[1]

x2_ref = model.body("via2").pos[0]
y2_ref = model.body("via2").pos[1]

x3_ref = model.body("via3").pos[0]
y3_ref = model.body("via3").pos[1]

z_ref = 0;

X0_ref = np.array([x0_ref,y0_ref,z_ref])
X1_ref = np.array([x1_ref,y1_ref,z_ref])
X2_ref = np.array([x2_ref,y2_ref,z_ref])
X3_ref = np.array([x3_ref,y3_ref,z_ref])


#print(model.body("via1").quat)
quat0 = model.body("via0").quat.copy()
quat1 = model.body("via1").quat.copy()
quat2 = model.body("via2").quat.copy()
quat3 = model.body("via3").quat.copy()

euler0 = ram.quat2bryant(quat0)
euler1 = ram.quat2bryant(quat1)
euler2 = ram.quat2bryant(quat2)
euler3 = ram.quat2bryant(quat3)

dt1 = 2;
dt2 = 2;
dt3 = 2;
t0 = 1;
t1 = t0+dt1;
t2 = t1+dt2;
t3 = t2+dt3;
simend = t3+1;

flag_cubic_spline = 1 #use cubic spline else quintic polynomial

if (flag_cubic_spline==1):
    t_sp = np.array([0,t0,t1,t2,t3,simend])
    x_sp = np.array([x0_ref,x0_ref,x1_ref,x2_ref,x3_ref,x3_ref])
    y_sp = np.array([y0_ref,y0_ref,y1_ref,y2_ref,y3_ref,y3_ref])
    #bc = 'clamped' #third derivative at the second and second to last knot point are continouous
    #bc = 'not-a-knot' #first derivative at the ends are zero
    bc = 'natural' #second derivative at the ends are zero
    cs_x = CubicSpline(t_sp, x_sp,bc_type=bc)
    cs_y = CubicSpline(t_sp, y_sp,bc_type=bc)


#initialize the controller
init_controller(model,data)

#set the controller
mj.set_mjcb_control(controller)

#opt.frame = 1 #mj.mjtFrame.mjFRAME_SITE #or use 3 (check https://mujoco.readthedocs.io/en/latest/APIreference/APItypes.html#mjvoption)

#to store the data
time_all = []
X_all = np.empty((0, 3))
Xref_all = np.empty((0, 3))

while not glfw.window_should_close(window):
    time_prev = data.time

    if (data.time<0.01):
        q = key_qpos.copy()

    while (data.time - time_prev < 1.0/60.0):

        if (flag_cubic_spline==1):
            #reference position (cubic spline)
            pos = np.array([cs_x(data.time),cs_y(data.time),z_ref])
            #vel = np.array([cs_x(data.time,1),cs_y(data.time,1),0])
            #acc = np.array([cs_x(data.time,2),cs_y(data.time,2),0])
        else:
            #reference position (piecewise quintic)
            if (data.time<t1):
                pos,_,_ = quintic_interp(data.time,t0,t1,X0_ref,X1_ref)
            if (data.time>=t1 and data.time<t2):
                pos,_,_ = quintic_interp(data.time,t1,t2,X1_ref,X2_ref)
            if (data.time>=t2):
                pos,_,_ = quintic_interp(data.time,t2,t3,X2_ref,X3_ref)

        #reference quaternion
        if (data.time<t1):
            ang,_,_ = quintic_interp(data.time,t0,t1,euler0,euler1)
        if (data.time>=t1 and data.time<t2):
            ang,_,_ = quintic_interp(data.time,t1,t2,euler1,euler2)
        if (data.time>=t2):
            ang,_,_ = quintic_interp(data.time,t2,t3,euler2,euler3)

        quat = ram.bryant2quat(ang)
        #quat = np.array([0.707,0,0,0.707])
        #pos = np.array([1,1,0])
        for i in range(0,4):
            data.mocap_quat[mocap_id,i] = quat[i];
            if (i<3):
                data.mocap_pos[mocap_id,i] = pos[i];

        x_ref = pos[0];
        y_ref = pos[1];
        theta_ref = ang[2]
        parms = [x_ref, y_ref, theta_ref]

        #use fsolve for root finding
        q = fsolve(inverse_kinematics, q,parms)

        #animate
        data.time += model.opt.timestep
        data.qpos = q.copy()
        mj.mj_forward(model, data)
        # mj.mj_step(model, data)

        if (data.time>=t0 and data.time<=t3):

            X_site = data.site('tip').xpos.copy();
            site_quat = np.zeros(4)
            mj.mju_mat2Quat(site_quat, data.site('tip').xmat)
            Ang_site = ram.quat2bryant(site_quat)

            time_all.append(data.time-t0)
            x_tmp = np.array([x_ref, y_ref, theta_ref])
            xref_tmp = np.array([X_site[0],X_site[1],Ang_site[2]])
            X_all = np.vstack([X_all, x_tmp.copy()])
            Xref_all = np.vstack([Xref_all, xref_tmp.copy()])


        # var = site_quat;
        # print(f"(mj) end_eff_quat: {np.array2string(var, precision=4, floatmode='fixed', separator=', ')}")




    if (data.time>=simend):
        break;

    # get framebuffer viewport
    viewport_width, viewport_height = glfw.get_framebuffer_size(
        window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

    #print camera configuration (help to initialize the view)
    if (print_camera_config==1):
        print('cam.azimuth =',cam.azimuth,';','cam.elevation =',cam.elevation,';','cam.distance = ',cam.distance)
        print('cam.lookat =np.array([',cam.lookat[0],',',cam.lookat[1],',',cam.lookat[2],'])')

    # Update scene and render
    mj.mjv_updateScene(model, data, opt, None, cam,
                       mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, context)

    # swap OpenGL buffers (blocking call due to v-sync)
    glfw.swap_buffers(window)

    # process pending GUI events, call GLFW callbacks
    glfw.poll_events()

glfw.terminate()


# colors_ref = ['black', 'black', 'black']
# linestyles_ref = ['--', '--', '--']
#
# colors = ['red', 'blue', 'green']
# linestyles = ['-', '-', '-']
#
# # Create subplots (3 rows, 1 column)
# fig1, axes1 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)  # Share x-axis for alignment
#
# for i in range(3):
#     axes1[i].plot(time_all, Xref_all[:, i], color=colors_ref[i], linestyle=linestyles_ref[i], linewidth=2, label='ref')
#     axes1[i].plot(time_all, X_all[:, i], color=colors[i], linestyle=linestyles[i], linewidth=4, label='act',alpha=0.5)
#     axes1[i].set_ylabel(f'X_{i}')  # Label for each subplot
#     axes1[i].grid(True)
#     axes1[i].legend()  # Add legend inside each subplot
#
# # Common x-label for the bottom row
# axes1[-1].set_xlabel('Time')
#
# plt.show(block=False)  # Non-blocking display
# plt.pause(10)  # Pause for 10 seconds
# plt.close()  # Close the plot

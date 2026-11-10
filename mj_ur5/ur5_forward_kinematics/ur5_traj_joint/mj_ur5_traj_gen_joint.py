import mujoco as mj
from mujoco.glfw import glfw
import pathlib
import numpy as np
import os
from forward_kinematics import forward_kinematics
from inverse_kinematics import inverse_kinematics
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from quintic_interp import quintic_interp

t0 = 2 #start time
tf = 6 #end time


#xml_path = '../universal_robots_ur5e/scene.xml' #xml file (assumes this is in the same folder as this file)
xml_path = 'universal_robots_ur5e/scene.xml' #xml file (assumes this is in the same folder as this file)
simend = 10 #t_init+(1/f)+0.05 #100 #simulation time
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
xml_path = pathlib.Path(__file__).parent.parent.parent / xml_path
# abspath = os.path.join(dirname + "/" + xml_path)
# xml_path = abspath

print(xml_path)

# MuJoCo data structures
model = mj.MjModel.from_xml_path(str(xml_path))  # MuJoCo model
data = mj.MjData(model)                # MuJoCo data
cam = mj.MjvCamera()                        # Abstract camera
opt = mj.MjvOption()                        # visualization options

#model.opt.timestep = 0.001

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
cam.azimuth = -143.92066650390623 ; cam.elevation = -30.10462036132817 ; cam.distance =  2.1879597135804874
cam.lookat =np.array([ 0.0 , 0.0 , 0.0 ])

#initialize the controller
init_controller(model,data)

#set the controller
mj.set_mjcb_control(controller)

# 1) End-effector site
site_id = model.site("attachment_site").id

# 2) Name of bodies we wish to apply gravity compensation to.
#we have not included the body called "base"
body_names = [
    "shoulder_link",
    "upper_arm_link",
    "forearm_link",
    "wrist_1_link",
    "wrist_2_link",
    "wrist_3_link",
]
body_ids = [model.body(name).id for name in body_names]
#print(body_ids)
#model.body_gravcomp[body_ids] = 1.0

# 3) Joint / actuators have the same names
joint_names = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
]

#4) Joint ids
joint_ids = np.array([model.joint(name).id for name in joint_names])

# 5) Note that actuator names are the same as joint names in this case.
actuator_ids = np.array([model.actuator(name).id for name in joint_names])

# 6) Initial joint configuration saved as a keyframe in the XML file.
key_id = model.key("home").id
key_qpos = model.key_qpos[key_id]  #Access qpos
key_ctrl = model.key_ctrl[key_id] #Access ctrl


#initialize and final pose
q0 = np.array([-1.5708, -1.5708,  1.5708, -1.5708, -1.5708,  0. ])
qf = np.array([np.pi, -np.pi,  0, 0, 0, np.pi ])

#to store the data
time_all = []
q_all = np.empty((0, 6))  # Initialize empty array with 2 columns
u_all = np.empty((0, 6))  # Initialize empty array with 2 columns


opt.frame = mj.mjtFrame.mjFRAME_SITE #or use 3 (check https://mujoco.readthedocs.io/en/latest/APIreference/APItypes.html#mjvoption)

while not glfw.window_should_close(window):
    time_prev = data.time

    if (data.time<0.01):
        # Reset the simulation to the initial keyframe.
        mj.mj_resetDataKeyframe(model, data, key_id)
        data.qpos = q0.copy()


    while (data.time - time_prev < 1.0/60.0):

        q,u,_ = quintic_interp(data.time,t0,tf,q0,qf)
        data.qpos = q.copy()
        data.qvel = u.copy()
        data.time += 0.1; #model.opt.timestep
        mj.mj_forward(model, data)

        # mj_end_eff_pos = data.site(site_id).xpos


        if (data.time>=t0 and data.time<=tf):
            time_all.append(data.time-t0)
            q_all = np.vstack([q_all, q.copy()])
            u_all = np.vstack([u_all, u.copy()])

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


colors = ['black', 'red', 'blue', 'green', 'purple', 'orange']
linestyles = ['--', '-', '-.', ':', '--', '-']

plt.figure(1)
for i in range(6):  # Loop over i = 0 to 5
    plt.plot(time_all, q_all[:, i], label=f'q_{i}', color=colors[i], linestyle=linestyles[i], linewidth=2)

plt.xlabel('time')
plt.ylabel('q')
plt.legend()
plt.title('Joint angles')
plt.grid()

plt.figure(2)
for i in range(6):  # Loop over i = 0 to 5
    plt.plot(time_all, u_all[:, i], label=f'u_{i}', color=colors[i], linestyle=linestyles[i], linewidth=2)


plt.xlabel('time')
plt.ylabel('u')
plt.legend()
plt.title('Joint velocities')
plt.grid()
# plt.show()
plt.show(block=False)  # Non-blocking display
plt.pause(20)  # Pause for 2 seconds
plt.close()  # Close the plot

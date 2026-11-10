import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os
from RNEA import RNEA
from TMT import TMT
import time

xml_path = '../universal_robots_ur5e/scene.xml'

simend = 200 #simulation time
print_camera_config = 0 #set to 1 to print camera config
                        #this is useful for initializing view of the model)

# For callback functions
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0


def print_matrix(Mat, decimals=6, name="Mat"):
    """
    Prints an m x n matrix in Octave/Matlab style with configurable decimal accuracy and an optional name.

    Args:
        Mat (numpy.ndarray): The matrix to display.
        decimals (int): The number of decimal places to display.
        name (str): The name of the matrix to display. Defaults to "Mat".
    """
    if not isinstance(Mat, np.ndarray):
        raise ValueError("Input must be a numpy.ndarray.")

    if Mat.ndim != 2:
        raise ValueError("Input must be a 2D numpy.ndarray (matrix).")

    format_string = f"{{:.{decimals}f}}"  # Dynamic formatting for specified decimals
    print(f"{name} = [ ...")
    for row in Mat:
        formatted_row = " ".join(format_string.format(elem) for elem in row)
        print(f"  {formatted_row};")
    print("];\n")



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
#model.body_gravcomp[body_ids] = 0

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


#initial guess
q = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0])
u = np.array([0.21,-0.38,0.41,-4.12,6.32,8.53])


while not glfw.window_should_close(window):
    time_prev = data.time

    if (data.time<0.01):
        # Reset the simulation to the initial keyframe.
        mj.mj_resetDataKeyframe(model, data, key_id)


    while (data.time - time_prev < 1.0/60.0):

        data.qpos = q.copy();
        data.qvel = u.copy();
        data.time += 0.2 #model.opt.timestep
        mj.mj_forward(model, data)

        flag = 0
        tic = time.time()  # Start timer
        C_py,G_py,CG_py,M_py = RNEA(q,u,flag)
        toc = time.time()  # End timer
        print(f"Elapsed time (RNEA): {toc - tic:.4f} seconds")

        tic = time.time()  # Start timer
        M_py_TMT = TMT(q)
        toc = time.time()  # End timer
        print(f"Elapsed time (TMT): {toc - tic:.4f} seconds")




        print("Gravity and Coriolis")
        qfrc_bias = np.zeros((model.nv,1))
        for i in range(0,model.nv):
            qfrc_bias[i]=data.qfrc_bias[i]
        print_matrix(qfrc_bias,4,"CG_mj")
        print_matrix(CG_py,4,"CG_py")
        print_matrix(CG_py-qfrc_bias,6,"CG_py-CG_mj ")

        print("Mass matrix")
        M_mj = np.zeros((model.nv,model.nv ))
        mj.mj_fullM(model, data, M_mj)
        print_matrix(M_mj,6,"M_mj")
        print_matrix(M_py,6,"M_py")
        print_matrix(M_py_TMT,6,"M_py_TMT")

        print_matrix(M_mj-M_py_TMT,6,"M_mj-M_py_TMT")
        print_matrix(M_py-M_mj,6,"M_py-M_mj ")



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

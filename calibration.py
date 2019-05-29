from camera import RealCamera
import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy import optimize
from iiwaaControler.sunrisePy import sunrisePy

from mpl_toolkits.mplot3d import Axes3D
import time

ip = '172.31.1.148'
iiwa = sunrisePy(ip)
cart_pos = iiwa.getEEFCartesianPosition()
iiwa.close()
take_picture = True
Cam = RealCamera()
Cam.start_pipe(usb3=False)

# Parameters of the camera
depth_scale = Cam.depth_scale
fx, fy, Cx, Cy = Cam.intr.fx, Cam.intr.fy, Cam.intr.ppx, Cam.intr.ppy

params = [fx, fy, Cx, Cy, depth_scale]
np.save('intrinsic_parameters.npy', params)

if take_picture:
    Cam.get_frame()
    Cam.stop_pipe()
    # np.save('mycalibcolor.npy', Cam.color_image)
    # np.save('mycalibdepth.npy', Cam.depth_image)
    camera_color_img = Cam.color_image
    camera_depth_img = Cam.depth_image

else:
    camera_color_img = np.load('mycalibcolor.npy')
    camera_depth_img = np.load('mycalibdepth.npy')

checkerboard_size = (8, 6)
refine_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

print(camera_color_img.shape, camera_depth_img.shape)

bgr_color_data = cv2.cvtColor(camera_color_img, cv2.COLOR_RGB2BGR)
gray_data = cv2.cvtColor(bgr_color_data, cv2.COLOR_RGB2GRAY)

checkerboard_found, corners = cv2.findChessboardCorners(gray_data, checkerboard_size, None,
                                                        cv2.CALIB_CB_ADAPTIVE_THRESH)

def camera2cartesian(u, v, d, params):
    [fx, fy, Cx, Cy, depth_scale] = params
    z = d/depth_scale
    x = (u-Cy) * z/fy
    y = (v-Cx) * z/fx
    return np.array([[x], [y], [z]])



if checkerboard_found:
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    corners2 = cv2.cornerSubPix(gray_data, corners, (11, 11), (-1, -1), refine_criteria)
    # img = cv2.drawChessboardCorners(gray_data, checkerboard_size, [corners2[7], corners2[0], corners2[47]], checkerboard_found)


    cv2.circle(gray_data, (corners2[0][0][0],corners2[0][0][1]), 5, -1)
    cv2.circle(gray_data, (corners2[47][0][0], corners2[47][0][1]), 5, -1)
    cv2.circle(gray_data, (corners2[40][0][0], corners2[40][0][1]), 5, -1)

    cv2.circle(camera_depth_img, (corners2[40][0][0], corners2[40][0][1]), 5, -1)
    cv2.circle(camera_depth_img, (corners2[47][0][0], corners2[47][0][1]), 5, -1)
    cv2.circle(camera_depth_img, (corners2[0][0][0], corners2[0][0][1]), 5, -1)

    plt.subplot(1, 2, 1)
    plt.imshow(gray_data)
    plt.subplot(1, 2, 2)
    plt.imshow(camera_depth_img)
    plt.show()

    P1 = corners2[40]
    P1_camera = camera2cartesian(P1[0][0], P1[0][1], camera_depth_img[int(P1[0][0])][int(P1[0][1])], params)
    P3 = corners2[47]
    P3_camera = camera2cartesian(P3[0][0], P3[0][1], camera_depth_img[int(P3[0][0])][int(P3[0][1])], params)
    P2 = corners2[0]
    P2_camera = camera2cartesian(P2[0][0], P2[0][1], camera_depth_img[int(P2[0][0])][int(P2[0][1])], params)



    try:
        assert camera_depth_img[int(P1[0][1])][int(P1[0][0])] != 0
        assert camera_depth_img[int(P3[0][1])][int(P3[0][0])] != 0
        assert camera_depth_img[int(P2[0][1])][int(P2[0][0])] != 0
    except AssertionError:
        print('Null depth value')

    print('Dans le repère caméra : ', P1_camera, P1, camera_depth_img[int(P1[0][1])][int(P1[0][0])])

if True:
    P1_robot, P2_robot, P3_robot = np.array([417.7, 143.15, 136.22]),\
                                   np.array([405.81, 229.73, 134.35]),\
                                   np.array([559.89, 117.24, 135.45])       # Position des points dans le repère robot

    print('distance ', (np.linalg.norm(P1_robot-P2_robot) - np.linalg.norm(P1-P2))/np.linalg.norm(P1-P2))
    print('distance ', (np.linalg.norm(P2_robot - P3_robot) - np.linalg.norm(P2 - P3))/np.linalg.norm(P2 - P3))
    print('distance ', (np.linalg.norm(P1_robot - P3_robot) - np.linalg.norm(P1 - P3))/np.linalg.norm(P1 - P3))

    print('distance ', np.linalg.norm(P1_robot - P3_robot), np.linalg.norm(P1 - P3))

    P_robot = np.array([np.transpose(P1_robot), np.transpose(P2_robot), np.transpose(P3_robot)])

    x_tcp, y_tcp, z_tcp, A_tcp, B_tcp, C_tcp = 10, 20, 5, 0, 0, 3.14
    x_tcp, y_tcp, z_tcp, A_tcp, B_tcp, C_tcp = cart_pos
    Ra = np.array([[1, 0, 0],
                   [0, np.cos(A_tcp), -np.sin(A_tcp)],
                   [0, np.sin(A_tcp), np.cos(A_tcp)]])
    Rb = np.array([[np.cos(B_tcp), 0, np.sin(B_tcp)],
                   [0, 1, 0],
                   [-np.sin(B_tcp), 0, np.cos(B_tcp)]])
    Rc = np.array([[np.cos(C_tcp), -np.sin(C_tcp), 0],
                   [np.sin(C_tcp), np.cos(C_tcp), 0],
                   [0, 0, 1]])

    Rbase_main = Ra.dot(Rb.dot(Rc))

    Tbase_main = np.array([[x_tcp], [y_tcp], [z_tcp]])
    R = Rbase_main
    camera = np.array([[P1_camera[0][0], P2_camera[0][0], P3_camera[0][0]],
                       [P1_camera[1][0], P2_camera[1][0], P3_camera[1][0]],
                       [P1_camera[2][0], P2_camera[2][0], P3_camera[2][0]]])

    T = np.tile(Tbase_main, (1, 3))
    camera = np.dot(R, camera) + T

    camera = np.dot(R, camera) + T

    #print(np.tile(Tbase_main, (1,3)))

def error(R):
    Rc2m = np.array([R[:3], R[3:6], R[6:9]])
    t =  np.array([[R[9]], [R[10]], [R[11]]])
    tc2m = np.tile(t, (1, 3))
    global camera, Tbase_main, Rbase_main, P_robot
    tm2r, Rm2r = -Tbase_main, np.transpose(Rbase_main)

    camera1 = camera/R[12]

    camera1 = np.dot(Rc2m, camera)  + tc2m
    camera1 = np.dot(Rm2r, camera1) + tm2r

    error = camera1 - P_robot
    error = np.sum(np.multiply(error, error))
    rmse = np.sqrt(error / P_robot.shape[0])
    return error

arg_opt = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 5, 4, 3, 1.2])

optim_result = optimize.minimize(error, arg_opt, method='Nelder-Mead')
result = optim_result.x
R_optim = np.array([[result[0], result[1], result[2]],
                    [result[3], result[4], result[5]],
                    [result[6], result[7], result[8]]])
t_optim = np.array([[result[9]],
                    [result[10]],
                    [result[11]]])
factor = result[12]

print(R_optim, t_optim, factor)

tm2r, Rm2r = -Tbase_main, np.transpose(Rbase_main)
camera1 = np.dot(R_optim, camera) + t_optim
camera1 = np.dot(Rm2r, camera1) + tm2r

print('Résultat : ', camera1)
print('Réalité : ', P_robot)
error = camera1 - P_robot

print(error)

error = np.sum(np.multiply(error, error))
print('erreur : ', error)

np.save('R_camera_poignet1.npy', R_optim)
np.save('T_camera_poignet1.npy', t_optim)

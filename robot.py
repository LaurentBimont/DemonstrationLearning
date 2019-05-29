from iiwaaControler.sunrisePy import sunrisePy
import time
import numpy as np
# from gripper import RobotiqGripper
import matplotlib.pyplot as plt

# TEST TEMPORAIRE POUR CREER LA FONCTION camera-->robot
from camera import RealCamera


class Robot:
    def __init__(self):
        ip = '172.31.1.148'
        self.iiwa = sunrisePy(ip)

        self.iiwa.setBlueOn()
        # self.grip = RobotiqGripper("/dev/ttyUSB0")
        # self.grip.reset()
        # self.grip.activate()
        time.sleep(2)
        self.iiwa.setBlueOff()
        self.relVel = 0.1
        self.vel = 10

        # Define Flange
        # self.iiwa.attachToolToFlange([-1.5, 1.54, 152.8, 0, 0, 0])
        # self.iiwa.attachToolToFlange([0., 0., 0., 0., 0., 0.])

        # Define CameraTransormation Matrix
        self.camera = RealCamera()
        self.camera.start_pipe(usb3=False)
        self.camera.get_frame()
        self.R_cam_poign = np.load('R_camera_poignet.npy')
        self.T_cam_poign = np.load('T_camera_poignet.npy')

    def getCart(self):
        return self.iiwa.getEEFCartesianPosition()

    def manual_click(self):
        fig = plt.figure()
        connection_id = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.imshow(self.camera.color_image)
        plt.show()
        while True:
            try:
                x = x_click
                y = y_click
                break
            except:
                pass
        position_camera = self.camera.transform_3D(x, y)
        return position_camera

    def camera2robot(self):
        global x_click, y_click
        # Compute tool orientation from heightmap rotation angle
        position = self.manual_click()

        Rbase_main, Tbase_main = self.RetT_Matrix()

        print('Position ', position)
        print('Rotation', self.R_cam_poign, Rbase_main)
        print('Translation', self.T_cam_poign, Tbase_main)
        position = np.dot(self.R_cam_poign, position) + self.T_cam_poign
        position = np.dot(np.transpose(Rbase_main), position) - Tbase_main
        print('La position est: ', position)

    def RetT_Matrix(self):
        x_tcp, y_tcp, z_tcp, A_tcp, B_tcp, C_tcp = self.iiwa.getEEFCartesianPosition()
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
        return Rbase_main, Tbase_main

    def checkTCPJoint(self):
        cartPos = self.iiwa.getEEFCartesianPosition()
        vel = 1
        self.iiwa.movePTPLineEEF(cartPos, vel)
        cartPos[5] += np.pi/8
        self.iiwa.movePTPLineEEF(cartPos, vel)
        cartPos[5] -= np.pi/8
        cartPos[4] += np.pi/8
        self.iiwa.movePTPLineEEF(cartPos, vel)
        cartPos[4] -= np.pi/8
        cartPos[3] += np.pi/8
        self.iiwa.movePTPLineEEF(cartPos, vel)
        cartPos[3] -= np.pi/8
        self.iiwa.movePTPLineEEF(cartPos, vel)

    def grasp(self, pos, ang, speed=40):
        grasp_above = [pos[0], pos[1], pos[2]+100, ang, 0, -np.pi]
        self.iiwa.movePTPLineEEF(grasp_above, speed)
        grasp = [pos[0], pos[1], pos[2], ang, 0, -np.pi]
        self.iiwa.movePTPLineEEF(grasp, speed)
        self.grip.closeGripper()
        print(self.grip.isObjectDetected())
        self.grip.openGripper()


def onclick(event):
    global x_click, y_click
    x_click, y_click = int(event.xdata), int(event.ydata)
    print(x_click, y_click)


if __name__=="__main__":
    rob = Robot()

    a = 1
    rob.camera2robot()

    rob.iiwa.close()

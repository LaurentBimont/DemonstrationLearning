import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
from scipy import optimize
import tensorflow as tf
from sklearn.decomposition import PCA

recompute = True

if recompute:
    from trainer import Trainer
    model = Trainer(load=True, snapshot_file='reference')
    im = np.zeros((1, 224, 224, 3), np.float32)
    im[:, 70:190, 100:105, :] = 1
    im[:, 70:80, 80:125, :] = 1
    im = tf.contrib.image.rotate(im, angles=45)
    model.forward(im)
    out = model.prediction_viz(model.output_prob, im)
else:
    out = np.load('trained_qmap.npy')

out[out < 0] = 0
def draw_rectangle(img, params, x0, y0, lp):
    #x1, y1, lx, ly, theta = params[0], params[1], params[2], params[3], params[4]
    [e, theta] = params
    theta_rad = theta * np.pi/180
    x1 = int(x0 - lp/2*np.cos(theta_rad) - e/2*np.sin(theta_rad))
    y1 = int(y0 + lp/2*np.sin(theta_rad) - e/2*np.cos(theta_rad))
    x2 = int(x0 + lp/2*np.cos(theta_rad) - e/2*np.sin(theta_rad))
    y2 = int(y0 - lp/2*np.sin(theta_rad) - e/2*np.cos(theta_rad))
    x3 = int(x0 - lp/2*np.cos(theta_rad) + e/2*np.sin(theta_rad))
    y3 = int(y0 + lp/2*np.sin(theta_rad) + e/2*np.cos(theta_rad))
    x4 = int(x0 + lp/2*np.cos(theta_rad) + e/2*np.sin(theta_rad))
    y4 = int(y0 - lp/2*np.sin(theta_rad) + e/2*np.cos(theta_rad))

    print(np.array([[x1, y1], [x2, y2], [x4, y4], [x3, y3]], dtype=np.int))
    return np.array([[x1, y1], [x2, y2], [x4, y4], [x3, y3]], dtype=np.int)

def grasping_rectangle_error(params):
    global out, x_max, y_max, lp
    img = out
    rect = draw_rectangle(img, params, x_max, y_max, lp)
    mask = cv2.fillConvexPoly(np.zeros(img.shape[:2]), rect, color=1)
    masked_img = img[:, :, 1] * mask
    score = (np.sum(masked_img)**3)/(np.sum(mask)**2)
    return -score

def heatmap2pointcloud(img):
    # Rescale between 0 and 1
    img = (img - np.min(img))/(np.max(img)-np.min(img))
    PointCloudList = []
    img = img - 0.6
    img[img<0] = 0.
    plt.imshow(img)
    plt.show()
    for index, x in np.ndenumerate(img):
        for i in range(int(x*10)):
            PointCloudList.append([index[1], 100-index[0]])

    return np.asarray(PointCloudList)

def py_ang(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'    """
    cosang = np.dot(v1, v2)
    sinang = np.linalg.norm(np.cross(v1, v2))

    det = v1[0] * v2[1] - v1[1] * v2[0]
    if det<0:
        return -np.arctan2(sinang, cosang)
    else:
        return np.arctan2(sinang, cosang)

(y_max, x_max) = np.unravel_index(out[:, :, 1].argmax(), out[:, :, 1].shape)
test_pca = out[y_max-50:y_max+50, x_max-50:x_max+50, 1]
PointCloud = heatmap2pointcloud(test_pca)
pca = PCA()
pca.fit(PointCloud)
vectors = pca.components_
sing_val = pca.singular_values_/np.linalg.norm(pca.singular_values_)
vectors[0] *= sing_val[0]
vectors[1] *= sing_val[1]
np.linalg.norm(pca.singular_values_)
origin = [50], [50]


plt.subplot(1, 2, 1)
plt.scatter(PointCloud[:, 0], PointCloud[:, 1])
plt.quiver(*origin, vectors[0, 0], vectors[0, 1], color='r', scale=1)
plt.quiver(*origin, vectors[1, 0], vectors[1, 1], color='b', scale=1)
plt.subplot(1, 2, 2)
plt.imshow(test_pca[:, :])
plt.show()

e = 2*pca.singular_values_[1]
theta = py_ang([0,1], vectors[0])*180/np.pi

print('Parametre du rectangle : ecartement {}, angle {}, x: {}, y: {}, longueur pince {}'.format(e, theta, x_max, y_max, 20))
#
rect = draw_rectangle(out, [e, theta], x_max, y_max, 20)
optim_rectangle = cv2.fillConvexPoly(np.zeros(out.shape[:2]), rect, color=1)


plt.subplot(2, 2, 1)
plt.imshow(optim_rectangle)
plt.subplot(2, 2, 2)
plt.imshow(out[:, :, 1] * optim_rectangle)
plt.subplot(2, 2, 3)
plt.imshow(out)
plt.plot([rect[2][0], rect[1][0]], [rect[2][1], rect[1][1]], linewidth=3, color='yellow')
plt.plot([rect[0][0], rect[3][0]], [rect[0][1], rect[3][1]], linewidth=3, color='yellow')

plt.subplot(2, 2, 4)
out[:, :, 0] = out[:, :, 0] * optim_rectangle
out[:, :, 1] = out[:, :, 1] * optim_rectangle
plt.imshow(out)
plt.show()
#

if __name__=="__main__":
    # (y_max, x_max) = np.unravel_index(out[:, :, 1].argmax(), out[:, :, 1].shape)
    # lp = 40
    # print(x_max, y_max)
    # params = [40, 40]    # [Ecartement pince, angle]
    # optim_result = optimize.minimize(grasping_rectangle_error, params, method='Nelder-Mead')
    # result = optim_result.x
    # rect = draw_rectangle(out, result, x_max, y_max, lp)
    # optim_rectangle = cv2.fillConvexPoly(np.zeros(out.shape[:2]), rect, color=1)

    test_rectangle = draw_rectangle(out, [20, 20], 50, 150, 60)
    print(test_rectangle)
    autre_rectangle = cv2.fillConvexPoly(np.zeros(out.shape[:2]), test_rectangle, color=1)

    plt.imshow(autre_rectangle)
    plt.show()
    #
    # plt.subplot(2, 2, 1)
    # plt.imshow(optim_rectangle)
    # plt.subplot(2, 2, 2)
    # plt.imshow(out[:, :, 1] * optim_rectangle)
    # plt.subplot(2, 2, 3)
    # plt.imshow(out)
    # plt.subplot(2, 2, 4)
    # out[:, :, 0] = out[:, :, 0] * optim_rectangle
    # out[:, :, 1] = out[:, :, 1] * optim_rectangle
    # plt.imshow(out)
    # plt.show()

# Problème d'orientation et revoir la manière de calculer l'écartement

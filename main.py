import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
from scipy import optimize

recompute = False

if recompute:
    from trainer import Trainer
    model = Trainer(load=True, snapshot_file='reference')
    im = np.zeros((1, 224, 224, 3), np.float32)
    im[:, 70:190, 100:105, :] = 1
    im[:, 70:80, 80:125, :] = 1
    model.forward(im)
    out = model.prediction_viz(model.output_prob, im)
else:
    out = np.load('trained_qmap.npy')

out[:, :, 1] = out[:, :, 1] - 0.2
out[out < 0] = 0
def draw_rectangle(img, params):
    x1, y1, lx, ly, theta = params[0], params[1], params[2], params[3], params[4]
    x2, y2 = int(x1 + ly * np.sin(theta * np.pi / 180)), int(y1 + ly * np.cos(theta * np.pi / 180))
    x3, y3 = int(x1 + lx * np.cos(theta * np.pi / 180)), int(y1 - lx * np.sin(theta * np.pi / 180))
    x4, y4 = int(x3 + ly * np.sin(theta * np.pi / 180)), int(y3 + ly * np.cos(theta * np.pi / 180))
    return np.array([[x1, y1], [x2, y2], [x4, y4], [x3, y3]], dtype=np.int)

def grasping_rectangle_error(params):
    global out
    img = out
    rect = draw_rectangle(img, params)
    mask = cv2.fillConvexPoly(np.zeros(img.shape[:2]), rect, color=1)
    masked_img = img[:, :, 1] * mask
    score = (np.sum(masked_img)**3)/(np.sum(mask)**2)
    return -score


if __name__=="__main__":
    (y_max, x_max) = np.unravel_index(out[:, :, 1].argmax(), out[:, :, 1].shape)
    print(x_max,y_max)
    params = [x_max-20, y_max-20, 40, 40, 0]
    optim_result = optimize.minimize(grasping_rectangle_error, params, method='Nelder-Mead')
    result = optim_result.x
    rect = draw_rectangle(out, result)
    optim_rectangle = cv2.fillConvexPoly(np.zeros(out.shape[:2]), rect, color=1)

    plt.subplot(2, 2, 1)
    plt.imshow(optim_rectangle)
    plt.subplot(2, 2, 2)
    plt.imshow(out[:, :, 1] * optim_rectangle)
    plt.subplot(2, 2, 3)
    plt.imshow(out)
    plt.subplot(2, 2, 4)
    out[:, :, 0] = out[:, :, 0] * optim_rectangle
    out[:, :, 1] = out[:, :, 1] * optim_rectangle
    plt.imshow(out)
    plt.show()

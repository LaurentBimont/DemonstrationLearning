import numpy as np
import argparse
import matplotlib.pyplot as plt
from trainer import Trainer
import cv2

model = Trainer(load=True, snapshot_file='reference')
im = np.zeros((1, 224, 224, 3), np.float32)
im[:, 70:190, 100:105, :] = 1
im[:, 70:80, 80:125, :] = 1
model.forward(im)
out = model.prediction_viz(model.output_prob, im)


out = cv2.fillConvexPoly(out, np.array([np.array([1, 5]), np.array([4, 5]), np.array([4, 10])]), color=2)

plt.imshow(out)
plt.show()

def grasping_rectangle(output):
    x1, x2, y1, y2, theta = 2, 5, 6, 10, 20
    mask = np.zeros(output.shape)

    pass


if __name__=="__main__":
    im = model.output_prob
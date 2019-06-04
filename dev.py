import numpy as np
import matplotlib.pyplot as plt
import divers as div
# Python Imaging Library imports
import cv2

best_pix_ind = [50, 100, 0, 70]    # x, y, angle, écartement (unité ??)
x, y, angle, e = best_pix_ind[0], best_pix_ind[1], best_pix_ind[2], best_pix_ind[3]
label_value = 1
rect = div.draw_rectangle([e, angle], x, y, 30)
print([angle, e], x, y)
label = np.zeros((224, 224, 3), dtype=np.float32)
cv2.fillConvexPoly(label, rect, color=1)
label *= label_value
plt.imshow(label)
plt.show()

########## Trainer.py class Trainer ###########

def max_primitive_pixel(self, prediction, viz=False):
    '''Locate the max value-pixel of the image
    Locate the highest pixel of a Q-map
    :param prediction: Q map
    :return: max_primitive_pixel_idx (tuple) : pixel of the highest Q value
             max_primitive_pixel_value : value of the highest Q-value
    '''
    # Transform the Q map tensor into a 2-size numpy array
    numpy_predictions = prediction.numpy()[0, :, :, 0]
    if viz:
        result = tf.reshape(prediction, (prediction.shape[1], prediction.shape[2]))
        plt.subplot(1, 2, 1)
        plt.imshow(result)
        plt.subplot(1, 2, 2)
        plt.imshow(numpy_predictions)
        plt.show()
    # Get the highest pixel
    max_primitive_pixel_idx = np.unravel_index(np.argmax(numpy_predictions),
                                               numpy_predictions.shape)
    # Get the highest score
    max_primitive_pixel_value = numpy_predictions[max_primitive_pixel_idx]
    print('Grasping confidence scores: {}, {}'.format(max_primitive_pixel_value, max_primitive_pixel_idx))
    return max_primitive_pixel_idx, max_primitive_pixel_value

def get_best_predicted_primitive(self):
    '''
    :param output_prob: Q-map
    :return: best_idx (tuple): best idx in raw-Q-map
             best_value : highest value in raw Q-map
             image_idx (tuple): best pixels in image format (224x224) Q-map
             image_value : best value in image format (224x224) Q-map
    '''

    # Best Idx in image frameadients(
    prediction = tf.image.resize_images(self.output_prob, (224, 224))
    image_idx, image_value = self.max_primitive_pixel(prediction)
    # Best Idx in network output frame
    best_idx, best_value = self.max_primitive_pixel(self.output_prob)

    self.best_idx, self.future_reward = best_idx, best_value
    return best_idx, best_value, image_idx, image_value

#### Main.py
def grasping_rectangle_error(params):
    global out, x_max, y_max, lp
    img = out
    rect = draw_rectangle(img, params, x_max, y_max, lp)
    mask = cv2.fillConvexPoly(np.zeros(img.shape[:2]), rect, color=1)
    masked_img = img[:, :, 1] * mask
    score = (np.sum(masked_img)**3)/(np.sum(mask)**2)
    return -score

##### FingerTracking.py

def detect_green_viz(self):
    '''
    For Vizualisation only of the
    :return:
    '''
    first_cont, second_cont = None, None

    # Min Distance over a period of 5 seconds
    if time.time() - self.t0 > 10:
        self.t0 = time.time()
        self.min_over_time = np.inf

    # Take each frame
    self.cam.get_frame()
    frame = self.cam.color_image
    _, frame = self.cam.get_frame()
    # Convert BGR to HSV
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

    # define range for green color in HSV
    lower_green = np.array([60, 40, 40])
    upper_green = np.array([90, 250, 250])

    # Threshold the HSV image to get only green colors
    mask = cv2.inRange(hsv, lower_green, upper_green)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame, frame, mask=mask)

    gray_mask_image = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(gray_mask_image, 0, 255, 0)
    cont, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    first_cont, second_cont = self.max_contour(cont)
    if first_cont is not None and second_cont is not None:
        cx1, cy1 = self.centroid(first_cont)
        cv2.circle(frame, (cx1, cy1), 5, [0, 0, 255], -1)

        cx2, cy2 = self.centroid(second_cont)
        cv2.circle(frame, (cx2, cy2), 5, [0, 255, 0], -1)

        xtcp, ytcp = (cx1 + cx2) // 2, (cy1 + cy2) // 2

        print(FT.get_ecartement(cx1, cy1, cx2, cy2), self.min_over_time)

        if FT.get_ecartement(cx1, cy1, cx2, cy2) < self.min_over_time:
            self.x_tcp, self.y_tcp = (cx1 + cx2) // 2, (cy1 + cy2) // 2
            self.min_over_time = FT.get_ecartement(cx1, cy1, cx2, cy2)
        cv2.circle(frame, (self.x_tcp, self.y_tcp), 5, [255, 0, 0], -1)
    else:
        cv2.circle(frame, (self.x_tcp, self.y_tcp), 5, [255, 0, 0], -1)
    # except Exception as e:
    #     print(str(e))
    #     print('Pas de contours')
    #     pass

    return frame, mask, res
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
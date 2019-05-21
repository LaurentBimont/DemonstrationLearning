import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import divers as div

if __name__=="__main__":
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    tf.enable_eager_execution(config)


class DensenetFeatModel(tf.keras.Model):
    def __init__(self):
        '''
        Dense net est entraîné sur des images 224x224, si l'image d'entrée est plus grande le réseau va appliquer
        fois Densenet sur des sous-régions, jusqu'à obtenir l'image complète
        '''

        super(DensenetFeatModel, self).__init__()
        baseModel = tf.keras.applications.densenet.DenseNet121(weights='imagenet')
        self.model = tf.keras.Model(inputs=baseModel.input, outputs=baseModel.get_layer(
            "conv4_block16_concat").output)

    def call(self, inputs):
        # inputs = tf.transpose(inputs,(0,3,2,1))
        output = self.model(inputs)
        return output


class VGGFeatModel(tf.keras.Model):
    def __init__(self):
        '''
        Dense net est entraîné sur des images 224x224, si l'image d'entrée est plus grande le réseau va appliquer
        fois Densenet sur des sous-régions, jusqu'à obtenir l'image complète
        '''

        super(VGGFeatModel, self).__init__()
        baseModel = tf.keras.applications.VGG19(weights='imagenet')
        self.model = tf.keras.Model(inputs=baseModel.input, outputs=baseModel.get_layer("block5_pool").output)

    def call(self, inputs):
        # inputs = tf.transpose(inputs,(0,3,2,1))
        output = self.model(inputs)
        return output


class BaseDeepModel(tf.keras.Model):
    def __init__(self):
        super(BaseDeepModel, self).__init__()
        pass


class GraspNet(BaseDeepModel):
    def __init__(self):
        super(GraspNet, self).__init__()
        # Batch Normalization speed up convergence by reducing the internal covariance shift between batches
        # We can use a higher learning rate and it acts like a regulizer
        # https://arxiv.org/abs/1502.03167
        self.bn0 = tf.keras.layers.BatchNormalization(name="grasp-b0")
        self.conv0 = tf.keras.layers.Convolution2D(3, kernel_size=1, strides=1, activation=tf.nn.relu,
                                            use_bias=False, padding='valid', name="grasp-conv0", trainable=True)
        self.bn1 = tf.keras.layers.BatchNormalization(name="grasp-b1")
        self.conv1 = tf.keras.layers.Convolution2D (3, kernel_size=1, strides=1, activation=tf.nn.relu,
                                            use_bias=False, padding='valid', name="grasp-conv1", trainable=True)
        self.bn2 = tf.keras.layers.BatchNormalization(name="grasp-b2")

    def call(self, inputs, bufferize=False, step_id=-1):
        # print('Entrée du réseau seondaire', inputs.shape)
        x = self.bn0(inputs)
        x = self.conv0(x)
        x = self.bn1(x)
        x = self.conv1(x)
        x = (x[:, :, :, 0]+x[:, :, :, 1]+x[:, :, :, 2])/3.
        x = tf.reshape(x, (*x.shape, 1))
        x = self.bn2(x)
        print('Sortie du réseau secondaire ', x.shape)
        return x


class Reinforcement(tf.keras.Model):
    def __init__(self):
        super(Reinforcement, self).__init__()
        self.Dense = DensenetFeatModel()
        # self.VGG = VGGFeatModel()
        self.QGrasp = GraspNet()
        self.my_trainable_variables = self.QGrasp.trainable_variables
        print("Number of Trainable Variables", self.QGrasp.trainable_variables)

        # Initialize variables
        self.in_height, self.in_width = 0, 0
        self.scale_factor = 2.0
        self.padding_width = 0
        self.target_height = 0
        self.target_width = 0

    def call(self, input):
        # x = self.QGrasp(self.VGG(input))
        # x = self.QGrasp(input)
        x = self.QGrasp(self.Dense(input))
        return x


if __name__ == "__main__":
    
    im = np.ndarray((3, 224, 224, 3), np.float32)
    Densenet = Reinforcement()
    print(Densenet(im))

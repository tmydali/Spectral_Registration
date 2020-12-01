import keras
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.layers import Conv2DTranspose, BatchNormalization, Dropout
from keras.layers.merge import concatenate, Add
from keras import Model, activations

class GreenUnet160a1(keras.Model):
    '''
    This model takes NIR image as input, and generate green image as output.
    Use Unet-like architecture with input size 160 * 160.
    Use Conv2DTranspose as up-conv layer.
    Depth = 4
    '''
    
    def __init__(self):
        super(GreenUnet160a1, self).__init__()
        ### Encoder
        # Input, NIR image of size 160 * 160
        self.conv1_1 = Conv2D(32, kernel_size=3, padding = 'same', activation='relu', input_shape=(160, 160, 1))
        self.conv1_2 = Conv2D(32, kernel_size=3, padding = 'same', activation='relu')
        
        self.conv2_1 = Conv2D(64, kernel_size=3, padding = 'same', activation='relu')
        self.conv2_2 = Conv2D(64, kernel_size=3, padding = 'same', activation='relu')
        
        self.conv3_1 = Conv2D(128, kernel_size=3, padding = 'same', activation='relu')
        self.conv3_2 = Conv2D(128, kernel_size=3, padding = 'same', activation='relu')
        
        self.conv4_1 = Conv2D(256, kernel_size=3, padding = 'same', activation='relu')
        self.conv4_2 = Conv2D(256, kernel_size=3, padding = 'same', activation='relu')
        
        ### Decoder     
        self.upconv3 = Conv2DTranspose(128, kernel_size=3, strides=2, padding = 'same', activation='relu')
        self.deconv3_1 = Conv2D(128, kernel_size=3, padding = 'same', activation='relu')
        self.deconv3_2 = Conv2D(128, kernel_size=3, padding = 'same', activation='relu')
        
        self.upconv2= Conv2DTranspose(64, kernel_size=3, strides=2, padding = 'same', activation='relu')
        self.deconv2_1 = Conv2D(64, kernel_size=3, padding = 'same', activation='relu')
        self.deconv2_2 = Conv2D(64, kernel_size=3, padding = 'same', activation='relu')
        
        self.upconv1= Conv2DTranspose(32, kernel_size=3, strides=2, padding = 'same', activation='relu')
        self.deconv1_1 = Conv2D(32, kernel_size=3, padding = 'same', activation='relu')
        self.deconv1_2 = Conv2D(32, kernel_size=3, padding = 'same', activation='relu')
        
        # Output, Green image of size 160 * 160
        self.outputs = Conv2D(1, kernel_size=3, padding = 'same', activation='relu')
        
    def call(self, inputs):
        x = self.conv1_1(inputs)
        shortcut1 = self.conv1_2(x)
        x = MaxPooling2D(2)(shortcut1)
        x = self.conv2_1(x)
        shortcut2 = self.conv2_2(x)
        x = MaxPooling2D(2)(shortcut2)
        x = self.conv3_1(x)
        shortcut3 = self.conv3_2(x)
        x = MaxPooling2D(2)(shortcut3)
        x = self.conv4_1(x)
        x = self.conv4_2(x)

        x = self.upconv3(x)
        merge3 = concatenate([x, shortcut3], axis=3)
        x = self.deconv3_1(merge3)
        x = self.deconv3_2(x)
        x = self.upconv2(x)
        merge2 = concatenate([x, shortcut2], axis=3)
        x = self.deconv2_1(merge2)
        x = self.deconv2_2(x)
        x = self.upconv1(x)
        merge1 = concatenate([x, shortcut1], axis=3)
        x = self.deconv1_1(merge1)
        x = self.deconv1_2(x)
        
        return self.outputs(x)
    
    def model(self):
        x = Input(shape=(160, 160, 1))
        return Model(inputs=x, outputs=self.call(x))
    
class GreenUnet160a2(keras.Model):
    '''
    This model takes NIR image as input, and generate green image as output.
    Use Unet-like architecture with input size 160 * 160.
    Use Conv2D + UpSampling as up-conv layer.
    Depth = 4
    '''
    
    def __init__(self):
        super(GreenUnet160a2, self).__init__()
        ### Encoder
        # Input, NIR image of size 160 * 160
        self.conv1_1 = Conv2D(32, kernel_size=3, padding = 'same', activation='relu', input_shape=(160, 160, 1))
        self.conv1_2 = Conv2D(32, kernel_size=3, padding = 'same', activation='relu')
        
        self.conv2_1 = Conv2D(64, kernel_size=3, padding = 'same', activation='relu')
        self.conv2_2 = Conv2D(64, kernel_size=3, padding = 'same', activation='relu')
        
        self.conv3_1 = Conv2D(128, kernel_size=3, padding = 'same', activation='relu')
        self.conv3_2 = Conv2D(128, kernel_size=3, padding = 'same', activation='relu')
        
        self.conv4_1 = Conv2D(256, kernel_size=3, padding = 'same', activation='relu')
        self.conv4_2 = Conv2D(256, kernel_size=3, padding = 'same', activation='relu')
        
        ### Decoder     
        self.upconv3 = Conv2D(128, kernel_size=2, padding = 'same', activation='relu')
        self.deconv3_1 = Conv2D(128, kernel_size=3, padding = 'same', activation='relu')
        self.deconv3_2 = Conv2D(128, kernel_size=3, padding = 'same', activation='relu')
        
        self.upconv2 = Conv2D(64, kernel_size=2, padding = 'same', activation='relu')
        self.deconv2_1 = Conv2D(64, kernel_size=3, padding = 'same', activation='relu')
        self.deconv2_2 = Conv2D(64, kernel_size=3, padding = 'same', activation='relu')
        
        self.upconv1 = Conv2D(32, kernel_size=2, padding = 'same', activation='relu')
        self.deconv1_1 = Conv2D(32, kernel_size=3, padding = 'same', activation='relu')
        self.deconv1_2 = Conv2D(32, kernel_size=3, padding = 'same', activation='relu')
        
        # Output, Green image of size 160 * 160
        self.outputs = Conv2D(1, kernel_size=3, padding = 'same', activation='relu')
        
    def call(self, inputs):
        x = self.conv1_1(inputs)
        shortcut1 = self.conv1_2(x)
        x = MaxPooling2D(2)(shortcut1)
        x = self.conv2_1(x)
        shortcut2 = self.conv2_2(x)
        x = MaxPooling2D(2)(shortcut2)
        x = self.conv3_1(x)
        shortcut3 = self.conv3_2(x)
        x = MaxPooling2D(2)(shortcut3)
        x = self.conv4_1(x)
        x = self.conv4_2(x)

        x = self.upconv3(UpSampling2D(2)(x))
        merge3 = concatenate([x, shortcut3], axis=3)
        x = self.deconv3_1(merge3)
        x = self.deconv3_2(x)
        x = self.upconv2(UpSampling2D(2)(x))
        merge2 = concatenate([x, shortcut2], axis=3)
        x = self.deconv2_1(merge2)
        x = self.deconv2_2(x)
        x = self.upconv1(UpSampling2D(2)(x))
        merge1 = concatenate([x, shortcut1], axis=3)
        x = self.deconv1_1(merge1)
        x = self.deconv1_2(x)
        
        return self.outputs(x)
    
    def model(self):
        x = Input(shape=(160, 160, 1))
        return Model(inputs=x, outputs=self.call(x))
    
class GreenUnet160_trans(keras.Model):
    '''
    This model takes NIR image as input, and generate green image as output.
    Use Unet-like architecture with input size 160 * 160.
    Use Conv2DTranspose as deconv layer.
    Use Conv2D + UpSampling as up-conv layer.
    Depth = 4
    '''
    
    def __init__(self):
        super(GreenUnet160_trans, self).__init__()
        ### Encoder
        # Input, NIR image of size 160 * 160
        self.conv1_1 = Conv2D(32, kernel_size=3, padding = 'same', activation='relu', input_shape=(160, 160, 1))
        self.conv1_2 = Conv2D(32, kernel_size=3, padding = 'same', activation='relu')
        
        self.conv2_1 = Conv2D(64, kernel_size=3, padding = 'same', activation='relu')
        self.conv2_2 = Conv2D(64, kernel_size=3, padding = 'same', activation='relu')
        
        self.conv3_1 = Conv2D(128, kernel_size=3, padding = 'same', activation='relu')
        self.conv3_2 = Conv2D(128, kernel_size=3, padding = 'same', activation='relu')
        
        self.conv4_1 = Conv2D(256, kernel_size=3, padding = 'same', activation='relu')
        self.conv4_2 = Conv2D(256, kernel_size=3, padding = 'same', activation='relu')
        
        ### Decoder
        self.upconv3 = Conv2D(128, kernel_size=2, padding = 'same', activation='relu')
        self.deconv3_1 = Conv2DTranspose(128, kernel_size=3, padding = 'same', activation='relu')
        self.deconv3_2 = Conv2DTranspose(128, kernel_size=3, padding = 'same', activation='relu')
        
        self.upconv2 = Conv2D(64, kernel_size=2, padding = 'same', activation='relu')
        self.deconv2_1 = Conv2DTranspose(64, kernel_size=3, padding = 'same', activation='relu')
        self.deconv2_2 = Conv2DTranspose(64, kernel_size=3, padding = 'same', activation='relu')
        
        self.upconv1 = Conv2D(32, kernel_size=2, padding = 'same', activation='relu')
        self.deconv1_1 = Conv2DTranspose(32, kernel_size=3, padding = 'same', activation='relu')
        self.deconv1_2 = Conv2DTranspose(32, kernel_size=3, padding = 'same', activation='relu')
        
        # Output, Green image of size 160 * 160
        self.outputs = Conv2DTranspose(1, kernel_size=3, padding = 'same', activation='relu')
        
    def call(self, inputs):
        x = self.conv1_1(inputs)
        shortcut1 = self.conv1_2(x)
        x = MaxPooling2D(2)(shortcut1)
        x = self.conv2_1(x)
        shortcut2 = self.conv2_2(x)
        x = MaxPooling2D(2)(shortcut2)
        x = self.conv3_1(x)
        shortcut3 = self.conv3_2(x)
        x = MaxPooling2D(2)(shortcut3)
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        
        x = self.upconv3(UpSampling2D(2)(x))
        merge3 = concatenate([x, shortcut3], axis=3)
        x = self.deconv3_1(merge3)
        x = self.deconv3_2(x)
        x = self.upconv2(UpSampling2D(2)(x))
        merge2 = concatenate([x, shortcut2], axis=3)
        x = self.deconv2_1(merge2)
        x = self.deconv2_2(x)
        x = self.upconv1(UpSampling2D(2)(x))
        merge1 = concatenate([x, shortcut1], axis=3)
        x = self.deconv1_1(merge1)
        x = self.deconv1_2(x)
        
        return self.outputs(x)
    
    def model(self):
        x = Input(shape=(160, 160, 1))
        return Model(inputs=x, outputs=self.call(x))
    
class GreenUnet160b(keras.Model):
    '''
    This model takes NIR image as input, and generate green image as output.
    Use Unet-like architecture with input size 160 * 160.
    Use UpSampling2D + Conv2D as up-conv layer.
    Depth = 5
    '''
    
    def __init__(self):
        super(GreenUnet160b, self).__init__()
        lrelu = lambda x: activations.relu(x, alpha=0.0)
        ### Encoder
        # Input, NIR image of size 160 * 160
        self.conv1_1 = Conv2D(32, kernel_size=3, padding = 'same', activation=lrelu, input_shape=(160, 160, 1))
        self.conv1_2 = Conv2D(32, kernel_size=3, padding = 'same', activation=lrelu)
        
        self.conv2_1 = Conv2D(64, kernel_size=3, padding = 'same', activation=lrelu)
        self.conv2_2 = Conv2D(64, kernel_size=3, padding = 'same', activation=lrelu)
        
        self.conv3_1 = Conv2D(128, kernel_size=3, padding = 'same', activation=lrelu)
        self.conv3_2 = Conv2D(128, kernel_size=3, padding = 'same', activation=lrelu)
        
        self.conv4_1 = Conv2D(256, kernel_size=3, padding = 'same', activation=lrelu)
        self.conv4_2 = Conv2D(256, kernel_size=3, padding = 'same', activation=lrelu)
        
        self.conv5_1 = Conv2D(512, kernel_size=3, padding = 'same', activation=lrelu)
        self.conv5_2 = Conv2D(512, kernel_size=3, padding = 'same', activation=lrelu)
        
        ### Decoder     
        self.upconv4 = Conv2D(256, kernel_size=2, padding = 'same', activation=lrelu)
        self.deconv4_1 = Conv2D(256, kernel_size=3, padding = 'same', activation=lrelu)
        self.deconv4_2 = Conv2D(256, kernel_size=3, padding = 'same', activation=lrelu)
        
        self.upconv3 = Conv2D(128, kernel_size=2, padding = 'same', activation=lrelu)
        self.deconv3_1 = Conv2D(128, kernel_size=3, padding = 'same', activation=lrelu)
        self.deconv3_2 = Conv2D(128, kernel_size=3, padding = 'same', activation=lrelu)
        
        self.upconv2= Conv2D(64, kernel_size=2, padding = 'same', activation=lrelu)
        self.deconv2_1 = Conv2D(64, kernel_size=3, padding = 'same', activation=lrelu)
        self.deconv2_2 = Conv2D(64, kernel_size=3, padding = 'same', activation=lrelu)
        
        self.upconv1= Conv2D(32, kernel_size=2, padding = 'same', activation=lrelu)
        self.deconv1_1 = Conv2D(32, kernel_size=3, padding = 'same', activation=lrelu)
        self.deconv1_2 = Conv2D(32, kernel_size=3, padding = 'same', activation=lrelu)
        
        # Output, Green image of size 160 * 160
        self.outputs = Conv2D(1, kernel_size=3, padding = 'same', activation='relu')
        
    def call(self, inputs):
        x = self.conv1_1(inputs)
        shortcut1 = self.conv1_2(x)
        x = MaxPooling2D(2)(shortcut1)
        x = self.conv2_1(x)
        shortcut2 = self.conv2_2(x)
        x = MaxPooling2D(2)(shortcut2)
        x = self.conv3_1(x)
        shortcut3 = self.conv3_2(x)
        x = MaxPooling2D(2)(shortcut3)
        x = self.conv4_1(x)
        shortcut4 = self.conv4_2(x)
        x = MaxPooling2D(2)(shortcut4)
        x = self.conv5_1(x)
        x = self.conv5_2(x)

        x = self.upconv4(UpSampling2D(2)(x))
        merge4 = concatenate([x, shortcut4], axis=3)
        x = self.deconv4_1(merge4)
        x = self.deconv4_2(x)
        x = self.upconv3(UpSampling2D(2)(x))
        merge3 = concatenate([x, shortcut3], axis=3)
        x = self.deconv3_1(merge3)
        x = self.deconv3_2(x)
        x = self.upconv2(UpSampling2D(2)(x))
        merge2 = concatenate([x, shortcut2], axis=3)
        x = self.deconv2_1(merge2)
        x = self.deconv2_2(x)
        x = self.upconv1(UpSampling2D(2)(x))
        merge1 = concatenate([x, shortcut1], axis=3)
        x = self.deconv1_1(merge1)
        x = self.deconv1_2(x)
        
        return self.outputs(x)
    
    def model(self):
        x = Input(shape=(160, 160, 1))
        return Model(inputs=x, outputs=self.call(x))
    
class GreenUnet320(keras.Model):
    '''
    This model takes NIR image as input, and generate green image as output.
    Use Unet-like architecture with input size 320 * 320.
    Use UpSampling2D + Conv2D as up-conv layer.
    Depth = 5
    '''
    
    def __init__(self):
        super(GreenUnet320, self).__init__()
        lrelu = lambda x: activations.relu(x, alpha=0.2)
        ### Encoder
        # Input, NIR image of size 320 * 320
        self.conv1_1 = Conv2D(32, kernel_size=3, padding = 'same', activation=lrelu, input_shape=(320, 320, 1))
        self.conv1_2 = Conv2D(32, kernel_size=3, padding = 'same', activation=lrelu)
        
        self.conv2_1 = Conv2D(64, kernel_size=3, padding = 'same', activation=lrelu)
        self.conv2_2 = Conv2D(64, kernel_size=3, padding = 'same', activation=lrelu)
        
        self.conv3_1 = Conv2D(128, kernel_size=3, padding = 'same', activation=lrelu)
        self.conv3_2 = Conv2D(128, kernel_size=3, padding = 'same', activation=lrelu)
        
        self.conv4_1 = Conv2D(256, kernel_size=3, padding = 'same', activation=lrelu)
        self.conv4_2 = Conv2D(256, kernel_size=3, padding = 'same', activation=lrelu)
        
        self.conv5_1 = Conv2D(512, kernel_size=3, padding = 'same', activation=lrelu)
        self.conv5_2 = Conv2D(512, kernel_size=3, padding = 'same', activation=lrelu)
        
        ### Decoder
        self.upconv4 = Conv2D(256, kernel_size=2, padding = 'same', activation=lrelu)
        self.deconv4_1 = Conv2D(256, kernel_size=3, padding = 'same', activation=lrelu)
        self.deconv4_2 = Conv2D(256, kernel_size=3, padding = 'same', activation=lrelu)
        
        self.upconv3 = Conv2D(128, kernel_size=2, padding = 'same', activation=lrelu)
        self.deconv3_1 = Conv2D(128, kernel_size=3, padding = 'same', activation=lrelu)
        self.deconv3_2 = Conv2D(128, kernel_size=3, padding = 'same', activation=lrelu)
        
        self.upconv2 = Conv2D(64, kernel_size=2, padding = 'same', activation=lrelu)
        self.deconv2_1 = Conv2D(64, kernel_size=3, padding = 'same', activation=lrelu)
        self.deconv2_2 = Conv2D(64, kernel_size=3, padding = 'same', activation=lrelu)
        
        self.upconv1 = Conv2D(32, kernel_size=2, padding = 'same', activation=lrelu)
        self.deconv1_1 = Conv2D(32, kernel_size=3, padding = 'same', activation=lrelu)
        self.deconv1_2 = Conv2D(32, kernel_size=3, padding = 'same', activation=lrelu)
        
        # Output, Green image of size 320 * 320
        self.outputs = Conv2D(1, kernel_size=3, padding = 'same', activation='sigmoid')
        
    def call(self, inputs):
        x = self.conv1_1(inputs)
        shortcut1 = self.conv1_2(x)
        x = MaxPooling2D(2)(shortcut1)
        x = self.conv2_1(x)
        shortcut2 = self.conv2_2(x)
        x = MaxPooling2D(2)(shortcut2)
        x = self.conv3_1(x)
        shortcut3 = self.conv3_2(x)
        x = MaxPooling2D(2)(shortcut3)
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        shortcut4 = Dropout(0.5)(x)
        x = MaxPooling2D(2)(shortcut4)
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = Dropout(0.5)(x)

        x = self.upconv4(UpSampling2D(2)(x))
        merge4 = concatenate([x, shortcut4], axis=3)
        x = self.deconv4_1(merge4)
        x = self.deconv4_2(x)
        x = self.upconv3(UpSampling2D(2)(x))
        merge3 = concatenate([x, shortcut3], axis=3)
        x = self.deconv3_1(merge3)
        x = self.deconv3_2(x)
        x = self.upconv2(UpSampling2D(2)(x))
        merge2 = concatenate([x, shortcut2], axis=3)
        x = self.deconv2_1(merge2)
        x = self.deconv2_2(x)
        x = self.upconv1(UpSampling2D(2)(x))
        merge1 = concatenate([x, shortcut1], axis=3)
        x = self.deconv1_1(merge1)
        x = self.deconv1_2(x)
        
        return self.outputs(x)
    
    def model(self):
        x = Input(shape=(320, 320, 1))
        return Model(inputs=x, outputs=self.call(x))
    
class GreenAutoencoder160(keras.Model):
    '''
    This model takes NIR image as input, and generate green image as output.
    Use encoder decoder architecture with input size 160 * 160.
    Use residuals.
    Use leaky relu.
    Depth = 5
    '''
    
    def __init__(self):
        super(GreenAutoencoder160, self).__init__()
        lrelu = lambda x: activations.relu(x, alpha=0.0)
        ### Encoder
        # Input, NIR image of size 160 * 160
        self.conv1_1 = Conv2D(32, kernel_size=3, padding = 'same', activation=lrelu, input_shape=(160, 160, 1))
        self.conv1_2 = Conv2D(32, kernel_size=3, padding = 'same', activation=lrelu)
        
        self.conv2_1 = Conv2D(64, kernel_size=3, padding = 'same', activation=lrelu)
        self.conv2_2 = Conv2D(64, kernel_size=3, padding = 'same', activation=lrelu)
        
        self.conv3_1 = Conv2D(128, kernel_size=3, padding = 'same', activation=lrelu)
        self.conv3_2 = Conv2D(128, kernel_size=3, padding = 'same', activation=lrelu)
        
        self.conv4_1 = Conv2D(256, kernel_size=3, padding = 'same', activation=lrelu)
        self.conv4_2 = Conv2D(256, kernel_size=3, padding = 'same', activation=lrelu)
        
        self.conv5_1 = Conv2D(512, kernel_size=3, padding = 'same', activation=lrelu)
        self.conv5_2 = Conv2D(512, kernel_size=3, padding = 'same', activation=lrelu)
        
        ### Decoder     
        self.upconv4 = Conv2D(256, kernel_size=2, padding = 'same', activation=lrelu)
        self.deconv4_1 = Conv2D(256, kernel_size=3, padding = 'same', activation=lrelu)
        self.deconv4_2 = Conv2D(256, kernel_size=3, padding = 'same', activation=lrelu)
        
        self.upconv3 = Conv2D(128, kernel_size=2, padding = 'same', activation=lrelu)
        self.deconv3_1 = Conv2D(128, kernel_size=3, padding = 'same', activation=lrelu)
        self.deconv3_2 = Conv2D(128, kernel_size=3, padding = 'same', activation=lrelu)
        
        self.upconv2= Conv2D(64, kernel_size=2, padding = 'same', activation=lrelu)
        self.deconv2_1 = Conv2D(64, kernel_size=3, padding = 'same', activation=lrelu)
        self.deconv2_2 = Conv2D(64, kernel_size=3, padding = 'same', activation=lrelu)
        
        self.upconv1= Conv2D(32, kernel_size=2, padding = 'same', activation=lrelu)
        self.deconv1_1 = Conv2D(32, kernel_size=3, padding = 'same', activation=lrelu)
        self.deconv1_2 = Conv2D(32, kernel_size=3, padding = 'same', activation=lrelu)
        
        ### Concatenate the layer itself so to double channel size
        self.chDoub = lambda x: concatenate([x, x], axis=3)
        
        ### 1x1 Conv
        self.chConv4u = Conv2D(256, kernel_size=1, activation=lrelu)
        self.chConv3u = Conv2D(128, kernel_size=1, activation=lrelu)
        self.chConv2u = Conv2D(64, kernel_size=1, activation=lrelu)
        self.chConv1u = Conv2D(32, kernel_size=1, activation=lrelu)
        
        # Output, Green image of size 160 * 160
        self.outputs = Conv2D(1, kernel_size=3, padding = 'same', activation='relu')
        
    def call(self, inputs):
        x = self.conv1_1(inputs)
        x = self.conv1_2(x)
        x = MaxPooling2D(2)(x)
        shortcut = self.chDoub(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = Add()([shortcut, x])
        x = MaxPooling2D(2)(x)
        shortcut = self.chDoub(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = Add()([shortcut, x])
        x = MaxPooling2D(2)(x)
        shortcut = self.chDoub(x)
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = Add()([shortcut, x])
        x = MaxPooling2D(2)(x)
        shortcut = self.chDoub(x)
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = Add()([shortcut, x])
        
        x = self.upconv4(UpSampling2D(2)(x))
        shortcut = self.chConv4u(x)
        x = self.deconv4_1(x)
        x = self.deconv4_2(x)
        x = Add()([shortcut, x])
        x = self.upconv3(UpSampling2D(2)(x))
        shortcut = self.chConv3u(x)
        x = self.deconv3_1(x)
        x = self.deconv3_2(x)
        x = Add()([shortcut, x])
        x = self.upconv2(UpSampling2D(2)(x))
        shortcut = self.chConv2u(x)
        x = self.deconv2_1(x)
        x = self.deconv2_2(x)
        x = Add()([shortcut, x])
        x = self.upconv1(UpSampling2D(2)(x))
        shortcut = self.chConv1u(x)
        x = self.deconv1_1(x)
        x = self.deconv1_2(x)
        x = Add()([shortcut, x])
        
        return self.outputs(x)
    
    def model(self):
        x = Input(shape=(160, 160, 1))
        return Model(inputs=x, outputs=self.call(x))
        
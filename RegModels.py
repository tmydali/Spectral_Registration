import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.layers import Conv2DTranspose, BatchNormalization, Dropout
from tensorflow.keras.layers import concatenate, Add
from tensorflow.keras import Model

class GreenUnet160(Model):
    '''
    This model takes NIR image as input, and generate green image as output.
    Using Unet-like architecture with input size 160 * 160.
    Using Conv2DTranspose as up-conv layer.
    Depth = 4
    '''
    
    def __init__(self):
        super(GreenUnet160, self).__init__()
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
    
class GreenUnet160b(Model):
    '''
    This model takes NIR image as input, and generate green image as output.
    Using Unet-like architecture with input size 160 * 160.
    Using UpSampling2D + Conv2D as up-conv layer.
    Depth = 5
    '''
    
    def __init__(self):
        super(GreenUnet160b, self).__init__()
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
        
        self.conv5_1 = Conv2D(512, kernel_size=3, padding = 'same', activation='relu')
        self.conv5_2 = Conv2D(512, kernel_size=3, padding = 'same', activation='relu')
        
        ### Decoder     
        self.upconv4 = Conv2D(256, kernel_size=2, padding = 'same', activation='relu')
        self.deconv4_1 = Conv2D(256, kernel_size=3, padding = 'same', activation='relu')
        self.deconv4_2 = Conv2D(256, kernel_size=3, padding = 'same', activation='relu')
        
        self.upconv3 = Conv2D(128, kernel_size=2, padding = 'same', activation='relu')
        self.deconv3_1 = Conv2D(128, kernel_size=3, padding = 'same', activation='relu')
        self.deconv3_2 = Conv2D(128, kernel_size=3, padding = 'same', activation='relu')
        
        self.upconv2= Conv2D(64, kernel_size=2, padding = 'same', activation='relu')
        self.deconv2_1 = Conv2D(64, kernel_size=3, padding = 'same', activation='relu')
        self.deconv2_2 = Conv2D(64, kernel_size=3, padding = 'same', activation='relu')
        
        self.upconv1= Conv2D(32, kernel_size=2, padding = 'same', activation='relu')
        self.deconv1_1 = Conv2D(32, kernel_size=3, padding = 'same', activation='relu')
        self.deconv1_2 = Conv2D(32, kernel_size=3, padding = 'same', activation='relu')
        
        ### BatchNormalization
        self.BN1 = BatchNormalization()
        self.BN2 = BatchNormalization()
        self.BN3 = BatchNormalization()
        self.BN4 = BatchNormalization()
        self.BN5 = BatchNormalization()
        
        # Output, Green image of size 160 * 160
        self.outputs = Conv2D(1, kernel_size=3, padding = 'same', activation='sigmoid')
        
    def call(self, inputs):
        x = self.conv1_1(inputs)
        x = self.conv1_2(x)
        shortcut1 = self.BN1(x)
        x = MaxPooling2D(2)(shortcut1)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        shortcut2 = self.BN2(x)
        x = MaxPooling2D(2)(shortcut2)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        shortcut3 = self.BN3(x)
        x = MaxPooling2D(2)(shortcut3)
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = Dropout(0.5)(x)
        shortcut4 = self.BN4(x)
        x = MaxPooling2D(2)(shortcut4)
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = Dropout(0.5)(x)
        x = self.BN5(x)

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
    
class GreenUnet320(Model):
    '''
    This model takes NIR image as input, and generate green image as output.
    Using Unet-like architecture with input size 320 * 320.
    Using UpSampling2D + Conv2D as up-conv layer.
    Depth = 5
    '''
    
    def __init__(self):
        super(GreenUnet320, self).__init__()
        ### Encoder
        # Input, NIR image of size 320 * 320
        self.conv1_1 = Conv2D(32, kernel_size=3, padding = 'same', activation='relu', input_shape=(320, 320, 1))
        self.conv1_2 = Conv2D(32, kernel_size=3, padding = 'same', activation='relu')
        
        self.conv2_1 = Conv2D(64, kernel_size=3, padding = 'same', activation='relu')
        self.conv2_2 = Conv2D(64, kernel_size=3, padding = 'same', activation='relu')
        
        self.conv3_1 = Conv2D(128, kernel_size=3, padding = 'same', activation='relu')
        self.conv3_2 = Conv2D(128, kernel_size=3, padding = 'same', activation='relu')
        
        self.conv4_1 = Conv2D(256, kernel_size=3, padding = 'same', activation='relu')
        self.conv4_2 = Conv2D(256, kernel_size=3, padding = 'same', activation='relu')
        
        self.conv5_1 = Conv2D(512, kernel_size=3, padding = 'same', activation='relu')
        self.conv5_2 = Conv2D(512, kernel_size=3, padding = 'same', activation='relu')
        
        ### Decoder
        self.upconv4 = Conv2D(256, kernel_size=2, padding = 'same', activation='relu')
        self.deconv4_1 = Conv2D(256, kernel_size=3, padding = 'same', activation='relu')
        self.deconv4_2 = Conv2D(256, kernel_size=3, padding = 'same', activation='relu')
        
        self.upconv3 = Conv2D(128, kernel_size=2, padding = 'same', activation='relu')
        self.deconv3_1 = Conv2D(128, kernel_size=3, padding = 'same', activation='relu')
        self.deconv3_2 = Conv2D(128, kernel_size=3, padding = 'same', activation='relu')
        
        self.upconv2 = Conv2D(64, kernel_size=2, padding = 'same', activation='relu')
        self.deconv2_1 = Conv2D(64, kernel_size=3, padding = 'same', activation='relu')
        self.deconv2_2 = Conv2D(64, kernel_size=3, padding = 'same', activation='relu')
        
        self.upconv1 = Conv2D(32, kernel_size=2, padding = 'same', activation='relu')
        self.deconv1_1 = Conv2D(32, kernel_size=3, padding = 'same', activation='relu')
        self.deconv1_2 = Conv2D(32, kernel_size=3, padding = 'same', activation='relu')
        
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
        
import tensorflow as tf

def mini_xception(input_shape, num_classes, l2_regularization=0.01):
    regularization = tf.keras.regularizers.l2(l2_regularization)
    
    # base
    img_input = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization, use_bias=False)(img_input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    
    x = tf.keras.layers.Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    
    # module 1
    residual = tf.keras.layers.Conv2D(16, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = tf.keras.layers.BatchNormalization()(residual)
    
    x = tf.keras.layers.SeparableConv2D(16, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    
    x = tf.keras.layers.SeparableConv2D(16, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.add([x, residual])
    
    # module 2
    residual = tf.keras.layers.Conv2D(32, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = tf.keras.layers.BatchNormalization()(residual)
    
    x = tf.keras.layers.SeparableConv2D(32, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    
    x = tf.keras.layers.SeparableConv2D(32, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.add([x, residual])
    
    # module 3
    residual = tf.keras.layers.Conv2D(64, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = tf.keras.layers.BatchNormalization()(residual)
    
    x = tf.keras.layers.SeparableConv2D(64, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    
    x = tf.keras.layers.SeparableConv2D(64, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.add([x, residual])
    
    # module 4
    residual = tf.keras.layers.Conv2D(128, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = tf.keras.layers.BatchNormalization()(residual)
    
    x = tf.keras.layers.SeparableConv2D(128, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    
    x = tf.keras.layers.SeparableConv2D(128, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    
    x = tf.keras.layers.add([x, residual])
    
    x = tf.keras.layers.Conv2D(num_classes, (3, 3), padding='same')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    output = tf.keras.layers.Activation('softmax', name='predictions')(x)
    model = tf.keras.models.Model(img_input, output)
    
    return model

if __name__ == "__main__":
    model = mini_xception((48, 48, 1), 7)
    model.summary()
    
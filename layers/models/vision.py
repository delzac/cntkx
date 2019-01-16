import cntk as C
import cntkx as Cx
import numpy as np
from cntk.layers import Convolution2D, Dense, Dropout, MaxPooling, BatchNormalization, AveragePooling
from cntk.layers.blocks import identity
from cntkx.layers import Conv2DMaxPool


def VGG16(num_classes: int):
    """ for image classification """

    layer1 = Conv2DMaxPool(2, conv_filter_shape=(3, 3), pool_filter_shape=(2, 2), conv_num_filters=64,
                           activation=C.relu, conv_pad=True, pool_strides=(2, 2), name_prefix='layer1')

    layer2 = Conv2DMaxPool(2, conv_filter_shape=(3, 3), pool_filter_shape=(2, 2), conv_num_filters=128,
                           activation=C.relu, conv_pad=True, pool_strides=(2, 2), name_prefix='layer2')

    layer3 = Conv2DMaxPool(2, conv_filter_shape=(3, 3), pool_filter_shape=(2, 2), conv_num_filters=256,
                           activation=C.relu, conv_pad=True, pool_strides=(2, 2), name_prefix='layer3')

    layer4 = Conv2DMaxPool(3, conv_filter_shape=(3, 3), pool_filter_shape=(2, 2), conv_num_filters=512,
                           activation=C.relu, conv_pad=True, pool_strides=(2, 2), name_prefix='layer4')

    layer5 = Conv2DMaxPool(3, conv_filter_shape=(3, 3), pool_filter_shape=(2, 2), conv_num_filters=512,
                           activation=C.relu, conv_pad=True, pool_strides=(2, 2), name_prefix='layer5')

    dense1 = Dense(4096, activation=C.relu, name='layer6')
    dropout1 = Dropout(0.5)
    dense2 = Dense(4096, activation=C.relu, name='layer7')
    dropout2 = Dropout(0.5)
    dense3 = Dense(num_classes, activation=C.relu, name='layer8')

    def model(x):
        x = layer1(x)
        x = layer2(x)
        x = layer3(x)
        x = layer4(x)
        x = layer5(x)
        x = dropout1(dense1(x))
        x = dropout2(dense2(x))
        x = dense3(x)
        return x

    return model


def VGG19(num_classes: int):
    """ for image classification """
    layer1 = Conv2DMaxPool(2, conv_filter_shape=(3, 3), pool_filter_shape=(2, 2), conv_num_filters=64,
                           activation=C.relu, conv_pad=True, pool_strides=(2, 2), name_prefix='layer1')

    layer2 = Conv2DMaxPool(2, conv_filter_shape=(3, 3), pool_filter_shape=(2, 2), conv_num_filters=128,
                           activation=C.relu, conv_pad=True, pool_strides=(2, 2), name_prefix='layer2')

    layer3 = Conv2DMaxPool(4, conv_filter_shape=(3, 3), pool_filter_shape=(2, 2), conv_num_filters=256,
                           activation=C.relu, conv_pad=True, pool_strides=(2, 2), name_prefix='layer3')

    layer4 = Conv2DMaxPool(4, conv_filter_shape=(3, 3), pool_filter_shape=(2, 2), conv_num_filters=512,
                           activation=C.relu, conv_pad=True, pool_strides=(2, 2), name_prefix='layer4')

    layer5 = Conv2DMaxPool(4, conv_filter_shape=(3, 3), pool_filter_shape=(2, 2), conv_num_filters=512,
                           activation=C.relu, conv_pad=True, pool_strides=(2, 2), name_prefix='layer5')

    dense1 = Dense(4096, activation=C.relu, name='layer6')
    dropout1 = Dropout(0.5)
    dense2 = Dense(4096, activation=C.relu, name='layer7')
    dropout2 = Dropout(0.5)
    dense3 = Dense(num_classes, activation=C.relu, name='layer8')

    def model(x):
        x = layer1(x)
        x = layer2(x)
        x = layer3(x)
        x = layer4(x)
        x = layer5(x)
        x = dropout1(dense1(x))
        x = dropout2(dense2(x))
        x = dense3(x)
        return x

    return model


def conv_bn(input, filter_size, num_filters, strides=(1, 1), init=C.he_normal(), bn_init_scale=1):
    c = Convolution2D(filter_size, num_filters, activation=None, init=init, pad=True, strides=strides, bias=False)(input)
    r = BatchNormalization(map_rank=1, normalization_time_constant=4096, use_cntk_engine=False, init_scale=bn_init_scale, disable_regularization=True)(c)
    return r


def conv_bn_relu(input, filter_size, num_filters, strides=(1, 1), init=C.he_normal()):
    r = conv_bn(input, filter_size, num_filters, strides, init, 1)
    return C.relu(r)


def resnet_bottleneck(input, out_num_filters, inter_out_num_filters):
    c1 = conv_bn_relu(input, (1, 1), inter_out_num_filters)
    c2 = conv_bn_relu(c1, (3, 3), inter_out_num_filters)
    c3 = conv_bn(c2, (1, 1), out_num_filters, bn_init_scale=0)
    p = c3 + input
    return C.relu(p)


def resnet_bottleneck_inc(input, out_num_filters, inter_out_num_filters, stride1x1, stride3x3):
    c1 = conv_bn_relu(input, (1, 1), inter_out_num_filters, strides=stride1x1)
    c2 = conv_bn_relu(c1, (3, 3), inter_out_num_filters, strides=stride3x3)
    c3 = conv_bn(c2, (1, 1), out_num_filters, bn_init_scale=0)
    stride = np.multiply(stride1x1, stride3x3)
    s = conv_bn(input, (1, 1), out_num_filters, strides=stride) # Shortcut
    p = c3 + s
    return C.relu(p)


def resnet_bottleneck_stack(input, num_stack_layers, out_num_filters, inter_out_num_filters):
    assert(num_stack_layers >= 0)
    l = input
    for _ in range(num_stack_layers):
        l = resnet_bottleneck(l, out_num_filters, inter_out_num_filters)
    return l


def create_imagenet_model_bottleneck(input, num_stack_layers, num_classes, stride1x1, stride3x3):
    c_map = [64, 128, 256, 512, 1024, 2048]

    # conv1 and max pooling
    conv1 = conv_bn_relu(input, (7, 7), c_map[0], strides=(2, 2))
    pool1 = MaxPooling((3,3), strides=(2,2), pad=True)(conv1)

    # conv2_x
    r2_1 = resnet_bottleneck_inc(pool1, c_map[2], c_map[0], (1, 1), (1, 1))
    r2_2 = resnet_bottleneck_stack(r2_1, num_stack_layers[0], c_map[2], c_map[0])

    # conv3_x
    r3_1 = resnet_bottleneck_inc(r2_2, c_map[3], c_map[1], stride1x1, stride3x3)
    r3_2 = resnet_bottleneck_stack(r3_1, num_stack_layers[1], c_map[3], c_map[1])

    # conv4_x
    r4_1 = resnet_bottleneck_inc(r3_2, c_map[4], c_map[2], stride1x1, stride3x3)
    r4_2 = resnet_bottleneck_stack(r4_1, num_stack_layers[2], c_map[4], c_map[2])

    # conv5_x
    r5_1 = resnet_bottleneck_inc(r4_2, c_map[5], c_map[3], stride1x1, stride3x3)
    r5_2 = resnet_bottleneck_stack(r5_1, num_stack_layers[3], c_map[5], c_map[3])

    # Global average pooling and output
    pool = AveragePooling(filter_shape=(7, 7), name='final_avg_pooling')(r5_2)
    z = Dense(num_classes, init=C.normal(0.01))(pool)
    return z


def UNET(num_classes, base_num_filters, pad=False):
    """ For semantic segmentation """
    # TODO: allow for depth to be varied
    f = [base_num_filters * 2 ** i for i in range(5)]

    down1 = Conv2DMaxPool(2, conv_filter_shape=(3, 3), pool_filter_shape=(2, 2), conv_num_filters=f[0],
                          activation=C.relu, conv_pad=pad, pool_strides=(2, 2), name_prefix='layer5')

    down2 = Conv2DMaxPool(2, conv_filter_shape=(3, 3), pool_filter_shape=(2, 2), conv_num_filters=f[1],
                          activation=C.relu, conv_pad=pad, pool_strides=(2, 2), name_prefix='layer5')

    down3 = Conv2DMaxPool(2, conv_filter_shape=(3, 3), pool_filter_shape=(2, 2), conv_num_filters=f[2],
                          activation=C.relu, conv_pad=pad, pool_strides=(2, 2), name_prefix='layer5')

    down4 = Conv2DMaxPool(2, conv_filter_shape=(3, 3), pool_filter_shape=(2, 2), conv_num_filters=f[3],
                          activation=C.relu, conv_pad=pad, pool_strides=(2, 2), name_prefix='layer5')

    centre_1 = Convolution2D((3, 3), f[4], activation=C.relu, pad=pad)
    centre_2 = Convolution2D((3, 3), f[4], activation=C.relu, pad=pad)

    up1_1 = Convolution2D((3, 3), f[3], activation=C.relu, pad=pad)
    up1_2 = Convolution2D((3, 3), f[3], activation=C.relu, pad=pad)

    up2_1 = Convolution2D((3, 3), f[2], activation=C.relu, pad=pad)
    up2_2 = Convolution2D((3, 3), f[2], activation=C.relu, pad=pad)

    up3_1 = Convolution2D((3, 3), f[1], activation=C.relu, pad=pad)
    up3_2 = Convolution2D((3, 3), f[1], activation=C.relu, pad=pad)

    up4_1 = Convolution2D((3, 3), f[0], activation=C.relu, pad=pad)
    up4_2 = Convolution2D((3, 3), f[0], activation=C.relu, pad=pad)

    # if num_classes > 1 then softmax is applied at loss function
    clf = Convolution2D((1, 1,), num_classes, activation=identity if num_classes > 1 else C.sigmoid, pad=pad)

    def model(x):
        feature_map0 = x

        # down path
        feature_map1 = down1(feature_map0)
        feature_map2 = down2(feature_map1)
        feature_map3 = down3(feature_map2)
        feature_map4 = down4(feature_map3)

        # latent
        feature_map5 = centre_2(centre_1(feature_map4))

        # up path
        feature_map6 = up1_2(up1_1(Cx.centre_crop_and_splice(feature_map3, Cx.upsample(feature_map5))))
        feature_map7 = up2_2(up2_1(Cx.centre_crop_and_splice(feature_map2, Cx.upsample(feature_map6))))
        feature_map8 = up3_2(up3_1(Cx.centre_crop_and_splice(feature_map1, Cx.upsample(feature_map7))))
        feature_map9 = up4_2(up4_1(Cx.centre_crop_and_splice(feature_map0, Cx.upsample(feature_map8))))

        prediction = clf(feature_map9)
        return prediction

    return model

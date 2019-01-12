import cntk as C
import cntkx as Cx
from cntk.layers import Convolution2D, MaxPooling, Dense, Dropout
from cntk.default_options import default_override_or
from cntk.layers.blocks import identity


def Conv2DMaxPool(n, conv_filter_shape,  # shape of receptive field, e.g. (3,3). Must be a 2-element tuple.
                  pool_filter_shape,  # shape of receptive field, e.g. (3,3)
                  conv_num_filters=None,  # e.g. 64 or None (which means 1 channel and don't add a dimension)
                  activation=default_override_or(identity),
                  init=default_override_or(C.glorot_uniform()),
                  conv_pad=default_override_or(False),
                  conv_strides=1,
                  bias=default_override_or(True),
                  init_bias=default_override_or(0),
                  reduction_rank=1,  # (0 means input has no depth dimension, e.g. audio signal or B&W image)
                  dilation=1,
                  groups=1,
                  pool_strides=1,
                  pool_pad=default_override_or(False),
                  name_prefix=''):
    """ n Convolution 2D followed by one max pooling layer. Convenience wrapper around Conv2D and MaxPooling. """

    convs = [Convolution2D(conv_filter_shape, conv_num_filters, activation, init, conv_pad, conv_strides, bias,
                           init_bias, reduction_rank, dilation, groups, name_prefix + f'_conv_{i}') for i in range(n)]
    maxpool = MaxPooling(pool_filter_shape, pool_strides, pool_pad, name_prefix + '_pool')

    def layer(x):

        for conv in convs:
            x = conv(x)

        x = maxpool(x)
        return x

    return layer


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

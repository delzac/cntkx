import cntk as C
from cntkx.layers.models import VGG16, VGG19, UNET, PretrainedWikitext103LanguageModel
from os.path import join


def test_vgg16():
    a = C.input_variable((3, 64, 64))
    b = VGG16(100)(a)

    assert b.shape == (100, )


def test_vgg19():
    a = C.input_variable((3, 64, 64))
    b = VGG19(100)(a)

    assert b.shape == (100,)


def test_unet():
    a = C.input_variable((3, 128, 128))
    b = UNET(num_classes=10, base_num_filters=8, pad=True)(a)

    assert b.shape == (10, 128, 128)

    a = C.input_variable((3, 256, 256))
    b = UNET(num_classes=10, base_num_filters=2, pad=False)(a)
    # TODO: assert the shape with no padding


def test_pretrained_wikitext103_lm():
    vocab_dim = 238462
    directory = 'C:/Users/Delzac/OneDrive/Pretrained Models/ulmfit/wt103'
    h5_file_path = join(directory, 'fwd_wt103.hdf5')

    a = C.sequence.input_variable(238462)
    lm = PretrainedWikitext103LanguageModel(h5_file_path, )
    prediction = lm(a)

    assert prediction.shape == (vocab_dim, )

    lm = PretrainedWikitext103LanguageModel(h5_file_path, 0.1, 0.1)
    prediction = lm(a)

    assert prediction.shape == (vocab_dim,)

# CNTKx
Deep learning library that builds on and extends Microsoft [CNTK](https://github.com/Microsoft/CNTK). 
This library is in active development, more models and pre-built components coming soon!

## Installation
cntk is a dependency to cntkx. Please get a working installation of cntk first. Then:

    pip install cntkx


## News
***2019-01-12.***
#### Added Vision models: VGG16, VGG19 and UNET
VGG is for image classification and UNET is for semantic segmentation. VGG is implemented for completeness 
sake and should not be used for any serious classification task.


Paper on VGG can be found [here](https://arxiv.org/abs/1409.1556) titled "Very Deep Convolutional Networks 
for Large-Scale Image Recognition"

Paper for UNET can be found [here](https://arxiv.org/abs/1505.04597) titled "U-Net: Convolutional Networks 
for Biomedical Image Segmentation"

VGG example:

    a = C.input_variable((3, 64, 64))
    b = VGG19(100)(a)

    assert b.shape == (100,)

UNET example:

    a = C.input_variable((3, 512, 512))
    b = UNET(num_classes=10, base_num_filters=64, pad=True)(a)

    assert b.shape == (10, 512, 512)


#### Added Transformer attention model and associated components
The Transformer was first introduced in the [paper](https://arxiv.org/abs/1706.03762) 'Attention is all you need'.
The architecture is based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.
More recently, [BERT](https://arxiv.org/abs/1810.04805) which broke almost all SOTA language task is also based on 
transformer and self-attention.

    a = C.sequence.input_variable(512)
    b = C.sequence.input_variable(512)

    transformer = Transformer()  # using default settings
    decoded = transformer(a, b)

    assert decoded.shape == (512, )


***2018-12-08.***
#### Added QRNN: Quasi-Recurrent Neural Network (QRNN) and `cntkx.ops.cumsum`
The QRNN provides similar accuracy to the LSTM but can be betwen 2 and 17 times faster than the 
highly optimized NVIDIA cuDNN LSTM implementation depending on the use case.

More details please refer to the original paper [here](https://arxiv.org/abs/1611.01576).

    input_seq = C.sequence.input_variable(input_dim)
    prediction_seq = QRNN(hidden_dim=50)(input_seq)

***2018-12-07.***
#### New sequence ops: `cntkx.ops.sequence.pad` and `cntkx.ops.sequence.length`
Added two new sequence ops. `cntkx.ops.sequence.pad` allows padding on the sequence axis and 
`cntkx.ops.sequence.length` calculates the length of the sequence.

***2018-12-05.***
#### Mixture Density Network
Mixture density networks are neural networks that can in principle represent arbitrary conditional 
probability distributions in the same way that a conventional neural network can represent arbitrary functions.
MDN are very useful when you need to map an input to several correct targets (aka. one-to-many problem).

Updated with Gaussian Mixture Density Network ops and loss function. Ops will allow you to extract mdn coefficients and sample from the network.

More details on mdn can be found in this [paper](https://publications.aston.ac.uk/373/1/NCRG_94_004.pdf) by Christopher Bishop.

    input_tensor = C.input_variable(1, name="input_tensor")
    target_tensor = C.input_variable(1, name="target_tensor")
    
    # model
    inner = Dense(50, activation=C.relu)(input_tensor)
    inner = Dense(50, activation=C.relu)(inner)
    prediction_tensor = Dense((ndim + 2) * nmix, activation=None)(inner)
    
    sampled = sample_gaussian_mdn(prediction_tensor, nmix, ndim)  # sampling node
    loss = gaussian_mdn_loss(prediction_tensor, target_tensor, nmix=nmix, ndim=ndim)  # loss function
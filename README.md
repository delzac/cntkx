# CNTKx
Deep learning library that builds on and extends Microsoft Cognitive Toolkit [CNTK](https://github.com/Microsoft/CNTK). 
This library is in active development, more models and pre-built components coming soon!

Contributions are very welcomed!

## Installation
cntk is a dependency to cntkx. Please get a working installation of cntk first. Then:

    pip install cntkx

cntkx only works with python>=3.6


## Available Components
| ops | Description |
| --- | ---|
| `cumsum` | Cumulative summation along axis |
| `upsample` | Upsample by 2x (for image) |
| `centre_crop` | Crop centre of image (convenience function) |
| `swish` | Activation (convenience function) |
| `hardmax` | Activation (convenience function) |
| `erf` | Error function |
| `sequence.pad` | Pad at start or end of sequence axis |
| `sequence.length` | length of sequence |
| `sequence.position` | position of every sequence element |
| `random.sample` | Samples a given probability distribution |
| `batchmatmul` | Batch Matrix Multiplication on a static batch axis, similar to tf.matmul |

| Layers | Description |
| --- | ---|
| `QRNN` | Quasi-Recurrent Neural Network |
| `WeightDroppedLSTM` | A form of regularised LSTM |
| `SinusoidalPositionalEmbedding` | Non-learnable positional embedding (no max sequence length) |
| `PositionalEmbedding` | Learnable Positional Embedding (used in BERT) |
| `BertEmbeddings` | BERT Embeddings (word + token_type + positional) |
| `BertPooler` | Pooler used in BERT |
| `SpatialPyramidPooling` | Fixed pooled representation regardless of image input size |
| `GatedLinearUnit` | Gated Convolutional Neural Network |
| `Variational Dropout` | Single binary dropout mask for entire sequence |
| `ScaledDotProductAttention` | Attention used in BERT and Transformer (aka 'attention is all you need') |
| `MultiHeadAttention` | Attention used in BERT and Transformer (aka 'attention is all you need') |
| `GaussianWindowAttention` | Windowed attention instead of conventional attention where everything is attended at the same time |
| `SequentialStride` | strides across sequential axis |
| `SequentialMaxPooling` | Max pool across sequential axis and static axes |

| Loss | Description |
| --- | ---|
| `gaussian_mdn_loss` | loss function when using Mixture density network |
| `focal_loss_with_softmax` | A kind of cross entropy that handles extreme class imbalance |

| Models | Description |
| --- | ---|
| `VGG` | Image Classification |
| `UNET` | Semantic Segmentation |
| `Transformer` | Language Modelling |
| `MDN` | Mixture Density Networks | 


| Pre-trained models | Description |
| --- | ---|
| `Bert` | Bidirectional Encoder Representations from Transformers |


| Learners | Description |
| --- | ---|
| `CyclicalLearningRate` | a method to eliminate the need to find best value and schedule for learning rate |


## C# CNTK Tutorials
This library is implemented in pure cntk python API. For help in cntk c#, you can refer to the two repository 
[deep-learning-with-csharp-and-cntk](https://github.com/anastasios-stamoulis/deep-learning-with-csharp-and-cntk) 
and [DeepBelief_Course4_Examples](https://github.com/AllanYiin/DeepBelief_Course4_Examples). 


## News
***2019-03-24.***
#### Added `cntkx.layers.SequentialMaxPooling` and `cntkx.layers.SequentialStride`
Add max pooling layer that works with sequential axis. Current cntk `MaxPooling` doesn't pool across sequence elements.
`SequentialStride` is added to allow striding across sequence axis.

Example on `cntkx.layers.SequentialMaxPooling`

    # rgb image of height 25 and variable width
    a = C.sequence.input_variable((3, 25))
    
    # Convolute across image with (3, 3) kernel with stride (1, 1)
    b = C.layers.SequentialConvolution(filter_shape=(3, 3), num_filters=16, stride=(1, 1), pad=True)(a)
    
    assert b.shape == (16, 25)
    
    # max pool (2,2) in height and width with stride (2,2) in height and width, no padding
    c = SequentialMaxPooling(filter_shape=(2, 2), strides=(2, 2), pad=False)(b)
    
    assert c.shape == (16, 12)


***2019-03-18.***
#### Added `cntkx.learners.CyclicalLearningRate`
Cyclical learning rate (CLR) is an implementation to that  practically eliminates the need to 
experimentally find the best values and schedule  for  the global  learning  rates.

Instead  of  monotonically decreasing the learning rate, this method lets the learning  
rate  cyclically  vary  between  reasonable  boundary  values. Training  with  
cyclical  learning  rates  instead of  fixed  values  achieves improved  classification 
accuracy without a need to tune and often in fewer iterations.

More details can be found in [Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/abs/1506.01186) 
by Leslie N. Smith

This CLR implementation can be used with the cntk training loop by adding only ***two lines of code***:

    model = C.layers.Dense(10)(C.input_variable(10))
    sgd_momentum = C.momentum_sgd(model.parameters, 0.1, 0.9)
    clr = CyclicalLeaningRate(sgd_momentum, minibatch_size=32)  # first line of code

    for epoch in range(10):
        for batch in range(100):
            trainer.train_minibatch(...)
            clr.batch_step()  # second line of code (to be called after every training update)




***2019-03-12.***
#### Added `cntkx.ops.batchmatmul`
Added Batch Matrix Multiplication. This implementation is similar 
to [tensorflow.matmul](https://www.tensorflow.org/api_docs/python/tf/linalg/matmul).

Example:

    a = C.sequence.input_variable((3, 4, 5))     # batch matrix
    b = C.sequence.input_variable((3, 5, 6))     # batch matrix
    c = Cx.batchmatmul(a, b)
    assert c.shape == (3, 4, 6)                  # 3 is treated as a batch axis



***2019-03-10.***
#### Added `PretrainedBertEncoder` and `PretrainedBertModel`
BERT, the state-of-the-art language model is now available as a CNTK pretrained model.

Currently, it is only tested to work with `BERT-Base, Uncased` (uncased_L-12_H-768_A-12) and can be
downloaded from [Google AI](https://github.com/google-research/bert)

When you have downloaded `BERT-Base, Uncased`, there should be 5 files inside. You will need to `.zip`
three of those files into a tensorflow checkpoint file before you can load it into `cntkx`.

Those three files are: `bert_model.ckpt.data-00000-of-00001`, `bert_model.ckpt.index`, `bert_model.ckpt.meta`.
Then rename the extension of `.zip` into `.ckpt` and you are good to go.

Example below

    text_tensor = C.sequence.input_variable(30522)
    token_type_tensor = C.sequence.input_variable(2)
    filepath_to_tf_bert_model = "YOUR_FILE_DIRECTORY/bert_model.ckpt"

    model = Cx.layers.PreTrainedBertModel(filepath_to_tf_bert_model, num_heads=12, dropout_rate=0.1)
    b = model(text_tensor, token_type_tensor)

    assert b.shape == (768,)

For more details about BERT, you can find the original paper [here](https://arxiv.org/abs/1810.04805), 
and some useful resources [here](https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270) 
and [here](http://jalammar.github.io/illustrated-bert/).

Note:
It goes without saying also that to use these pre-trained models you will need to have tensorflow installed
since we are convert them from tensorflow models.


***2019-03-06.***
#### Added `PositionalEmbedding`, `BertEmbeddings` and `PretrainedBertEmbeddings`
CNTK implementation of `PositionalEmbedding`, `BertEmbeddings` and tf-to-cntk `PreTrainedBertEmbeddings`.
BERT is a state-of-the-art language model from Google AI, more details can be found in
[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805).

Google AI's pre-trained BERT tensorflow model can be downloaded [here](https://github.com/google-research/bert).
Tensorflow would need to be installed in your environment if you intend to use `PreTrainedBertEmbeddings`, which
takes a tensorflow model and convert it cntk.

Example for `PositionalEmbedding`

    a = C.sequence.input_variable(12)
    b = PositionalEmbedding(max_seq_length, hidden_dim)(a)

    assert b.shape == (hidden_dim, )

Example for `BertEmbeddings`

    text_tensor = C.sequence.input_variable(100)
    token_type_tensor = C.sequence.input_variable(2)
    b = BertEmbeddings(max_seq_length, hidden_dim, 0.1)(text_tensor, token_type_tensor)

    assert b.shape == (hidden_dim, )

Example for `PreTrainedBertEmbeddings`

    text_tensor = C.sequence.input_variable(30522)
    token_type_tensor = C.sequence.input_variable(2)
    filepath_to_tf_bert_model = "YOURFILEPATH"
    embeddings = PreTrainedBertEmbeddings(filepath_to_tf_bert_model, 0.1, False)
    b = embeddings(text_tensor, token_type_tensor)
    
    assert b.shape == (768, )

    
***2019-03-02.***
#### Added `VariationalDrpoout` and `WeightDroppedLSTM`
CNTK implementation of `VariationalDrpoout` found in 
[A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](https://arxiv.org/abs/1512.05287)
and `WeightDroppedLSTM` proposed in a salesforce research paper 
[Regularizing and Optimizing LSTM Language Models](https://arxiv.org/abs/1708.02182).

`WeightDroppedLSTM` is a regularised LSTM that uses DropConnect on hidden-to-hidden weights as a form of recurrent
regularisation. It also include application of variational dropout on the inputs and outputs of the recurrent units
for further regularisation.

`VariationalDrpoout` is a regularisation that uses same dropout mask at each time step 
(i.e. across the dynamic sequence axis) as opposed to the naive application of `C.layers.Dropout` to a sequence
which will result in a different dropout mask for every tensor along the sequence axis.


    import cntkx as Cx
    import cntk as C
    
    seq = C.sequence.input_variable(56)
    hidden = Cx.layers.WeightDroppedLSTM(100,
                                         dropconnect_rate=0.1,
                                         variational_dropout_rate_input=0.1,
                                         variational_dropout_rate_output=0.1)(seq)
    
    assert hidden.shape == (100, )
    
    seq_dropped = VariationalDropout(0.1)(seq)
    
    assert seq_dropped.shape == seq.shape


***2019-02-02.***
#### Added Gated Linear Unit / Gated CNN
CNTK implementation of Gated Linear Unit (Gated CNN) founnd in Facebook AI Research Lab's paper:
[Language Modeling with Gated Convolutional Networks](https://arxiv.org/abs/1612.08083).
This paper applies a convolutional approach to language modelling with a novel Gated-CNN model.

    import cntkx as Cx
    import cntk as C
    
    seq = C.sequence.input_variable(56)
    hidden = Cx.layers.GatedLinearUnit(window=2, hidden_dim=100)(seq)
    
    assert hidden.shape == (100, )


***2019-01-21.***
#### Added `Focal Loss` for multi-class and binary classification
CNTK implementation of `Focal Loss` enables the training of highly accurate dense object detectors in the
presence of vast numbers of easy background examples or dataset with extreme class imbalance (e.g. 1:1000).

`Focal Loss` focuses training on a sparse set of hard examples and prevents the vast number of easy negatives from
overwhelm-ing the model during training. 

For more details please refer to [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)

    import cntkx as Cx
    
    Cx.focal_loss_with_softmax([[0, 0, 0.8, 0.2]], [[0, 0, 1, 0]]).eval()
    array([[0.31306446]], dtype=float32)



***2019-01-18.***
#### Added Gaussian Window Attention Model
Gaussian window attention model was first introduced by Alex Graves in 
"Generating sequences with recurrent neural networks".

It uses a mixture of gaussian windows to attend to 
portions of the sequence as oppose to the widely used attention model introduced in 
"Neural machine translation by jointly learning to align and translate" by Bahdanau, et al. that attends
to the entire sequence all at once.

Gaussian window attention is also directional in its attention on the context sequence. When modeling
strongly ordered sequences, gaussian window attention will be a natural choice due to this inductive bias.
    
    import cntk as C
    import cntkx as Cx
    
    seq1 = C.Axis.new_unique_dynamic_axis('seq1')
    seq2 = C.Axis.new_unique_dynamic_axis('seq2')

    encoded = C.sequence.input_variable(30, sequence_axis=seq1)
    query = C.sequence.input_variable(28, sequence_axis=seq2)

    a = Cx.layers.GaussianWindowAttention(10)(encoded, query)

    assert a.shape == (30, )

"Generating sequences with recurrent neural networks" can be found [here](https://arxiv.org/abs/1308.0850).
"Neural machine translation by jointly learning to align and translate" can be found [here](https://arxiv.org/abs/1409.0473).

***2019-01-16.***
#### Added Spatial Pyramid Pooling layer
Spatial pyramid pooling layer is a pooling layer than returns a fixed length representation regardless of the 
image size/scale. It is frequently used for multi-size image training. It reported SOTA classification results using
a single full-image representation without fine-tuning. For more details on the paper
"Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition" by K. He, X. Zhang, S. Ren, J. Sun,
link [here](https://arxiv.org/abs/1406.4729).

    import cntk as C
    import cntkx as Cx

    n = np.random.random((3, 3, 32, 32)).astype(np.float32)
    a = C.input_variable((3, 32, 32))
    b = Cx.layers.SpatialPyramidPooling((1, 2, 4))(a)

    assert b.shape == (3 * (4 * 4 + 2 * 2 + 1),)  # representation not dependent on image size


***2019-01-15.***
#### Added Sinusoidal Positional Embedding and `cntkx.ops.erf`
Added sinusoidal positional embedding used in [Transformer](https://arxiv.org/abs/1706.03762). For an accessible
explanation of transformer, you may look up [here](http://jalammar.github.io/illustrated-transformer/).

    import cntk as C
    import cntkx as Cx
    
    a = C.sequence.input_variable(10)
    b = SinusoidalPositionalEmbedding()(a)

    assert b.shape == (10, )

Added `cntkx.ops.erf` error function.

***2019-01-12.***
#### Added Vision models: VGG16, VGG19 and UNET
VGG is for image classification and UNET is for semantic segmentation. VGG is implemented for completeness 
sake and should not be used for any serious classification task.


Paper on VGG can be found [here](https://arxiv.org/abs/1409.1556) titled "Very Deep Convolutional Networks 
for Large-Scale Image Recognition"

Paper for UNET can be found [here](https://arxiv.org/abs/1505.04597) titled "U-Net: Convolutional Networks 
for Biomedical Image Segmentation"

VGG example:

    import cntk as C
    import cntkx as Cx
    
    a = C.input_variable((3, 64, 64))
    b = Cx.layers.VGG19(100)(a)

    assert b.shape == (100,)

UNET example:

    import cntk as C
    import cntkx as Cx
    
    a = C.input_variable((3, 512, 512))
    b = Cx.layers.UNET(num_classes=10, base_num_filters=64, pad=True)(a)

    assert b.shape == (10, 512, 512)

Convenience functions such as `cntkx.ops.upsample` and `centre_crop` have also been added.
`cntkx.ops.upsample` upsamples an image twice on each spatial dim. `centre_crop` crops a smaller image from
a bigger one in the centre given a reference smaller image.


#### Added Transformer attention model and associated components
The Transformer was first introduced in the [paper](https://arxiv.org/abs/1706.03762) 'Attention is all you need'.
The architecture is based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.
More recently, [BERT](https://arxiv.org/abs/1810.04805) which broke almost all SOTA language task is also based on 
transformer and self-attention.

    import cntk as C
    import cntkx as Cx
    
    a = C.sequence.input_variable(512)
    b = C.sequence.input_variable(512)

    transformer = Cx.layers.Transformer()  # using default settings
    decoded = transformer(a, b)

    assert decoded.shape == (512, )


***2018-12-08.***
#### Added QRNN: Quasi-Recurrent Neural Network (QRNN) and `cntkx.ops.cumsum`
The QRNN provides similar accuracy to the LSTM but can be betwen 2 and 17 times faster than the 
highly optimized NVIDIA cuDNN LSTM implementation depending on the use case.

More details please refer to the original paper [here](https://arxiv.org/abs/1611.01576).

    import cntk as C
    import cntkx as Cx
    
    input_seq = C.sequence.input_variable(input_dim)
    prediction_seq = Cx.layers.QRNN(hidden_dim=50)(input_seq)



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
    
    import cntk as C
    import cntkx as Cx
    
    input_tensor = C.input_variable(1, name="input_tensor")
    target_tensor = C.input_variable(1, name="target_tensor")
    
    # model
    inner = Dense(50, activation=C.relu)(input_tensor)
    inner = Dense(50, activation=C.relu)(inner)
    prediction_tensor = Dense((ndim + 2) * nmix, activation=None)(inner)
    
    sampled = Cx.sample_gaussian_mdn(prediction_tensor, nmix, ndim)  # sampling node
    loss = Cx.gaussian_mdn_loss(prediction_tensor, target_tensor, nmix=nmix, ndim=ndim)  # loss function
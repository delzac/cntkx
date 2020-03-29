# CNTKx
CNTKx is a deep learning library that builds on and extends Microsoft Cognitive Toolkit [CNTK](https://github.com/Microsoft/CNTK). 
Despite the last planned release of cntk 2.7, cntkx will continue to be in active development, more models and pre-built components coming soon!

Feel free to open an issue for any request or a PR to contribute :)

## Installation
cntkx is written in pure python and cntk is a dependency to it. Please get a working installation of cntk first. Then:

    pip install cntkx

cntkx only works with `python>=3.6`


## Available Components
| ops | Description |
| --- | ---|
| `floor_division` | element-wise floor_division |
| `remainder` | element-wise remainder of division |
| `scalar` | cast tensor to scalar (1,) |
| `cumsum` | Cumulative summation along axis |
| `upsample` | Upsample by 2x (for image) |
| `centre_crop` | Crop centre of image |
| `swish` | Activation |
| `mish` | Activation |
| `hardmax` | Activation |
| `erf` | Error function |
| `gelu` | Gaussian Error Linear Unit function |
| `gelu_fast` | fast approximation of Gaussian Error Linear Unit function |
| `sequence.pad` | Pad at start or end of sequence axis |
| `sequence.length` | length of sequence |
| `sequence.position` | position of every sequence element |
| `sequence.stride` | strides across sequential axis  |
| `sequence.join` | joins two sequence along their sequential axis  |
| `sequence.window` | creates sliding window along the sequence axis  |
| `sequence.window_causal` | creates causal sliding window along the sequence axis  |
| `sequence.reverse` | reverses the items along the dynamic sequence axis  |
| `sequence.reduce_mean` | calculates the mean along the dynamic sequence axis  |
| `sequence.reduce_concat_pool` | drop-in replace for sequence.last  |
| `random.sample` | Samples an unnormalised log probability distribution |
| `random.sample_with_bias` | Samples an unnormalised log probability distribution over-weighted to more probable classes |
| `random.sample_top_k` | Samples from the top_k of an unnormalised log probability distribution |
| `batchmatmul` | Batch Matrix Multiplication on a static batch axis, similar to tf.matmul |

| Layers | Description |
| --- | ---|
| `QRNN` | Quasi-Recurrent Neural Network |
| `Recurrence` | With option to apply `VariationalDroppout` |
| `PyramidalBiRecurrence` | Pyramidal bi-directional recurrence |
| `VariationalDropout` | Single binary dropout mask for entire sequence |
| `SinusoidalPositionalEmbedding` | Non-learnable positional embedding (no max sequence length) |
| `PositionalEmbedding` | Learnable Positional Embedding (used in BERT) |
| `BertEmbeddings` | BERT Embeddings (word + token_type + positional) |
| `BertPooler` | Pooler used in BERT |
| `SpatialPyramidPooling` | Fixed pooled representation regardless of image input size |
| `GatedLinearUnit` | Gated Convolutional Neural Network |
| `ScaledDotProductAttention` | Attention used in BERT and Transformer (aka 'attention is all you need') |
| `MultiHeadAttention` | Attention used in BERT and Transformer (aka 'attention is all you need') |
| `GaussianWindowAttention` | Windowed attention instead of conventional attention where everything is attended at the same time |
| `SequentialDense` | Applies Dense to a window of sequence item along sequence axis |
| `SequentialMaxPooling` | Max pool across sequential axis and static axes |
| `SequentialAveragePooling` | Average pool across sequential axis and static axes |
| `SequentialConcatPooling` | Concat Average and Mean pool across sequential axis and static axes |
| `vFSMN` | Vectorised Feedforward Sequential Memory Networks |
| `cFSMN` | Compact Feedforward Sequential Memory Networks |
| `BiRecurrence` | BiRecurrence recurrent layer with weight tying option to half parameter requirement |
| `GlobalConcatPooling` | Global spatial concat pooling of ave and mean |
|`FilterResponseNormalization`| Drop in replacement for batch norm with superior performance |
|`Boom`| More parametrically efficient alternative to Position-Wise FeedForward layer found in Transformer |

| Blocks | Description |
| --- | ---|
| `WeightDroppedLSTM` | A form of regularised LSTM |
| `IndyLSTM` | A parameter efficient form of LSTM |
| `IndRNN` | a RNN with long memory and can be stacked deeply |

| Loss | Description |
| --- | ---|
| `gaussian_mdn_loss` | loss function when using Mixture density network |
| `focal_loss_with_softmax` | A kind of cross entropy that handles extreme class imbalance |
| `cross_entropy_with_softmax` | Added `label smoothing regularisation` in cross entropy with softmax |
| `adaptive_robust_barron_loss` | drop-in replacement for l1 or l2 loss |

| Models | Description |
| --- | ---|
| `VGG` | Image Classification |
| `UNET` | Semantic Segmentation |
| `Transformer` | Language Modelling |
| `MDN` | Mixture Density Networks | 


| Pre-trained models | Description |
| --- | ---|
| `Bert` | Bidirectional Encoder Representations from Transformers |
| [fwd_wt103.hdf5](https://1drv.ms/u/s!AjJ4XyC3prp8mItNxiawGK4gD8iMhA?e=wh7PLB) | The weight parameters of the fastai's pytorch model. To be used to initialise `PretrainedWikitext103LanguageModel` |
| [fwd_wt103.cntk](https://1drv.ms/u/s!AjJ4XyC3prp8mItPBdfmDYr9QP7J4w?e=k1BXlW) | The converted cntk model of fastai's pytorch model. To be used with `C.load_model` |
| [fwd_wt103.onnx](https://1drv.ms/u/s!AjJ4XyC3prp8mItO70T_q8HOPwa6aQ?e=h2Fiv5) | The converted ONNX model of fastai's pytorch model. |


| Learners | Description |
| --- | ---|
| `CyclicalLearningRate` | a method to eliminate the need to find best value and schedule for learning rate |
| `RAdam` | a variant of `Adam` that doesn't require any warmup |

| Misc | Description |
| --- | ---|
| `CTCEncoder` | Helper class to convert data into a format acceptable for cntk's ctc implementation |


## C# CNTK Tutorials
This library is implemented in pure cntk python API. For help in cntk c#, you can refer to the two repository 
[deep-learning-with-csharp-and-cntk](https://github.com/anastasios-stamoulis/deep-learning-with-csharp-and-cntk) 
and [DeepBelief_Course4_Examples](https://github.com/AllanYiin/DeepBelief_Course4_Examples). 


## F# CNTK
For the F# wrapper of CNTK please visit [FsCNTK](https://github.com/fwaris/FsCNTK), 
it also contains some example implementations like seq2seq, autoencoder, LSTM, GAN.


## News
***2020-03-29.***
#### Added `adaptive_robust_barron_loss` and `sample_with_bias`
`adaptive_robust_barron_loss` is a generalisation for generalization of the Cauchy/Lorentzian,
Geman-McClure, Welsch/Leclerc, generalized Charbonnier, Charbonnier/pseudo-Huber/L1-L2, and
L2 loss functions.

Can be used as a drop-in replacement in any regression task that you have.

For more details, please refer to [A General and Adaptive Robust Loss Function](https://arxiv.org/abs/1701.03077), Jonathan T. Barron, 
It is the Best Paper Award Finalist in CVPR 2019.


Implemented `sample_with_bias` to sample more likely classes as a replacement 
for `sample_top_k` which cannot be used inside a `UnfoldFrom`



***2020-02-11.***
#### Added `Boom`
Boom layer from SHA-RNN by S. Merity creator of QRNN. Alternative to PositionwiseFeedForward. Serves the same function as
PositionwiseFeedForward found in transformer.

Boom layer minimizes computation and removes an entire matrix of parameters compared to traditional down-projection layers.

For more detail please read: [Single Headed Attention RNN: Stop Thinking With Your Head](https://arxiv.org/abs/1911.11423) by Stephen Merity.


***2019-12-03.***
#### Added `FilterResponseNormalization` and `ThresholdedLinearUnit`
Added cntk implementation of `FilterResponseNormalization`. Filter Response Normalization (FRN) layer 
is a novel combination of a normalization and an activation function,
that can be used as a drop-in replacement for other normalizations and activations.

The method operates on each activation map of each batch sample
independently, eliminating the dependency on other batch samples or channels of the same sample.
The method outperforms BN and all alternatives in a variety of settings for all batch sizes.
FRN layer performs ≈0.7−1.0% better on top-1 validation accuracy than BN with large mini-batch sizes on
Imagenet classification on InceptionV3 and ResnetV2-50 architectures. Further, it performs >1% better
than GN on the same problem in the small mini-batch size regime. For object detection problem on COCO dataset,
FRN layer outperforms all other methods by at least 0.3−0.5% in all batch size regimes.

Please refer to the paper [Filter Response Normalization Layer: Eliminating Batch Dependence 
in the Training of Deep Neural Networks](https://arxiv.org/abs/1911.09737v1) for more details .



***2019-11-04.***
#### Added `ops.floor_division`, `ops.remainder`, `sequence.window_causal` and `SequentialDense`
Add in new operations stated above. `sequence.window` now has an additional argument that lets you control striding.
`sequence.window_causal` creates causal window that doesn't leak future information into the past (preserve causality).
`SequentialDense` convenience layer added to apply dense to a window of sequence item,
much like `SequentialConvolution` but with better memory performance.



***2019-11-04.***
#### Added `SequentialConcatPooling`, `Cx.Sequence.reduce_concat_pool` and `GlobalConcatPooling`
`Cx.Sequence.reduce_concat_pool` concatenates the last item in the sequence axis with the summarisation
of the sequence represented by `reduce_max` and `reduce_mean` of the sequence axis. Anytime `C.sequence.last` is used,
this can be a drop-in replacement.

Example:

    n = 32
    a = C.sequence.input_variable(n)
    b = Cx.sequence.reduce_concat_pool(a)

    assert b.shape == (n * 3, )


`SequentialConcatPooling` does spatial concat pooling over the sequential axis.
Concat pooling is the concatenation of both average pooling and max pooling. In any situation where max or ave 
pooling is appropriate, concat pooling can be used as a drop-in replacement and achieve improvements in performance.

Example:

    a = C.sequence.input_variable((3, 10))
    b = SequentialConcatPooling(filter_shape=(2, 2), strides=2)(a)
    
    assert b.shape == (6, 10)
            
    n = [np.random.random((3, 10)).astype(np.float32),
         np.random.random((3, 10)).astype(np.float32),
         np.random.random((3, 10)).astype(np.float32),
         np.random.random((3, 10)).astype(np.float32), ]
    
    print(b.eval({a: n}))


`GlobalConcatPooling` is the standard spatial concat pooling of both max pool and ave pool.


***2019-10-15.***
#### Added `mish` activation function and `RAdam` learner.
`mish` is an activation function is introduced in [Mish: A Self Regularized Non-Monotonic Neural Activation Function](https://arxiv.org/abs/1908.08681v2)
by Diganta Misra. Experiments show that `mish` tends to work better than both ReLU and Swish along with other standard
activation functions in many deep networks across challenging datasets. For instance,
in Squeeze Excite Net-18 for CIFAR 100 classification, the network with Mish had an increase in
Top-1 test accuracy by 0.494% and 1.671% as compared to the same network with Swish and ReLU respectively.
The similarity to Swish along with providing a boost in performance and its simplicity in implementation
makes it easier for researchers and developers to use Mish in their Neural Network Models.

This activation function is adopted in `fast ai`.

Rectified Adam or `RAdam` is a `Adam` optimiser variant that doesn't require a warmup schedule, which `Adam` tends
to need to maintain stability. In this cntk implementation, we added a `RAdam` like optimiser based
on the work of [On the adequacy of untuned warmup for adaptive optimization](https://arxiv.org/abs/1910.04209) by Jerry Ma and Denis Yarats.

`RAdam` is adopted in `fast ai` too.



***2019-09-30.***
#### Added `BiRecurrence` with weight tying
Made improvement to weight tying of BiRecurrence by have one parameter tensor token for every state 
in the step function per direction (forward and backward). This will allow forward and backward token
to have more representational flexibility. Previously, all states use the same forward or backward token.


***2019-09-06.***
#### Added `BiRecurrence` with weight tying
Add a wrapper to create a bidirectional recurrent layer using `BiRecurrence`. Included in the implementation
is an option to half the number of parameters required by  bidirectional recurrent layer. This is done
by only using one recurrent unit to do both forward and backward computation instead of the usual two.
A forward and backward token is used to initialise the hidden state so that the recurrent unit can tell
the directionality.

More details can be found in the paper [Efficient Bidirectional Neural Machine Translation](https://arxiv.org/abs/1908.09329)

Example:

    a = C.sequence.input_variable(10)
    b = BiRecurrence(LSTM(100), weight_tie=True)(a)
    
    assert b.shape == (200, )


***2019-08-29.***
#### Added `PretrainedWikitext103LanguageModel`
CNTK implementation of Fast AI's Universal  Language  ModelFine-tuning (ULMFiT) English model has been added.
This language model was trained on Wikitext-103 and can be used as a base model for any downstream language task.

It is also much more efficient to run compare to BERT and other Transformer language models.

For more details, you can refer to the original paper [here](https://arxiv.org/abs/1801.06146)

Example:

    vocab_size = 238462
    converted_hdf5_model_file_path = 'PATH/fwd_wt103.hdf5'  # this is not the original pytorch model
    lm = PretrainedWikitext103LanguageModel(converted_hdf5_model_file_path)

    a = C.sequence.input_variable(vocab_size)
    prediction = lm(a)  # next-word-prediction
    features = prediction.features  # features of tokens

    assert prediction.shape == (vocab_size, )
    assert features.shape == (400, )

| Model | Description |
| --- | ---|
| [fwd_wt103.hdf5](https://1drv.ms/u/s!AjJ4XyC3prp8mItNxiawGK4gD8iMhA?e=wh7PLB) | The weight parameters of the fastai's pytorch model. To be used to initialise `PretrainedWikitext103LanguageModel` |
| [fwd_wt103.cntk](https://1drv.ms/u/s!AjJ4XyC3prp8mItPBdfmDYr9QP7J4w?e=k1BXlW) | The converted cntk model of fastai's pytorch model. To be used with `C.load_model` |
| [fwd_wt103.onnx](https://1drv.ms/u/s!AjJ4XyC3prp8mItO70T_q8HOPwa6aQ?e=h2Fiv5) | The converted ONNX model of fastai's pytorch model. |
| [itos_wt103.pkl](http://files.fast.ai/models/wt103/) | Tokens used in pretrained model |


***2019-08-08.***
#### Added `cntkx.ops.gelu` and `cntkx.ops.gelu_fast`
Added two cntk implementation of `gelu` activation function. `gelu` activation is used in `BERT`
and OpenAI's `GPT` and `GPT-2`.

Gaussian Error Linear Unit (GELU), a high-performing neuralnetwork activation function.
The GELU nonlinearity is the expected transforma-tion of a stochastic regularizer which randomly
applies the identity or zero mapto a neuron’s input.  The GELU nonlinearity weights inputs by their
magnitude,rather than gates inputs by their sign as in ReLUs.

For more detail please refer to [Gaussian Error Linear Units (GELU)](https://arxiv.org/abs/1606.08415)
by Hendrycks and Gimpel.


***2019-07-04.***
#### Added `cntkx.ops.sequence.reduce_mean`
Calculates the mean along the dynamic sequential axis in CNTK.


***2019-06-26.***
#### Added `cntkx.ops.sequence.reverse`
Allows the sequence items along the sequence axis to be reversed. This is useful when you want to create a 
Bi-directional Auto-Regressive layer because using UnfoldFrom does not work with Recurrence(go_backwards=True) and
 will result in a ValueError.


Example:
    
    import cntk as C
    import cntkx as Cx
    from cntk.layers import Recurrence, UnfoldFrom, LSTM
    
    hidden_dim = 50
    start_token = C.Constant(0, shape=(hidden_dim,))
    a = C.sequence.input_variable(1, name='seq1')
    
    b = UnfoldFrom(Recurrence(LSTM(hidden_dim), go_backwards=True))(start_token, a)
    
    n = [np.random.random((10, hidden_dim)).astype(np.float32),]
    
    # This raise 'ValueError: It is not allowed to have multiple different stepping directions in the same loop'
    b.eval({b.arguments[0]: n})
    

The workaround would be:
    
    import cntk as C
    import cntkx as Cx
    from cntk.layers import Recurrence, UnfoldFrom, LSTM
    
    hidden_dim = 50
    start_token = C.Constant(0, shape=(hidden_dim,))
    a = C.sequence.input_variable(1, name='seq1')
    a_reversed = Cx.sequence.reverse(a)
    
    b = UnfoldFrom(Recurrence(LSTM(hidden_dim)))(start_token, a_reversed)  # remove go_backwards=True

    n = [np.random.random((10, hidden_dim)).astype(np.float32),]    
    b.eval({b.arguments[0]: n})  # this will run just fine


    
***2019-06-21.***
#### Added `cntkx.ops.random.sample_top_k`
CNTK implementation of that allows sampling of the top_k unnormalised log-probabilities distribution.
This is useful text (or sequence) generation where it is known that greedy decoding will cause
text degeneration.

For more details on this, please refer to [A curious case of neural text degeneration](https://arxiv.org/abs/1904.09751)


Example:
    
    import cntk as C
    import cntkx as Cx
    
    a = C.input_variable(5)
    b = Cx.random.sample_top_k(a, k=3, num_classes=5)
        
    n = np.array([[1, 2, 3, 4, 5],] * 1000)

    results = b.eval({a: n})
    assert np.sum(results[:, :2]) == 0
    assert np.sum(results[:, 2:]) == 1000

***2019-05-04.***
#### Added `cntkx.layers.vFSMN` and `cntkx.layers.cFSMN`
CNTK implementation of Bidirectional vectorised Feedforward Sequential Memory Network (vFSMN)
and Compact Feedforward Sequential Memory Network (cFSMN).

FSMN is a standard fully-connected feedforward neural network equipped
with some learnable memory blocks in its hidden layers. The memory blocks
use a tapped-delay line structure to encode the long context information into
a fixed-size representation as short-term memory mechanism.

The authors claim that FSMNs can be learned much more reliably and faster than
RNNs or LSTMs due to the inherent non-recurrent model structure while significantly
outperforming RNNs in language and speech modeling.

For more details please refer to [Feedforward Sequential Memory Networks: A 
New Structure to Learn Long-term Dependency](https://arxiv.org/abs/1512.08301)

cFSMN is a compact version of FSMN that can result in a reduction of up
to 60% in model size and speed up the learning by more than 7 times while
still significantly outperforming the popular bi-direction LSTMs for both
frame-level cross-entropy (CE) criterion based training and MMI based sequence training.

For more details please refer to "Compact Feedforward Sequential Memory Networks for
Large VocabularyContinuous Speech Recognition" by Zhang, et al.

Example:
    
    import cntk as C
    from cntkx.layers import vFSMN, cFSMN
    
    # unidirectional vFSMN (only past conext used)
    a = C.sequence.input_variable(10)
    b = vFSMN(100, C.relu, num_past_context=3, num_future_context=0)(a)

    assert b.shape == (100,)
    
    # bidirectional vFSMN (enable both past and future context)
    a = C.sequence.input_variable(10)
    b = vFSMN(120, C.relu, num_past_context=3, num_future_context=3)(a)

    assert b.shape == (120,)
    
    # bidirectional cFSMN (enable both past and future context)
    a = C.sequence.input_variable(100)
    b = cFSMN(120, 50, C.relu, num_past_context=3, num_future_context=3)(a)

    assert b.shape == (120,)


***2019-04-19.***
#### Added `cntkx.misc.CTCEncoder`
CNTK's CTC implementation requires that data be formatted in a particular way that's typically in acoustic
modeling but unusual in other applications. So class provides an easy way to convert data between
what users typically expect and what cntk demands.

Example:
    labels = ['a', 'b', 'c']
    encoder = CTCEncoder(labels)

    labels_tensor = C.sequence.input_variable(len(encoder.classes_))  # number of classes = 4
    input_tensor = C.sequence.input_variable(100)

    labels_graph = cntk.labels_to_graph(labels_tensor)
    network_out = model(input_tensor)

    fb = C.forward_backward(labels_graph, network_out, blankTokenId=encoder.blankTokenId)

    ground_truth = ['a', 'b', 'b', 'b', 'c']
    seq_length = 10  # must be the same length as the sequence length in network_out

    fb.eval({input_tensor: [...],
             labels_tensor: [encoder.transform(ground_truth, seq_length=seq_length)]})



***2019-04-14.***
#### Added `Label Smoothing Regularization`, `seqeuence.window` and `PyramidalBiRecurrence`
Added `Label Smoothing Regularization` in `cross_entropy_with_softmax`.
Added `sequence.window` that creates non-overlapping window along the sequence axis thereby reducing the 
sequence length and increasing the dimension by the same factor.

Implemented a convenience layer used in acoustic modelling known as `PyramidalBiRecurrence`. Used to create
pyramidal bi-directional LSTM (BLSTM) found in "Listen, attend and spell" by Chan et al. (https://arxiv.org/abs/1508.01211)
Typically used to down sample the sequence length to make memory and runtime manageable.


***2019-04-08.***
#### Added `cntkx.ops.sequence.join`
Added a new `op` called `join` where two sequence tensors can be joined along with sequence axis forming a longer sequence.


***2019-04-08.***
#### Added `cntkx.layers.SequentialAveragePooling`
Add average pooling layer that works with sequential axis. Current cntk `AveragePooling` doesn't pool across sequence elements.

Example on `cntkx.layers.SequentialAveragePooling`

    # rgb image of height 25 and variable width
    a = C.sequence.input_variable((3, 25))
    
    # Convolute across image with (3, 3) kernel with stride (1, 1)
    b = C.layers.SequentialConvolution(filter_shape=(3, 3), num_filters=16, stride=(1, 1), pad=True)(a)
    
    assert b.shape == (16, 25)
    
    # max pool (2,2) in height and width with stride (2,2) in height and width, no padding
    c = SequentialAveragePooling(filter_shape=(2, 2), strides=(2, 2), pad=False)(b)
    
    assert c.shape == (16, 12)


***2019-04-07.***
#### Added `cntkx.sequence.stride` and `cntkx.ops.scalar`
`Cx.sequence.stride` enables striding across the sequential axis, selecting every integer items along the sequence.
`Cx.scalar` converts tensor into scalar of shape `(1,)` 



***2019-04-06.***
#### Added `IndyLSTM` and `IndRNN`
CNTK implementation of [Independently Recurrent Long Short-term Memory cells: IndyLSTMs](https://arxiv.org/abs/1903.08023)
by Gonnet and Deselaers, and [Independently Recurrent Neural Network (IndRNN): Building A Longer andDeeper RNN](https://arxiv.org/abs/1803.04831)
by Li, et al.

Both `IndyLSTM` and `IndRNN` have hidden-to-hidden weights that are diagonal matrix instead of the usual full matrix.
Thus neurons in each layer are independent from each other, and the cross-channel information is 
obtained through stacking multiple layers.

`IndRNN` allows for the use of `C.relu` activation thus allowing multiple `IndRNN` layers to be stacked together deeply.

`IndyLSTM` has parameters linear to the number of nodes in the linear, as opposed to standard LSTM that is quadratic
making `IndyLSTM` potentially faster and smaller as a model.

Authors of both `IndRNN` and `IndyLSTM` have claimed performance as good as or even better than regular LSTMs.

Example:

    import cntk as C
    from cntkx.layers import IndyLSTM, IndRNN, Recurrence
    
    a = C.sequence.input_variable(10)
    b = Recurrence(IndRNN(20))(a)
    c = Recurrence(IndyLSTM(20))(a)
    
    assert b.shape == c.shape == (20,)



***2019-03-24.***
#### Added `cntkx.layers.SequentialMaxPooling`
Add max pooling layer that works with sequential axis. Current cntk `MaxPooling` doesn't pool across sequence elements.

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
#### Added `VariationalDropout` and `WeightDroppedLSTM`
CNTK implementation of `Variational Dropout` found in 
[A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](https://arxiv.org/abs/1512.05287)
and `Weight Dropped LSTM` proposed in a salesforce research paper 
[Regularizing and Optimizing LSTM Language Models](https://arxiv.org/abs/1708.02182).

`Weight Dropped LSTM` is a regularised LSTM that uses DropConnect on hidden-to-hidden weights as a form of recurrent
regularisation. It also include application of variational dropout on the inputs and outputs of the recurrent units
for further regularisation.

`Variational Drpoout` is a regularisation that uses same dropout mask at each time step 
(i.e. across the dynamic sequence axis) as opposed to the naive application of `C.layers.Dropout` to a sequence
which will result in a different dropout mask for every tensor along the sequence axis.


    import cntkx as Cx
    from cntkx.layers import Recurrence, WeightDroppedLSTM
    import cntk as C
    
    dropconnect_rate = 0.2
    variationaldrop_rate = 0.1
    
    seq = C.sequence.input_variable(56)
    b = Recurrence(WeightDroppedLSTM(20, dropconnect_rate),
                   variational_dropout_rate_input=variationaldrop_rate,
                   variational_dropout_rate_output=variationaldrop_rate)(seq)
    
    assert b.shape == (100, )
    
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
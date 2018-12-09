# CNTKx
Deep learning library that builds on and extends Microsoft [CNTK](https://github.com/Microsoft/CNTK). 
This library is in active development, more models and pre-built components coming soon!

## Installation
cntk is a dependency to cntkx. Please get a working installation of cntk first. Then:

    pip install cntkx


## News
***2018-12-08***
#### Added QRNN: Quasi-Recurrent Neural Network (QRNN) and `cntkx.ops.cumsum`
The QRNN provides similar accuracy to the LSTM but can be betwen 2 and 17 times faster than the 
highly optimized NVIDIA cuDNN LSTM implementation depending on the use case.

More details please refer to the original paper [here](https://arxiv.org/abs/1611.01576).


***2018-12-07***
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
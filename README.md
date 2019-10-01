<a href="https://captum.org">
  <img width="350" src="./captum_logo_lockup.svg" alt="Captum Logo" />
</a>

<hr/>

[![Conda](https://img.shields.io/conda/v/pytorch/captum.svg)](https://anaconda.org/pytorch/captum)
[![PyPI](https://img.shields.io/pypi/v/captum.svg)](https://pypi.org/project/captum)
[![CircleCI](https://circleci.com/gh/pytorch/captum.svg?style=shield)](https://circleci.com/gh/pytorch/captum)
[![Codecov](https://img.shields.io/codecov/c/github/pytorch/captum.svg)](https://codecov.io/github/pytorch/captum)

Captum is a model interpretability and understanding library for PyTorch.
Captum means comprehension in latin and contains general purpose implementations
of integrated gradients, saliency maps, smoothgrad, vargrad and others for
PyTorch models. It has quick integration for models built with domain-specific
libraries such as torchvision, torchtext, and others.

*Captum is currently in beta and under active development!*


#### About Captum

With the increase in model complexity and the resulting lack of transparency, model interpretability methods have become increasingly important. Model understanding is both an active area of research as well as an area of focus for practical applications across industries using machine learning. Captum provides state-of-the-art algorithms, including Integrated Gradients, to provide researchers and developers with an easy way to understand which features are contributing to a model’s output.

For model developers, Captum can be used to improve and troubleshoot models by facilitating the identification of different features that contribute to a model’s output in order to design better models and troubleshoot unexpected model outputs.

Captum helps ML researchers more easily implement interpretability algorithms that can interact with PyTorch models. Captum also allows researchers to quickly benchmark their work against other existing algorithms available in the library.

#### Target Audience

The primary audiences for Captum are model developers who are looking to improve their models and understand which features are important and interpretability researchers focused on identifying algorithms that can better interpret many types of models.

Captum can also be used by application engineers who are using trained models in production. Captum provides easier troubleshooting through improved model interpretability, and the potential for delivering better explanations to end users on why they’re seeing a specific piece of content, such as a movie recommendation.

## Installation

**Installation Requirements**
- Python >= 3.6
- PyTorch >= 1.2


##### Installing the latest release

The latest release of Captum is easily installed either via
[Anaconda](https://www.anaconda.com/distribution/#download-section) (recommended):
```bash
conda install captum -c pytorch
```
or via `pip`:
```bash
pip install captum
```


##### Installing from latest master

If you'd like to try our bleeding edge features (and don't mind potentially
running into the occasional bug here or there), you can install the latest
master directly from GitHub:
```bash
pip install git+https://github.com/pytorch/captum.git
```

**Manual / Dev install**

Alternatively, you can do a manual install. For a basic install, run:
```bash
git clone https://github.com/pytorch/captum.git
cd captum
pip install -e .
```

To customize the installation, you can also run the following variants of the
above:
* `pip install -e .[insights]`: Also installs all packages necessary for running Captum Insights.
* `pip install -e .[dev]`: Also installs all tools necessary for development
  (testing, linting, docs building; see [Contributing](#contributing) below).
* `pip install -e .[tutorials]`: Also installs all packages necessary for running the tutorial notebooks.

To execute unit tests from a manual install, run:
```bash
# running a single unit test
python -m unittest -v tests.attr.test_saliency
# running all unit tests
pytest -ra
```

## Getting Started
Captum helps you interpret and understand predictions of PyTorch models by
exploring features that contribute to a prediction the model makes.
It also helps understand which neurons and layers are important for
model predictions.

Currently, the library uses gradient-based interpretability algorithms
and attributes contributions to each input of the model with respect to
different neurons and layers, both intermediate and final.

Let's apply some of those algorithms to a toy model we have created for
demonstration purposes.
For simplicity, we will use the following architecture, but users are welcome
to use any PyTorch model of their choice.


```
import numpy as np

import torch
import torch.nn as nn

from captum.attr import (
    GradientShap,
    DeepLift,
    DeepLiftShap,
    IntegratedGradients,
    LayerConductance,
    NeuronConductance,
    NoiseTunnel,
)

class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.lin = nn.Linear(3, 2, bias=True)
        self.sigmoid = nn.Sigmoid()

        # initialize weights and biases
        self.lin.weight = nn.Parameter(torch.arange(1.0, 7.0).view(2, 3))
        self.lin.bias = nn.Parameter(torch.zeros(2))

    def forward(self, input):
        return self.sigmoid(self.lin(self.relu(input)))
```

Let's create an instance of our model and set it to eval mode.
```
model = ToyModel()
model.eval()
```

Next, we need to define simple input and baseline tensors.
Baselines belong to the input space and often carry no predictive signal.
Zero tensor can serve as a baseline for many tasks.
Some interpretability algorithms such as Integrated
Gradients, Deeplift and GradientShap are designed to attribute the change
between the input and baseline to a predictive class or a value that the neural
network outputs.

We will apply model interpretability algorithms on the network
mentioned above in order to understand the importance of individual
neurons/layers and the parts of the input that play an important role in the
final prediction.

Let's fix random seeds to make computations deterministic.
```
torch.manual_seed(0)
np.random.seed(0)
```

Let's define our input and baseline tensors. Baselines are used in some
interpretability algorithms such as `IntegratedGradients, DeepLift,
GradientShap, NeuronConductance, LayerConductance, InternalInfluence and
NeuronIntegratedGradients`.

```
input = torch.rand(2, 3)
baseline = torch.zeros(2, 3)
```
Next we will use `IntegratedGradients` algorithms to assign attribution
scores to each input feature with respect to the second target output.
```
ig = IntegratedGradients(model)
attributions, delta = ig.attribute(input, baseline, target=1)
print('IG Attributions: ', attributions, ' Approximation error: ', delta)
```
Output:
```
IG Attributions:  tensor([[0.1556, 0.3011, 0.0416],
                          [0.0447, 0.1302, 0.3223]])
Approximation Error: 2.980232238769531e-07
```
The algorithm outputs an attribution score for each input element and an
approximation error that we would like to minimize. It can also serve as a proxy
of how much we can trust the attribution scores assigned by an attribution algorithm.
If the approximation error is large, we can try larger number of integral
approximation steps by setting `n_steps` to a larger value. Not all algorithms
return approximation error. Those which do, they compute it based on the
completeness property of the algorithms.

Positive attribution score means that the input in that particular position positively
contributed to the final prediction and negative means the opposite.
The magnitude of the attribution score signifies the strength of the contribution.
Zero attribution score means no contribution from that particular feature.

Similarly, we can apply `GradientShap`, `DeepLift` and other attribution algorithms to the model.

```
gs = GradientShap(model)

# We define a distribution of baselines and draw `n_samples` from that
# distribution in order to estimate the expectations of gradients across all baselines
baseline_dist = torch.randn(1000, 3) * 0.01
attributions, delta = gs.attribute(input, stdevs=0.001, n_samples=2000, baselines=baseline_dist, target=0)
print('GradientShap Attributions: ', attributions, ' Approximation error: ', delta)
```
Output
```
GradientShap Attributions:  tensor([[0.0878, 0.2730, 0.0447],
                                    [0.0209, 0.0989, 0.3109]])
Approximation Error: 0.00531

```

Below is an example of how we can apply `DeepLift` and `DeepLiftShap` on the
`ToyModel` described above. Current implementation of DeepLift supports only
`Rescale` rule.
```
dl = DeepLift(model)
attributions, delta = dl.attribute(input, baseline, target=0)
print('DeepLift Attributions: ', attributions, ' Approximation error: ', delta)
```
Output
```
DeepLift Attributions:  tensor([[0.0883, 0.2733, 0.0472],
                                [0.0216, 0.1007, 0.3116]])
        Approximation error:  0.0
```
DeepLift assigns similar attribution scores as Integrated Gradients to inputs,
however it has lower execution time. Another important thing to remember about
DeepLift is that it currently doesn't support all non-linear activation types.
For more details, about the limitation of current implementation, please,
read DeepLift's documentation.

Now let's look into `DeepLiftShap`. Similar to `GradientShap`, `DeepLiftShap` uses
baseline distribution. In the example below, we use the same baseline distribution
as for the `GradientShap`.

```
dl = DeepLiftShap(model)
attributions, delta = dl.attribute(input, baseline_dist, target=0)
print('DeepLiftSHAP Attributions: ', attributions, ' Approximation error: ', delta)
```
Output
```
DeepLift Attributions: tensor([0.0872, 0.2707, 0.0450],
                              [0.0209, 0.0989, 0.3082]], grad_fn=<MeanBackward1>)
Approximation error:  5.066394805908203e-07
```
`DeepLiftShap` uses `DeepLift` to compute attribution score for each
input-baseline pair and averages it for each input across all baselines.

In order to smooth and improve the quality of the attributions we can run
`IntegratedGradients` and other attribution methods through a `NoiseTunnel`.
`NoiseTunnel` allows to use SmoothGrad, SmoothGrad_Sq and VarGrad techniques
to smoothen the attributions by aggregating them for multiple noisy
samples that were generated by adding gaussian noise.

Here is an example how we can use `NoiseTunnel` with `IntegratedGradients`.

```
ig = IntegratedGradients(model)
nt = NoiseTunnel(ig)
attributions, delta = nt.attribute(input, nt_type='smoothgrad', stdevs=0.2, baselines=baseline, target=0)
print('IG + SmoothGrad Attributions: ', attributions, ' Approximation error: ', delta)
```
Output
```
IG + SmoothGrad Attributions:  tensor([[0.0717, 0.2328, 0.0789],
                                       [0.0338, 0.0532, 0.3317]])
Approximation Error:  4.470348358154297e-07

```

Let's look into the internals of our network and understand which layers
and neurons are important for the predictions.
We will start with the neuron conductance. Neuron conductance helps us to identify
input features that are important for a particular neuron in a given
layer. In this case, we choose to analyze the first neuron in the first layer.

```
nc = NeuronConductance(model, model.relu)
attributions = nc.attribute(input, neuron_index=0, target=0)
print('Neuron Attributions: ', attributions)
```
Output
```
Neuron Attributions:  tensor([[0.0000, 0.2854, 0.0000],
                              [0.0000, 0.1238, 0.0000]])
```

Layer conductance shows the importance of neurons for a layer and given input.
It doesn't attribute the contribution scores to the input features
but shows the importance of each neuron in selected layer.
```
lc = LayerConductance(model, model.relu)
attributions, delta = lc.attribute(input, baselines=baseline, target=0)
print('Layer Attributions: ', attributions, ' Approximation Error: ', delta)
```
Outputs
```
Layer Attributions: tensor([[0.0891, 0.2758, 0.0476],
                            [0.0219, 0.1019, 0.3152]], grad_fn=<SumBackward1>)
Approximation error: 0.008803457021713257
```

More details on the list of supported algorithms and how to apply
Captum on different types of models can be found in our tutorials.


## Captum Insights

Captum provides a web interface called Insights for easy visualization and
access to a number of our interpretability algorithms.

To analyze a sample model on CIFAR10 via Captum Insights run

```
python -m captum.insights.example
```

and navigate to the URL specified in the output.

To build Insights you will need [Node](https://nodejs.org/en/) >= 8.x
and [Yarn](https://yarnpkg.com/en/) >= 1.5.

To build and launch from a checkout in a conda environment run

```
conda install -c conda-forge yarn
BUILD_INSIGHTS=1 python setup.py develop
python captum/insights/example.py
```

## Contributing
See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.


## References

* [Axiomatic Attribution for Deep Networks, Mukund Sundararajan et al. 2017](https://arxiv.org/abs/1703.01365)
* [Did the Model Understand the Question? Pramod K. Mudrakarta, et al. 2018](https://www.aclweb.org/anthology/P18-1176)
* [Investigating the influence of noise and distractors on the interpretation of neural networks, Pieter-Jan Kindermans et al. 2016](https://arxiv.org/abs/1611.07270)
* [SmoothGrad: removing noise by adding noise, Daniel Smilkov et al. 2017](https://arxiv.org/abs/1706.03825)
* [Local Explanation Methods for Deep Neural Networks Lack Sensitivity to Parameter Values, Julius Adebayo et al. 2018](https://arxiv.org/abs/1810.03307)
* [Sanity Checks for Saliency Maps, Julius Adebayo et al. 2018](https://arxiv.org/abs/1810.03292)
* [How Important is a neuron?, Kedar Dhamdhere et al. 2018](https://arxiv.org/abs/1805.12233)
* [Learning Important Features Through Propagating Activation Differences, Avanti Shrikumar et al. 2017](https://arxiv.org/pdf/1704.02685.pdf)
* [Computationally Efficient Measures of Internal Neuron Importance, Avanti Shrikumar et al. 2018](https://arxiv.org/pdf/1807.09946.pdf)
* [A Unified Approach to Interpreting Model Predictions, Scott M. Lundberg et al. 2017](http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions)
* [Influence-Directed Explanations for Deep Convolutional Networks, Klas Leino et al. 2018](https://arxiv.org/pdf/1802.03788.pdf)
* [Towards better understanding of gradient-based attribution methods for deep neural networks, Marco Ancona et al. 2018](https://openreview.net/pdf?id=Sy21R9JAW)

## License
Captum is BSD licensed, as found in the [LICENSE](LICENSE) file.

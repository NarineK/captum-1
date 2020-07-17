#!/usr/bin/env python3
from enum import Enum
from typing import Any, Tuple, Union

import torch
from torch import Tensor

from captum.log import log_usage

from ..._utils.common import (
    _expand_and_update_additional_forward_args,
    _expand_and_update_baselines,
    _expand_and_update_target,
    _format_input,
    _format_output,
    _format_tensor_into_tuples,
    _is_tuple,
)
from .._utils.attribution import Attribution, GradientAttribution
from .._utils.common import _validate_noise_tunnel_type


class NoiseTunnelType(Enum):
    smoothgrad = 1
    smoothgrad_sq = 2
    vargrad = 3


SUPPORTED_NOISE_TUNNEL_TYPES = list(NoiseTunnelType.__members__.keys())


class NoiseTunnel(Attribution):
    r"""
    Adds gaussian noise to each input in the batch `n_samples` times
    and applies the given attribution algorithm to each of the samples.
    The attributions of the samples are combined based on the given noise
    tunnel type (nt_type):
    If nt_type is `smoothgrad`, the mean of the sampled attributions is
    returned. This approximates smoothing the given attribution method
    with a Gaussian Kernel.
    If nt_type is `smoothgrad_sq`, the mean of the squared sample attributions
    is returned.
    If nt_type is `vargrad`, the variance of the sample attributions is
    returned.

    More details about adding noise can be found in the following papers:
        https://arxiv.org/abs/1810.03292
        https://arxiv.org/abs/1810.03307
        https://arxiv.org/abs/1706.03825
        https://arxiv.org/pdf/1806.10758
    This method currently also supports batches of multiple examples input,
    however it can be computationally expensive depending on the model,
    the dimensionality of the data and execution environment.
    It is assumed that the batch size is the first dimension of input tensors.
    """

    def __init__(self, attribution_method: Attribution) -> None:
        r"""
        Args:
            attribution_method (Attribution): An instance of any attribution algorithm
                        of type `Attribution`. E.g. Integrated Gradients,
                        Conductance or Saliency.
        """
        self.attribution_method = attribution_method
        self.is_delta_supported = self.attribution_method.has_convergence_delta()
        self._use_input_marginal_effects = (
            self.attribution_method.uses_input_marginal_effects
        )
        self.is_gradient_method = isinstance(
            self.attribution_method, GradientAttribution
        )
        Attribution.__init__(self, self.attribution_method.forward_func)

    @property
    def uses_input_marginal_effects(self):
        return self._use_input_marginal_effects

    @log_usage()
    def attribute(
        self,
        inputs: Union[Tensor, Tuple[Tensor, ...]],
        nt_type: str = "smoothgrad",
        n_samples: int = 5,
        stdevs: Union[float, Tuple[float, ...]] = 1.0,
        draw_baseline_from_distrib: bool = False,
        **kwargs: Any,
    ):
        r"""
        Args:

            inputs (tensor or tuple of tensors):  Input for which integrated
                        gradients are computed. If forward_func takes a single
                        tensor as input, a single input tensor should be provided.
                        If forward_func takes multiple tensors as input, a tuple
                        of the input tensors should be provided. It is assumed
                        that for all given input tensors, dimension 0 corresponds
                        to the number of examples, and if multiple input tensors
                        are provided, the examples must be aligned appropriately.
            nt_type (string, optional): Smoothing type of the attributions.
                        `smoothgrad`, `smoothgrad_sq` or `vargrad`
                        Default: `smoothgrad` if `type` is not provided.
            n_samples (int, optional):  The number of randomly generated examples
                        per sample in the input batch. Random examples are
                        generated by adding gaussian random noise to each sample.
                        Default: `5` if `n_samples` is not provided.
            stdevs    (float, or a tuple of floats optional): The standard deviation
                        of gaussian noise with zero mean that is added to each
                        input in the batch. If `stdevs` is a single float value
                        then that same value is used for all inputs. If it is
                        a tuple, then it must have the same length as the inputs
                        tuple. In this case, each stdev value in the stdevs tuple
                        corresponds to the input with the same index in the inputs
                        tuple.
                        Default: `1.0` if `stdevs` is not provided.
            draw_baseline_from_distrib (bool, optional): Indicates whether to
                        randomly draw baseline samples from the `baselines`
                        distribution provided as an input tensor.
                        Default: False
            **kwargs (Any, optional): Contains a list of arguments that are passed
                        to `attribution_method` attribution algorithm.
                        Any additional arguments that should be used for the
                        chosen attribution method should be included here.
                        For instance, such arguments include
                        `additional_forward_args` and `baselines`.

        Returns:
            **attributions** or 2-element tuple of **attributions**, **delta**:
            - **attributions** (*tensor* or tuple of *tensors*):
                        Attribution with
                        respect to each input feature. attributions will always be
                        the same size as the provided inputs, with each value
                        providing the attribution of the corresponding input index.
                        If a single tensor is provided as inputs, a single tensor is
                        returned. If a tuple is provided for inputs, a tuple of
                        corresponding sized tensors is returned.
            - **delta** (*float*, returned if return_convergence_delta=True):
                        Approximation error computed by the
                        attribution algorithm. Not all attribution algorithms
                        return delta value. It is computed only for some
                        algorithms, e.g. integrated gradients.
                        Delta is computed for each input in the batch
                        and represents the arithmetic mean
                        across all `n_sample` perturbed tensors for that input.


        Examples::

            >>> # ImageClassifier takes a single input tensor of images Nx3x32x32,
            >>> # and returns an Nx10 tensor of class probabilities.
            >>> net = ImageClassifier()
            >>> ig = IntegratedGradients(net)
            >>> input = torch.randn(2, 3, 32, 32, requires_grad=True)
            >>> # Creates noise tunnel
            >>> nt = NoiseTunnel(ig)
            >>> # Generates 10 perturbed input tensors per image.
            >>> # Computes integrated gradients for class 3 for each generated
            >>> # input and averages attributions accros all 10
            >>> # perturbed inputs per image
            >>> attribution = nt.attribute(input, nt_type='smoothgrad',
            >>>                            n_samples=10, target=3)
        """

        def add_noise_to_inputs() -> Tuple[Tensor, ...]:
            if isinstance(stdevs, tuple):
                assert len(stdevs) == len(inputs), (
                    "The number of input tensors "
                    "in {} must be equal to the number of stdevs values {}".format(
                        len(inputs), len(stdevs)
                    )
                )
            else:
                assert isinstance(
                    stdevs, float
                ), "stdevs must be type float. " "Given: {}".format(type(stdevs))
                stdevs_ = (stdevs,) * len(inputs)
            return tuple(
                add_noise_to_input(input, stdev).requires_grad_()
                if self.is_gradient_method
                else add_noise_to_input(input, stdev)
                for (input, stdev) in zip(inputs, stdevs_)
            )

        def add_noise_to_input(input: Tensor, stdev: float) -> Tensor:
            # batch size
            bsz = input.shape[0]

            # expand input size by the number of drawn samples
            input_expanded_size = (bsz * n_samples,) + input.shape[1:]

            # expand stdev for the shape of the input and number of drawn samples
            stdev_expanded = torch.tensor(stdev, device=input.device).repeat(
                input_expanded_size
            )

            # draws `np.prod(input_expanded_size)` samples from normal distribution
            # with given input parametrization
            # FIXME it look like it is very difficult to make torch.normal
            # deterministic this needs an investigation
            noise = torch.normal(0, stdev_expanded)
            return input.repeat_interleave(n_samples, dim=0) + noise

        def compute_expected_attribution_and_sq(attribution):
            bsz = attribution.shape[0] // n_samples
            attribution_shape = (bsz, n_samples)
            if len(attribution.shape) > 1:
                attribution_shape += attribution.shape[1:]

            attribution = attribution.view(attribution_shape)
            expected_attribution = attribution.mean(dim=1, keepdim=False)
            expected_attribution_sq = torch.mean(attribution ** 2, dim=1, keepdim=False)
            return expected_attribution, expected_attribution_sq

        with torch.no_grad():
            # Keeps track whether original input is a tuple or not before
            # converting it into a tuple.
            is_inputs_tuple = isinstance(inputs, tuple)

            inputs = _format_input(inputs)

            _validate_noise_tunnel_type(nt_type, SUPPORTED_NOISE_TUNNEL_TYPES)

            delta = None
            inputs_with_noise = add_noise_to_inputs()
            # if the algorithm supports targets, baselines and/or
            # additional_forward_args they will be expanded based
            # on the n_steps and corresponding kwargs
            # variables will be updated accordingly
            _expand_and_update_additional_forward_args(n_samples, kwargs)
            _expand_and_update_target(n_samples, kwargs)
            _expand_and_update_baselines(
                inputs,
                n_samples,
                kwargs,
                draw_baseline_from_distrib=draw_baseline_from_distrib,
            )

            # smoothgrad_Attr(x) = 1 / n * sum(Attr(x + N(0, sigma^2))
            # NOTE: using __wrapped__ such that it does not log the inner logs
            attr_func = self.attribution_method.attribute
            attributions = attr_func.__wrapped__(  # type: ignore
                self.attribution_method,  # self
                inputs_with_noise if is_inputs_tuple else inputs_with_noise[0],
                **kwargs,
            )

            return_convergence_delta = (
                "return_convergence_delta" in kwargs
                and kwargs["return_convergence_delta"]
            )

            if self.is_delta_supported and return_convergence_delta:
                attributions, delta = attributions

            is_attrib_tuple = _is_tuple(attributions)
            attributions = _format_tensor_into_tuples(attributions)

            expected_attributions = []
            expected_attributions_sq = []
            for attribution in attributions:
                expected_attr, expected_attr_sq = compute_expected_attribution_and_sq(
                    attribution
                )
                expected_attributions.append(expected_attr)
                expected_attributions_sq.append(expected_attr_sq)

            if NoiseTunnelType[nt_type] == NoiseTunnelType.smoothgrad:
                return self._apply_checks_and_return_attributions(
                    tuple(expected_attributions),
                    is_attrib_tuple,
                    return_convergence_delta,
                    delta,
                )

            if NoiseTunnelType[nt_type] == NoiseTunnelType.smoothgrad_sq:
                return self._apply_checks_and_return_attributions(
                    tuple(expected_attributions_sq),
                    is_attrib_tuple,
                    return_convergence_delta,
                    delta,
                )

            vargrad = tuple(
                expected_attribution_sq - expected_attribution * expected_attribution
                for expected_attribution, expected_attribution_sq in zip(
                    expected_attributions, expected_attributions_sq
                )
            )

            return self._apply_checks_and_return_attributions(
                vargrad, is_attrib_tuple, return_convergence_delta, delta
            )

    def _apply_checks_and_return_attributions(
        self,
        attributions: Tuple[Tensor, ...],
        is_attrib_tuple: bool,
        return_convergence_delta: bool,
        delta: Union[None, Tensor],
    ):
        attributions = _format_output(is_attrib_tuple, attributions)

        return (
            (attributions, delta)
            if self.is_delta_supported and return_convergence_delta
            else attributions
        )

    def has_convergence_delta(self) -> bool:
        return self.is_delta_supported

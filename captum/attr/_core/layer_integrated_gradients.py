#!/usr/bin/env python3
import torch

from captum.attr._utils.approximation_methods import approximation_parameters
from captum.attr._utils.batching import _batched_operator
from captum.attr._utils.common import (
    _validate_input,
    _format_additional_forward_args,
    _format_attributions,
    _format_input_baseline,
    _reshape_and_sum,
    _tensorize_baseline,
    _expand_additional_forward_args,
    _expand_target,
)

from captum.attr._utils.attribution import LayerAttribution, GradientAttribution
from captum.attr._utils.gradient import _run_forward, _forward_layer_distributed_eval, _gather_distributed_tensors, _extract_device_ids

class LayerIntegratedGradients(LayerAttribution, GradientAttribution):
    def __init__(self, forward_func, layer, device_ids=None):
        r"""
        Args:

            forward_func (callable):  The forward function of the model or any
                          modification of it
        """
        LayerAttribution.__init__(self, forward_func, layer, device_ids=device_ids)
        GradientAttribution.__init__(self, forward_func)

    def attribute(
        self,
        inputs,
        baselines=None,
        target=None,
        additional_forward_args=None,
        n_steps=50,
        method="gausslegendre",
        internal_batch_size=None,
        return_convergence_delta=False,
        attribute_to_layer_input=False, # TODO remove this?
    ):
        # Keeps track whether original input is a tuple or not before
        # converting it into a tuple.
        is_inputs_tuple = isinstance(inputs, tuple)

        inputs, baselines = _format_input_baseline(inputs, baselines)
        _validate_input(inputs, baselines, n_steps, method)
        baselines = _tensorize_baseline(inputs, baselines)

        additional_forward_args = _format_additional_forward_args(
            additional_forward_args
        )

        expanded_inputs = tuple(torch.cat([input] * n_steps, dim=0) for input in inputs)
        expanded_baselines = tuple(torch.cat([baseline] * n_steps, dim=0) for baseline in baselines)

        expanded_additional_args = ( \
            _expand_additional_forward_args(additional_forward_args, n_steps)
            if additional_forward_args is not None
            else None
        )

        device_ids = getattr(self.forward_func, 'device_ids', None)

        expanded_target = _expand_target(target, n_steps)

        # retrieve step size and scaling factor for specified approximation method
        step_sizes_func, alphas_func = approximation_parameters(method)
        step_sizes, alphas = step_sizes_func(n_steps), alphas_func(n_steps)

        num_examples = len(inputs[0])
        expanded_alphas = torch.tensor(alphas).repeat_interleave(num_examples)[:, None]

        if device_ids is None:
            scattered_expanded_alphas = (expanded_alphas.to(inputs[0].device), )
        else:
            scattered_expanded_alphas = scatter(expanded_alphas, target_gpus=device_ids)

        scattered_expanded_alphas_dict = {scattered_expanded_alpha.device: scattered_expanded_alpha for scattered_expanded_alpha in scattered_expanded_alphas}

        inputs_layer_dict = _forward_layer_distributed_eval(self.forward_func, expanded_inputs, self.layer,
                            additional_forward_args=expanded_additional_args,
                            attribute_to_layer_input=attribute_to_layer_input)

        baselines_layer_dict = _forward_layer_distributed_eval(self.forward_func, expanded_baselines, self.layer,
                            additional_forward_args=expanded_additional_args,
                            attribute_to_layer_input=attribute_to_layer_input)

        # scale features and compute gradients. (batch size is abbreviated as bsz)
        # scaled_features' dim -> (bsz * #steps x inputs[0].shape[1:], ...)
        scaled_features_dict = {input_device:
            baselines_layer_dict[input_device] + (scattered_expanded_alphas_dict[input_device] * (inputs_layer_dict[input_device] - \
            baselines_layer_dict[input_device]).view(inputs_layer_dict[input_device].shape[0], -1)).view(inputs_layer_dict[input_device].shape)
        for input_device in inputs_layer_dict.keys()}

        # apply number of steps to additional forward args
        # currently, number of steps is applied only to additional forward arguments
        # that are nd-tensors. It is assumed that the first dimension is
        # the number of batches.
        # dim -> (bsz * #steps x additional_forward_args[0].shape[1:], ...)

        def layer_forward_hook(module, inputs, outputs):
            outputs = scaled_features_dict[outputs.device] #scaled_features_tpl[0].clone()
            return outputs

        hook = self.layer.register_forward_hook(layer_forward_hook)
        #expand inputs before passing
        output = _run_forward(self.forward_func, expanded_inputs, expanded_target, expanded_additional_args)
        hook.remove()

        if device_ids is None:
            scattered_ouputs = (output, )
        else:
            scattered_ouputs = scatter(output, target_gpus=device_ids)

        # grads: dim -> (bsz * #steps x inputs[0].shape[1:], ...)
        grads_dict = {scattered_ouput.device: self.gradient_func(
            self.forward_func,
            scaled_features_dict[scattered_ouput.device],
            output=scattered_ouput
        )[0] for scattered_ouput in scattered_ouputs}

        grads = _gather_distributed_tensors(grads_dict, device_ids=device_ids)
        grads = (grads, )
        # flattening grads so that we can multilpy it with step-size
        # calling contiguous to avoid `memory whole` problems
        scaled_grads = [
            grad.contiguous().view(n_steps, -1)
            * torch.tensor(step_sizes).view(n_steps, 1).to(grad.device)
            for grad in grads
        ]

        # aggregates across all steps for each tensor in the input tuple
        # total_grads has the same dimensionality as inputs
        total_grads = [
            _reshape_and_sum(
                scaled_grad, n_steps, grad.shape[0] // n_steps, grad.shape[1:]
            )
            for (scaled_grad, grad) in zip(scaled_grads, grads)
        ]

        inputs_layer = _gather_distributed_tensors(inputs_layer_dict, device_ids=device_ids)
        inputs_layer = (inputs_layer, )
        baselines_layer = _gather_distributed_tensors(baselines_layer_dict, device_ids=device_ids)
        baselines_layer = (baselines_layer, )

        # computes attribution for each tensor in input tuple
        # attributions has the same dimensionality as inputs
        attributions = tuple(
            total_grad * (input[:num_examples] - baseline[:num_examples])
            for total_grad, input, baseline in zip(total_grads, inputs_layer, baselines_layer)
        )
        if return_convergence_delta:
            start_point, end_point = baselines, inputs
            # computes approximation error based on the completeness axiom
            delta = self.compute_convergence_delta(
                attributions,
                start_point,
                end_point,
                additional_forward_args=additional_forward_args,
                target=target,
            )
            return _format_attributions(is_inputs_tuple, attributions), delta
        return _format_attributions(is_inputs_tuple, attributions)

    def has_convergence_delta(self):
        return True

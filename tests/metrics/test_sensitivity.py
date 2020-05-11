#!/usr/bin/env python3

import torch

from captum.attr import DeepLift, GuidedBackprop, IntegratedGradients, Saliency
from captum.metrics import sensitivity_max
from captum.metrics._core.sensitivity import default_perturb_func

from ..helpers.basic import BaseTest, assertArraysAlmostEqual, assertTensorAlmostEqual
from ..helpers.basic_models import (
    BasicModel2,
    BasicModel4_MultiArgs,
    BasicModel_ConvNet_One_Conv,
    BasicModel_MultiLayer,
)


def _perturb_func(inputs):
    def perturb_ratio(input):
        return (
            torch.arange(-torch.numel(input[0]) // 2, torch.numel(input[0]) // 2)
            .view(input[0].shape)
            .float()
            / 100
        )

    if isinstance(inputs, tuple):
        input1 = inputs[0]
        input2 = inputs[1]
    else:
        input1 = inputs
        input2 = None

    perturbed_input1 = input1 + perturb_ratio(input1)

    if input2 is None:
        return perturbed_input1

    return perturbed_input1, input2 + perturb_ratio(input2)


class Test(BaseTest):
    def test_basic_sensitivity_max_single(self):
        model = BasicModel2()
        sa = Saliency(model)

        input1 = torch.tensor([3.0])
        input2 = torch.tensor([1.0])
        self.sensitivity_max_assert(
            sa.attribute, (input1, input2), [0.0], perturb_func=default_perturb_func
        )

    def test_basic_sensitivity_max_multiple(self):
        model = BasicModel2()
        sa = Saliency(model)

        input1 = torch.tensor([3.0] * 20)
        input2 = torch.tensor([1.0] * 20)
        self.sensitivity_max_assert(
            sa.attribute, (input1, input2), [0.0] * 20, max_examples_per_batch=21
        )
        self.sensitivity_max_assert(
            sa.attribute, (input1, input2), [0.0] * 20, max_examples_per_batch=60
        )

    def test_convnet_multi_target(self):
        r"""
        Another test with Saliency, local sensitivity and more
        complex model with higher dimensional input.
        """
        model = BasicModel_ConvNet_One_Conv()
        sa = Saliency(model)

        input = torch.stack([torch.arange(1, 17).float()] * 20, dim=0).view(20, 1, 4, 4)

        self.sensitivity_max_assert(
            sa.attribute,
            input,
            [0.0] * 20,
            target=torch.tensor([1] * 20),
            n_perturb_samples=10,
            max_examples_per_batch=40,
        )

    def test_convnet_multi_target_and_default_pert_func(self):
        r"""
        Similar to previous example but here we also test default
        perturbation function.
        """
        model = BasicModel_ConvNet_One_Conv()
        gbp = GuidedBackprop(model)

        input = torch.stack([torch.arange(1, 17).float()] * 20, dim=0).view(20, 1, 4, 4)

        sens1 = self.sensitivity_max_assert(
            gbp.attribute,
            input,
            [0.0] * 20,
            perturb_func=default_perturb_func,
            target=torch.tensor([1] * 20),
            n_perturb_samples=10,
            max_examples_per_batch=40,
        )

        sens2 = self.sensitivity_max_assert(
            gbp.attribute,
            input,
            [0.0] * 20,
            perturb_func=default_perturb_func,
            target=torch.tensor([1] * 20),
            n_perturb_samples=10,
            max_examples_per_batch=5,
        )
        assertTensorAlmostEqual(self, sens1, sens2)

    def test_sensitivity_max_multi_dim(self):
        model = BasicModel_MultiLayer()

        input = torch.arange(1.0, 13.0).view(4, 3)

        additional_forward_args = (None, True)
        targets = [(0, 1, 1), (0, 1, 1), (1, 1, 1), (0, 1, 1)]

        ig = IntegratedGradients(model)
        sens = self.sensitivity_max_assert(
            ig.attribute,
            input,
            [0.006, 0.01, 0.001, 0.008],
            n_perturb_samples=1,
            max_examples_per_batch=4,
            perturb_func=_perturb_func,
            target=targets,
            additional_forward_args=additional_forward_args,
        )

    def test_sensitivity_max_multi_dim_batching(self):
        model = BasicModel_MultiLayer()

        input = torch.arange(1.0, 16.0).view(5, 3)

        additional_forward_args = (torch.ones(5, 3).float(), False)
        targets = [0, 0, 0, 0, 0]

        sa = Saliency(model)

        sensitivity1 = self.sensitivity_max_assert(
            sa.attribute,
            input,
            [0.0] * 5,
            n_perturb_samples=1,
            max_examples_per_batch=None,
            perturb_func=_perturb_func,
            target=targets,
            additional_forward_args=additional_forward_args,
        )
        sensitivity2 = self.sensitivity_max_assert(
            sa.attribute,
            input,
            [0.0] * 5,
            n_perturb_samples=10,
            max_examples_per_batch=10,
            perturb_func=_perturb_func,
            target=targets,
            additional_forward_args=additional_forward_args,
        )
        assertTensorAlmostEqual(self, sensitivity1, sensitivity2, 0.0)

    def test_sensitivity_additional_forward_args_multi_args(self):
        model = BasicModel4_MultiArgs()

        input1 = torch.tensor([[1.5, 2.0, 3.3]])
        input2 = torch.tensor([[3.0, 3.5, 2.2]])

        args = torch.tensor([[1.0, 3.0, 4.0]])
        ig = DeepLift(model)

        sensitivity1 = self.sensitivity_max_assert(
            ig.attribute,
            (input1, input2),
            [0.0],
            additional_forward_args=args,
            n_perturb_samples=1,
            max_examples_per_batch=1,
            perturb_func=_perturb_func,
        )

        sensitivity2 = self.sensitivity_max_assert(
            ig.attribute,
            (input1, input2),
            [0.0],
            additional_forward_args=args,
            n_perturb_samples=4,
            max_examples_per_batch=2,
            perturb_func=_perturb_func,
        )
        assertTensorAlmostEqual(self, sensitivity1, sensitivity2, 0.0)

    def test_classification_sensitivity_tpl_target_w_baseline(self):
        model = BasicModel_MultiLayer()
        input = torch.arange(1.0, 13.0).view(4, 3)
        baseline = torch.ones(4, 3)
        additional_forward_args = (torch.arange(1, 13).view(4, 3).float(), True)
        targets = [(0, 1, 1), (0, 1, 1), (1, 1, 1), (0, 1, 1)]
        dl = DeepLift(model)

        sens1 = self.sensitivity_max_assert(
            dl.attribute,
            input,
            [0.01, 0.003, 0.001, 0.001],
            additional_forward_args=additional_forward_args,
            baselines=baseline,
            target=targets,
            n_perturb_samples=10,
            perturb_func=_perturb_func,
        )
        sens2 = self.sensitivity_max_assert(
            dl.attribute,
            input,
            [0.0, 0.0, 0.0, 0.0],
            additional_forward_args=additional_forward_args,
            baselines=baseline,
            target=targets,
            n_perturb_samples=10,
            perturb_func=_perturb_func,
            max_examples_per_batch=30,
        )
        assertTensorAlmostEqual(self, sens1, sens2)

    def sensitivity_max_assert(
        self,
        expl_func,
        inputs,
        expected_sensitivity,
        perturb_func=_perturb_func,
        n_perturb_samples=5,
        max_examples_per_batch=None,
        baselines=None,
        target=None,
        additional_forward_args=None,
    ):
        if baselines is None:
            sens = sensitivity_max(
                expl_func,
                inputs,
                perturb_func=perturb_func,
                target=target,
                additional_forward_args=additional_forward_args,
                n_perturb_samples=n_perturb_samples,
                max_examples_per_batch=max_examples_per_batch,
            )
        else:
            sens = sensitivity_max(
                expl_func,
                inputs,
                perturb_func=perturb_func,
                baselines=baselines,
                target=target,
                additional_forward_args=additional_forward_args,
                n_perturb_samples=n_perturb_samples,
                max_examples_per_batch=max_examples_per_batch,
            )
        assertArraysAlmostEqual(sens, expected_sensitivity)
        return sens

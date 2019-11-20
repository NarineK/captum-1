#!/usr/bin/env python3
import torch

from ..helpers.utils import BaseTest

from ..helpers.utils import assertTensorAlmostEqual
from ..helpers.classification_models import SoftmaxModel
from ..helpers.basic_models import BasicModel_MultiLayer

from captum.attr._core.gradient_shap import GradientShap
from captum.attr._core.layer.layer_gradient_shap import LayerGradientShap

from ..test_gradient_shap import _assert_attribution_delta


class Test(BaseTest):
    def test_basic_multilayer(self):
        model = BasicModel_MultiLayer(inplace=True)
        model.eval()

        inputs = torch.tensor([[1.0, -20.0, 10.0]])
        baselines = torch.randn(30, 3)
        expected = [[-2.147, 0.0]]

        self._assert_attributions(
            model, model.linear2, (inputs,), (baselines,), 0, expected
        )

    def test_basic_multilayer_compare_w_inp_features(self):
        model = BasicModel_MultiLayer()
        model.eval()

        inputs = torch.tensor([[10.0, 20.0, 10.0]])
        baselines = torch.randn(30, 3)

        gs = GradientShap(model)
        expected, delta = gs.attribute(
            inputs, baselines, target=0, return_convergence_delta=True
        )
        self.setUp()
        self._assert_attributions(
            model,
            model.linear0,
            (inputs,),
            (baselines,),
            0,
            expected,
            expected_delta=delta,
            attribute_to_layer_input=True,
        )

    def test_classification(self):
        def custom_baseline_fn(inputs):
            num_in = inputs[0].shape[1]
            return (torch.arange(0.0, num_in * 4.0).reshape(4, num_in),)

        num_in = 40
        n_samples = 10

        # 10-class classification model
        model = SoftmaxModel(num_in, 20, 10)
        model.eval()

        inputs = torch.arange(0.0, num_in * 2.0).reshape(2, num_in)
        baselines = custom_baseline_fn
        expected = torch.zeros(2, 20)

        self._assert_attributions(
            model, model.relu1, (inputs,), baselines, 1, expected, n_samples=n_samples
        )

    def _assert_attributions(
        self,
        model,
        layer,
        inputs,
        baselines,
        target,
        expected,
        expected_delta=None,
        n_samples=5,
        attribute_to_layer_input=False,
    ):
        lgs = LayerGradientShap(model, layer)
        attrs, delta = lgs.attribute(
            inputs,
            baselines=baselines,
            target=target,
            n_samples=n_samples,
            stdevs=0.0009,
            return_convergence_delta=True,
            attribute_to_layer_input=attribute_to_layer_input,
        )
        assertTensorAlmostEqual(self, attrs[0], expected, 0.005)
        if expected_delta is None:
            _assert_attribution_delta(self, inputs, attrs, n_samples, delta, True)
        else:
            for delta_i, expected_delta_i in zip(delta, expected_delta):
                self.assertAlmostEqual(delta_i, expected_delta_i, delta=0.01)

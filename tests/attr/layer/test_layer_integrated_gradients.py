#!/usr/bin/env python3

import torch

from ..helpers.utils import assertArraysAlmostEqual

from captum.attr._models.base import (
    configure_interpretable_embedding_layer,
    remove_interpretable_embedding_layer,
)

from captum.attr._core.integrated_gradients import IntegratedGradients
from captum.attr._core.layer.layer_integrated_gradients import LayerIntegratedGradients
from captum.attr._core.layer.layer_conductance import LayerConductance

from ..helpers.utils import BaseTest
from ..helpers.basic_models import (
    BasicEmbeddingModel,
    BasicModel_MultiLayer,
)


class Test(BaseTest):
    def test_compare_with_emb_patching(self):
        input1 = torch.tensor([[2, 5, 0, 1]])
        baseline1 = torch.tensor([[0, 0, 0, 0]])
        # these ones will be use as an additional forward args
        input2 = torch.tensor([[0, 2, 4, 1]])
        input3 = torch.tensor([[2, 3, 0, 1]])

        self._assert_compare_with_emb_patching(
            input1, baseline1, additional_args=(input2, input3)
        )

    def test_compare_with_emb_patching_batch(self):
        input1 = torch.tensor([[2, 5, 0, 1], [3, 1, 1, 0]])
        baseline1 = torch.tensor([[0, 0, 0, 0]])
        # these ones will be use as an additional forward args
        input2 = torch.tensor([[0, 2, 4, 1], [2, 3, 5, 7]])
        input3 = torch.tensor([[3, 5, 6, 7], [2, 3, 0, 1]])

        self._assert_compare_with_emb_patching(
            input1, baseline1, additional_args=(input2, input3)
        )

    def test_compare_with_layer_conductance(self):
        model = BasicModel_MultiLayer()
        lc = LayerConductance(model, model.linear0)
        # when we use input=torch.tensor()[[50.0, 50.0, 50.0]]),
        # F(x) - F(x - 1) is equal to 1
        # therefore layer conductance is nearly the same as layer integrated gradients
        # for large number of steps.
        input = torch.tensor([[50.0, 50.0, 50.0]], requires_grad=True)
        attribution, delta = lc.attribute(
            input, target=0, n_steps=1500, return_convergence_delta=True,
        )
        lig = LayerIntegratedGradients(model, model.linear0)
        attributions2, delta2 = lig.attribute(
            input, target=0, n_steps=1500, return_convergence_delta=True,
        )
        assertArraysAlmostEqual(attribution, attributions2, 0.01)
        assertArraysAlmostEqual(delta, delta2, 0.05)

    def _assert_compare_with_emb_patching(self, input, baseline, additional_args):
        model = BasicEmbeddingModel(nested_second_embedding=True)
        lig = LayerIntegratedGradients(model, model.embedding1)

        attributions, delta = lig.attribute(
            input,
            baselines=baseline,
            additional_forward_args=additional_args,
            return_convergence_delta=True,
        )

        # now let's interpret with standard integrated gradients and
        # the embeddings for monkey patching
        interpretable_embedding = configure_interpretable_embedding_layer(
            model, "embedding1"
        )
        input_emb = interpretable_embedding.indices_to_embeddings(input)
        baseline_emb = interpretable_embedding.indices_to_embeddings(baseline)
        ig = IntegratedGradients(model)
        attributions_with_ig, delta_with_ig = ig.attribute(
            input_emb,
            baselines=baseline_emb,
            additional_forward_args=additional_args,
            target=0,
            return_convergence_delta=True,
        )
        remove_interpretable_embedding_layer(model, interpretable_embedding)

        assertArraysAlmostEqual(attributions, attributions_with_ig)
        assertArraysAlmostEqual(delta, delta_with_ig)

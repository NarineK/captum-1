import sys

import torch
import torch.optim as optim


class DeepDream:
    def __init__(self, forward_func):
        self.forward_func = forward_func

    # TODO move some of the args into kwargs and support additional forward args
    def dream(
        self,
        inputs,
        optimizer=optim.SGD,
        optimizer_params={"lr": 12, "weight_decay": 1e-4},
        objective=torch.mean,
        num_iter=500,
        target=None,
        layer=None,
        transform=None,
        return_generated_inputs=True,
        verbose=True,
        logger=sys.stdout,
        lag=100,
    ):

        optimizer = self._create_optimizer(optimizer, inputs, optimizer_params)

        if return_generated_inputs:
            generated_inputs = []

        layer_out = None

        def forward_hook(module, inputs, outputs):
            nonlocal layer_out
            # In a general case, it is good to clone the tensor
            # in case inplace operations on the tensor take place
            layer_out = outputs.clone()

        # if layer is not provided, forward hook will not be used and
        # the output of the model will be used instead
        if layer is not None:
            handle = layer.register_forward_hook(forward_hook)

        for i in range(num_iter):
            optimizer.zero_grad()

            input_transformed = inputs
            if transform is not None:
                input_transformed = transform(inputs)

            output = self.forward_func(input_transformed)

            if layer_out is not None:
                output = layer_out

            if target is not None:
                # TODO replace this with `_select_targets` function
                output = layer_out[target]

            loss = -1 * objective(output)

            # compute gradient of the loss with respect to data points
            loss.backward()

            # update data points
            # instead of using optimizer.step we could, alternatively, update
            # data here using the learning rate but using optimizers instead
            # since they can offer more flexibility
            optimizer.step()

            if i % lag == 0:
                if verbose:
                    logger.write("Loss: {}\n".format(-loss.cpu().detach()))
                if return_generated_inputs is not None:
                    generated_inputs.append(input_transformed.clone().detach())

        if layer is not None:
            handle.remove()

        if return_generated_inputs:
            return generated_inputs
        return input_transformed.clone().detach()

    def _create_optimizer(self, optimizer, input, optimizer_params=None):
        return optimizer([input], **optimizer_params)

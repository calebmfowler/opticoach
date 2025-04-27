from keras.src.layers import Layer
from keras.src.saving import register_keras_serializable

@register_keras_serializable()
class Slicer(Layer):
    def __init__(self, num_steps, **kwargs):
        super(Slicer, self).__init__(**kwargs)
        self.num_steps = num_steps

    def call(self, inputs, mask=None):
        # Slice the inputs to keep only the last `num_steps` time steps
        sliced_inputs = inputs[:, -self.num_steps:, :]
        if mask is not None:
            # Slice the mask to match the sliced inputs
            sliced_mask = mask[:, -self.num_steps:]
            self._last_mask = sliced_mask  # Store the mask for compute_mask
        else:
            self._last_mask = None
        return sliced_inputs

    def compute_mask(self, inputs, mask=None):
        # Return the sliced mask if it exists
        if mask is not None:
            return self._last_mask
        return None

    def get_config(self):
        config = super(Slicer, self).get_config()
        config.update({"num_steps": self.num_steps})
        return config

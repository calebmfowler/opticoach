from tensorflow import expand_dims
from keras.src.layers import Layer
from keras.src.saving import register_keras_serializable

@register_keras_serializable()
class Expander(Layer):
    def __init__(self, **kwargs):
        super(Expander, self).__init__(**kwargs)

    def call(self, inputs, mask=None):
        self._mask = mask
        return expand_dims(inputs, axis=1)
        self._last_mask = mask

    def compute_mask(self, inputs, mask=None):
        if mask is not None:
            return self._mask
        return None

from keras.src.layers import Layer
from keras.src.saving import register_keras_serializable

@register_keras_serializable()
class Slicer(Layer):
    def __init__(self, num_steps, **kwargs):
        super(Slicer, self).__init__(**kwargs)
        self.num_steps = num_steps

    def call(self, inputs):
        return inputs[:, -self.num_steps:, :]

    def get_config(self):
        config = super(Slicer, self).get_config()
        config.update({"num_steps": self.num_steps})
        return config

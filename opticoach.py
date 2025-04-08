from aggregate import Aggregator
from preprocess import Preprocessor
from opticoachmodel import OpticoachModel
from postprocess import Postprocessor

aggregator = Aggregator()
aggregator.aggregate()
preprocessor = Preprocessor(aggregator)
preprocessor.preprocess()
model = OpticoachModel(preprocessor)
model.train()
model.predict()
postprocessor = Postprocessor(preprocessor, model)
postprocessor.postprocess()
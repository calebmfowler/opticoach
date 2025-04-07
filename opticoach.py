from aggregate import Aggregator
from preprocess import Preprocessor
from model import Model
from postprocess import Postprocessor

aggregator = Aggregator()
preprocessor = Preprocessor()
model = Model()
postprocessor = Postprocessor()

aggregator.aggregate()
preprocessor.preprocess(aggregator.aggregatedFiles)
model.train(preprocessor.preprocessedFiles)
model.predict(preprocessor.preprocessedFiles)
postprocessor.postprocess(model.predictedFiles)
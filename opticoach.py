from aggregate import Aggregator
from preprocess import Preprocessor
from opticoachmodel import OpticoachModel
from postprocess import Postprocessor

# See https://calebfowler.notion.site/opticoach for a class diagram and description of the program

aggregator = Aggregator()
# aggregator.aggregate()
preprocessor = Preprocessor(aggregator, 2000, 2024, 15, 5)
preprocessor.preprocess()
model = OpticoachModel(preprocessor)
model.train()
model.predict()
# postprocessor = Postprocessor(preprocessor, model)
# postprocessor.postprocess()
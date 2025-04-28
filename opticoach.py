from aggregate import Aggregator
from preprocess import Preprocessor
from opticoachmodel import OpticoachModel
from postprocess import Postprocessor

# See https://calebfowler.notion.site/opticoach for a class diagram and description of the program

aggregator = Aggregator()
# aggregator.aggregate()
preprocessor = Preprocessor(aggregator, 1985, 2020, 15, 3)
preprocessor.preprocess()
model = OpticoachModel(preprocessor)
model.build_and_train_directly()
postprocessor = Postprocessor(preprocessor, model)
postprocessor.postprocess()
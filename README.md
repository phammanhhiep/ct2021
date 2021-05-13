# CT2021
## Code Structure
- src/train.py, src/evaluate.py: entry points for training and testing respectively.
- src/search_hyperparameters.py: look for the best combination of hyperparameters.
- src/synthesize_result.py: compare experiments, and produce charts and tables of statistics. 
- src/trainer/XXX_trainer.py: implement a training algorithm, i.e. combine subnetworks, specify loss functions and optimizer.
- src/models/XXX_model.py: construct a model using networks defined in src/models/networks, including to define forward propagation, loss functions, and so on.
- src/options/XXX_option.py: collect options from users and provide default options to control the network.
- src/data/: contains scripts to load, transform, and save different datasets. 
- experiments/: contains trained models along with hyperparameters


## src/options/XXX_option.py
- It define a class that provide default options; and methods to collect user options, to save, and update options.  
- The options being passed arround within the network is the object return by the built-in method `parse_args` of class argparse.ArgumentParser.
- Structure of object option [REVIEW]
++ network
+++ each base_network
+++ each block
+++ each discriminator
+++ each encoder
+++ each generator
+++ each loss
+++ each normalization

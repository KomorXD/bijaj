# Training script for segmentation models of camouflaged animals (university project)

This project's goal is to train a model from [segmentation models for pytorch](https://github.com/qubvel/segmentation_models.pytorch) library to be able to find and recognize camouflaged animals in the environment they can camouflage in using different models, loss functions and other variables and then compare results for different configurations.  
  
The main.py file holds a few "config" variables (directories and parameters [model and loss function strings are for generated folder structures, models and loss functions themselves you have to edit directly in the code]).  
  
You should organize your datasets in this way:

    bijaj/
	    datasets/
	      train/
	      train_masks/
	      valid/
	      valid_masks/
	      test/
	      test_masks/
, as it is coded for that directory tree (or you can change that in the code). 

Mask image may consist of up to 5 colored layers:

 - Animal (blue) - animal itself
 - Masking background (green) - things that cover, camouflage the animal
 - Non-masking background (red) - things that are not important for the camouflage
 - Non-masking foreground attention (white) - things that are not camouflaging, but catch one's eye
 - None (black) - not looked at by model, it exists for "holes" in the masks image

Results (model predictions and training/validation score graphs) are saved in appropriate directories, in the output directory.

# MelanomaClassification
*PyTorch model for melanoma detection*

In this project, I decided to build a model able to detect if a mole is either risky or benign.

I build the model using PyTorch based on this [train/test images set](https://www.kaggle.com/drscarlat/melanoma) from the HAM10k.
 
 1. The model is defined, trained and tested in the notebook [model.ipynb](model.ipynb).
 2. The [load_model.py](load_model.py) file contains the functions to transform the input image and initialize the model.
 3. The model is built with its trained weights and bias and launched as a demo interface using [Gradio](https://www.gradio.app/) in the [main.py](main.py) file.

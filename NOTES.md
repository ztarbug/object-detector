# Setting up the EfficientDet project by Google 

## Load pre-trained model and export as Tensorflow SavedModel (all inside the efficientdet automl repo)
- Clone https://github.com/google/automl/tree/master/efficientdet
- Create and activate virtual Python environment if you like (e.g. `python3 -m venv .venv`, followed by `source .venv/bin/activate`)
- Install dependencies
  ```
  pip install numpy
  sudo apt install python3-dev
  pip install -r requirements.txt
  ```
- Download pre-trained model of desired size (refer to efficientdet repo above; use the "ckpt" link)
- Convert model into SavedModel
  `python model_inspect.py --runmode=saved_model --model_name=efficientdet-d0 --ckpt_path=efficientdet-d0 --saved_model_dir=savedmodeldir`
- The SavedModel is now in `./savedmodeldir`, you can move that directory into this repository to make things easier (this is what below instruction will assume)

## Set up the experimenting notebook in this repository
- Create and activate virtual Python environment if you like (e.g. `python3 -m venv .venv`, followed by `source .venv/bin/activate`)
- Install dependencies\
  `pip install -r requirements.txt`


# How-To progress
- Does our seemingly loaded model work or not?
- Look through the code of Google`s repository for how to load the model properly
- Try .h5 file? (which is apparently an old format to be used with Keras)
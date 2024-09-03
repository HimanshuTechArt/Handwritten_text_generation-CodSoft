# Handwritten text Generation

This repository presents a solution for handwritten text (HT) generation, using image processing techniques, in order to build custom HT datasets that are as realistic as possible. 

### Installation
This repository was run with:
- Python 3.7.3

Before installing the required packages, set up a virtual environment :
- using venv: python -m venv venv
- using conda: conda create venv

Then activate it:
- using venv: source venv/bin/activate
- using conda: conda activate venv

To install the required packages, run :
- pip install -r requirements.txt

### Procedure
To use this repo, the first is to download a character dataset with different handwriting styles, Chars74k was chosen.
Then the "Generate_Handwritten_text" notebook can be used to build datasets for various tasks (text segmentation, text recognition, ...). 
This notebook uses methods in "Generation" class to generate images from strings, by :
- Taking a handwritten character image from Char74k, for each character in the string.
- Processing character image (dilation, resizing, border removing, ....)
- Merging the processed character images to form a single image. 

Depending on the task type, the result would be multiple image sets (train, test, ...) with :
- COCO annotations for segmentation tasks
- Ground-truth text files for recognition tasks.

If another character dataset is to be used, it should follow the same structure as Chars74k.
Generation samples are provided in "Sample results" directory.

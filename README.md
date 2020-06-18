# Predict Age and Gender using Python
This module will help you **determine the gender and age** of people from the image. The predict method **returns a list** of faces of people who were found in the image with a possible age and gender of the person.

**Available for Python** 2.7, 3.4, 3.5, 3.6, 3.7, 3.8

![img](https://raw.githubusercontent.com/mowshon/age-and-gender/master/example/result.jpg)

© [Bill Gates family](https://www.businessinsider.com/microsoft-bill-melinda-gates-drive-daughter-to-school-2019-4)

# Instalation

```bash
git clone git@github.com:mowshon/age-and-gender.git
cd age-and-gender
python3 setup.py install --user
```

## Download the pre-trained models

We use already trained models. Thanks for the provided models from: https://github.com/davisking/dlib-models

**Author**: [Davis E. King](https://github.com/davisking)

1. **shape_predictor_5_face_landmarks.dat.bz2** [Download](https://github.com/davisking/dlib-models/raw/master/shape_predictor_5_face_landmarks.dat.bz2)

_This is a 5 point landmarking model which identifies the corners of the eyes and bottom of the nose. It is trained on the dlib 5-point face landmark dataset, which consists of 7198 faces. [@davisking](https://github.com/davisking) created this dataset by downloading images from the internet and annotating them with dlib's imglab tool._
    
2. **dnn_age_predictor_v1.dat.bz2** [Download](https://github.com/davisking/dlib-models/raw/master/age-predictor/dnn_age_predictor_v1.dat.bz2)
    
_The initial source for the model's creation came from the document of Z. Qawaqneh et al.: "Deep Convolutional Neural Network for Age Estimation based on VGG-Face Model". However, our research has led us to significant improvements in the CNN model, allowing us to estimate the age of a person outperforming the state-of-the-art results in terms of the exact accuracy and for 1-off accuracy._

_This model is thus an age predictor leveraging a ResNet-10 architecture and trained using a private dataset of about 110k different labelled images. During the training, we used an optimization and data augmentation pipeline and considered several sizes for the entry image._

_This age predictor model is provided for free by Cydral Technology and is licensed under the Creative Commons Zero v1.0 Universal._
    
3. **dnn_gender_classifier_v1.dat.bz2** [Download](https://github.com/davisking/dlib-models/raw/master/gender-classifier/dnn_gender_classifier_v1.dat.bz2)

_This model is a gender classifier trained using a private dataset of about 200k different face images and was generated according to the network definition and settings given in [Minimalistic CNN-based ensemble model for gender prediction from face images](http://www.eurecom.fr/fr/publication/4768/download/mm-publi-4768.pdf). Even if the dataset used for the training is different from that used by G. Antipov et al, the classification results on the LFW evaluation are similar overall (± 97.3%). To take up the authors' proposal to join the results of three networks, a simplification was made by finally presenting RGB images, thus simulating three "grayscale" networks via the three image planes. Better results could be probably obtained with a more complex and deeper network, but the performance of the classification is nevertheless surprising compared to the simplicity of the network used and thus its very small size._

_This gender model is provided for free by Cydral Technology and is licensed under the Creative Commons Zero v1.0 Universal._
    
4. Unpack the `*.bz2` archives, you need only the `.dat` file.

## Folder structure

```
test_example
-- shape_predictor_5_face_landmarks.dat
-- dnn_age_predictor_v1.dat
-- dnn_gender_classifier_v1.dat
-- test-image.jpg
-- example.py
```

# Example

```python
from age_and_gender import AgeAndGender
from PIL import Image

data.load_shape_predictor('shape_predictor_5_face_landmarks.dat')
data.load_dnn_gender_classifier('dnn_gender_classifier_v1.dat')
data.load_dnn_age_predictor('dnn_age_predictor_v1.dat')

image = Image.open('test-image.jpg').convert("RGB")
result = data.predict(image)

print(result)
```

Result:

```
[{'age': {'confidence': 85, 'value': 26},
  'face': [414, 265, 504, 355],
  'gender': {'confidence': 100, 'value': 'female'}},
 {'age': {'confidence': 58, 'value': 62},
  'face': [223, 199, 330, 307],
  'gender': {'confidence': 99, 'value': 'female'}},
 {'age': {'confidence': 73, 'value': 19},
  'face': [593, 128, 700, 235],
  'gender': {'confidence': 99, 'value': 'male'}},
 {'age': {'confidence': 50, 'value': 24},
  'face': [342, 534, 450, 641],
  'gender': {'confidence': 100, 'value': 'female'}},
 {'age': {'confidence': 92, 'value': 61},
  'face': [782, 116, 872, 206],
  'gender': {'confidence': 99, 'value': 'male'}}]
```

### Examples of determining the gender and age of people from the image
Code: https://github.com/mowshon/age-and-gender/tree/master/example

# How to increase efficiency with [face_recognition](https://github.com/ageitgey/face_recognition) ?

The module will try to determine where the faces of people are on the image. But, it is better for us to provide a variable with people's faces using the library [face_recognition](https://github.com/ageitgey/face_recognition) and method `face_locations()`.

```
python -m pip install numpy --user
python -m pip install face_recognition --user
```

Code:

```python
from age_and_gender import *
from PIL import Image
import face_recognition
import numpy


data = AgeAndGender()
data.load_shape_predictor('models/shape_predictor_5_face_landmarks.dat')
data.load_dnn_gender_classifier('models/dnn_gender_classifier_v1.dat')
data.load_dnn_age_predictor('models/dnn_age_predictor_v1.dat')

filename = 'test-image-2.jpg'

img = Image.open(filename).convert("RGB")
face_bounding_boxes = face_recognition.face_locations(
    numpy.asarray(img),  # Convert to numpy array
    model='hog'  # 'hog' for CPU | 'cnn' for GPU (NVIDIA with CUDA)
)

result = data.predict(img, face_bounding_boxes)
```

## Module `age-and-gender` without `face_recognition`

![img](https://raw.githubusercontent.com/mowshon/age-and-gender/master/example/result-2-default.jpg)

## Module `age-and-gender` with `face_recognition` and `face_bounding_boxes`

![img](https://raw.githubusercontent.com/mowshon/age-and-gender/master/example/result-2.jpg)

**Full example of code**: https://github.com/mowshon/age-and-gender/blob/master/example/example-with-face-recognition.py



# Changelog

**Version 1.0.1**
- The method `predict(pillow_img)` now require a PIL.Image object. Thanks to [@arrufat](https://github.com/arrufat) for the [piece of code](https://github.com/arrufat/wallyfinder/blob/2a3ddc1af2b676ad434574fecd9be0004c0fcc23/src/wallyfinder.cpp#L8-L42) that successfully performs the matrix conversion.
- The method `predict(pillow_img, face_bounding_boxes)` takes another argument `face_bounding_boxes` with a list of faces in the image. Check out this example. 
- If the method `predict(pillow_img)` does not get the second argument `face_bounding_boxes` with a list of faces, then the module will try to find the faces in the image itself.

**Version 1.0.0**
- Initial commit and code
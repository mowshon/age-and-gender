# Predict Age and Gender using Python
This module will help you **determine the gender and age** of people from the image. The predict method **returns a list** of faces of people who were found in the image with a possible age and gender of the person.

![img](https://raw.githubusercontent.com/mowshon/age-and-gender/master/example/result.jpg)

© [Bill Gates family](https://www.businessinsider.com/microsoft-bill-melinda-gates-drive-daughter-to-school-2019-4)

# Instalation

```bash
git clone git@github.com:mowshon/age-and-gender.git
cd age-and-gender
python3 setup.py install
```

Or

```
python3 -m pip install age-and-gender
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

data.load_shape_predictor('shape_predictor_5_face_landmarks.dat')
data.load_dnn_gender_classifier('dnn_gender_classifier_v1.dat')
data.load_dnn_age_predictor('dnn_age_predictor_v1.dat')

result = data.predict("test-image.jpg")

print(result)
```

Result:

```
[{'age': {'confidence': 84.60948944091797, 'value': 26},
  'face': [419, 266, 506, 352],
  'gender': {'confidence': 100.0, 'value': 'female'}},
 {'age': {'confidence': 46.72210693359375, 'value': 58},
  'face': [780, 112, 883, 215],
  'gender': {'confidence': 99.71077728271484, 'value': 'male'}},
 {'age': {'confidence': 72.31566619873047, 'value': 19},
  'face': [595, 135, 699, 238],
  'gender': {'confidence': 98.93635559082031, 'value': 'male'}},
 {'age': {'confidence': 53.86639404296875, 'value': 62},
  'face': [227, 198, 314, 285],
  'gender': {'confidence': 99.9999771118164, 'value': 'female'}},
 {'age': {'confidence': 89.23471069335938, 'value': 25},
  'face': [352, 544, 438, 630],
  'gender': {'confidence': 100.0, 'value': 'female'}}]
```

### Example with Pillow
Code: https://github.com/mowshon/age-and-gender/tree/master/example
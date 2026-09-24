<p align="center">
  <img src="https://raw.githubusercontent.com/mowshon/age-and-gender/master/example/logo.png" alt="age-and-gender" width="500">
</p>

<p align="center">
  Age and gender estimation from face images.<br>
  Pure Python install. No compiler, no model downloads.
</p>

---

## Credits

This package uses pretrained models from
**[dlib-models](https://github.com/davisking/dlib-models)** by
**[Davis E. King](https://github.com/davisking)**. The age predictor and gender
classifier were contributed to dlib-models by Cydral Technology.
We are grateful to them for making these models freely available.

## Installation

```bash
pip install age-and-gender
```

Requires Python 3.11+. Pretrained models are bundled with the package.

## Quick start

```python
from PIL import Image
from age_and_gender import AgeAndGender

predictor = AgeAndGender()
image = Image.open("photo.jpg").convert("RGB")

print(predictor.predict(image))
```

```python
[{'gender': {'value': 'female', 'confidence': 100},
  'age': {'value': 26, 'confidence': 84},
  'face': [419, 266, 506, 352]},
 ...]
```

![result](https://raw.githubusercontent.com/mowshon/age-and-gender/master/example/result.jpg)

<sub>© [Bill Gates family](https://www.businessinsider.com/microsoft-bill-melinda-gates-drive-daughter-to-school-2019-4)</sub>

## Using any face detector

The built-in detector is optional. Crop faces with any detector (OpenCV, MediaPipe,
RetinaFace, face_recognition, etc.) and pass each crop to `predict_face()`:

```python
from PIL import Image
from age_and_gender import AgeAndGender

predictor = AgeAndGender()
image = Image.open("photo.jpg").convert("RGB")

for left, top, right, bottom in detect_faces(image):  # your detector
    face = image.crop((left, top, right, bottom))  # cropped face only
    print(predictor.predict_face(face))
```

```python
{'gender': {'value': 'female', 'confidence': 100}, 'age': {'value': 25, 'confidence': 80}}
```

If you only need one attribute, `gender()` and `age()` each run a single model:

```python
predictor.gender(face)  # {'value': 'female', 'confidence': 100}
predictor.age(face)     # {'value': 25, 'confidence': 80}
```

> `face` can also be an RGB NumPy array. Convert OpenCV (BGR) frames first with
> `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)`.

## API

| Method | Returns |
| --- | --- |
| `AgeAndGender()` | Predictor using the bundled models |
| `predict(image, face_bounding_boxes=None)` | `list` of `{gender, age, face}`, one per detected face |
| `predict_face(face)` | `{gender, age}` for one cropped face |
| `gender(face)` | `{value, confidence}` |
| `age(face)` | `{value, confidence}` |
| `AgeAndGender.from_model_dir(path)` | Predictor using a custom model bundle (`manifest.json` + models) |

- `image` / `face`: RGB `PIL.Image` or `[H, W, 3]` `uint8` NumPy array.
- `face_bounding_boxes`: optional `(top, right, bottom, left)` boxes. When given, detection is skipped.
- `face` in results: `[left, top, right, bottom]`.
- `confidence`: integer percentage.

## Models

The pretrained models come from
[davisking/dlib-models](https://github.com/davisking/dlib-models) and are bundled with
the package, so nothing needs to be downloaded.

| Model | Description | In this package |
| --- | --- | --- |
| HOG face detector | dlib's built-in frontal face detector. Used only by `predict()` when no boxes are given. | built into `dlib` |
| `shape_predictor_5_face_landmarks.dat` | 5-point landmark model (eye corners and bottom of the nose) used to align each face. Trained on 7,198 annotated faces. | unchanged |
| `dnn_age_predictor_v1.dat` | ResNet-10 age predictor trained on about 110k labelled face images. Estimates ages from 0 to 80. | `age-v1.onnx` |
| `dnn_gender_classifier_v1.dat` | Compact CNN gender classifier trained on about 200k face images. Around 97.3% accuracy on LFW. | `gender-v1.onnx` |

`dnn_age_predictor_v1.dat` and `dnn_gender_classifier_v1.dat` were converted to
**ONNX** so they run on [ONNX Runtime](https://onnxruntime.ai/) and are easier to use from
Python. The weights are unchanged, and the outputs match the original dlib networks.

## What's new in 2.0

- **No legacy C++ code.** The C++ extension and the vendored dlib sources from 1.x
  have been removed. Nothing is compiled during installation.
- **ONNX inference.** Age and gender run on ONNX Runtime. Face detection and landmarks
  use the prebuilt `dlib-bin` wheel.
- **Bundled models.** `AgeAndGender()` works immediately. The `load_*` calls are no
  longer needed.
- **Any face detector.** `predict_face()`, `gender()` and `age()` accept cropped faces.

## License

The package is released under the [MIT License](LICENSE). The bundled models are
dedicated to the public domain under
[CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/).

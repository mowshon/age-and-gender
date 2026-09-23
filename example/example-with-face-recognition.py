from age_and_gender import *
from PIL import Image, ImageDraw, ImageFont
import face_recognition
import numpy

"""

Install numpy and face_recognition

python -m pip install numpy --user
python -m pip install face_recognition --user

"""

# Uses the models installed with the package; no explicit paths needed. The
# original dnn_age_predictor_v1.dat/dnn_gender_classifier_v1.dat cannot be
# loaded directly any more (see spec/PR-5.md's "Design change" section) since
# they are a proprietary dlib format with no runtime ONNX reader, so
# load_dnn_age_predictor/load_dnn_gender_classifier now only accept a
# converted .onnx file. To use a custom five-point landmark model instead of
# the bundled one, call:
#   data.load_shape_predictor('models/shape_predictor_5_face_landmarks.dat')
data = AgeAndGender()

filename = 'test-image-2.jpg'

img = Image.open(filename).convert("RGB")
face_bounding_boxes = face_recognition.face_locations(
    numpy.asarray(img),  # Convert to numpy array
    model='hog'  # 'hog' for CPU | 'cnn' for GPU (NVIDIA with CUDA)
)

result = data.predict(img, face_bounding_boxes)

font = ImageFont.truetype("Acme-Regular.ttf", 15)

for info in result:
    shape = [(info['face'][0], info['face'][1]), (info['face'][2], info['face'][3])]
    draw = ImageDraw.Draw(img)

    gender = info['gender']['value'].title()
    gender_percent = int(info['gender']['confidence'])
    age = info['age']['value']
    age_percent = int(info['age']['confidence'])

    draw.text(
        (info['face'][0] - 10, info['face'][3] + 10), f"{gender} (~{gender_percent}%)\n{age} y.o. (~{age_percent}%).",
        fill='red', font=font, align='center'
    )

    draw.rectangle(shape, outline="red", width=5)

img.show()
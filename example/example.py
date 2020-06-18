from age_and_gender import *
from PIL import Image, ImageDraw, ImageFont

data = AgeAndGender()
data.load_shape_predictor('models/shape_predictor_5_face_landmarks.dat')
data.load_dnn_gender_classifier('models/dnn_gender_classifier_v1.dat')
data.load_dnn_age_predictor('models/dnn_age_predictor_v1.dat')

filename = 'test-image.jpg'

img = Image.open(filename).convert("RGB")
result = data.predict(img)

font = ImageFont.truetype("Acme-Regular.ttf", 20)

for info in result:
    shape = [(info['face'][0], info['face'][1]), (info['face'][2], info['face'][3])]
    draw = ImageDraw.Draw(img)

    gender = info['gender']['value'].title()
    gender_percent = int(info['gender']['confidence'])
    age = info['age']['value']
    age_percent = int(info['age']['confidence'])

    draw.text(
        (info['face'][0] - 10, info['face'][3] + 10), f"{gender} (~{gender_percent}%)\n{age} y.o. (~{age_percent}%).",
        fill='white', font=font, align='center'
    )

    draw.rectangle(shape, outline="red", width=5)

img.show()

import os
from flask import Flask, jsonify, request
from flask_cors import CORS
from paddleocr import PaddleOCR
import utility

ocr = PaddleOCR(use_textline_orientation=True, lang='es')

app = Flask(__name__)

CORS(app, resources={
  r"/*":{
    "origins":"*"
  }
}, supports_credentials=True)
app.config['CORS_HEADER'] = 'Content-type'

@app.route('/', methods=['GET'])
def base():

  return 'servicio activo'

@app.route('/ocr', methods=['POST'])
def ocr_endpoint():

    testing = request.args.get('testing', 'false').lower() == 'true'
    if testing:
      image = request.files.get('image')
      image = utility.convertFilestorage(image)
    else:
      data = request.get_json()
      image = utility.convertB64(data.get('image'))

    # temp_dir = '/temp'
    # os.makedirs(temp_dir, exist_ok=True)
    # image_path = os.path.join(temp_dir, image.filename)
    # image.save(image_path)

    result = ocr.predict(image)

    text = []

    for data in result:
      if 'rec_texts' in data:
        text = data['rec_texts']
    return jsonify(text)


# def read_text_from_image():

#   # reqBody = request.get_json()

#   imagen = request.files.get('image')

#   data =  utility.convertFilestorage(imagen)

#   ocr = PaddleOCR(use_textline_orientation=True, lang='en')
#   result = ocr.predict(data)
#   print(result)
#   # for line in result:
#   #   for box in line:
#   #     print(box[1][0])  # Print recognized text

#   return 'asdasd'

if __name__ == '__main__':
  app.run(debug=True, port=5000)
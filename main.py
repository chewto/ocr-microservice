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
      image = utility.preprocessing(image, 1080, filters='sharp')
    else:
      data = request.get_json()
      image = utility.convertB64(data.get('image'))
      image = utility.preprocessing(image, 1080, filters='sharp')

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


if __name__ == '__main__':
  app.run(debug=True, host="0.0.0.0",port=4000)
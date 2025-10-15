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

    angle = 0

    text = []

    for data in result:

      if 'doc_preprocessor_res' in data:
        if 'angle' in data['doc_preprocessor_res']:
          angle = data['doc_preprocessor_res']['angle']
          print(angle)
      if 'rec_texts' in data:
        text = data['rec_texts']
    return jsonify({"ocr":text, "textAngle": angle})

@app.route('/yolo-ocr', methods=['POST'])
def yoloOCR():

  bodyReq = request.get_json()
  image = bodyReq.get('image', None)
  image = utility.convertB64(image)
  image = utility.preprocessing(image, 1080, filters='sharp')
  noOcrLabels = bodyReq.get('noOcrLabels', None)
  labelsCoords = bodyReq['labels']

  extractedData = []


  for label in labelsCoords:
    crop = label["crop"]
    y = crop[0]
    x = crop[1]

    cropImage = image[y[0]:y[1], x[0]:x[1]]

    if cropImage is None or cropImage.size == 0:
        print(f"ADVERTENCIA: Recorte vacío para la etiqueta {label['label']} con coords Y:{y}, X:{x}")
        continue 

    if label["label"] not in noOcrLabels:
      result = ocr.predict(cropImage)[0]

      data = {"label": label["label"], "coords": {"y": y, "x": x}}

      if 'doc_preprocessor_res' in result:
        print(result['doc_preprocessor_res'])
        if 'angle' in result['doc_preprocessor_res']:
          angle = result['doc_preprocessor_res']['angle']
          data["angle"] = angle
      if 'rec_texts' in result:
        text = result['rec_texts']
        data["text"] = text

      extractedData.append(data)

  return extractedData

@app.route('/rotate', methods=['POST'])
def rotate():

    testing = request.args.get('testing', 'false').lower() == 'true'
    if testing:
      image = request.files.get('image')
      image = utility.convertFilestorage(image)
      image = utility.preprocessing(image, 150, filters='sharp')
    else:
      data = request.get_json()
      image = utility.convertB64(data.get('image'))
      image = utility.preprocessing(image, 150, filters='sharp')

    # temp_dir = '/temp'
    # os.makedirs(temp_dir, exist_ok=True)
    # image_path = os.path.join(temp_dir, image.filename)
    # image.save(image_path)

    result = ocr.predict(image)

    angle = 0

    text = []

    for data in result:

      if 'doc_preprocessor_res' in data:
        if 'angle' in data['doc_preprocessor_res']:
          angle = data['doc_preprocessor_res']['angle']
          print(angle)
      if 'rec_texts' in data:
        text = data['rec_texts']
    return jsonify({"ocr":text, "textAngle": angle})

if __name__ == '__main__':
  app.run(debug=True, host="0.0.0.0",port=4500)
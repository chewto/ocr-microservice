from flask import Flask, jsonify, request
from flask_cors import CORS
from paddleocr import PaddleOCR
import utility
import time

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

    # --- Bloque Try/Except GENERAL ---
    # Captura cualquier error inesperado (mala request, error de preprocesamiento, etc.)
    try:
        testing = request.args.get('testing', 'false').lower() == 'true'
        image = None # Inicializar

        if testing:
            image_file = request.files.get('image')
            if not image_file:
                return jsonify({"error": "No image file provided in 'image' form-data."}), 400
            
            image = utility.convertFilestorage(image_file)
            image = utility.preprocessing(image, 1080, filters='sharp')
        else:
            data = request.get_json()
            if not data or 'image' not in data:
                return jsonify({"error": "No 'image' key found in JSON payload."}), 400

            b64_image = data.get('image')
            if not b64_image:
                return jsonify({"error": "Image data (base64) is missing."}), 400
                
            image = utility.convertB64(b64_image)
            image = utility.preprocessing(image, 1080, filters='sharp')

        if image is None:
            return jsonify({"error": "Image processing or conversion failed."}), 400

        
        # --- Bloque Try/Except ESPECÍFICO (El que tú escribiste) ---
        # Intenta solo la predicción de OCR, que es la línea propensa a fallar.
        try:
            # Esta es la línea que está fallando
            result = ocr.predict(image) 

        except RuntimeError as e:
            # 1. Imprime el error en el log
            print(f"Error de PaddleOCR: {e}")

            # 2. (Opcional pero recomendado) Guarda la imagen problemática para analizarla
            # Asumiendo que 'image' es un objeto de imagen de PIL o similar
            try:
                timestamp = int(time.time())
                # Asegúrate de que este directorio '/usr/src/app/' exista y tengas permisos
                save_path = f"/imagenes_ocr/problematic_image_{timestamp}.png"
                image.save(save_path)
                print(f"Imagen problemática guardada como {save_path}")
            except Exception as save_e:
                print(f"No se pudo guardar la imagen problemática: {save_e}")

            # 3. Devuelve un error JSON claro al cliente
            # (Tu código original devolvía un dict, lo he envuelto en jsonify)
            return jsonify({"error": "Error al procesar la imagen con OCR.", "details": str(e)}), 500
        
        # --- Procesamiento del resultado (si el try/except específico NO falló) ---
        
        angle = 0
        text = []

        for data in result:
            if 'doc_preprocessor_res' in data:
                if 'angle' in data['doc_preprocessor_res']:
                    angle = data['doc_preprocessor_res']['angle']
                    print(angle)
            if 'rec_texts' in data:
                text = data['rec_texts']
        
        # Respuesta exitosa
        return jsonify({"ocr": text, "textAngle": angle})

    # --- Captura del Try/Except GENERAL ---
    except Exception as e:
        # Captura cualquier otro error (ej. JSON mal formado, error en utility.convertB64, etc.)
        print(f"Error inesperado en /ocr endpoint: {e}")
        # import traceback
        # print(traceback.format_exc()) # Descomenta para más detalles
        
        return jsonify({
            "error": "Ocurrió un error interno general.",
            "details": str(e)
        }), 500

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
  app.run(debug=True, host="0.0.0.0",port=4000)
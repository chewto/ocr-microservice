
import base64

import cv2
import numpy as np


def convertB64(imagenUrl):
  imagenData = base64.b64decode(imagenUrl.split(",")[1])

  npArray = np.frombuffer(imagenData, np.uint8)

  imagen = cv2.imdecode(npArray, cv2.IMREAD_COLOR)

  return imagen

def convertFilestorage(file_storage):
    file_bytes = file_storage.read()
    np_array = np.frombuffer(file_bytes, np.uint8)
    imagen = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    return imagen
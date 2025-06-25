
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

def preprocessing(img, resolution, filters):

    h0, w0 = img.shape[:2]

    print(img.shape[:2], 'previo')
    if resolution < w0:
        h1 = int(h0 * resolution / w0)
        img = cv2.resize(img, (resolution, h1), interpolation=cv2.INTER_AREA)

    proc = img.copy()
    print(proc.shape[:2], 'post')
    if 'gray' in filters:
        proc = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
    if 'hist' in filters:
        gray = proc if proc.ndim == 2 else cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
        proc = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    if 'sharp' in filters:
        kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
        proc = cv2.filter2D(proc, -1, kernel)
    if 'blur' in filters:
        proc = cv2.medianBlur(proc, 3)
    if 'thresh' in filters:
        if proc.ndim == 2:
            _, proc = cv2.threshold(proc, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    proc_rgb = proc if proc.ndim == 3 else cv2.cvtColor(proc, cv2.COLOR_GRAY2BGR)

    return proc_rgb
import cv2
import requests

# For Android IP Webcam, which I'm not actively using now.
import numpy as np
from requests.auth import HTTPBasicAuth

from ultralytics import YOLO

import threading

import time
import datetime

import base64

"""
Load .env content, check out README.md for more details 
"""
from dotenv import load_dotenv
import os

# This is only for Android IP webcam, which I'm not actively using.
def _get_img_from_ipcam_stream(
    url:str = None,
    user:str = None,
    password:str = None
    ):
  if url is None:
    raise ValueError("url is Empty")
  if user is None or password is None:
    raise Warning("Username and password not set, check if your server is set to with password.")
  
  img_resp = requests.get(url, auth=HTTPBasicAuth(username=user, password=password)) 
  img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8) 
  img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
  return img


def _extract_model_prediction(model, img) -> dict:
  result = model(img)[0]
  max_conf = 0

  for conf, cs in zip(result.boxes.conf, result.boxes.cls):
    if conf >= 0.3 and result.names[int(cs)] == 'cat' and (conf >= max_conf):
      max_conf = conf
  
  msg_dict = None
  if max_conf >= 0.8:
    msg_dict = {
      "pr": "default",
      "title": "Cat Detected!",
      "msg": f"Found cat at confedence level: {conf}",
      "tags": "tada"
    }
  elif max_conf >= 0.3:
    msg_dict = {
      "pr": "low",
      "title": "Cat Detected! Probably...",
      "msg": f"Found cat at confedence level: {conf}. This could be wrong.",
      "tags": "tada"
    }

  return msg_dict


def _push_ntfy(
    host:str = None,
    topic:str = None,
    msg_dict:dict= None,
    ntfy_user:str = None,
    ntfy_pass:str = None,
    img_path = None,
    ):
  auth = base64.b64encode((ntfy_user+":"+ntfy_pass).encode('UTF-8'))

  requests.post(
    f"https://{host}/{topic}",               
    data=msg_dict['msg'].encode(encoding='utf-8'),
    headers={
      "Authorization": auth,
      "Title": msg_dict['title'],
      "Priority": msg_dict['pr'],
      "Tags": msg_dict['tags']
    }
  )

  if img_path is not None:
    data = open(img_path, "rb")
    filename = img_path.split('/')[-1]
    requests.put(
      f"https://{host}/{topic}",
      data=data,
      headers={"Filename": filename}
    )


def _chk_crt_path(path):
  if not os.path.exists(path):
    os.mkdir(path)
  elif not os.path.isdir(path):
    raise ValueError("path given exists and is not a directory.")


class RTSPStream:
  def __init__(self, rtsp_url):
    self.rtsp_url = rtsp_url
    self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
    self.frame = None
    self.lock = threading.Lock()
    self.running = True
    threading.Thread(target=self.update, daemon=True).start()

  def update(self):
    while self.running:
      ret, frame = self.cap.read()
      if ret:
        with self.lock:
          self.frame = frame

  def get_frame(self):
    with self.lock:
      return self.frame.copy() if self.frame is not None else None

  def stop(self):
    self.running = False
    self.cap.release()


if __name__ == "__main__":
  load_dotenv()

  # webcam related information
  user = os.getenv('USER')
  password = os.getenv('PASSWORD')
  url = os.getenv('URL')

  # ntfy related information
  ntfy_user=os.getenv('NTFY_USER')
  ntfy_pass=os.getenv('NTFY_PASS')
  host = os.getenv('HOSTNAME')
  topic = os.getenv('TOPIC')

  # path related, if IMG_SAVE_PATH is not set, save_img would be set to PROJ_PATH/saved_img 
  proj_path = os.getenv('PROJ_PATH')
  img_path = os.getenv('IMG_SAVE_PATH')

  # Load model for object recognition
  model = YOLO(os.getenv('MODEL'))

  # notification/process suspend after activation (in seconds)
  suspend = int(os.getenv('SUSPEND'))

  # Set environment for rtsp_transport
  os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;0"

  if img_path == "" or img_path is None:
    img_path = proj_path+"/saved_img"
  _chk_crt_path(img_path)

  stream = RTSPStream(url)

  threshold = 500
  activation = 0
  step = 20
  if suspend == "" or suspend is None:
    suspend = 600 # in seconds

  while True:
    try:
      frame = stream.get_frame()
      filename=f"{img_path}/{datetime.datetime.now().strftime('%c')}.jpg"
      
      if frame is not None:
        msg_dict = _extract_model_prediction(model, frame)
        
        if msg_dict is not None:
          activation += step
        else:
          activation -= (step/10)

        if activation >= threshold:
          cv2.imwrite(filename, frame)
          _push_ntfy(
            host = host, 
            topic = topic,
            msg_dict = msg_dict,
            ntfy_user = ntfy_user,
            ntfy_pass = ntfy_pass,
            img_path = filename)
          

          time.sleep(suspend)
          activation = 0
        
    except KeyboardInterrupt:
      break

  stream.stop()

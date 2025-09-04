import cv2
import requests

# For Android IP Webcam, which I'm not actively using now.
import numpy as np
from requests.auth import HTTPBasicAuth

from ultralytics import YOLO
import torch

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


def _setup_torch_device():
  # Priority of torch.device: cuda > mps > cpu
  if torch.cuda.is_available():
    torch_device = torch.device("cuda")
  elif torch.backends.mps.is_available():
    torch_device = torch.device("mps")
  else:
    torch_device = torch.device("cpu")
  return torch_device


def _extract_model_prediction(model, img, device) -> dict:
  result = model(img, device=device)[0]
  max_conf = 0

  for conf, cs in zip(result.boxes.conf, result.boxes.cls):
    if conf >= 0.3 and result.names[int(cs)] == 'cat' and (conf >= max_conf):
      max_conf = conf
  
  msg_dict = None
  if max_conf >= 0.8:
    msg_dict = {
      "pr": "default",
      "title": "Cat Detected!",
      "msg": f"Found cat at conf lv.: {conf*100:.2f}%",
      "tags": "tada"
    }
  elif max_conf >= 0.3:
    msg_dict = {
      "pr": "low",
      "title": "Cat Detected! Probably...",
      "msg": f"Found cat at conf lv: {conf*100:.2f}%. This could be wrong.",
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


def _extract_info_diff(prev_frame, frame):
  fram_diff = None
  info_KB = 0
  if prev_frame is not None and frame is not None:
    frame = cv2.fastNlMeansDenoisingColored(frame)
    prev_frame = cv2.fastNlMeansDenoisingColored(prev_frame)
    fram_diff = frame - prev_frame
    
    info_bits = torch.tensor(fram_diff).abs().log2()
    info_bits = (info_bits.nan_to_num() * ~torch.isneginf(info_bits)) # in bits
    info_KB = info_bits.sum()/(8*1024)
  return info_KB, fram_diff


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

  if (user is not None or user == "") and (password is not None or password == ""):
    url_li = url.split('//')
    url = f"{url[0]}//{user}:{password}@{url_li[1]}"

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
  
  show = True

  # Set environment for rtsp_transport
  os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;0"

  if img_path == "" or img_path is None:
    img_path = proj_path+"/saved_img"
  _chk_crt_path(img_path)

  stream = RTSPStream(url)

  msg_n_init_act = 500
  det_thres = 80 # info difference threshold for activate model (in KB)

  msg_act_thres = 1000
  msg_activation = msg_n_init_act
  msg_act_step = 20

  if suspend == "" or suspend is None:
    suspend = 600 # in seconds
  device = _setup_torch_device()
  
  while True:
    try:
      try:
        if frame is not None:
          prev_frame = frame
          fram_diff = np.empty(frame.shape)
        else:
          prev_frame = None
          fram_diff = None
      except NameError:
        prev_frame = None

      frame = stream.get_frame()
      filename=f"{img_path}/{datetime.datetime.now().strftime('%c')}.jpg"
      
      info_KB, fram_diff = _extract_info_diff(prev_frame=prev_frame, frame=frame)

      if frame is not None and info_KB > det_thres:
        print(f"info differences in approx. KB: {info_KB:.2f}KB")

        if fram_diff is not None and show:
          cv2.imshow(winname="diff", mat=fram_diff)
          cv2.waitKey(1)

        model_activation = 100
        model_act_step = 10

        while model_activation > 0 and msg_activation >= 0:
          frame = stream.get_frame()
          filename=f"{img_path}/{datetime.datetime.now().strftime('%c')}.jpg"

          msg_dict = _extract_model_prediction(model, frame, device)
          
          if msg_dict is not None:
            msg_activation += msg_act_step
            model_activation += model_act_step
          elif msg_activation >= 0:
            msg_activation -= (msg_act_step/10)
            model_activation -= (model_act_step/5)

          if msg_activation >= msg_act_thres:
            cv2.imwrite(filename, frame)
            _push_ntfy(
              host = host, 
              topic = topic,
              msg_dict = msg_dict,
              ntfy_user = ntfy_user,
              ntfy_pass = ntfy_pass,
              img_path = filename
            )
            
            time.sleep(suspend)
            break
        
        if msg_activation != msg_n_init_act:
          msg_activation = msg_n_init_act
        
    except KeyboardInterrupt:
      break
  
  cv2.destroyAllWindows()
  cv2.waitKey(1)
  stream.stop()
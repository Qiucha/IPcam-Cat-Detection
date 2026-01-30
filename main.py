import requests
import numpy as np
from requests.auth import HTTPBasicAuth
from ultralytics import YOLO
import torch
import threading
import datetime
import base64

# Load .env content, check out README.md for more details
from dotenv import load_dotenv
import os
import time
import gc

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|rtsp_timeout;500000"
import cv2

# This is only for Android IP webcam, which I'm not actively using.
def _get_img_from_ipcam_stream(
        url:str = None,
        user:str = None,
        password:str = None
    ) -> np.ndarray:
    if url is None:
        raise ValueError("url is Empty")
    if user is None or password is None:
        raise Warning("Username and password not set, check if your server is set to with password.")

    img_resp = requests.get(url, auth=HTTPBasicAuth(username=user, password=password))
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    return img

def _extract_model_prediction(model, img, device, verbose:bool = False) -> dict:
    with torch.no_grad():
        result = model(img, device=device, stream=False, verbose=verbose)[0]
    max_conf = 0

    if result.boxes:
        for box in result.boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            label = result.names[cls_id]

            if conf >= 0.35 and label == 'cat' and (conf >= max_conf):
                max_conf = conf

    for conf, cs in zip(result.boxes.conf, result.boxes.cls):
        conf = float(conf.item())
        cs = int(cs.item())
        if conf >= 0.35 and result.names[cs] == 'cat' and (conf >= max_conf):
            max_conf = conf

    del result

    msg_dict = None
    if max_conf >= 0.7:
        if verbose:
            print("Cat detected!")
        msg_dict = {
            "pr": "default",
            "title": "Cat Detected!",
            "msg": f"Found cat at conf lv.: {max_conf*100:.2f}%",
            "tags": "tada"
        }
    elif max_conf >= 0.35:
        if verbose:
            print("Cat detected...? Probably...?")
        msg_dict = {
            "pr": "low",
            "title": "Cat Detected! Probably...",
            "msg": f"Found cat at conf lv.: {max_conf*100:.2f}%. This could be wrong.",
            "tags": "tada"
        }

    return msg_dict

def _push_cat_ntfy(
           host:str = None,
           topic:str = None,
           msg_dict:dict= None,
           ntfy_user:str = None,
           ntfy_pass:str = None,
           img_path:str = None
    ) -> None:
    auth = base64.b64encode((ntfy_user+":"+ntfy_pass).encode('UTF-8'))

    requests.post(
        f"https://{host}/{topic}",
        data=msg_dict['msg'].encode(encoding='utf-8'),
        headers={
            "Authorization": "Basic " + auth.decode('utf-8'),
            "Title": msg_dict['title'],
            "Priority": msg_dict['pr'],
            "Tags": msg_dict['tags']
        }
    )

    if img_path is not None:
        with open(img_path, "rb") as data:
            filename = img_path.split('/')[-1]
            requests.put(
                 f"https://{host}/{topic}",
                 data=data,
                 headers={
                        "Filename": filename,
                        "Authorization": "Basic " + auth.decode('utf-8'),
                        "Priority": msg_dict['pr'],
                        "Tags": "camera_flash"
                 }
              )
    return None

def _extract_info_diff(prev_frame, frame) -> tuple[float, np.ndarray]:
    fram_diff = None
    info_KB = 0.0
    if prev_frame is not None and frame is not None:
        fram_diff = cv2.absdiff(prev_frame, frame)

        diff_float = fram_diff.astype(np.float32)

        with np.errstate(divide='ignore'):
            info_bits = np.log2(diff_float)

        info_bits[np.isneginf(info_bits)] = 0
        info_bits = np.nan_to_num(info_bits)
        info_KB = np.sum(info_bits) / (8 * 1024)
    return info_KB, fram_diff

class RTSPStream:
    def __init__(self, rtsp_url, host, discon_topic, ntfy_user, ntfy_pass):
        self.rtsp_url = rtsp_url
        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        self.frame = None
        self.lock = threading.Lock()
        self.running = True

        # Ntfy params for disconnection notification
        self.host = host
        self.discon = discon_topic
        self.ntfy_user = ntfy_user
        self.ntfy_pass = ntfy_pass

        ## set activation params
        self.neuron = 0
        self.neuron_neutral = 0
        self.act_thres = 10

        threading.Thread(target=self.update, daemon=True).start()

    def update(self) -> None:
        tried = False
        while self.running:

            if not self.cap.isOpened():
                self.neuron += 1
                if self.neuron >= self.act_thres:
                    self._push_discon_ntfy(mode="disconnect")
                    self.neuron = self.neuron_neutral
                    try:
                        time.sleep(3600) # wait for 1 hour
                    except KeyboardInterrupt:
                        print("KeyboardInterrupt")

            ret, frame = self.cap.read()
            if ret:
                if self.neuron >= self.neuron_neutral:
                    self.neuron = self.neuron_neutral

                with self.lock:
                    self.frame = frame
            else:
                self.neuron += 1
                if self.neuron >= self.act_thres and not tried:
                    self.cap.release()
                    self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
                    tried = True
                elif self.neuron >= self.act_thres and tried:
                    self._push_discon_ntfy(mode="timeout")
                    self.neuron = self.neuron_neutral
                    try:
                        time.sleep(3600) # wait for 1 hour
                    except KeyboardInterrupt:
                        print("KeyboardInterrupt")
                else:
                    self._push_discon_ntfy(mode=None)
                    print(f"something goes wrong.")
                    print(f"Current neuron: {self.neuron}\nCurrent act_threshold: {self.act_thres}")
                    try:
                        time.sleep(600) # wait for 10 minutes
                    except KeyboardInterrupt:
                        print("KeyboardInterrupt")

    def _push_discon_ntfy(self, mode:str = None) -> None:
        # Three mode accepted ["disconnect", "timeout", others(fallback)]
        auth = base64.b64encode((self.ntfy_user+":"+self.ntfy_pass).encode('UTF-8'))

        match mode:
            case "timeout":
                requests.post(
                    f"https://{self.host}/{self.discon}",
                    data="restart the program!".encode(encoding='utf-8'),
                    headers={
                        "Authorization": "Basic " + auth.decode('utf-8'),
                        "Title": "IPCam framecap timeout!",
                        "Priority": "default",
                        "Tags": "warning"
                    }
                )
            case "disconnect":
                requests.post(
                    f"https://{self.host}/{self.discon}",
                    data="check IPCam connection!".encode(encoding='utf-8'),
                    headers={
                        "Authorization": "Basic " + auth.decode('utf-8'),
                        "Title": "IPCam disconnected!",
                        "Priority": "default",
                        "Tags": "warning"
                    }
                )
            case None:
                requests.post(
                    f"https://{self.host}/{self.discon}",
                    data="Mode not set!".encode(encoding='utf-8'),
                    headers={
                        "Authorization": "Basic " + auth.decode('utf-8'),
                        "Title": "discon_ntfy with mode not set!",
                        "Priority": "default",
                        "Tags": "warning"
                    }
                )
        return None

    def get_frame(self) -> np.ndarray or None:
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self) -> None:
        self.running = False
        self.cap.release()
        return None

    def clear_memory(self):
        gc.collect()
        # Clear CUDA cache if using NVIDIA GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Clear MPS cache if using Apple Silicon (Mac Mini M4/Air M1)
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()


class Config:
    def __init__(self):
        load_dotenv()
        # webcam related information
        self.user = os.getenv('USER')
        self.password = os.getenv('PASSWORD')
        self.url = os.getenv('URL')
        if (self.user != '') and (self.password != ''):
            self.url = f"{self.url.split('//')[0]}//{self.user}:{self.password}@{self.url.split('//')[1]}"

        # ntfy related information
        self.ntfy_user=os.getenv('NTFY_USER')
        self.ntfy_pass=os.getenv('NTFY_PASS')
        self.host = os.getenv('HOSTNAME')
        self.topic = os.getenv('TOPIC')

        # disconnect ntfy topic
        self.discon_topic = os.getenv('DISCON_TOPIC')

        # path related, if IMG_SAVE_PATH is not set, save_img would be set to PROJ_PATH/saved_img
        self.proj_path = os.getenv('PROJ_PATH')
        self.img_path = os.getenv('IMG_SAVE_PATH')
        if (self.img_path == "" or self.img_path is None):
            self.img_path = self.proj_path+"/saved_img"

        # model name for object recognition
        self.model_name = os.getenv('MODEL')

        # notification/process suspend after activation (in seconds)
        self.suspend = int(os.getenv('SUSPEND'))
        if self.suspend == "" or self.suspend is None:
            self.suspend = 600 # in seconds

        # show difference between frame
        self.show = False
        self.verbose = os.getenv('VERBOSE')
        self.verbose = True if self.verbose == "True" or self.verbose == "1" else False

if __name__ == "__main__":
    config = Config()

    # Load model for object recognition
    model = YOLO(config.model_name)

    # Set environment for rtsp_transport
    # os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
    # os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] += "|rtsp_timeout;500000"

    # Make sure the path exist and is actually a pathname
    if not os.path.exists(config.img_path):
        os.mkdir(config.img_path)
    elif not os.path.isdir(config.img_path):
        raise ValueError("path given exists and is not a directory.")

    stream = RTSPStream(
        rtsp_url=config.url,
        host=config.host,
        discon_topic=config.discon_topic,
        ntfy_user=config.ntfy_user,
        ntfy_pass=config.ntfy_pass
    )

    msg_n_init_act = 500
    det_thres = 80 # info difference threshold for activate model (in KB)

    msg_act_thres = 1000
    msg_activation = msg_n_init_act
    msg_act_step = 20

    device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    frame = None
    prev_frame = None
    frame_diff = None

    # Counter for manual Garbage Collection
    loop_counter = 0

    while True:
        try:
            # Manual GC to prevent memory creep in long running scripts
            loop_counter += 1
            if loop_counter % 100 == 0:
                gc.collect()
                # Clear CUDA cache if using NVIDIA GPU
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # Clear MPS cache if using Apple Silicon (Mac Mini M4/Air M1)
                elif torch.backends.mps.is_available():
                    torch.mps.empty_cache()

            if frame is not None:
                prev_frame = frame

            frame = stream.get_frame()

            if frame is not None and prev_frame is not None:
                frame = cv2.fastNlMeansDenoisingColored(frame)

            info_KB, fram_diff = _extract_info_diff(prev_frame=prev_frame, frame=frame)

            if frame is not None and info_KB > det_thres:
                if config.verbose:
                    print(f"\rinfo differences in approx. KB: {info_KB:.2f}KB")

                if fram_diff is not None and config.show:
                    cv2.imshow(winname="diff", mat=fram_diff)
                    cv2.waitKey(1)

                model_activation = 100
                model_act_step = 10

                while model_activation > 0 and msg_activation >= 0:
                    frame = stream.get_frame()

                    msg_dict = _extract_model_prediction(model, frame, device)

                    if msg_dict is not None:
                        msg_activation += msg_act_step
                        model_activation += model_act_step
                    elif msg_activation >= 0:
                        msg_activation -= (msg_act_step/10)
                        model_activation -= (model_act_step/5)

                    if msg_activation >= msg_act_thres:
                        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                        filename = f"{config.img_path}/{timestamp}.jpg"

                        cv2.imwrite(filename, frame)
                        _push_cat_ntfy(
                            host = config.host,
                            topic = config.topic,
                            msg_dict = msg_dict,
                            ntfy_user = config.ntfy_user,
                            ntfy_pass = config.ntfy_pass,
                            img_path = filename
                        )

                        print("Press ANY key to keep detecting and send message!")
                        cv2.waitKey(1)
                        time.sleep(config.suspend)
                        stream.clear_memory()
                        prev_frame = None
                        break
                if msg_activation != msg_n_init_act:
                    msg_activation = msg_n_init_act
            else:
                time.sleep(0.01)
        except KeyboardInterrupt:
            stream.stop()
            break

        if config.show and cv2.waitKey(1) == ord('q'):
            break
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    stream.stop()

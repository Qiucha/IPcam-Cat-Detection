import requests
import numpy as np
from requests.auth import HTTPBasicAuth
from ultralytics import YOLO
import torch
import threading
import datetime
import base64
from dotenv import load_dotenv
import os
import time
import gc

# Optimize OpenCV for RTSP
# Increased timeout to 5000000 (5s) for stability
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|rtsp_timeout;5000000"
import cv2

def _get_img_from_ipcam_stream(
        url:str = None,
        user:str = None,
        password:str = None
    ) -> np.ndarray:
    if url is None:
        raise ValueError("url is Empty")
    if user is None or password is None:
        raise Warning("Username and password not set.")

    img_resp = requests.get(url, auth=HTTPBasicAuth(username=user, password=password))
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    return img

def _extract_model_prediction(model, img, device, verbose:bool = False) -> dict:
    # Use no_grad to ensure no gradient history is stored (saves memory)
    with torch.no_grad():
        results = model(img, device=device, stream=False, verbose=verbose)
        result = results[0]

    max_conf = 0

    # Efficient box processing
    if result.boxes:
        for box in result.boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            label = result.names[cls_id]

            if conf >= 0.35 and label == 'cat' and (conf >= max_conf):
                max_conf = conf

    # Clean up heavy tensor objects immediately
    del result
    del results

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
    try:
        auth = base64.b64encode((ntfy_user+":"+ntfy_pass).encode('UTF-8'))

        requests.post(
            f"https://{host}/{topic}",
            data=msg_dict['msg'].encode(encoding='utf-8'),
            headers={
                "Authorization": "Basic " + auth.decode('utf-8'),
                "Title": msg_dict['title'],
                "Priority": msg_dict['pr'],
                "Tags": msg_dict['tags']
            },
            timeout=10 # Added timeout to prevent hanging
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
                    },
                    timeout=30
                )
    except Exception as e:
        print(f"Error pushing notification: {e}")
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

        # Explicit cleanup
        del diff_float
        del info_bits

    return info_KB, fram_diff

class RTSPStream:
    def __init__(self, rtsp_url, host, discon_topic, ntfy_user, ntfy_pass):
        self.rtsp_url = rtsp_url
        self.cap = None
        self.frame = None
        self.lock = threading.Lock()
        self.running = True

        # State tracking
        self.connected = False
        self.reconnect_delay = 5  # seconds

        # Ntfy params
        self.host = host
        self.discon = discon_topic
        self.ntfy_user = ntfy_user
        self.ntfy_pass = ntfy_pass

        threading.Thread(target=self.update, daemon=True).start()

    def update(self) -> None:
        while self.running:
            # 1. Connect if not initialized
            if self.cap is None:
                self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
                if not self.cap.isOpened():
                    print(f"Failed to open stream. Retrying in {self.reconnect_delay}s...")
                    self.cap = None
                    time.sleep(self.reconnect_delay)
                    continue
                else:
                    print("Stream Connected.")
                    self.connected = True
                    # Optional: Notify reconnection
                    # self._push_discon_ntfy(mode="connected")

            # 2. Read Frame
            ret, frame = self.cap.read()

            if ret:
                with self.lock:
                    self.frame = frame
                # Reset connection state if we just recovered
                if not self.connected:
                    self.connected = True
            else:
                # 3. Handle Disconnection
                print("Frame read failed. Reconnecting...")
                if self.connected:
                    self._push_discon_ntfy(mode="disconnect")
                    self.connected = False

                self.cap.release()
                self.cap = None
                time.sleep(1) # Short cooldown before retry loop

    def _push_discon_ntfy(self, mode:str = None) -> None:
        if not self.host or not self.discon:
            return

        try:
            auth = base64.b64encode((self.ntfy_user+":"+self.ntfy_pass).encode('UTF-8'))
            title = "IPCam Disconnected!"
            msg = "Check IPCam connection! Stream read failed."

            requests.post(
                f"https://{self.host}/{self.discon}",
                data=msg.encode(encoding='utf-8'),
                headers={
                    "Authorization": "Basic " + auth.decode('utf-8'),
                    "Title": title,
                    "Priority": "high",
                    "Tags": "warning"
                },
                timeout=10
            )
        except Exception as e:
            print(f"Failed to send disconnect notification: {e}")

    def get_frame(self) -> np.ndarray or None:
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self) -> None:
        self.running = False
        if self.cap:
            self.cap.release()

    def clear_memory(self):
        # Force garbage collection
        gc.collect()
        # Clear Hardware Caches
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()


class Config:
    def __init__(self):
        load_dotenv()
        self.user = os.getenv('USER')
        self.password = os.getenv('PASSWORD')
        self.url = os.getenv('URL')
        if (self.user != '') and (self.password != ''):
            # Safe URL construction
            base = self.url.split('//')[1] if '//' in self.url else self.url
            scheme = self.url.split('//')[0] + '//' if '//' in self.url else 'rtsp://'
            self.url = f"{scheme}{self.user}:{self.password}@{base}"

        self.ntfy_user=os.getenv('NTFY_USER')
        self.ntfy_pass=os.getenv('NTFY_PASS')
        self.host = os.getenv('HOSTNAME')
        self.topic = os.getenv('TOPIC')
        self.discon_topic = os.getenv('DISCON_TOPIC')

        self.proj_path = os.getenv('PROJ_PATH')
        self.img_path = os.getenv('IMG_SAVE_PATH')
        if (self.img_path == "" or self.img_path is None):
            self.img_path = (self.proj_path if self.proj_path else ".") + "/saved_img"

        self.model_name = os.getenv('MODEL')
        self.suspend = int(os.getenv('SUSPEND')) if os.getenv('SUSPEND') else 600

        self.show = False
        self.verbose = os.getenv('VERBOSE')
        self.verbose = True if self.verbose == "True" or self.verbose == "1" else False

if __name__ == "__main__":
    config = Config()

    # Initialize Model
    model = YOLO(config.model_name)

    if not os.path.exists(config.img_path):
        os.makedirs(config.img_path, exist_ok=True)

    stream = RTSPStream(
        rtsp_url=config.url,
        host=config.host,
        discon_topic=config.discon_topic,
        ntfy_user=config.ntfy_user,
        ntfy_pass=config.ntfy_pass
    )

    msg_n_init_act = 500
    det_thres = 80
    msg_act_thres = 1000
    msg_activation = msg_n_init_act
    msg_act_step = 20

    # Device selection
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    frame = None
    prev_frame = None
    fram_diff = None

    loop_counter = 0

    while True:
        try:
            # Memory Management Cycle
            loop_counter += 1
            if loop_counter % 100 == 0:
                stream.clear_memory()
                loop_counter = 0

            # --- Logic Start ---
            if frame is not None:
                prev_frame = frame

            frame = stream.get_frame()

            if frame is not None and prev_frame is not None:
                # Processing
                frame_denoised = cv2.fastNlMeansDenoisingColored(frame)
                info_KB, fram_diff = _extract_info_diff(prev_frame=prev_frame, frame=frame_denoised)

                if info_KB > det_thres:
                    if config.verbose:
                        print(f"\rDifference info: {info_KB:.2f}KB")

                    if fram_diff is not None and config.show:
                        cv2.imshow(winname="diff", mat=fram_diff)
                        cv2.waitKey(1)

                    model_activation = 100
                    model_act_step = 10

                    # --- Detection Sub-loop ---
                    while model_activation > 0 and msg_activation >= 0:
                        # Grab FRESH frame for detection to avoid lag
                        det_frame = stream.get_frame()
                        if det_frame is None:
                            break

                        msg_dict = _extract_model_prediction(model, det_frame, device, config.verbose)

                        # Explicit cleanup of detection frame
                        del det_frame

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

                            print(f"Notification sent! Suspending for {config.suspend}s...")
                            # Deep Clean before Sleep
                            stream.clear_memory()
                            prev_frame = None
                            time.sleep(config.suspend)
                            break # Exit detection loop

                    if msg_activation != msg_n_init_act:
                        msg_activation = msg_n_init_act

                # Loop cleanup
                if fram_diff is not None:
                    del fram_diff
                    fram_diff = None
                if 'frame_denoised' in locals():
                    del frame_denoised
            else:
                # Prevent CPU spin if stream is reconnecting
                time.sleep(0.1)

        except KeyboardInterrupt:
            print("Stopping...")
            break
        except Exception as e:
            print(f"Main loop error: {e}")
            time.sleep(1)

        if config.show and cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()
    stream.stop()

import cv2
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

def _extract_model_prediction(model, img, device) -> dict:
	result = model(img, device=device)[0]
	max_conf = 0

	for conf, cs in zip(result.boxes.conf, result.boxes.cls):
		if conf >= 0.35 and result.names[int(cs)] == 'cat' and (conf >= max_conf):
			max_conf = conf

	msg_dict = None
	if max_conf >= 0.7:
		msg_dict = {
			"pr": "default",
			"title": "Cat Detected!",
			"msg": f"Found cat at conf lv.: {conf*100:.2f}%",
			"tags": "tada"
		}
	elif max_conf >= 0.35:
		msg_dict = {
			"pr": "low",
			"title": "Cat Detected! Probably...",
			"msg": f"Found cat at conf lv.: {conf*100:.2f}%. This could be wrong.",
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
        data = open(img_path, "rb")
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
	info_KB = 0
	if prev_frame is not None and frame is not None:
		frame = cv2.fastNlMeansDenoisingColored(frame)
		prev_frame = cv2.fastNlMeansDenoisingColored(prev_frame)
		fram_diff = cv2.absdiff(prev_frame, frame)

		info_bits = torch.tensor(fram_diff).abs().log2()
		info_bits = (info_bits.nan_to_num() * ~torch.isneginf(info_bits)) # in bits
		info_KB = info_bits.sum()/(8*1024)
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
					cv2.waitKey(3600*1000) # wait for 1 hour

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
					cv2.waitKey(3600*1000) # wait for 1 hour
				else:
					print(f"something goes wrong.")

	def _push_discon_ntfy(self, mode) -> None:
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
		    case _:
    			requests.post(
    				f"https://{self.host}/{self.discon}",
    				data="mode not set!".encode(encoding='utf-8'),
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


if __name__ == "__main__":
	load_dotenv()
	# webcam related information
	user = os.getenv('USER')
	password = os.getenv('PASSWORD')
	url = os.getenv('URL')

	if (user != '') and (password != ''):
		url_li = url.split('//')
		url = f"{url_li[0]}//{user}:{password}@{url_li[1]}"

	# ntfy related information
	ntfy_user=os.getenv('NTFY_USER')
	ntfy_pass=os.getenv('NTFY_PASS')
	host = os.getenv('HOSTNAME')
	topic = os.getenv('TOPIC')

	# disconnect ntfy topic
	discon_topic = os.getenv('DISCON_TOPIC')

	# path related, if IMG_SAVE_PATH is not set, save_img would be set to PROJ_PATH/saved_img
	proj_path = os.getenv('PROJ_PATH')
	img_path = os.getenv('IMG_SAVE_PATH')

	# Load model for object recognition
	model = YOLO(os.getenv('MODEL'))

	# notification/process suspend after activation (in seconds)
	suspend = int(os.getenv('SUSPEND'))

	show = False

	# Set environment for rtsp_transport
	os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
	os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_timeout;500000"

	if img_path == "" or img_path is None:
		img_path = proj_path+"/saved_img"

	# Make sure the path exist and is actually a pathname
	if not os.path.exists(img_path):
		os.mkdir(img_path)
	elif not os.path.isdir(img_path):
		raise ValueError("path given exists and is not a directory.")

	stream = RTSPStream(
		rtsp_url=url,
		host=host,
		discon_topic=discon_topic,
		ntfy_user=ntfy_user,
		ntfy_pass=ntfy_pass
	)

	msg_n_init_act = 500
	det_thres = 80 # info difference threshold for activate model (in KB)

	msg_act_thres = 1000
	msg_activation = msg_n_init_act
	msg_act_step = 20

	if suspend == "" or suspend is None:
		suspend = 600 # in seconds
	device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

	frame = None
	prev_frame = None
	frame_diff = None
	while True:
		try:
			if frame is not None:
				prev_frame = frame
				fram_diff = np.empty(frame.shape)

			frame = stream.get_frame()
			filename=f"{img_path}/{datetime.datetime.now().strftime('%c')}.jpg"
			info_KB, fram_diff = _extract_info_diff(prev_frame=prev_frame, frame=frame)

			if frame is not None and info_KB > det_thres:
				print(f"\ninfo differences in approx. KB: {info_KB:.2f}KB")

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
						_push_cat_ntfy(
							host = host,
							topic = topic,
							msg_dict = msg_dict,
							ntfy_user = ntfy_user,
							ntfy_pass = ntfy_pass,
							img_path = filename
						)

						print("Press ANY key to keep detecting and send message!")
						cv2.waitKey(suspend*1000)
						break

				if msg_activation != msg_n_init_act:
					msg_activation = msg_n_init_act

		except KeyboardInterrupt:
			break

		if cv2.waitKey(50) == ord('q'):
			break

	cv2.destroyAllWindows()
	cv2.waitKey(1)
	stream.stop()

import os

def chk_crt_pth(path:str = None):
  if path is None:
    pass
  if not os.path.exists(path):
    os.mkdir(path)
  elif not os.path.isdir(path):
    raise ValueError("path given exists and is not a directory.")

def _crt_main_env(proj_path):
  env_content = f"""
# webcam related information
USER=
PASSWORD=
URL=

# ntfy related information
NTFY_USER=
NTFY_PASS=
HOSTNAME=
TOPIC=

# Load model for object recognition
MODEL=

# path related, if IMG_SAVE_PATH is not set, save_img would be set to PROJ_PATH/saved_img 
PROJ_PATH={proj_path}
IMG_SAVE_PATH=

# notification/process suspend after activation (in seconds)
SUSPEND=
"""
  env_path = proj_path + "/.env"
  f = open(env_path, "w+")
  f.write(env_content)
  f.close()


def _crt_docker_env(proj_path):
  env_content = """
# time zone settings
TZ=

# base_url settings for ntfy
BASE_URL=

# UID:GID settings for system security
UID=1000
GID=1000

# NTFY path settings
NTFY_CACHE=./ntfy/cache
NTFY_ETC=./ntfy/etc

# tailscale path settings
TS_NTFY_CONFIG=./ts/ntfy/config
TS_NTFY_STATE=./ts/ntfy/state
"""
  env_path = proj_path + "/ntfy-docker/.env"
  f = open(env_path, "w+")
  f.write(env_content)
  f.close()


def create_envs():
  file_path = os.path.abspath(__file__)
  proj_path = '/'.join(file_path.split('/')[:-2])
  
  _crt_main_env(proj_path)
  _crt_docker_env(proj_path)

  print(f"Please fill in the information missing in .env file!")


if __name__ == "__main__":
  create_envs()
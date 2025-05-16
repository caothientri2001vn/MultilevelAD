#collapse-output
import gdown
# url = "https://drive.google.com/drive/folders/1HWFHKCprFzR7H7TYhrE-W7v4bz2Vc7Ia"
url = 'https://drive.google.com/drive/folders/1fvF1RFeOCWIraWhTUu71ZX1TX5Za8_kb'
gdown.download_folder(url, quiet=True, use_cookies=False)
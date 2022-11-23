import os
import gdown

url = 'https://drive.google.com/uc?id=1vc_IkhxhNfEeEbiFPHxt_AsDclDNW8d5'
output = 'peract_600k.zip'

gdown.download(url, output, quiet=False)
os.system(f'unzip {output}')
os.system(f'rm {output}')
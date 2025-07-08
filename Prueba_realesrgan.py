from PIL import Image
import torch
from py_real_esrgan.model import RealESRGAN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RealESRGAN(device, scale=4)
model.load_weights('RealESRGAN_x4plus.pth', download=False) # Adaptar el nombre del modelo en caso de ser diferente
img = Image.open('inputImages/multitud.jpg').convert('RGB')
sr = model.predict(img)
sr.save('output.png')

import os
import torch
import numpy as np
from PIL import Image
from basicsr.archs.rrdbnet_arch import RRDBNet
from Prueba_realesrgan import RealESRGANer

def upscale_image(input_path, output_path, model_path, device=None, tile=0, pre_pad=0, half=True):
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Crea arquitectura RRDBNet para escala ×4
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=23, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        tile=tile,
        pre_pad=pre_pad,
        half=half
    )
    img = Image.open(input_path).convert('RGB')
    img_np = np.array(img)
    output, _ = upsampler.enhance(img_np, outscale=4)
    out_img = Image.fromarray(output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_img.save(output_path)
    print(f"Imagen procesada guardada en: {output_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Upscale utilizando Real‑ESRGAN oficial")
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model', default='RealESRGAN_x4plus.pth')
    parser.add_argument('--tile', type=int, default=0, help="Tamaño de tile para evitar OOM")
    parser.add_argument('--half', action='store_true', help="FP16 para GPU compatibles")
    args = parser.parse_args()

    upscale_image(args.input, args.output, args.model, None, args.tile, 0, args.half)

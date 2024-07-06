import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
import io
from rembg import remove
from io import BytesIO
from load_img import fetch_images
import requests
from concurrent.futures import ProcessPoolExecutor
from typing import List
import os
import argparse

# Cartoonizer setup
_sess_options = ort.SessionOptions()
_sess_options.intra_op_num_threads = 15
MODEL_SESS = ort.InferenceSession(
    "cartoonizer.onnx", _sess_options, providers=["CPUExecutionProvider"]
)


# Remove background function
def remove_background(image: Image) -> Image:
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    no_bg_bytes = remove(img_byte_arr)
    no_bg_image = Image.open(BytesIO(no_bg_bytes))
    return no_bg_image


def preprocess_image(image: Image) -> np.ndarray:
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    image_rgb = image.convert("RGB")
    image_array = np.array(image_rgb)
    image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    h, w, c = np.shape(image_array)
    if min(h, w) < 1800:
        if h < w:
            h, w = int(1800 * h / w), 1800
        else:
            h, w = 1800, int(1800 * w / h)
    h, w = (h // 8) * 8, (w // 8) * 8
    image_array = cv2.resize(image_array, (w, h), interpolation=cv2.INTER_AREA)
    image_array = image_array.astype(np.float32) / 127.5 - 1
    return np.expand_dims(image_array, axis=0)


def inference(image: Image) -> Image:
    if image.mode == 'RGBA':
        original_alpha = image.split()[3]
    else:
        original_alpha = Image.new('L', image.size, 255)
    processed_image = preprocess_image(image)
    results = MODEL_SESS.run(None, {"input_photo:0": processed_image})
    output = (np.squeeze(results[0]) + 1.0) * 127.5
    output = np.clip(output, 0, 255).astype(np.uint8)
    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    cartoonized = Image.fromarray(output)
    original_alpha = original_alpha.resize(cartoonized.size)
    cartoonized.putalpha(original_alpha)
    return cartoonized


def cartoonize_image(image: Image, output_path, remove_bg=False):
    if remove_bg:
        image = remove_background(image)
    cartoonized_img = inference(image)
    cartoonized_img.save(output_path)


def load_image_from_url(url: str) -> Image:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "image/webp,image/apng,image/*,*/*;q=0.8"
    }
    response = requests.get(url, headers=headers)
    print(response)
    image = Image.open(io.BytesIO(response.content))
    return image


def process_image(index, image_url):
    input_image = load_image_from_url(image_url)

    # Cartoonize the image with background
    output_image_path_with_bg = f"cartoonized_image_with_bg_{index + 1}.png"
    cartoonize_image(input_image, output_image_path_with_bg)

    # Remove the background from the cartoonized image
    output_image_path_without_bg = f"cartoonized_image_without_bg_{index + 1}.png"
    cartoonize_image(input_image, output_image_path_without_bg, remove_bg=True)


def preprocess_batch(images: List[Image]) -> np.ndarray:
    batch = []
    for image in images:
        processed_image = preprocess_image(image)
        batch.append(processed_image)
    return np.stack(batch)


def inference_batch(batch: np.ndarray) -> List[Image]:
    results = MODEL_SESS.run(None, {"input_photo:0": batch})
    cartoonized_images = []
    for result in results:
        output = (np.squeeze(result) + 1.0) * 127.5
        output = np.clip(output, 0, 255).astype(np.uint8)
        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        cartoonized = Image.fromarray(output)
        cartoonized_images.append(cartoonized)
    return cartoonized_images


def cartoonize_batch(images: List[Image], output_paths: List[str], remove_bg=False):
    if remove_bg:
        images = [remove_background(image) for image in images]
    batch = preprocess_batch(images)
    cartoonized_images = inference_batch(batch)
    for cartoonized, path in zip(cartoonized_images, output_paths):
        cartoonized.save(path)


# Function to remove PNG images from a directory
def remove_png_images(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            file_path = os.path.join(directory, filename)
            if os.path.exists(file_path):
                os.remove(file_path)


def process_query_images(output_query):
    images = fetch_images(output_query)[:3]

    with ProcessPoolExecutor() as executor:
        executor.map(process_image, range(len(images)), images)


parser = argparse.ArgumentParser(description="Transforme les images en caricatures et supprime l'arrière-plan.")
parser.add_argument('--image_name', type=str, required=True, help="Nom de l'image.")

args = parser.parse_args()

process_query_images(args.image_name)

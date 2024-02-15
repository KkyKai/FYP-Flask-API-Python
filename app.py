from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import io
from PIL import Image

import numpy as np
import torch
import argparse
from argparse import Namespace
from torchvision import transforms
from torch.nn import functional as F
import torchvision
import matplotlib.pyplot as plt
import base64

import dlib
import os
import cv2

from models.util import save_image, load_image, visualize
from models.dualstylegan import DualStyleGAN
from models.psp import pSp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# os.chdir('DualStyleGAN')
CODE_DIR = 'models'
# os.chdir(f'{CODE_DIR}')
# device = 'cuda'
MODEL_DIR = os.path.join('models\checkpoint')
DATA_DIR = os.path.join('models\data')

MODEL_PATHS = {
    "encoder": {"name": 'encoder.pt'},
    "cartoon-G": {"name": 'generator-001700.pt'},
    "cartoon-S": {"name": 'refined_exstyle_code.npy'},
}

style_types = ['cartoon']
style_type = style_types[0]

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return "Hello, this is your Flask application running on Google Cloud Run!"

@app.route('/process_image', methods=['POST'])
def process_image():
    print(request.files, flush = True)

    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    try:
        number = request.form['number']
        print('Number:', number, flush=True)
    except KeyError:
        print('Number not found in form data', flush=True)

    image = request.files['image']
    print('Image:', image.filename, flush=True)
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    print(request.files, flush = True)

    # Process the image (e.g., pass it through your ML model)

    result_data = process_image_function(image, number)

    return jsonify(result_data)

transform = transforms.Compose(
[
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
]
)

# load DualStyleGAN
generator = DualStyleGAN(1024, 512, 8, 2, res_index=6)
generator.eval()
# ckpt = torch.load(os.path.join(MODEL_DIR, style_type, 'generator-001700.pt'), map_location=lambda storage, loc: storage)
script_dir = os.path.dirname(os.path.abspath(__file__))
ckpt = torch.load(os.path.join(script_dir, 'models', 'checkpoint', 'cartoon', 'generator-001700.pt'), map_location=lambda storage, loc: storage)
generator.load_state_dict(ckpt["g_ema"])
generator = generator.to(device)

# load encoder
model_path = os.path.join(script_dir,'models' , 'checkpoint', 'encoder.pt')
ckpt = torch.load(model_path, map_location='cpu')
opts = ckpt['opts']
opts['checkpoint_path'] = model_path
opts = Namespace(**opts)
opts.device = device
encoder = pSp(opts)
encoder.eval()
encoder = encoder.to(device)

# load extrinsic style code
exstyles = np.load(os.path.join(script_dir,'models' , 'checkpoint', style_type, MODEL_PATHS[style_type+'-S']["name"]), allow_pickle='TRUE').item()

print('Model successfully loaded!')


def run_alignment(image_path):
    from models.align_all_parallel import align_face
    modelname = os.path.join(script_dir,'models', 'checkpoint', 'shape_predictor_68_face_landmarks.dat')
    predictor = dlib.shape_predictor(modelname)
    aligned_image = align_face(filepath=image_path, predictor=predictor)
    return aligned_image

if_align_face = True

def process_image_function(image, number):
    # Save the uploaded image to a temporary file
    temp_image_path = f'models/holder/{image.filename}'
    image.save(temp_image_path)

    # current_dir = os.path.dirname(os.path.abspath(__file__))

    convert_and_save_cv2('models/holder/', 'models/holder/')

    # Load the image using util.py load_image function to get img tensor
    # original_image = load_image(temp_image_path)

    if if_align_face:
        I = transform(run_alignment(temp_image_path)).unsqueeze(dim=0).to(device)
    else:
        I = F.adaptive_avg_pool2d(load_image(temp_image_path).to(device), 256)

    # try to load the style image
    if number == '1':
        style_id = 28 # tangled_009_04680_optimized.jpg
    elif number == '2':
        style_id = 53 #turningred_002_00624_optimized.jpg
    elif number == '3':
        style_id = 126 #Cartoons_00448_01.jpg (How to train your dragon)
    elif number == '4':
        style_id = 206 #Cartoons_00170_03.jpg (Rapunzel)
    else:
        style_id = 99 # Cartoons_00511_01.jpg

    stylename = list(exstyles.keys())[style_id]
    stylepath = os.path.join(script_dir, 'models', 'data', 'cartoon', 'images', 'train', stylename)

    print('loading %s'%stylepath)
    if os.path.exists(stylepath):
        S = load_image(stylepath)
    else:
        print('%s is not found'%stylename)
    print(type(number), flush=True)
    if number == '1':
        with torch.no_grad():
            img_rec, instyle = encoder(I, randomize_noise=False, return_latents=True,
                                    z_plus_latent=True, return_z_plus_latent=True, resize=False)
            img_rec = torch.clamp(img_rec.detach(), -1, 1)

            latent = torch.tensor(exstyles[stylename]).repeat(2,1,1).to(device)
            # latent[0] for both color and structrue transfer and latent[1] for only structrue transfer
            exstyle = generator.generator.style(latent.reshape(latent.shape[0]*latent.shape[1], latent.shape[2])).reshape(latent.shape)

            # deactivate color-related layers by setting w_c = 0
            img_gen, _ = generator([instyle], exstyle[0:1], z_plus_latent=True,
                                    truncation=0.7, truncation_latent=0, use_res=True, interp_weights=[0.7]*7+[0.8]*11)
            img_gen = torch.clamp(img_gen.detach(), -1, 1)
            
            img_gen = img_gen.cpu()
            save_image (img_gen, f'models/holder/output_{image.filename}')
            img1_np = (img_gen.permute(0, 2, 3, 1).cpu().numpy() + 1) / 2.0 * 255.0
            img1_np = img1_np.astype('uint8')[0]
            img1_np_bgr = cv2.cvtColor(img1_np, cv2.COLOR_BGR2RGB)

            '''results = []
            for i in range(6): # change weights of structure codes
                if i == 5:
                    i = 3.5
                w = [i/5.0]*7+[0]*11

                img_gen, _ = generator([instyle], exstyle[0:1], z_plus_latent=True,
                                        truncation=0.7, truncation_latent=0, use_res=True, interp_weights=w)
                img_gen = torch.clamp(F.adaptive_avg_pool2d(img_gen.detach(), 1024), -1, 1)
                results += [img_gen] '''

            results = []

            for i in range(6):  # iterate over rows
                for j in range(6):  # iterate over columns
                    # Change weights of structure codes
                    w = [i / 5.0] * 7 + [j / 5.0] * 11

                    img_gen, _ = generator([instyle], exstyle[0:1], z_plus_latent=True,
                                            truncation=0.7, truncation_latent=0, use_res=True, interp_weights=w)
                    img_gen = torch.clamp(F.adaptive_avg_pool2d(img_gen.detach(), 1024), -1, 1)

                    # Store the left diagonal element in results
                    if i == j:
                        results += [img_gen]
                
    elif number == '2':
        with torch.no_grad():
            img_rec, instyle = encoder(I, randomize_noise=False, return_latents=True,
                                    z_plus_latent=True, return_z_plus_latent=True, resize=False)
            img_rec = torch.clamp(img_rec.detach(), -1, 1)

            latent = torch.tensor(exstyles[stylename]).to(device)
            # latent[0] for both color and structrue transfer and latent[1] for only structrue transfer
            # latent[1,7:18] = instyle[0,7:18]
            exstyle = generator.generator.style(latent.reshape(latent.shape[0]*latent.shape[1], latent.shape[2])).reshape(latent.shape)

            img_gen, _ = generator([instyle], exstyle, z_plus_latent=True,
                                truncation=0.7, truncation_latent=0, use_res=True, interp_weights=[0.6]*7+[0.8]*11)
            img_gen = torch.clamp(img_gen.detach(), -1, 1)

            img_gen = img_gen.cpu()
            save_image (img_gen, f'models/holder/output_{image.filename}')
            img1_np = (img_gen.permute(0, 2, 3, 1).cpu().numpy() + 1) / 2.0 * 255.0
            img1_np = img1_np.astype('uint8')[0]
            img1_np_bgr = cv2.cvtColor(img1_np, cv2.COLOR_BGR2RGB)
            
            results = []
            
            for i in range(6):  # iterate over rows
                for j in range(6):  # iterate over columns
                    # Change weights of structure codes
                    w = [i / 5.0] * 7 + [j / 5.0] * 11

                    img_gen, _ = generator([instyle], exstyle[0:1], z_plus_latent=True,
                                            truncation=0.7, truncation_latent=0, use_res=True, interp_weights=w)
                    img_gen = torch.clamp(F.adaptive_avg_pool2d(img_gen.detach(), 1024), -1, 1)

                    # Store the left diagonal element in results
                    if i == j:
                        results += [img_gen]


    elif number == '3':
        with torch.no_grad():
            img_rec, instyle = encoder(I, randomize_noise=False, return_latents=True,
                                    z_plus_latent=True, return_z_plus_latent=True, resize=False)
            img_rec = torch.clamp(img_rec.detach(), -1, 1)

            latent = torch.tensor(exstyles[stylename]).to(device)
            # latent[0] for both color and structrue transfer and latent[1] for only structrue transfer
            # latent[1,7:18] = instyle[0,7:18]
            exstyle = generator.generator.style(latent.reshape(latent.shape[0]*latent.shape[1], latent.shape[2])).reshape(latent.shape)

            img_gen, _ = generator([instyle], exstyle, z_plus_latent=True,
                                truncation=0.7, truncation_latent=0, use_res=True, interp_weights=[0.5]*7+[0.9]*11)
            img_gen = torch.clamp(img_gen.detach(), -1, 1)
            
            img_gen = img_gen.cpu()
            save_image (img_gen, f'models/holder/output_{image.filename}')
            img1_np = (img_gen.permute(0, 2, 3, 1).cpu().numpy() + 1) / 2.0 * 255.0
            img1_np = img1_np.astype('uint8')[0]
            img1_np_bgr = cv2.cvtColor(img1_np, cv2.COLOR_BGR2RGB)
            
            results = []

            for i in range(5):  # iterate over rows
                for j in range(5):  # iterate over columns
                    # Change weights of structure codes
                    w = [i / 5.0] * 7 + [j / 5.0] * 11

                    img_gen, _ = generator([instyle], exstyle[0:1], z_plus_latent=True,
                                            truncation=0.7, truncation_latent=0, use_res=True, interp_weights=w)
                    img_gen = torch.clamp(F.adaptive_avg_pool2d(img_gen.detach(), 1024), -1, 1)

                    # Store the left diagonal element in results
                    if i == j:
                        results += [img_gen]
    else:
        with torch.no_grad():
            img_rec, instyle = encoder(I, randomize_noise=False, return_latents=True,
                                    z_plus_latent=True, return_z_plus_latent=True, resize=False)
            img_rec = torch.clamp(img_rec.detach(), -1, 1)

            latent = torch.tensor(exstyles[stylename]).repeat(2,1,1).to(device)
            # latent[0] for both color and structrue transfer and latent[1] for only structrue transfer
            latent[1,7:18] = instyle[0,7:18]
            exstyle = generator.generator.style(latent.reshape(latent.shape[0]*latent.shape[1], latent.shape[2])).reshape(latent.shape)

            # deactivate color-related layers by setting w_c = 0
            img_gen, _ = generator([instyle], exstyle[0:1], z_plus_latent=True,
                                    truncation=0.7, truncation_latent=0, use_res=True, interp_weights=[1]*7+[1]*11)
            img_gen = torch.clamp(img_gen.detach(), -1, 1)

            img_gen = img_gen.cpu()
            save_image (img_gen, f'models/holder/output_{image.filename}')
            img1_np = (img_gen.permute(0, 2, 3, 1).cpu().numpy() + 1) / 2.0 * 255.0
            img1_np = img1_np.astype('uint8')[0]
            img1_np_bgr = cv2.cvtColor(img1_np, cv2.COLOR_BGR2RGB)

            results = []

            for j in range(0, 6):
                # traversing vertically
                for i in range(0, 6 - j):
                    w = [i / (6 - 1) for _ in range(7)] + [j / (6 - 1) for _ in range(11)]
                    img_gen, _ = generator([instyle], exstyle[0:1], z_plus_latent=True,
                                            truncation=0.7, truncation_latent=0, use_res=True, interp_weights=w)
                    img_gen = torch.clamp(F.adaptive_avg_pool2d(img_gen.detach(), 1024), -1, 1)
                    results.append(img_gen)

                # traverse horizontally
                for k in range(j + 1, 6):
                    w = [(6 - 1 - j) / (6 - 1) for _ in range(7)] + [k / (6 - 1) for _ in range(11)]
                    img_gen, _ = generator([instyle], exstyle[0:1], z_plus_latent=True,
                                            truncation=0.7, truncation_latent=0, use_res=True, interp_weights=w)
                    img_gen = torch.clamp(F.adaptive_avg_pool2d(img_gen.detach(), 1024), -1, 1)
                    results.append(img_gen)

            # Extract the images for visualization
            first_row_images = results[1:6]
            second_row_images = results[6:6+5]

            # Combine first_row_images and second_row_images
            combined_images = first_row_images + second_row_images

            input_np = (I.permute(0, 2, 3, 1).cpu().numpy() + 1) / 2.0 * 255.0
            input_np = input_np.astype('uint8')[0]
            img_resized = cv2.resize(input_np, (1024, 1024))
            input_bgr = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

            # Create interpolation video
            output_video_path = 'models/holder/interpolation_video.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Change the codec to mp4v for MP4
            fps = 4.0  # Set the frames per second
            video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (1024, 1024))

            # Include resized source image as the first frame
            video_writer.write(input_bgr)

            # Convert PyTorch tensors to NumPy arrays and save frames
            for img_tensor in combined_images:
                img_np = (img_tensor.permute(0, 2, 3, 1).cpu().numpy() + 1) / 2.0 * 255.0
                img_np = img_np.astype('uint8')[0]
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)  # Convert from RGB to BGR
                video_writer.write(img_bgr)

            # Release video writer
            video_writer.release()

            print(f'Interpolation video saved at {output_video_path}')

    if number == '1' or number == '2' or number == '3':
        input_np = (I.permute(0, 2, 3, 1).cpu().numpy() + 1) / 2.0 * 255.0
        input_np = input_np.astype('uint8')[0]
        img_resized = cv2.resize(input_np, (1024, 1024))
        input_bgr = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        # Create interpolation video
        output_video_path = 'models/holder/interpolation_video.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Change the codec to mp4v for MP4
        fps = 4.0  # Set the frames per second
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (1024, 1024))

        # Include resized source image as the first frame
        video_writer.write(input_bgr)

        # Convert PyTorch tensors to NumPy arrays and save frames
        for img_tensor in results:
            img_np = (img_tensor.permute(0, 2, 3, 1).cpu().numpy() + 1) / 2.0 * 255.0
            img_np = img_np.astype('uint8')[0]
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)  # Convert from RGB to BGR
            video_writer.write(img_bgr)

        # Release video writer
        video_writer.release()

        print(f'Interpolation video saved at {output_video_path}')  

    # return img

    # Convert the processed image to bytes
    img_byte_array = cv2.imencode('.jpg', img1_np_bgr)[1].tobytes()

    # Convert the video to bytes
    with open(output_video_path, 'rb') as video_file:
        video_bytes = video_file.read()

    return {'image': base64.b64encode(img_byte_array).decode('utf-8'), 'video': base64.b64encode(video_bytes).decode('utf-8')}

def convert_and_save_cv2(folder_path, output_folder):
    # Get a list of all files in the folder
    file_list = os.listdir(folder_path)

    print(file_list, flush=True)

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through each file in the folder
    for file_name in file_list:
        # Construct the full file path
        file_path = os.path.join(folder_path, file_name)

        # Check if the file is an image (supports png, jpg, jpeg, and bmp)
        if not file_name.lower().endswith('.mp4'):
            # Read the image using cv2
            img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

            # Convert RGBA to RGB
            if img.shape[2] == 4:  # Check if the image has an alpha channel
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

            # Save the converted image to the output folder
            output_path = os.path.join(output_folder, file_name)
            cv2.imwrite(output_path, img)

        else:
            print(f"Ignoring MP4 file: {file_name}")


if __name__ == '__main__':
    # port = int(os.environ.get('PORT', 8080))
    # app.run(host='0.0.0.0', port=port)
    # app.run(host = 'localhost', port = 5000, debug=True)

    app.run(port=int(os.environ.get("PORT", 8080)),host='0.0.0.0',debug=True)

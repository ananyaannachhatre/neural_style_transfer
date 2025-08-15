## 🎨 Neural Style Transfer with TensorFlow
This project implements Neural Style Transfer (NST) using TensorFlow 2 and a pretrained VGG19 network.
It blends the content of one image with the style of another, producing a completely new, stylized image.

The code is modularized for clarity, reusability, and easy customization.

## 📂 Project Structure
- main.py                 # Entry point to run style transfer pipeline
- image_loader.py         # Loads and preprocesses images
- style_content_model.py  # Defines VGG19 feature extraction model
- style_transfer.py       # Implements the training & optimization loop
- smoothing.py            # Post-processing (smoothing / denoising)

## ✨ Features
- Load content/style images from local files or URLs.
- Uses VGG19 pretrained on ImageNet for feature extraction.
- Separates content and style representations from different network layers.
- Computes Gram matrices for multi-scale style matching.
- Optimizes the generated image with Adam optimizer.
- Optional post-processing for smoother and cleaner results.
- Fully decomposed into reusable modules.

## ⚙️ Installation
1. Clone this repository
git clone https://github.com/yourusername/neural-style-transfer.git
cd neural-style-transfer
2. Install dependencies
pip install -r requirements.txt

## 📦 Requirements
requirements.txt contains:
- tensorflow>=2.8.0
- numpy
- matplotlib
- opencv-python
- requests
- scikit-image
  
## 🚀 Usage
In main.py, change: Content & Style image paths
- content_url = "/path/to/your/content.jpg"
- style_url = "/path/to/your/style.jpg"
Training params
- epochs = 200
- steps_per_epoch = 100
- style_weight = 1e-2
- content_weight = 1e4

🖼 Workflow
Here’s how the system works step-by-step:
- Image Loading (image_loader.py)
Reads content and style images from local disk/URL. Resizes to a max dimension while maintaining aspect ratio. Normalizes pixel values.

- VGG19 Feature Extraction (style_content_model.py)
Loads pretrained VGG19 without the fully connected layers. Extracts intermediate feature maps from selected layers. Choosing Layers for Content & Style

Content Layer: 
- block5_conv2 → Deep enough to capture semantic structure and object layout.

Style Layers:
- block1_conv1 → very fine details (edges, colors)
- block2_conv1 → small textures & patterns
- block3_conv1, block4_conv1 → mid-level structures
- block5_conv1 → large abstract style patterns

These give multi-scale style representation via Gram matrices.

- Style Transfer Optimization (style_transfer.py)
Initializes generated image as the content image. Computes content loss (MSE in content layer features). Computes style loss (MSE in Gram matrices) across style layers.
(Minimizes: total_loss = content_weight * content_loss + style_weight * style_loss). Uses Adam optimizer with frequent clipping to keep pixel range valid.

- Post-Processing (smoothing.py)
Uses Total Variation Denoising to reduce edge noise for a cleaner image.

## 🎯 Why These Specific VGG19 Layers?
- Content: We want high-level structures (shapes, object locations) without the low-level pixel noise — hence block5_conv2 from deep in the network.
- Style: Style is multi-scale, fine pixel textures and broad color & pattern distributions both matter. So we take multiple layers from low, mid, and high positions in VGG19 to capture everything from small strokes to large color blobs.

This approach is based on the original Gatys et al. 2015 neural style transfer paper.

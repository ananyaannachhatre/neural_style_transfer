import tensorflow as tf
from image_loader import load_image
from style_transfer import style_transfer
from smoothing import smooth_image

# Define layers
content_layers = ['block5_conv2']
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

content_url = "D:\VS Python\neural_transfer_image\images\neuraltransferimagecontent1.jpg"
style_url = "D:\VS Python\neural_transfer_image\images\neuraltransferimagestyle.jpg"

content_image = load_image(content_url)
style_image = load_image(style_url)

# Style transfer parameters
epochs = 200
steps_per_epoch = 100
style_weight = 1e-2
content_weight = 1e4

generated_image = style_transfer(content_image, style_image, style_layers, content_layers, epochs, steps_per_epoch, style_weight, content_weight)
smooth_image(generated_image)

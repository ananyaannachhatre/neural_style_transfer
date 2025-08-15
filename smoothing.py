from skimage.restoration import denoise_tv_chambolle
import numpy as np
import matplotlib.pyplot as plt

def smooth_image(generated_image):
    generated_image_np = generated_image.numpy()
    if generated_image_np.max() > 1:
        generated_image_np = generated_image_np / 255.0
    smoothed_img = denoise_tv_chambolle(generated_image_np, weight=0.13, channel_axis=-1)
    smoothed_img = (smoothed_img * 255).astype(np.uint8)
    plt.imshow(np.squeeze(smoothed_img, 0))
    plt.axis('off')
    plt.title('Smoothed Image')
    plt.show()
    return smoothed_img

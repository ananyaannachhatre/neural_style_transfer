import tensorflow as tf
import requests

def load_image(img_path, max_dim=512):
    """Load and preprocess image from URL or local path"""
    if img_path.startswith('http'):
        response = requests.get(img_path)
        img = tf.image.decode_image(response.content, channels=3)
    else:
        img = tf.io.read_file(img_path)
        img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    scale = max_dim / max(shape)
    new_shape = tf.cast(shape * scale, tf.int32)
    img = tf.image.resize(img, new_shape)
    return img[tf.newaxis, :]

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def style_transfer(content_image, style_image, style_layers, content_layers, epochs, steps_per_epoch, style_weight, content_weight):
    from style_content_model import StyleContentModel

    extractor = StyleContentModel(style_layers, content_layers)
    style_targets = extractor(style_image)['style']
    content_targets = extractor(content_image)['content']
    generated_image = tf.Variable(content_image)
    opt = tf.optimizers.Adam(learning_rate=0.02)

    for epoch in range(epochs):
        for step in range(steps_per_epoch):
            with tf.GradientTape() as tape:
                outputs = extractor(generated_image)
                content_outputs = outputs['content']
                style_outputs = outputs['style']
                content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name])**2)
                                         for name in content_outputs.keys()])
                style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name])**2)
                                       for name in style_outputs.keys()])
                total_loss = content_weight * content_loss + style_weight * style_loss
            gradients = tape.gradient(total_loss, generated_image)
            opt.apply_gradients([(gradients, generated_image)])
            generated_image.assign(tf.clip_by_value(generated_image, 0.0, 1.0))

        print(f"Epoch {epoch+1}, Total Loss: {total_loss.numpy()}")

    plt.imshow(np.squeeze(generated_image.read_value(), 0))
    plt.axis('off')
    plt.show()
    return generated_image

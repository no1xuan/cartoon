import os
import cv2
import numpy as np
import tensorflow._api.v2.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
import network
import guided_filter
from tqdm import tqdm



def resize_crop(image):
    h, w, c = np.shape(image)
    if min(h, w) > 720:
        if h > w:
            h, w = int(720*h/w), 720
        else:
            h, w = 720, int(720*w/h)
    image = cv2.resize(image, (w, h),
                       interpolation=cv2.INTER_AREA)
    h, w = (h//8)*8, (w//8)*8
    image = image[:h, :w, :]
    return image
    

def cartoonize(load_folder, save_folder, model_path):
    input_photo = tf.placeholder(tf.float32, [1, None, None, 3])
    network_out = network.unet_generator(input_photo)
    final_out = guided_filter.guided_filter(input_photo, network_out, r=1, eps=5e-3)

    all_vars = tf.trainable_variables()
    gene_vars = [var for var in all_vars if 'generator' in var.name]
    saver = tf.train.Saver(var_list=gene_vars)
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    sess.run(tf.global_variables_initializer())
    saver.restore(sess, tf.train.latest_checkpoint(model_path))
    name_list = os.listdir(load_folder)
    for name in tqdm(name_list):
        try:
            load_path = os.path.join(load_folder, name)
            save_path = os.path.join(save_folder, name)
            image = cv2.imread(load_path)
            image = resize_crop(image)
            batch_image = image.astype(np.float32)/127.5 - 1
            batch_image = np.expand_dims(batch_image, axis=0)
            output = sess.run(final_out, feed_dict={input_photo: batch_image})
            output = (np.squeeze(output)+1)*127.5
            output = np.clip(output, 0, 255).astype(np.uint8)
            cv2.imwrite(save_path, output)
        except:
            print('cartoonize {} failed'.format(load_path))


def cartoonize_inference(input_image, model_path):
    input_photo = tf.placeholder(tf.float32, [1, None, None, 3])
    network_out = network.unet_generator(input_photo)
    final_out = guided_filter.guided_filter(input_photo, network_out, r=1, eps=5e-3)

    all_vars = tf.trainable_variables()
    gene_vars = [var for var in all_vars if 'generator' in var.name]
    saver = tf.train.Saver(var_list=gene_vars)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    sess.run(tf.global_variables_initializer())
    saver.restore(sess, tf.train.latest_checkpoint(model_path))

    image = resize_crop(input_image)
    batch_image = image.astype(np.float32) / 127.5 - 1
    batch_image = np.expand_dims(batch_image, axis=0)
    output = sess.run(final_out, feed_dict={input_photo: batch_image})
    output = (np.squeeze(output) + 1) * 127.5
    output = np.clip(output, 0, 255).astype(np.uint8)
    return output


class Cartoonizer():
    def __init__(self, model_path="./saved_models"):
        self.model_path = model_path

    def load_model(self):
        input_photo = tf.placeholder(tf.float32, [1, None, None, 3])
        network_out = network.unet_generator(input_photo)
        final_out = guided_filter.guided_filter(input_photo, network_out, r=1, eps=5e-3)

        all_vars = tf.trainable_variables()
        gene_vars = [var for var in all_vars if 'generator' in var.name]
        saver = tf.train.Saver(var_list=gene_vars)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        sess.run(tf.global_variables_initializer())
        saver.restore(sess, tf.train.latest_checkpoint(self.model_path))

        self.sess = sess
        self.final_out= final_out
        self.input_photo = input_photo

    def inference(self, input_image):
        image = resize_crop(input_image)
        batch_image = image.astype(np.float32) / 127.5 - 1
        batch_image = np.expand_dims(batch_image, axis=0)
        output = self.sess.run(self.final_out, feed_dict={self.input_photo: batch_image})
        output = (np.squeeze(output) + 1) * 127.5
        output = np.clip(output, 0, 255).astype(np.uint8)
        return output


if __name__ == '__main__':
    image = cv2.imread("test_images/china6.jpg")
    model = Cartoonizer()
    model.load_model()

    output = model.inference(image)
    cv2.imwrite("actress2.jpg", output)
import tensorflow as tf
import numpy as np
import os
from PIL import Image
from PIL import ImageOps

# generate image
def generator(inp, keep_prob=0.5, reuse=False):
    with tf.variable_scope("G", reuse=reuse):
        conv1 = tf.layers.conv2d(inp, 64, [4, 4], strides=(2, 2), padding='same',
                                 activation=tf.nn.leaky_relu, kernel_regularizer=tf.layers.batch_normalization)
        conv2 = tf.layers.conv2d(conv1, 128, [4, 4], strides=(2, 2), padding='same',
                                 activation=tf.nn.leaky_relu, kernel_regularizer=tf.layers.batch_normalization)
        conv3 = tf.layers.conv2d(conv2, 256, [4, 4], strides=(2, 2), padding='same',
                                 activation=tf.nn.leaky_relu, kernel_regularizer=tf.layers.batch_normalization)
        conv4 = tf.layers.conv2d(conv3, 512, [4, 4], strides=(2, 2), padding='same',
                                 activation=tf.nn.leaky_relu, kernel_regularizer=tf.layers.batch_normalization)
        conv5 = tf.layers.conv2d(conv4, 512, [4, 4], strides=(2, 2), padding='same',
                                 activation=tf.nn.leaky_relu, kernel_regularizer=tf.layers.batch_normalization)
        conv6 = tf.layers.conv2d(conv5, 512, [4, 4], strides=(2, 2), padding='same',
                                 activation=tf.nn.leaky_relu, kernel_regularizer=tf.layers.batch_normalization)
        conv7 = tf.layers.conv2d(conv6, 512, [4, 4], strides=(2, 2), padding='same',
                                 activation=tf.nn.leaky_relu, kernel_regularizer=tf.layers.batch_normalization)
        conv8 = tf.layers.conv2d(conv7, 512, [4, 4], strides=(2, 2), padding='same', activation=tf.nn.leaky_relu)
        dconv1 = tf.layers.conv2d_transpose(conv8, 512, [4, 4], strides=(2, 2), padding='same',
                                            activation=tf.nn.relu, kernel_regularizer=tf.layers.batch_normalization)
        dconv1 = tf.nn.dropout(dconv1, keep_prob)
        dconv2 = tf.layers.conv2d_transpose(tf.concat([dconv1, conv7], 3), 512, [4, 4], strides=(2, 2), padding='same',
                                            activation=tf.nn.relu, kernel_regularizer=tf.layers.batch_normalization)
        dconv2 = tf.nn.dropout(dconv2, keep_prob)
        dconv3 = tf.layers.conv2d_transpose(tf.concat([dconv2, conv6], 3), 512, [4, 4], strides=(2, 2), padding='same',
                                            activation=tf.nn.relu, kernel_regularizer=tf.layers.batch_normalization)
        dconv3 = tf.nn.dropout(dconv3, keep_prob)
        dconv4 = tf.layers.conv2d_transpose(tf.concat([dconv3, conv5], 3), 512, [4, 4], strides=(2, 2), padding='same',
                                            activation=tf.nn.relu, kernel_regularizer=tf.layers.batch_normalization)
        dconv5 = tf.layers.conv2d_transpose(tf.concat([dconv4, conv4], 3), 256, [4, 4], strides=(2, 2), padding='same',
                                            activation=tf.nn.relu, kernel_regularizer=tf.layers.batch_normalization)
        dconv6 = tf.layers.conv2d_transpose(tf.concat([dconv5, conv3], 3), 128, [4, 4], strides=(2, 2), padding='same',
                                            activation=tf.nn.relu, kernel_regularizer=tf.layers.batch_normalization)
        dconv7 = tf.layers.conv2d_transpose(tf.concat([dconv6, conv2], 3), 64, [4, 4], strides=(2, 2), padding='same',
                                            activation=tf.nn.relu, kernel_regularizer=tf.layers.batch_normalization)
        dconv8 = tf.layers.conv2d_transpose(dconv7, 3, [4, 4], strides=(2, 2), padding='same')
        o = tf.nn.sigmoid(dconv8)
    return o


# discriminate image
def discriminator(inp, label_map, reuse=False):
    with tf.variable_scope("D", reuse=reuse):
        conv1_1 = tf.layers.conv2d(inp, 64, [4, 4], strides=(2, 2), padding='same',
                                   activation=tf.nn.relu, kernel_regularizer=tf.layers.batch_normalization)
        conv1_2 = tf.layers.conv2d(label_map, 64, [4, 4], strides=(2, 2), padding='same',
                                   activation=tf.nn.relu, kernel_regularizer=tf.layers.batch_normalization)
        conv1 = tf.concat([conv1_1, conv1_2], 3)
        conv2 = tf.layers.conv2d(conv1, 128, [4, 4], strides=(2, 2), padding='same',
                                 activation=tf.nn.relu, kernel_regularizer=tf.layers.batch_normalization)
        conv3 = tf.layers.conv2d(conv2, 256, [4, 4], strides=(2, 2), padding='same',
                                 activation=tf.nn.relu, kernel_regularizer=tf.layers.batch_normalization)
        conv4 = tf.layers.conv2d(conv3, 512, [4, 4], strides=(2, 2), padding='same',
                                 activation=tf.nn.relu, kernel_regularizer=tf.layers.batch_normalization)
        conv5 = tf.layers.conv2d(conv4, 1, [4, 4], strides=(2, 2), padding='same')
        o = tf.nn.sigmoid(conv5)
    return o


# load data from directory
def load_data(img_dir, train=False):
    imgs_name = os.listdir(img_dir)
    img_map = np.zeros([len(imgs_name), 256, 256, 3], 'float16')
    label_map = np.zeros([len(imgs_name), 256, 256, 3], 'float16')
    for i in range(len(imgs_name)):
        img_and_label = Image.open(img_dir + str(i+1) + '.jpg')
        img_and_label = img_and_label.resize((256*2, 256))
        img_and_label = np.array(img_and_label)
        img, label = np.split(img_and_label, 2, 1)
        # augment image for training
        if train:
            img = Image.fromarray(img)
            label = Image.fromarray(label)
            # flip
            if np.random.rand() > 0.5:
                img = ImageOps.mirror(img)
                label = ImageOps.mirror(label)
            # crop
            img = img.resize((256 + 12, 256 + 12))
            label = label.resize((256 + 12, 256 + 12))
            x = np.random.randint(0, 12)
            y = np.random.randint(0, 12)
            img = img.crop((x, y, x + 256, y + 256))
            label = label.crop((x, y, x + 256, y + 256))
            # rotate
            z = np.random.randint(0, 4)
            img = img.rotate(90*z)
            label = label.rotate(90 * z)
        img_map[i] = np.array(img)
        label_map[i] = np.array(label)
    img_map = img_map / 255
    label_map = label_map / 255
    return img_map, label_map


def main():

    train_path = 'maps/train/'
    val_path = 'maps/val/'
    # make folder
    if not os.path.exists('p2p_img/'):
        os.makedirs('p2p_img/')

    # input placeholder
    img_inp = tf.placeholder(tf.float32, [None, 256, 256, 3])
    label_inp = tf.placeholder(tf.float32, [None, 256, 256, 3])
    keep_prob = tf.placeholder(tf.float32)

    # generate image
    generated_sample = generator(label_inp, keep_prob)
    # discriminate real and fake
    d_real = discriminator(img_inp, label_inp)
    d_gen = discriminator(generated_sample, label_inp, True)

    # loss for input image and generated image
    d_loss = tf.reduce_mean(-tf.log(d_real) - tf.log(1 - d_gen))
    g_loss = tf.reduce_mean(-tf.log(d_gen)) + tf.reduce_mean(tf.abs(generated_sample - img_inp))

    # variables for discriminator and generator
    t_vars = tf.trainable_variables()
    d_var = [var for var in t_vars if 'D' in var.name]
    g_var = [var for var in t_vars if 'G' in var.name]

    # optimize discriminator and generator
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        d_optim = tf.train.AdamOptimizer(2e-6).minimize(d_loss, var_list=d_var)
        g_optim = tf.train.AdamOptimizer(2e-4).minimize(g_loss, var_list=g_var)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        init.run()
        # load val data
        img_val, label_val = load_data(val_path)

        # training network
        for step in range(1000):
            batch_size = 1
            # load training data
            if step % 5 == 0:
                img_map, label_map = load_data(train_path, True)
            # feed training data
            for ite in range(img_map.shape[0] // batch_size):
                img_np = img_map[ite*batch_size:(ite+1)*batch_size]
                label_np = label_map[ite * batch_size:(ite + 1) * batch_size]
                _, d_loss_p = sess.run([d_optim, d_loss], feed_dict={img_inp: img_np, label_inp: label_np, keep_prob: 0.5})
                _, g_loss_p = sess.run([g_optim, g_loss], feed_dict={img_inp: img_np, label_inp: label_np, keep_prob: 0.5})
                print(str(step) + '/' + str(ite))
                print('d_loss: ' + str(d_loss_p))
                print('g_loss: ' + str(g_loss_p))

            # generate image
            batch_size = 4
            img_np = img_val[0:batch_size]
            label_np = label_val[0:batch_size]
            p_samples = sess.run(generated_sample, feed_dict={label_inp: label_np, keep_prob: 1.0})
            # save image
            p_samples = p_samples * 255
            p_samples = p_samples.astype('uint8')
            img_np = img_np * 255
            img_np = img_np.astype('uint8')
            im = Image.fromarray(np.concatenate([img_np[0], p_samples[0],
                                                 img_np[1], p_samples[1],
                                                 img_np[2], p_samples[2],
                                                 img_np[3], p_samples[3]], 1))
            im.save("p2p_img/" + str(int(step + 1)) + ".jpg")


if __name__ == '__main__':
    main()

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import os
import time
import glob
import matplotlib.pyplot as plt
from IPython.display import clear_output

generated_image_dir = '/data3/dongjy/Celeb_Ablation/photos_test'
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

strategy = tf.distribute.MirroredStrategy()

#generated_image_dir = './photos_test'
if not os.path.exists(generated_image_dir):
    os.makedirs(generated_image_dir)

PATH = '/data3/dongjy/Celeb_Ablation/photos/'
#PATH = '/home/jiayuan/Ablation_study/photos/'

BUFFER_SIZE = 400
BATCH_SIZE_PER_REPLICA = 64
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
IMG_WIDTH = 256
IMG_HEIGHT = 256
EPOCHS = 60

def get_dataset(training = True):
  if training == True:
    input_paths = glob.glob(PATH+'train/*/*')
  else:
    input_paths = glob.glob(PATH+'val/*/*')

  if len(input_paths) == 0:
    raise Exception("input_dir contains no image files")

  def get_name(path):
    name, _ = os.path.splitext(os.path.basename(path))
    return name

  # if the image names are numbers, sort by the value rather than asciibetically
  # having sorted inputs means that the outputs are sorted in test mode
  if all(get_name(path).isdigit() for path in input_paths):
    input_paths = sorted(input_paths, key=lambda path: int(get_name(path)))
  else:
    input_paths = sorted(input_paths)

  all_image_paths = input_paths
  image_count = len(all_image_paths)

  label_names = []
  for item in list(glob.glob(PATH + 'train/*')):
    label_names.append(os.path.basename(item))
  label_names = sorted(label_names)

  label_to_index = dict((name, index) for index,name in enumerate(label_names))

  all_image_labels = [label_to_index[os.path.basename(os.path.dirname(path))] for path in all_image_paths]

  return all_image_paths, all_image_labels

def load(image_file):
  image = tf.io.read_file(image_file)
  image = tf.image.decode_png(image)

  w = tf.shape(image)[1]

  w = w // 2
  real_image = image[:, :w, :]
  input_image = image[:, w:, :]

  input_image = tf.cast(input_image, tf.float32)
  real_image = tf.cast(real_image, tf.float32)

  return input_image, real_image

def resize(input_image, real_image, height, width):
  input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  real_image = tf.image.resize(real_image, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  return input_image, real_image

def random_crop(input_image, real_image):
  stacked_image = tf.stack([input_image, real_image], axis=0)
  cropped_image = tf.image.random_crop(
      stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

  return cropped_image[0], cropped_image[1]

# normalizing the images to [-1, 1]

def normalize(input_image, real_image):
  input_image = (input_image / 127.5) - 1
  real_image = (real_image / 127.5) - 1

  return input_image, real_image

@tf.function()
def random_jitter(input_image, real_image):
  # resizing to 286 x 286 x 3
  input_image, real_image = resize(input_image, real_image, 286, 286)

  # randomly cropping to 256 x 256 x 3
  input_image, real_image = random_crop(input_image, real_image)

  if tf.random.uniform(()) > 0.5:
    # random mirroring
    input_image = tf.image.flip_left_right(input_image)
    real_image = tf.image.flip_left_right(real_image)

  return input_image, real_image

# As you can see in the images below
# that they are going through random jittering
# Random jittering as described in the paper is to
# 1. Resize an image to bigger height and width
# 2. Randomnly crop to the original size
# 3. Randomnly flip the image horizontally

def load_image_train(image_file):
  input_image, real_image = load(image_file)
  input_image, real_image = random_jitter(input_image, real_image)
  input_image, real_image = normalize(input_image, real_image)

  return image_file, input_image, real_image

def load_image_test(image_file):
  input_image, real_image = load(image_file)
  input_image, real_image = resize(input_image, real_image,
                                   IMG_HEIGHT, IMG_WIDTH)
  input_image, real_image = normalize(input_image, real_image)

  return image_file, input_image, real_image

def train_split_image(image,label):
  return image[1],image[2]

def split_image(image,label):
  return image[0],image[1],image[2],label

#create a dataset with filenames(random order in default)
all_image_paths,all_image_labels = get_dataset(training=True)
path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
image_ds = path_ds.map(load_image_train, tf.data.experimental.AUTOTUNE)
label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))
image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
image_label_ds = image_label_ds.map(train_split_image, tf.data.experimental.AUTOTUNE)
train_dataset = image_label_ds.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(GLOBAL_BATCH_SIZE)
print('A:',train_dataset)
train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
print('B:',train_dist_dataset)

all_image_paths,all_image_labels = get_dataset(training=False)
path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
image_ds = path_ds.map(load_image_test)
label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))
image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
image_label_ds = image_label_ds.map(split_image)
test_dataset = image_label_ds.batch(1)

OUTPUT_CHANNELS = 3

def downsample(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result

def upsample(filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result

def Generator():
  down_stack = [
    downsample(64, 4, apply_batchnorm=False), # (bs, 128, 128, 64)
    downsample(128, 4), # (bs, 64, 64, 128)
    downsample(256, 4), # (bs, 32, 32, 256)
    downsample(512, 4), # (bs, 16, 16, 512)
    downsample(512, 4), # (bs, 8, 8, 512)
    downsample(512, 4), # (bs, 4, 4, 512)
    downsample(512, 4), # (bs, 2, 2, 512)
    downsample(512, 4), # (bs, 1, 1, 512)
  ]

  up_stack = [
    upsample(512, 4, apply_dropout=True), # (bs, 2, 2, 1024)
    upsample(512, 4, apply_dropout=True), # (bs, 4, 4, 1024)
    upsample(512, 4, apply_dropout=True), # (bs, 8, 8, 1024)
    upsample(512, 4), # (bs, 16, 16, 1024)
    upsample(256, 4), # (bs, 32, 32, 512)
    upsample(128, 4), # (bs, 64, 64, 256)
    upsample(64, 4), # (bs, 128, 128, 128)
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh') # (bs, 256, 256, 3)

  concat = tf.keras.layers.Concatenate()

  inputs = tf.keras.layers.Input(shape=[None,None,3])
  x = inputs

  # Downsampling through the model
  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)

  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = concat([x, skip])

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)

with strategy.scope():
  generator = Generator()

def Discriminator():
  initializer = tf.random_normal_initializer(0., 0.02)

  inp = tf.keras.layers.Input(shape=[None, None, 3], name='input_image')
  tar = tf.keras.layers.Input(shape=[None, None, 3], name='target_image')

  x = tf.keras.layers.concatenate([inp, tar]) # (bs, 256, 256, channels*2)

  down1 = downsample(64, 4, False)(x) # (bs, 128, 128, 64)
  down2 = downsample(128, 4)(down1) # (bs, 64, 64, 128)
  down3 = downsample(256, 4)(down2) # (bs, 32, 32, 256)

  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3) # (bs, 34, 34, 256)
  conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1) # (bs, 31, 31, 512)

  batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

  leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)

  last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2) # (bs, 30, 30, 1)

  return tf.keras.Model(inputs=[inp, tar], outputs=last)

with strategy.scope():
  discriminator = Discriminator()

LAMBDA = 100

with strategy.scope():
  loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True,reduction=tf.keras.losses.Reduction.NONE)

  def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return tf.nn.compute_average_loss(total_disc_loss, global_batch_size=GLOBAL_BATCH_SIZE)

  def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return tf.nn.compute_average_loss(total_gen_loss, global_batch_size=GLOBAL_BATCH_SIZE)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

with strategy.scope():
  generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
  discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

  checkpoint = tf.train.Checkpoint(step=tf.Variable(1),
                                   generator_optimizer=generator_optimizer,
                                   discriminator_optimizer=discriminator_optimizer,
                                   generator=generator,
                                   discriminator=discriminator)
#manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)
#checkpoint.restore(manager.latest_checkpoint)
#if manager.latest_checkpoint:
#  print("Restored from {}".format(manager.latest_checkpoint))
#else:
#  print("Initializing from scratch.")

def generate_images(model, path, inp, tar):
  # the training=True is intentional here since
  # we want the batch statistics while running the model
  # on the test dataset. If we use training=False, we will get
  # the accumulated statistics learned from the training dataset
  # (which we don't want)
  prediction = model(inp, training=True)
  path_byte = path.numpy()[0]
  path = path_byte.decode("utf-8")
  base_name = os.path.splitext(os.path.basename(path))
  inp_name = base_name[0]+'-input'+ base_name[1]
  tar_name = base_name[0]+'-target'+ base_name[1]
  pred_name = base_name[0]+'-output'+ base_name[1]
  outputs = {inp_name:inp[0], tar_name:tar[0], pred_name:prediction[0]}
  for key,value in outputs.items():
    out_path = os.path.join(generated_image_dir,key)
    plt.imsave(out_path,np.array((value+1)/2))

with strategy.scope():
  def train_step(inputs):
    input_image, target = inputs

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      gen_output = generator(input_image, training=True)

      disc_real_output = discriminator([input_image, target], training=True)
      disc_generated_output = discriminator([input_image, gen_output], training=True)

      gen_loss = generator_loss(disc_generated_output, gen_output, target)
      disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_loss,
                                            generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                                 discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients,
                                            generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                discriminator.trainable_variables))

with strategy.scope():
  @tf.function
  def distributed_train_step(dataset_inputs):
    per_replica_losses = strategy.experimental_run_v2(train_step, args=(dataset_inputs,))

  def train(dataset, epochs):
    for epoch in range(epochs):
      start = time.time()

      for x in dataset:
        distributed_train_step(x)
        checkpoint.step.assign_add(1)
      # saving (checkpoint) the model every 20 epochs
      if (epoch + 1) % 1 == 0:
        checkpoint.save(file_prefix = checkpoint_prefix)

      print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                                     time.time()-start))

  train(train_dist_dataset, EPOCHS)

  #restoring the latest checkpoint in checkpoint_dir
  checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

# Run the trained model on the entire test dataset
for path, inp, tar, label in test_dataset:
  generate_images(generator, path, inp, tar)



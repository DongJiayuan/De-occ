from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import os
import time
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import clear_output

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

strategy = tf.distribute.MirroredStrategy()

#generated_image_dir = './photos_test2'
generated_image_dir = '/data2/dongjy/photos_test3'
if not os.path.exists(generated_image_dir):
    os.makedirs(generated_image_dir)

#PATH = './photos/'
PATH = '/data2/dongjy/photos/'

BUFFER_SIZE = 1000
BATCH_SIZE_PER_REPLICA = 64
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
IMG_WIDTH = 256
IMG_HEIGHT = 256
EPOCHS = 10

def get_dataset(training = True):
  if training == True:
    input_paths = glob.glob(PATH+'train2/*/*')
  else:
    input_paths = glob.glob(PATH+'val2/*/*')

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
  for item in list(glob.glob(PATH + 'train2/*')):
    label_names.append(os.path.basename(item))
  label_names = sorted(label_names)
  label_to_index = dict((name, index) for index,name in enumerate(label_names))
  all_image_labels = [label_to_index[os.path.basename(os.path.dirname(path))] for path in all_image_paths]

  return all_image_paths, all_image_labels

def load(image_file):
  image = tf.io.read_file(image_file)
  image = tf.image.decode_png(image)

  w = tf.shape(image)[1]

  w = w // 3
  real_image = image[:, :w, :]
  input_image = image[:, w:2*w, :]
  occlusion_image = image[:,2*w:,:]

  input_image = tf.cast(input_image, tf.float32)
  real_image = tf.cast(real_image, tf.float32)
  occlusion_image = tf.cast(occlusion_image, tf.float32)

  return input_image, real_image, occlusion_image

def resize(input_image, real_image,occlusion_image, height, width):
  input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  real_image = tf.image.resize(real_image, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  occlusion_image = tf.image.resize(occlusion_image, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  return input_image, real_image, occlusion_image

def random_crop(input_image, real_image,occlusion_image):
  stacked_image = tf.stack([input_image, real_image, occlusion_image], axis=0)
  cropped_image = tf.image.random_crop(
      stacked_image, size=[3, IMG_HEIGHT, IMG_WIDTH, 3])

  return cropped_image[0], cropped_image[1], cropped_image[2]

# normalizing the images to [-1, 1]

def normalize(input_image, real_image,occlusion_image):
  input_image = (input_image / 127.5) - 1
  real_image = (real_image / 127.5) - 1
  occlusion_image = (occlusion_image / 127.5) - 1

  return input_image, real_image, occlusion_image

@tf.function()
def random_jitter(input_image, real_image,occlusion_image):
  # resizing to 286 x 286 x 3
  input_image, real_image,occlusion_image = resize(input_image, real_image,occlusion_image, 286, 286)

  # randomly cropping to 256 x 256 x 3
  input_image, real_image,occlusion_image = random_crop(input_image, real_image,occlusion_image)

  if tf.random.uniform(()) > 0.5:
    # random mirroring
    input_image = tf.image.flip_left_right(input_image)
    real_image = tf.image.flip_left_right(real_image)
    occlusion_image = tf.image.flip_left_right(occlusion_image)

  return input_image, real_image,occlusion_image

# As you can see in the images below
# that they are going through random jittering
# Random jittering as described in the paper is to
# 1. Resize an image to bigger height and width
# 2. Randomnly crop to the original size
# 3. Randomnly flip the image horizontally

def load_image_train(image_file):
  input_image, real_image,occlusion_image= load(image_file)
  input_image, real_image,occlusion_image = random_jitter(input_image, real_image,occlusion_image)
  input_image, real_image,occlusion_image = normalize(input_image, real_image,occlusion_image)

  return image_file, input_image, real_image,occlusion_image

def load_image_test(image_file):
  input_image, real_image,occlusion_image = load(image_file)
  input_image, real_image,occlusion_image = resize(input_image, real_image,occlusion_image,
                                   IMG_HEIGHT, IMG_WIDTH)
  input_image, real_image,occlusion_image = normalize(input_image, real_image,occlusion_image)

  return image_file, input_image, real_image, occlusion_image

def train_split_image(image,label):
  #return path,input_image_real_image_occlusion_image,label(int)
  return image[1],image[2],image[3]

def split_image(image,label):
  #return path,input_image_real_image_occlusion_image,label(int)
  return image[0],image[1],image[2],image[3],label

#create a dataset with filenames(random order in default)
all_image_paths,all_image_labels = get_dataset(training=True)
path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
image_ds = path_ds.map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))
image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
image_label_ds = image_label_ds.map(train_split_image,num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = image_label_ds.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(GLOBAL_BATCH_SIZE)
train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)

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

def Generator(One_input = True):
  down_stack = [
    downsample(64, 4, apply_batchnorm=False), # (bs, 64, 64, 64)
    downsample(128, 4), # (bs, 32, 32, 128)
    downsample(256, 4), # (bs, 16, 16, 256)
    downsample(512, 4), # (bs, 8, 8, 512)
    downsample(512, 4), # (bs, 4, 4, 512)
    downsample(512, 4),
    downsample(512, 4), # (bs, 2, 2, 512)
    downsample(512, 4), # (bs, 1, 1, 512)
  ]

  up_stack = [
    upsample(512, 4, apply_dropout=True), # (bs, 2, 2, 1024)
    upsample(512, 4, apply_dropout=True), # (bs, 4, 4, 1024)
    upsample(512, 4, apply_dropout=True), # (bs, 8, 8, 1024)
    upsample(512, 4),
    upsample(256, 4), # (bs, 16, 16, 512)
    upsample(128, 4), # (bs, 32, 32, 256)
    upsample(64, 4), # (bs, 64, 64, 128)
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh') # (bs, 128, 128, 3)

  concat = tf.keras.layers.Concatenate()

  if One_input == True:
    inputs = tf.keras.layers.Input(shape=[None,None,3])
    x = inputs
  else:
    inputs = tf.keras.layers.Input(shape=[None,None,6])
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
  generator1 = Generator(One_input=True)
  generator2 = Generator(One_input=False)

def Discriminator():
  initializer = tf.random_normal_initializer(0., 0.02)

  inp = tf.keras.layers.Input(shape=[None, None, 3], name='input_image')
  tar = tf.keras.layers.Input(shape=[None, None, 3], name='target_image')

  x = tf.keras.layers.concatenate([inp, tar]) # (bs, 128, 128, channels*2)

  down1 = downsample(64, 4, False)(x) # (bs, 64, 64, 64)
  down2 = downsample(128, 4)(down1) # (bs, 32, 32, 128)
  down3 = downsample(256, 4)(down2) # (bs, 16, 16, 256)

  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3) # (bs, 18, 18, 256)
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
  discriminator1 = Discriminator()
  discriminator2 = Discriminator()

LAMBDA = 100

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True,
                                                reduction = tf.keras.losses.Reduction.NONE)

def discriminator_loss(disc_real_output, disc_generated_output):
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

  total_disc_loss = real_loss + generated_loss

  return tf.nn.compute_average_loss(total_disc_loss,global_batch_size = GLOBAL_BATCH_SIZE)

def generator_loss(disc_generated_output, gen_output, target):
  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

  # mean absolute error
  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

  total_gen_loss = gan_loss + (LAMBDA * l1_loss)

  return tf.nn.compute_average_loss(total_gen_loss,global_batch_size = GLOBAL_BATCH_SIZE)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
with strategy.scope():
  generator1_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
  discriminator1_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

  generator2_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
  discriminator2_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)


  checkpoint = tf.train.Checkpoint(step=tf.Variable(1),
                                   generator1_optimizer=generator1_optimizer,
                                   discriminator1_optimizer=discriminator1_optimizer,
                                   generator2_optimizer=generator2_optimizer,
                                   discriminator2_optimizer=discriminator2_optimizer,
                                   generator1=generator1,
                                   discriminator1=discriminator1,
                                   generator2=generator2,
                                   discriminator2=discriminator2)



def generate_images(model1,model2, path, inp, occ, tar):
  # the training=True is intentional here since
  # we want the batch statistics while running the model
  # on the test dataset. If we use training=False, we will get
  # the accumulated statistics learned from the training dataset
  # (which we don't want)
  prediction = model1(inp, training=True)
  model2_input = tf.keras.layers.concatenate([inp,prediction])
  path_byte = path.numpy()[0]
  path = path_byte.decode("utf-8")
  base_name = os.path.splitext(os.path.basename(path))
  #apendix = 'jpg'
  inp_name = base_name[0]+'-input'+ base_name[1]
  tar_name = base_name[0]+'-target'+ base_name[1]
  occ_name = base_name[0]+'-g1'+ base_name[1]
  pred_name = base_name[0]+'-g2'+ base_name[1]
  #inp_name = base_name[0]+'-input'+ apendix
  #tar_name = base_name[0]+'-target'+ apendix
  #occ_name = base_name[0]+'-g1'+ apendix
  #pred_name = base_name[0]+'-g2'+ apendix
  pred_image = model2(model2_input,training=True)
  outputs = {inp_name:inp[0], tar_name:tar[0], occ_name:prediction[0],pred_name:pred_image[0]}
  for key,value in outputs.items():
    #print(generated_image_dir,key,value)
    out_path = os.path.join(generated_image_dir,key)
    #print(out_path)
    v = np.array((value+1)/2)
    plt.imsave(out_path,v)

with strategy.scope():
  def train_step_for_GAN1(input):
    input_image,_,target = input
    with tf.GradientTape() as gen1_tape, tf.GradientTape() as disc1_tape:
      gen_output = generator1(input_image, training=True)

      disc_real_output = discriminator1([input_image, target], training=True)
      disc_generated_output = discriminator1([input_image, gen_output], training=True)

      gen_loss = generator_loss(disc_generated_output, gen_output, target)
      disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator1_gradients = gen1_tape.gradient(gen_loss,
                                            generator1.trainable_variables)
    discriminator1_gradients = disc1_tape.gradient(disc_loss,
                                                 discriminator1.trainable_variables)

    generator1_optimizer.apply_gradients(zip(generator1_gradients,
                                            generator1.trainable_variables))
    discriminator1_optimizer.apply_gradients(zip(discriminator1_gradients,
                                                discriminator1.trainable_variables))

  def train_step_for_GAN2(input, occ_generated):
    with tf.GradientTape() as gen2_tape, tf.GradientTape() as disc2_tape:
      input_image, target,_ = input
      input_fake = tf.keras.layers.concatenate([input_image,occ_generated])
      gen_output = generator2(input_fake, training=True)

      disc_real_output = discriminator2([input_image, target], training=True)
      disc_generated_output = discriminator2([input_image, gen_output], training=True)

      gen2_loss = generator_loss(disc_generated_output, gen_output, target)
      disc2_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator2_gradients = gen2_tape.gradient(gen2_loss,
                                            generator2.trainable_variables)
    discriminator2_gradients = disc2_tape.gradient(disc2_loss,
                                                 discriminator2.trainable_variables)

    generator2_optimizer.apply_gradients(zip(generator2_gradients,
                                            generator2.trainable_variables))
    discriminator2_optimizer.apply_gradients(zip(discriminator2_gradients,
                                                discriminator2.trainable_variables))

  def inferrence_GAN1(input):
    input_image,_,_ = input
    return generator1(input_image)

with strategy.scope():
  @tf.function
  def distributed_train_step_for_GAN1(input):
    per_replica_losses = strategy.experimental_run_v2(train_step_for_GAN1,
                                                    args=(input,))

  @tf.function
  def distributed_train_step_for_GAN2(input, occ_generated):
    per_replica_losses = strategy.experimental_run_v2(train_step_for_GAN2,
                                                    args=(input,occ_generated))

  @tf.function
  def distributed_inferrence_GAN1(input):
    output = strategy.experimental_run_v2(inferrence_GAN1,args=(input,))
    return output

  for epoch in range(EPOCHS):
    start = time.time()
    for x in train_dist_dataset:
      distributed_train_step_for_GAN1(x)
      checkpoint.step.assign_add(1)

    # saving (checkpoint) the model every 20 epochs
    #if (epoch + 1) % 10 == 0:
    checkpoint.save(file_prefix = checkpoint_prefix)

    print ('GAN1: Time taken for epoch {} is {} sec\n'.format(epoch + 1,time.time()-start))

  for epoch in range(5*EPOCHS):
    start = time.time()

    for x in train_dist_dataset:
      occ_generated = distributed_inferrence_GAN1(x)
      #occ_dist_dataset = strategy.experimental_distribute_dataset(occ_generated)
      distributed_train_step_for_GAN2(x, occ_generated)
      checkpoint.step.assign_add(1)
    # saving (checkpoint) the model every 20 epochs
    #if (epoch + 1) % 10 == 0:
    checkpoint.save(file_prefix = checkpoint_prefix)

    print ('GAN2: Time taken for epoch {} is {} sec\n'.format(epoch + 1,time.time()-start))

  # restoring the latest checkpoint in checkpoint_dir
  checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

# Run the trained model on the entire test dataset
for path, inp, tar, occ, label in test_dataset:
  generate_images(generator1,generator2, path, inp, occ, tar)

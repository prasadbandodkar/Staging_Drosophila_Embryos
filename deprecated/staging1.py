
# checking if tf is working
#
import tensorflow as tf
from tensorflow import keras    # type: ignore
print(tf.__version__)
if tf.config.list_physical_devices('GPU'):
  print("TensorFlow **IS** using the GPU")
else:
  print("TensorFlow **IS NOT** using the GPU")


########################################################################################     
#

import matplotlib.pyplot as plt
from sdata import DataGenerator as DGen

# folders and files
#
sys_folder   = '/Volumes/X2/Projects/staging/Data/'
drive_folder = '/Volumes/X2/Projects/staging/Data/'
data_folder  = sys_folder + 'data/'


# training parameters
#  
image_length = 128
shufflenum   = 100
nBatchTrain  = 25
nBatchVal    = 25
AUTOTUNE     = tf.data.AUTOTUNE


# define train, val, test data
test = [6,7]
val  = [8,9]
ignore = [26, 31, 20, 21, 22, 23, 24, 25, 27]
model_name = f"staging_nl_test_{'-'.join(map(str, test))}_val_{'-'.join(map(str, val))}_ignore_{'-'.join(map(str, ignore))}"
print(f"Model name: {model_name}")


# run data generator and test it
#
size        = (512, 512)
padding     = 44
npoints     = 60
inward      = 40
outward     = -24
length      = 128
d = DGen(data_folder, test, val, ignore, size, padding, npoints, inward, outward, length)
image, id = next(d('test'))
print(f"Image shape: {image.shape}") # type: ignore
print(f"Image id: {id}")
print(f"Image dtype: {image.dtype}") # type: ignore
print(f"Id dtype: {id.dtype}")


# specify tf dataset and print a few data
#
output_signature = (
  tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32),   # type: ignore
  tf.TensorSpec(shape=(), dtype=tf.float64))                # type: ignore

train_ds = tf.data.Dataset.from_generator(
  lambda: d('test'), 
  output_signature=output_signature
)
# train_ds = train_ds.shuffle(shufflenum).batch(nBatchTrain).prefetch(AUTOTUNE)

print(train_ds)

i = 0
try:
  for I, id in train_ds.take(10):
    # plot in figure that is 512 by 512
    plt.figure(figsize=(20, 20))
    print(f"i = {i}")
    print(f"Image shape: {I.shape}")
    print(f"Image id: {id}")
    plt.imshow(I[:, :, 0], cmap='gray')
    plt.show()
    plt.close()
    i += 1
except tf.errors.OutOfRangeError:
  print("Reached the end of the dataset.")


# Model
#
# define image_length from first image of train_ds
#
# Get the image shape from the first element of the dataset
for image, _ in train_ds.take(1):
    image_shape = image.shape
    break

# Define the CNN model
def create_cnn_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)  # Assuming regression task, adjust if classification
    ])
    return model

# Create the model
image_shape = image.shape
model = create_cnn_model(image_shape)

# Compile the model
model.compile(optimizer='adam',
              loss='mse',       # Mean Squared Error for regression
              metrics=['mae'])  # Mean Absolute Error

# Print model summary
model.summary()









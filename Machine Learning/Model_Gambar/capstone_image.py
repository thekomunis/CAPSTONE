# -*- coding: utf-8 -*-
"""Capstone Image.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1l353Q1NHw0dH-f_KWoP0CHzYW7fWeoGJ

# **Klasifikasi Gambar Penyakit Kulit Kucing**

## **Objective:**   
Membangun sebuah model menggunakan CNN yang dapat mengklasifikasikan penyakit kulit kucing menggunakan gambar.

## **Sumber Dataset**
https://www.kaggle.com/datasets/adityavahreza/cat-skin-disease-v2

# **Import Libraries**
"""

# Commented out IPython magic to ensure Python compatibility.
# Mengimpor libraries umum yang sering digunakan
import os, shutil
import zipfile
import random
from random import sample
import shutil
from shutil import copyfile
import pathlib
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm as tq

# Mengimpor libraries untuk visualisasi
# %matplotlib inline
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.image import imread

# Mengimpor libraries untuk pemrosesan data gambar
import cv2
from PIL import Image
import skimage
from skimage import io
from skimage.transform import resize
from skimage.transform import rotate, AffineTransform, warp
from skimage import img_as_ubyte
from skimage.exposure import adjust_gamma
from skimage.util import random_noise

# Mengimpor libraries untuk pembuatan dan evaluasi model
import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.layers import InputLayer, Conv2D, SeparableConv2D, MaxPooling2D, MaxPool2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau

# Mengabaikan peringatan
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Mencetak versi TensorFlow yang sedang digunakan
print(tf.__version__)

"""# **Data Loading**"""

# Import module yang disediakan google colab untuk kebutuhan upload file
from google.colab import files
files.upload()

!rm -rf cat-skin-disease-v2/

# Download kaggle dataset and unzip the file
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d adityavahreza/cat-skin-disease-v2
!unzip -o cat-skin-disease-v2.zip

"""## **Plot gambar sampel untuk semua kelas**"""

# Membuat kamus yang menyimpan nama gambar untuk setiap kelas
cat_image = {}

# Path ke direktori data
path = "CAT SKIN DISEASE/"

# Isi lung_image dengan daftar gambar dari masing-masing kelas
for class_name in os.listdir(path):
    class_path = os.path.join(path, class_name)
    if os.path.isdir(class_path):
        image_files = [img for img in os.listdir(class_path) if img.endswith(('.png', '.jpg', '.jpeg'))]
        cat_image[class_name] = image_files

# Pastikan lung_image terisi
if not cat_image:
    raise ValueError("Tidak ada data ditemukan di folder.")

# Menampilkan secara acak 5 gambar untuk setiap kelas
fig, axs = plt.subplots(len(cat_image.keys()), 5, figsize=(15, 15))

for i, (class_name, image_list) in enumerate(cat_image.items()):
    images = np.random.choice(image_list, 5, replace=False)

    for j, image_name in enumerate(images):
        img_path = os.path.join(path, class_name, image_name)
        img = Image.open(img_path).convert("L")  # Konversi ke grayscale
        axs[i, j].imshow(img, cmap='gray')
        axs[i, j].set(xlabel=class_name, xticks=[], yticks=[])

fig.tight_layout()
plt.show()

"""## **Plot distribusi gambar di seluruh kelas**"""

# Define source path
lung_path = "CAT SKIN DISEASE/"

# Create a list that stores data for each filenames, filepaths, and labels in the data
file_name = []
labels = []
full_path = []

# Get data image filenames, filepaths, labels one by one with looping, and store them as dataframe
for path, subdirs, files in os.walk(lung_path):
    for name in files:
        full_path.append(os.path.join(path, name))
        labels.append(path.split('/')[-1])
        file_name.append(name)

distribution_train = pd.DataFrame({"path":full_path,'file_name':file_name,"labels":labels})

# Plot the distribution of images across the classes
Label = distribution_train['labels']
plt.figure(figsize = (6,6))
sns.set_style("darkgrid")
plot_data = sns.countplot(Label)

"""# **Data Augmentation**

Proses augmentasi gambar adalah teknik yang digunakan untuk membuat variasi baru dari setiap gambar dalam dataset, sehingga model memiliki lebih banyak variasi untuk dipelajari. Ini membantu mencegah overfitting, di mana model terlalu terbiasa dengan data pelatihan dan tidak dapat menggeneralisasi dengan baik ke data baru.

Berikut adalah beberapa strategi augmentasi gambar yang dapat kita terapkan:

- `anticlockwise_rotation` adalah ketika gambar diputar ke arah berlawanan dengan arah jarum jam.
- `clockwise_rotation` adalah ketika gambar diputar ke arah searah dengan arah jarum jam.
- `flip_up_down` adalah ketika gambar dibalik secara vertikal dari atas ke bawah.
- `sheared` adalah ketika gambar diberikan efek pergeseran acak.
- `blur` adalah ketika gambar diberikan efek kabur atau blur.
- `wrap_shift` adalah ketika gambar diberikan efek pergeseran melengkung.
- `brightness` adalah ketika gambar diberikan efek peningkatan kecerahan.
"""

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import random

# Membuat fungsi untuk melakukan rotasi berlawanan arah jarum jam
def anticlockwise_rotation(img):
    img = tf.image.resize(img, (224, 224))
    img = tf.image.rot90(img, k=random.randint(1, 4))  # Rotasi 90, 180, atau 270 derajat secara acak
    return img

# Membuat fungsi untuk melakukan rotasi searah jarum jam
def clockwise_rotation(img):
    img = tf.image.resize(img, (224, 224))
    img = tf.image.rot90(img, k=random.randint(1, 4))  # Rotasi 90, 180, atau 270 derajat secara acak
    return img

# Membuat fungsi untuk membalik gambar secara vertikal dari atas ke bawah
def flip_up_down(img):
    img = tf.image.resize(img, (224, 224))
    img = tf.image.flip_up_down(img)
    return img

# Membuat fungsi untuk memberikan efek peningkatan kecerahan pada gambar
def add_brightness(img):
    img = tf.image.resize(img, (224, 224))
    img = tf.image.adjust_brightness(img, delta=random.uniform(0.1, 0.5))  # Sesuaikan nilai delta sesuai kebutuhan
    return img

# Membuat fungsi untuk memberikan efek blur pada gambar
def blur_image(img):
    img = tf.image.resize(img, (224, 224))
    img = tf.image.random_blur(img, (3, 3))  # Ukuran kernel blur bisa disesuaikan
    return img

# Membuat fungsi untuk memberikan efek pergeseran acak pada gambar
def sheared(img):
    img = tf.image.resize(img, (224, 224))
    # Buat objek ImageDataGenerator dengan parameter shearing range
    datagen = ImageDataGenerator(shear_range=0.2)
    img = next(iter(datagen.flow(tf.expand_dims(img, 0))))[0]
    return img

# Membuat fungsi untuk melakukan pergeseran melengkung pada gambar
def warp_shift(img):
    img = tf.image.resize(img, (224, 224))
    # Buat objek ImageDataGenerator dengan parameter width_shift_range dan height_shift_range
    datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1)
    img = next(iter(datagen.flow(tf.expand_dims(img, 0))))[0]
    return img

# Define source path
cat_path = "CAT SKIN DISEASE/"

# Create a list that stores data for each filenames, filepaths, and labels in the data
file_name = []
labels = []
full_path = []

# Get data image filenames, filepaths, labels one by one with looping, and store them as dataframe
for path, subdirs, files in os.walk(cat_path):
    for name in files:
        full_path.append(os.path.join(path, name))
        labels.append(path.split('/')[-1])
        file_name.append(name)

distribution_train = pd.DataFrame({"path":full_path,'file_name':file_name,"labels":labels})

# Plot the distribution of images across the classes
Label = distribution_train['labels']
plt.figure(figsize = (6,6))
sns.set_style("darkgrid")
plot_data = sns.countplot(Label)

"""# **Data Splitting : Training and Testing**"""

# Panggil variabel mypath yang menampung folder dataset gambar
mypath= 'CAT SKIN DISEASE/'

file_name = []
labels = []
full_path = []
for path, subdirs, files in os.walk(mypath):
    for name in files:
        full_path.append(os.path.join(path, name))
        labels.append(path.split('/')[-1])
        file_name.append(name)


# Memasukan variabel yang sudah dikumpulkan pada looping di atas menjadi sebuah dataframe agar rapih
df = pd.DataFrame({"path":full_path,'file_name':file_name,"labels":labels})
# Melihat jumlah data gambar pada masing-masing label
df.groupby(['labels']).size()

# Variabel yang digunakan pada pemisahan data ini dimana variabel x = data path dan y = data labels
X= df['path']
y= df['labels']

# Pertama, bagi data menjadi train dan sementara (temp)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=300)

# Kemudian, bagi temp menjadi test dan validation (misalnya 50% masing-masing dari temp)
X_test, X_valid, y_test, y_valid = train_test_split(X_temp, y_temp, test_size=0.5, random_state=300)

# Menyatukan kedalam masing-masing dataframe
df_tr = pd.DataFrame({'path': X_train, 'labels': y_train, 'set': 'train'})
df_valid = pd.DataFrame({'path': X_valid, 'labels': y_valid, 'set': 'validation'})
df_te = pd.DataFrame({'path': X_test, 'labels': y_test, 'set': 'test'})

# Print hasil diatas untuk melihat panjang size data training dan testing
print('train size', len(df_tr))
print('test size', len(df_te))
print('valid size', len(df_valid))

# Gabungkan DataFrame df_tr dan df_te
df_all = pd.concat([df_tr, df_te, df_valid], ignore_index=True)

print('===================================================== \n')
print(df_all.groupby(['set', 'labels']).size(), '\n')
print('===================================================== \n')

# Cek sample data
print(df_all.sample(5))

# Memanggil dataset asli yang berisi keseluruhan data gambar yang sesuai dengan labelnya
datasource_path = "CAT SKIN DISEASE/"
# Membuat variabel Dataset, dimana nanti menampung data yang telah dilakukan pembagian data training dan testing
dataset_path = "Dataset-Final/"

for index, row in tq(df_all.iterrows()):
    # Deteksi filepath
    file_path = row['path']
    if os.path.exists(file_path) == False:
            file_path = os.path.join(datasource_path,row['labels'],row['image'].split('.')[0])

    # Buat direktori tujuan folder
    if os.path.exists(os.path.join(dataset_path,row['set'],row['labels'])) == False:
        os.makedirs(os.path.join(dataset_path,row['set'],row['labels']))

    # Tentukan tujuan file
    destination_file_name = file_path.split('/')[-1]
    file_dest = os.path.join(dataset_path,row['set'],row['labels'],destination_file_name)

    # Salin file dari sumber ke tujuan
    if os.path.exists(file_dest) == False:
        shutil.copy2(file_path,file_dest)

"""# **Image Data Generator**

In TensorFlow you can do this through the `tf.keras.preprocessing.image.ImageDataGenerator` class. This class allows you to do:
- Configure the random transformation and normalization operations to be performed on the image data during training
- Instantiate generator of augmented image sets (and their labels) via `.flow(data, labels)` or `.flow_from_directory(directory)`. This generator can then be used with `tf.keras` model methods which accept generator data as input, `fit`, `evaluate` and `predict`

Prepare the training and validation data, to begin with using `.flow_from_directory()` which generates image datasets and their labels directly in their respective folders by setting the `WIDTH` and `HEIGHT` size, predefined `BATCH SIZE` size and mode its class. Here we use `"binary"` class mode because the number of classes used is 2.
"""

# Define training and test directories
TRAIN_DIR = "Dataset-Final/train"
TEST_DIR = "Dataset-Final/test"
VALID_DIR = "Dataset-Final/validation"

train_health = os.path.join(TRAIN_DIR + '/Health')
train_flea = os.path.join(TRAIN_DIR + '/Flea_Allergy')
train_ringworm = os.path.join(TRAIN_DIR + '/Ringworm')
train_scabies = os.path.join(TRAIN_DIR + '/Scabies')

test_health = os.path.join(TEST_DIR + '/Health')
test_flea = os.path.join(TEST_DIR + '/Flea_Allergy')
test_ringworm = os.path.join(TEST_DIR + '/Ringworm')
test_scabies = os.path.join(TEST_DIR + '/Scabies')

valid_health = os.path.join(VALID_DIR + '/Health')
valid_flea = os.path.join(VALID_DIR + '/Flea_Allergy')
valid_ringworm = os.path.join(VALID_DIR + '/Ringworm')
valid_scabies = os.path.join(VALID_DIR + '/Scabies')

print("Total number of health images in training set: ",len(os.listdir(train_health)))
print("Total number of flea allergy images in training set: ",len(os.listdir(train_flea)))
print("Total number of ringworm images in training set: ",len(os.listdir(train_ringworm)))
print("Total number of scabies images in training set: ",len(os.listdir(train_scabies)))

print("Total number of health images in testing set: ",len(os.listdir(test_health)))
print("Total number of flea allergy images in testing set: ",len(os.listdir(test_flea)))
print("Total number of ringworm images in training set: ",len(os.listdir(test_ringworm)))
print("Total number of scabies images in training set: ",len(os.listdir(test_scabies)))

print("Total number of health images in validation set: ",len(os.listdir(valid_health)))
print("Total number of flea allergy images in validation set: ",len(os.listdir(valid_flea)))
print("Total number of ringworm images in training set: ",len(os.listdir(valid_ringworm)))
print("Total number of scabies images in training set: ",len(os.listdir(valid_scabies)))

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Image size harus sama dengan input MobileNetV2 (150x150 dalam kasus ini)
IMG_SIZE = (150, 150)
BATCH_SIZE = 32

# Data augmentation + validation split untuk train & val
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',  # subset untuk training
    shuffle=True
)

validation_generator = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',  # subset untuk validasi
    shuffle=False
)

test_datagen = ImageDataGenerator(rescale=1. / 255)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)

"""# **Model Transfer Learning using MobileNetV2:**"""

from tensorflow.keras.applications import MobileNetV2

# Load pretrained base model
base_model = MobileNetV2(input_shape=(150,150,3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze weights

# Tambahkan custom CNN layers setelah base_model
x = base_model.output
x = Conv2D(32, (3,3), padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2,2))(x)

x = Conv2D(64, (3,3), padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2,2))(x)

x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(4, activation='softmax')(x)  # 4 kelas output

# Final model
model = Model(inputs=base_model.input, outputs=output)

# Compile
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True)
]

count_health = len(os.listdir(train_health))
count_flea = len(os.listdir(train_flea))
count_ringworm = len(os.listdir(train_ringworm))
count_scabies = len(os.listdir(train_scabies))

# Menghitung weight untuk setiap kelas
weight_0 = (1 / count_health) * (count_health + count_flea + count_ringworm + count_scabies) / 4.0
weight_1 = (1 / count_flea) * (count_health + count_flea + count_ringworm + count_scabies) / 4.0
weight_2 = (1 / count_ringworm) * (count_health + count_flea + count_ringworm + count_scabies) / 4.0
weight_3 = (1 / count_scabies) * (count_health + count_flea + count_ringworm + count_scabies) / 4.0

# Membuat dictionary untuk class_weight
class_weights = {0: weight_0, 1: weight_1, 2: weight_2, 3: weight_3}

history_1 = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=50,
    callbacks=callbacks,
    class_weight=class_weights
)

test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc:.2f}")

acc = history_1.history['accuracy']
val_acc = history_1.history['val_accuracy']
loss = history_1.history['loss']
val_loss = history_1.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r')
plt.plot(epochs, val_acc, 'b')
plt.title('Training and Validation Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(epochs, loss, 'r')
plt.plot(epochs, val_loss, 'b')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.title('Training and Validation Loss')
plt.show()

test_generator.reset()

preds_1 = model.predict(test_generator, verbose=0)
preds_1 = np.argmax(preds_1, axis=1)

# Print Confusion Matrix
cm = pd.DataFrame(data=confusion_matrix(test_generator.classes, preds_1, labels=[0, 1, 2, 3]),index=["Actual Health", "Actual Flea", "Actual Ringworm", "Actual Scabies"],
columns=["Predicted Health", "Predicted Flea", "Predicted Ringworm", "Predicted Scabies"])
sns.heatmap(cm,annot=True,fmt="d")

# Print Classification Report
print("\n")
print(classification_report(y_true=test_generator.classes,y_pred=preds_1,target_names =['Health','Flea', 'Ringworm', 'Scabies'], digits=4))

"""## **Konversi Model**"""

# Konversi ke Saved_model
model.export('saved_model')

# Cek apakah folder saved_model dan isinya sudah disimpan
if os.path.exists('saved_model'):
    print("Model saved in 'saved_model' folder.")
    print("Contents of 'saved_model':", os.listdir('saved_model'))
else:
    print("Model not saved.")

# Konversi ke TFLite
converter = tf.lite.TFLiteConverter.from_saved_model("saved_model")
converter.experimental_new_converter = True
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()

# Simpan model.tflite
os.makedirs("tflite", exist_ok=True)
with open("tflite/model.tflite", "wb") as f:
    f.write(tflite_model)

# Simpan label.txt
labels = ["Flea_Allergy", "Health", "Ringworm", "Scabies"]
with open("tflite/label.txt", "w") as f:
    for label in labels:
        f.write(label + "\n")

os.makedirs('tfjs_model', exist_ok=True)
model.save("tfjs_model/model.h5")

!pip install tensorflowjs

import tensorflow as tf
import tensorflowjs as tfjs

# Load model dari .h5
model = tf.keras.models.load_model('tfjs_model/model.h5')

# Save ke format TensorFlow.js
tfjs.converters.save_keras_model(model, 'tfjs_model')

"""## **Testing**"""

import textwrap
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from google.colab import files
from IPython.display import display, Markdown

# Load model
model = load_model('model.h5')  # Ganti dengan nama model kamu

# Label kelas
class_labels = ['Flea_Allergy', 'Health', 'Ringworm', 'Scabies']

# Saran perawatan berdasarkan prediksi
treatment_suggestions = {
    'Health': "Kulit kucing tampak sehat dan bebas dari gejala penyakit kulit. Untuk menjaga kesehatan ini, pastikan kucing Anda tetap berada di lingkungan yang bersih dan bebas dari parasit seperti kutu dan tungau. Berikan makanan bernutrisi tinggi yang mengandung omega-3 dan omega-6 untuk menjaga kesehatan kulit dan bulu. Mandikan kucing secara rutin (sekitar 1-2 kali per bulan) dengan shampo khusus kucing yang lembut. Lakukan grooming untuk menghindari bulu kusut dan memeriksa adanya tanda-tanda penyakit secara dini. Jangan lupa untuk menjadwalkan pemeriksaan rutin ke dokter hewan minimal 1 kali setiap 6 bulan.",
    'Flea_Allergy': "Kucing menunjukkan gejala alergi terhadap gigitan kutu, yang biasanya berupa gatal parah, kerontokan rambut lokal, dan luka akibat garukan. Langkah pertama adalah segera memberikan pengobatan antiparasit seperti spot-on atau obat oral yang direkomendasikan oleh dokter hewan. Mandikan kucing dengan shampo anti-kutu khusus kucing, dan hindari penggunaan produk manusia karena bisa berbahaya. Bersihkan seluruh area tempat kucing sering berada—termasuk kasur, karpet, dan furnitur—karena kutu juga bisa bersarang di sana. Gunakan vacuum cleaner secara rutin dan pertimbangkan menyemprotkan obat pembasmi kutu di lingkungan sekitar. Bila kondisi tidak membaik dalam beberapa hari, segera konsultasikan ulang ke dokter hewan untuk penanganan lanjutan.",
    'Ringworm': "Ringworm atau dermatofitosis adalah infeksi jamur yang sangat menular, baik ke sesama kucing, hewan lain, maupun manusia. Tanda-tandanya termasuk kerontokan bulu berbentuk melingkar, kulit kemerahan, dan bersisik. Perawatan dimulai dengan mengisolasi kucing dari hewan peliharaan lainnya untuk mencegah penyebaran. Berikan salep anti-jamur seperti miconazole atau ketoconazole sesuai resep dokter. Mandikan kucing dengan shampo anti-jamur secara teratur. Semua barang yang pernah disentuh kucing harus dibersihkan dan didesinfeksi, termasuk tempat tidur dan mainan. Pengobatan oral mungkin dibutuhkan dalam kasus berat. Karena infeksi jamur bisa berlangsung lama, tetap disiplin dalam perawatan dan terus kontrol ke dokter hingga kucing benar-benar sembuh.",
    'Scabies': "Scabies atau kudis pada kucing disebabkan oleh infestasi tungau Sarcoptes atau Notoedres, dan sangat menular. Gejalanya meliputi rasa gatal ekstrem, kulit menebal dan bersisik, luka akibat garukan, dan bisa disertai infeksi sekunder. Perawatan dimulai dengan membawa kucing ke dokter hewan untuk diagnosis pasti (kadang melalui scraping kulit). Dokter akan memberikan obat seperti ivermectin, selamectin, atau milbemycin. Mandikan kucing dengan shampo khusus antiparasit dan rawat kulit yang terinfeksi agar tidak terjadi luka terbuka. Isolasi kucing selama masa perawatan agar tidak menularkan ke hewan atau manusia lain. Lingkungan juga harus dibersihkan secara intensif, dan perawatan harus dilanjutkan sampai tungau benar-benar hilang."
}

# Upload gambar dari user
uploaded = files.upload()

for file_name in uploaded.keys():
    # Load dan preprocess gambar
    img = image.load_img(file_name, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Prediksi
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_label = class_labels[predicted_class]

    # Output
    print(f"\nGambar: {file_name}")
    print(f"Prediksi: {predicted_label}")
    print(f"Probabilitas: {prediction}")
    # Tambahkan enter dan buat heading tebal
    display(Markdown("\n\nSaran Perawatan:"))

    # Wrap dan tampilkan isi saran
    wrapped_text = textwrap.fill(treatment_suggestions[predicted_label], width=100)
    print(wrapped_text)


    # Tampilkan gambar
    plt.imshow(img)
    plt.title(f"Prediksi: {predicted_label}")
    plt.axis('off')
    plt.show()
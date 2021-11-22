<%@ Page Title="" Language="vb" AutoEventWireup="false" MasterPageFile="~/Template.Master" CodeBehind="imgSet1.aspx.vb" Inherits="CIS4290_Image_Classification_website.WebForm1" %>
<asp:Content ID="Content1" ContentPlaceHolderID="head" runat="server">
</asp:Content>
<asp:Content ID="Content2" ContentPlaceHolderID="ContentPlaceHolder1" runat="server">
    <asp:Label ID="lblImgSet1Code" runat="server" Text="">
import PIL.ImageShow <br />
import matplotlib.pyplot as plt<br />
import numpy as np<br />
import PIL <br />
import tensorflow as tf <br />
import pathlib <br />
from tensorflow.keras import layers <br />
from tensorflow.keras import models<br />
from tensorflow.keras.models import Sequential<br />
    <br />
dataset = pathlib.Path('targetdir/Train') <br />
print(dataset) <br />
image_count = len(list(dataset.glob('*/*.jpg'))) <br />
print("image count " + str(image_count)) <br />
plt.show() <br />
        <br />
batch_size = 64 <br />
img_height = 250  <br />
img_width = 250  <br />
<br />
        <br />
train_ds = tf.keras.preprocessing.image_dataset_from_directory( <br />
  dataset, <br />
  validation_split=0.2, <br />
  subset="training", <br />
  seed=123, <br />
  image_size=(img_height, img_width), <br />
  batch_size=batch_size) <br />
        <br />
        <br />
# Validation <br />
val_ds = tf.keras.preprocessing.image_dataset_from_directory( <br />
  dataset, <br />
  validation_split=0.2, <br />
  subset="validation", <br />
  seed=123, <br />
  image_size=(img_height, img_width), <br />
  batch_size=batch_size) <br />
        <br />
class_names = train_ds.class_names <br />
print(class_names) <br />
        <br />
        <br />
plt.figure(figsize=(10, 10)) <br />
for images, labels in train_ds.take(1): <br />
    for i in range(9):  <br />
        ax = plt.subplot(3, 3, i + 1) <br />
        plt.imshow(images[i].numpy().astype("uint8")) <br />
        plt.title(class_names[labels[i]]) <br />
        plt.axis("on") <br />
        <br />
for image_batch, labels_batch in train_ds:<br />
    print(image_batch.shape) <br />
    print(labels_batch.shape) <br />
    break <br />
        <br />
# display images for val_ds <br />
for images, labels in val_ds.take(1): <br />
    for i in range(9):  <br />
        ax = plt.subplot(3, 3, i + 1) <br />
        plt.imshow(images[i].numpy().astype("uint8")) <br />
        plt.title(class_names[labels[i]]) <br />
        plt.axis("on") <br />
        <br />
for image_batch, labels_batch in val_ds: <br />
    print("val_ds images") <br />
    print(image_batch.shape) <br />
    print(labels_batch.shape) <br />
    break <br />
        <br />
 <br />
AUTOTUNE = tf.data.AUTOTUNE
<br />
        <br />
val_size = int(image_count*0.2) <br />
print("val dataset size " + str(val_size))<br />
train_size = int(image_count*0.8)<br />
print("train dataset size " + str(train_size))<br />
train_ds = train_ds.take(500).cache()<br />
val_ds = val_ds.take(500).cache()<br />
<br />
        <br />
normalization_layer = layers.experimental.preprocessing.Rescaling(1./255) <br />
        <br />
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y)) <br />
image_batch, labels_batch = next(iter(normalized_ds)) <br />
first_image = image_batch[0]<br />
# Notice the pixels values are now in `[0,1]`. <br />
print(np.min(first_image), np.max(first_image))<br />
        <br />
        <br />
model = Sequential([ <br />
  layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),<br />
  layers.Conv2D(8, 3, padding='same', activation='relu'),<br />
  layers.MaxPooling2D(),<br />
  layers.Conv2D(16, 3, padding='same', activation='relu'), <br />
  layers.MaxPooling2D(),<br />
  layers.Conv2D(32, 3, padding='same', activation='relu'), <br />
  layers.MaxPooling2D(),<br />
  layers.Conv2D(64, 3, padding='same', activation='relu'),<br />
  layers.MaxPooling2D(),<br />
  layers.Conv2D(128, 3, padding='same', activation='relu'), <br />
  layers.MaxPooling2D(),<br />
  layers.Flatten(),<br />
  layers.Dense(256, activation='relu'), <br />
  layers.Dense(len(class_names)) <br />
])<br />

model.compile(optimizer='adam', <br />
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),<br />
              metrics=['accuracy'])<br />
        <br />
        <br />
model.summary()<br />
        <br />
epochs = 15<br />
        <br />
history = model.fit(train_ds, <br />
                    validation_data=val_ds,<br />
                    batch_size=batch_size,<br />
                    epochs=epochs)<br />
acc = history.history['accuracy']<br />
val_acc = history.history['val_accuracy']<br />
        <br />
loss = history.history['loss']<br />
val_loss = history.history['val_loss']<br />
        <br />
epochs_range = range(epochs)<br />
        <br />
plt.figure(figsize=(8, 8))<br />
plt.subplot(1, 2, 1)<br />
plt.plot(epochs_range, acc, label='Training Accuracy')<br />
plt.plot(epochs_range, val_acc, label='Validation Accuracy')<br />
plt.legend(loc='lower right')<br />
plt.title('Training and Validation Accuracy')<br />
        <br />
plt.subplot(1, 2, 2)<br />
plt.plot(epochs_range, loss, label='Training Loss')<br />
plt.plot(epochs_range, val_loss, label='Validation Loss')<br />
plt.legend(loc='upper right')<br />
plt.title('Training and Validation Loss')<br />
plt.show()<br />


</asp:Label>
</asp:Content>

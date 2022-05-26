from telegram.ext import Updater, Filters, \
    CommandHandler, MessageHandler
import tensorflow as tf
from tensorflow.keras.layers import RandomRotation, RandomFlip,\
    Resizing, Rescaling, RandomContrast, RandomZoom, RandomCrop
from PIL import Image
import numpy as np
import json
import os

data_dir = 'data'

IMG_SIZE = 224

resize_and_rescale = tf.keras.Sequential([
  Resizing(IMG_SIZE, IMG_SIZE),
  Rescaling(1./255)
])

model = tf.keras.models.load_model(os.path.join(data_dir, 'model_best.hdf5'))

with open(os.path.join(data_dir, 'labels.txt'), 'r', encoding='utf-8') as f:
    labels = f.read()
labels = labels.split('\n')


def start(updater, context):
    updater.message.reply_text('Начнем работу! Пришлите изображение цветка.')


def help_(updater, context):
    updater.message.reply_text('Пришлите изображение цветка. С остальным разберется скрипт.')


def message(updater, context):
    msg = updater.message.text
    print(msg)
    updater.message.reply_text(msg)


def preprocess_image(updater, context):
    img = updater.message.photo[-1].get_file()
    img_path = os.path.join(data_dir, 'img.jpg')
    img.download(img_path)
    img = tf.keras.utils.load_img(img_path)
    img = np.array(img)
    img = tf.expand_dims(img, axis=0)
    img = resize_and_rescale(img)
    preds = model.predict(img)
    any_class = tf.math.reduce_any(preds>0.5)
    print(any_class)
    if any_class:
        index = tf.argmax(preds, axis=1).numpy()[0]
        # print(index)
        print(labels[index])
        updater.message.reply_text(labels[index])
    else:
        print('No  class')
        updater.message.reply_text('Кажется, это не цветок')

configs_path = os.path.join(data_dir, 'configs.json')
with open(configs_path) as f:
    configs_data = json.load(f)

updater = Updater(configs_data['bot_token'])
dispatcher = updater.dispatcher
dispatcher.add_handler(CommandHandler('start', start))
dispatcher.add_handler(CommandHandler('help', help_))
dispatcher.add_handler(MessageHandler(Filters.text, message))
dispatcher.add_handler(MessageHandler(Filters.photo, preprocess_image))

updater.start_polling()
updater.idle()

# import os
#
# from flask import Flask, request
#
# import telebot
#
# TOKEN = '1584253686:AAHEA0l_O4BPLD-DUe3oe_-u1NfnKnGccD0'
# bot = telebot.TeleBot(TOKEN)
# server = Flask(__name__)
#
#
# @bot.message_handler(commands=['start'])
# def start(message):
#     bot.reply_to(message, 'Hello, ' + message.from_user.first_name)
#
#
# @bot.message_handler(func=lambda message: True, content_types=['text'])
# def echo_message(message):
#     bot.reply_to(message, message.text)
#
#
# @server.route(https://api.telegram.org/bot + TOKEN + '/', methods=['POST'])
# def getMessage():
#     bot.process_new_updates([telebot.types.Update.de_json(request.stream.read().decode("utf-8"))])
#     return "!", 200
#
#
# @server.route("/")
# def webhook():
#     bot.remove_webhook()
#     bot.set_webhook(url='https://safe-savannah-20654.herokuapp.com/')
#     return "!", 200
#
#
# if __name__ == "__main__":
#     server.run(host="0.0.0.0", port=int(os.environ.get('PORT', 5000)))


import os

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F
import skimage.io
from skimage.io import imsave
from skimage.transform import resize

from torchvision import transforms
import torch.nn.functional as F
import random
import sklearn
from sklearn.neighbors import NearestNeighbors
import joblib
import neural_net

# Подключаем модуль для Телеграма
from symbol import decorator

import telebot
from flask import Flask, request
# Указываем токен

TOKEN = '1584253686:AAHEA0l_O4BPLD-DUe3oe_-u1NfnKnGccD0'
bot = telebot.TeleBot(TOKEN)


if "HEROKU" in list(os.environ.keys()):
    server = Flask(__name__)
    @server.route('/bot', methods=['POST'])
    def getMessage():
        bot.process_new_updates([telebot.types.Update.de_json(request.stream.read().decode("utf-8"))])
        return "!", 200

    @server.route("/")
    def webhook():
        bot.remove_webhook()
        bot.set_webhook(url='https://safe-savannah-20654.herokuapp.com/bot')
        return "?", 200

    server.run(host="0.0.0.0", port=int(os.environ.get('PORT', 80)))
else:
    bot.remove_webhook()
    bot.polling()
# Импортируем типы из модуля, чтобы создавать кнопки

from telebot import types

# Метод, который получает сообщения и обрабатывает их

@bot.message_handler(commands=['start'])
def start(message):
    bot.reply_to(message, 'Здравствуйте, ' + message.from_user.first_name)

@bot.message_handler(content_types=['text'])
def get_text_messages(message):
    # Если написали «Привет»

    if message.text == "Привет":

        # Пишем приветствие

        bot.send_message(message.from_user.id, "Привет, пришли мне фотографию, чтобы пририсовать хмурому человеку на ней улыбку.")

        # Готовим кнопки

        keyboard = types.InlineKeyboardMarkup()

        # По очереди готовим текст и обработчик для каждого знака зодиака

        key_smile = types.InlineKeyboardButton(text='Приделать улыбку', callback_data='smile')

        # И добавляем кнопку на экран

        keyboard.add(key_smile)

        key_find = types.InlineKeyboardButton(text='Найти совпадения', callback_data='find')

        keyboard.add(key_find)


        bot.send_message(message.from_user.id, text='Выберите что вам интересно сделать с этой фотографией', reply_markup=keyboard)

    elif message.text == "/help":

        bot.send_message(message.from_user.id, "Напиши привет")

    else:

        bot.send_message(message.from_user.id, "Я тебя не понимаю. Напиши /help.")


# Обработчик нажатий на кнопки

# @bot.callback_query_handler(func=lambda call: True)
# def callback_worker(call):
#     # Если нажали на одну из 12 кнопок — выводим гороскоп
#
#     if call.data == "zodiac":
#         # Формируем гороскоп
#
#         msg = random.choice(first) + ' ' + random.choice(second) + ' ' + random.choice(
#             second_add) + ' ' + random.choice(third)
#
#         # Отправляем текст в Телеграм
#
#         bot.send_message(call.message.chat.id, msg)


import pandas as pd


@bot.message_handler(content_types=['photo'])
def photo(message):
    print ('message.photo =', message.photo)
    fileID = message.photo[-1].file_id
    print ('fileID =', fileID)
    file_info = bot.get_file(fileID)
    print ('file.file_path =', file_info.file_path)
    downloaded_file = bot.download_file(file_info.file_path)

    with open("image.jpg", 'wb') as new_file:
        new_file.write(downloaded_file)

    # ресайзим изображение
    dx = dy = 80
    dimx = dimy = 45
    img = skimage.io.imread("image.jpg")
    crop = lambda img: img[dy:-dy, dx:-dx]
    res = lambda img: resize(img, [dimx, dimy])
    img = res(crop(img))
    # img = resize(img, (45, 45))
    imsave("image.jpg",  img)


    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225] )
    ])
    img = img.astype(np.float32)
    img = transform(img)


    #загружаем модель
    model = neural_net.AE(input_shape=6075)
    model.load_state_dict(torch.load("autoencoderConv1layer.pth"))

    #тензор улыбки
    mean_smiling_code = torch.load("mean_smiling_code.pt")

    with torch.no_grad():
        # x = x.view(-1, 6075).to(device)
        img = img.view(1, 3, 45, 45)
        witcher_code = model.encode(img)
        witcher_code = witcher_code  + mean_smiling_code
        reconstruction = model.decode(witcher_code)

    out = reconstruction[0].numpy().transpose((1, 2, 0))
    out = np.clip(out, 0, 1)
    out = resize(out, (100, 100))
    imsave("image.jpg", out)

    # # # подбор похожих фото
    #используем другую модель с меньшим latent space, иначе knn весит гигабайт
    model = neural_net.AE2(input_shape=6075)
    model.load_state_dict(torch.load("autoencodermodel.pth"))

    with torch.no_grad():
        witcher_code = model.encode(img)

    knn = NearestNeighbors(n_neighbors=4, radius=3.0, algorithm='kd_tree', metric='euclidean')
    knn = joblib.load('knn.pth')

    n_neighbors = 5
    (distances,), (idx,) = knn.kneighbors(witcher_code.reshape(1, -1).detach().numpy(), n_neighbors=n_neighbors)

    df_attrs = pd.read_csv("lfw_attributes.txt", sep='\t',
                           skiprows=1, )
    df_attrs = pd.DataFrame(df_attrs.iloc[:, :-1].values, columns=df_attrs.columns[1:])

    bot.send_message(message.from_user.id, "Вероятнее всего на фото " + df_attrs['person'][idx[0]])

    # код для отправки фото клиенту
    img = open("image.jpg", 'rb')
    bot.send_photo(message.from_user.id, img)

    #код который отправлял бы пять похожих фотографий, если бы на хостинге было место под датасет
    # for i in range(n_neighbors):
    #     imsave("lfw-deepfunneled/" + df_attrs['person'][idx[0]] + "/" + df_attrs['person'][idx[0]] //
    #     + "_0001".jpg", out)
    #     img = open("image.jpg", 'rb')
    #     bot.send_photo(message.from_user.id, img)

    # out = reconstruction[0].numpy().transpose((1, 2, 0))
    # out = np.clip(out, 0, 1)
    # out = data[666]
    # imsave("image.jpg", out)



# Запускаем постоянный опрос бота в Телеграме
# bot.polling()


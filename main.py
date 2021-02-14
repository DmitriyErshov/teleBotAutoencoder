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

# Подключаем модуль для Телеграма
from symbol import decorator

import telebot
from flask import Flask, request
# Указываем токен

TOKEN = '1584253686:AAHEA0l_O4BPLD-DUe3oe_-u1NfnKnGccD0'
bot = telebot.TeleBot(TOKEN)
server = Flask(__name__)

@server.route('/safe-savannah-20654', methods=['POST'])
def getMessage():
    bot.process_new_updates([telebot.types.Update.de_json(request.stream.read().decode("utf-8"))])
    return "!", 200


@server.route("/")
def webhook():
    bot.remove_webhook()
    bot.set_webhook(url='https://safe-savannah-20654.herokuapp.com/')
    return "!", 200


if __name__ == "__main__":
    server.run(host="0.0.0.0", port=int(os.environ.get('PORT', 5000)))
    
# Импортируем типы из модуля, чтобы создавать кнопки

from telebot import types

# Заготовки для трёх предложений

first = ["Сегодня — идеальный день для новых начинаний.",
         "Оптимальный день для того, чтобы решиться на смелый поступок!",
         "Будьте осторожны, сегодня звёзды могут повлиять на ваше финансовое состояние.",
         "Лучшее время для того, чтобы начать новые отношения или разобраться со старыми.",
         "Плодотворный день для того, чтобы разобраться с накопившимися делами."]

second = ["Но помните, что даже в этом случае нужно не забывать про", "Если поедете за город, заранее подумайте про",
          "Те, кто сегодня нацелен выполнить множество дел, должны помнить про",
          "Если у вас упадок сил, обратите внимание на",
          "Помните, что мысли материальны, а значит вам в течение дня нужно постоянно думать про"]

second_add = ["отношения с друзьями и близкими.",
              "работу и деловые вопросы, которые могут так некстати помешать планам.",
              "себя и своё здоровье, иначе к вечеру возможен полный раздрай.",
              "бытовые вопросы — особенно те, которые вы не доделали вчера.",
              "отдых, чтобы не превратить себя в загнанную лошадь в конце месяца."]

third = ["Злые языки могут говорить вам обратное, но сегодня их слушать не нужно.",
         "Знайте, что успех благоволит только настойчивым, поэтому посвятите этот день воспитанию духа.",
         "Даже если вы не сможете уменьшить влияние ретроградного Меркурия, то хотя бы доведите дела до конца.",
         "Не нужно бояться одиноких встреч — сегодня то самое время, когда они значат многое.",
         "Если встретите незнакомца на пути — проявите участие, и тогда эта встреча посулит вам приятные хлопоты."]


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

        key_oven = types.InlineKeyboardButton(text='Овен', callback_data='zodiac')

        # И добавляем кнопку на экран

        keyboard.add(key_oven)

        key_telec = types.InlineKeyboardButton(text='Телец', callback_data='zodiac')

        keyboard.add(key_telec)

        key_bliznecy = types.InlineKeyboardButton(text='Близнецы', callback_data='zodiac')

        keyboard.add(key_bliznecy)

        key_rak = types.InlineKeyboardButton(text='Рак', callback_data='zodiac')

        keyboard.add(key_rak)

        key_lev = types.InlineKeyboardButton(text='Лев', callback_data='zodiac')

        keyboard.add(key_lev)

        key_deva = types.InlineKeyboardButton(text='Дева', callback_data='zodiac')

        keyboard.add(key_deva)

        key_vesy = types.InlineKeyboardButton(text='Весы', callback_data='zodiac')

        keyboard.add(key_vesy)

        key_scorpion = types.InlineKeyboardButton(text='Скорпион', callback_data='zodiac')

        keyboard.add(key_scorpion)

        key_strelec = types.InlineKeyboardButton(text='Стрелец', callback_data='zodiac')

        keyboard.add(key_strelec)

        key_kozerog = types.InlineKeyboardButton(text='Козерог', callback_data='zodiac')

        keyboard.add(key_kozerog)

        key_vodoley = types.InlineKeyboardButton(text='Водолей', callback_data='zodiac')

        keyboard.add(key_vodoley)

        key_ryby = types.InlineKeyboardButton(text='Рыбы', callback_data='zodiac')

        keyboard.add(key_ryby)

        # Показываем все кнопки сразу и пишем сообщение о выборе

        bot.send_message(message.from_user.id, text='Выбери свой знак зодиака', reply_markup=keyboard)

    elif message.text == "/help":

        bot.send_message(message.from_user.id, "Напиши привет")

    else:

        bot.send_message(message.from_user.id, "Я тебя не понимаю. Напиши /help.")


# Обработчик нажатий на кнопки

@bot.callback_query_handler(func=lambda call: True)
def callback_worker(call):
    # Если нажали на одну из 12 кнопок — выводим гороскоп

    if call.data == "zodiac":
        # Формируем гороскоп

        msg = random.choice(first) + ' ' + random.choice(second) + ' ' + random.choice(
            second_add) + ' ' + random.choice(third)

        # Отправляем текст в Телеграм

        bot.send_message(call.message.chat.id, msg)


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

    from skimage import color
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

    import torch.nn.functional as F

    class AE(nn.Module):
        def __init__(self, input_shape):
            super().__init__()

            # Encoder
            self.conv1 = nn.Conv2d(3, 26, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(26)
            self.pool = nn.MaxPool2d(2, 2)

            # Decoder
            self.t_conv3 = nn.ConvTranspose2d(26, 3, 3, stride=2)

        def encode(self, x):
            x = F.relu(self.conv1(x))
            x = self.pool(x)
            self.bn1
            return x

        def decode(self, x):
            x = F.sigmoid(self.t_conv3(x))
            return x

        def forward(self, x):
            code = self.encode(x)
            return self.decode(code)

    model = AE(input_shape=6075)
    model.load_state_dict(torch.load("autoencoderConv1layer.pth"))

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

    # код для отправки фото клиенту
    img = open("image.jpg", 'rb')


    # # # подбор похожих фото
    # knn = NearestNeighbors(n_neighbors=4, radius=3.0, algorithm='kd_tree', metric='euclidean')
    # knn = joblib.load('knn2.pth')
    #
    # n_neighbors = 5
    # (distances,), (idx,) = knn.kneighbors(witcher_code.reshape(1, -1).detach().numpy(), n_neighbors=n_neighbors)
    #
    # df_attrs = pd.read_csv("lfw_attributes.txt", sep='\t', skiprows=1, )
    # bot.send_message(message.from_user.id, "Вероятнее всего на фото " + df_attrs['person'][idx[0]])
    # for i in range(n_neighbors):
    #     imsave("lfw-deepfunneled/" + df_attrs['person'][idx[0]] + "/" + df_attrs['person'][idx[0]] //
    #     + "_0001".jpg", out)
    #     img = open("image.jpg", 'rb')
    #     bot.send_photo(message.from_user.id, img)

    # out = reconstruction[0].numpy().transpose((1, 2, 0))
    # out = np.clip(out, 0, 1)
    # out = data[666]
    # imsave("image.jpg", out)

    # img = open("image.jpg", 'rb')
    bot.send_photo(message.from_user.id, img)
# Запускаем постоянный опрос бота в Телеграме
# bot.polling()


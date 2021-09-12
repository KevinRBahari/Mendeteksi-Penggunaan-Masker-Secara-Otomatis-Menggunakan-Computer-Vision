import tensorflow.keras
import numpy as np
import cv2
import discord

# discord client logic
client = discord.Client()
message_di_discord = "Ada orang tidak memakai masker!!!"
# masukan token yang diberikan oleh discord dibawah ini
token_dsc = "#token discord"
# Setting
np.set_printoptions(suppress=True)

# Masukan Model sesuai nama file
model = tensorflow.keras.models.load_model('keras_machine_learning_model.h5')

# Data Array
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Camera, Ubah "1" menjadi 0,2,3 bila Kamera tidak terdeteksi
cam = cv2.VideoCapture(1)

text = ""

#statement while
while True:
    _,img = cam.read("training1.jpg")
    img = cv2.resize(img,(224, 224))
    image_array = np.asarray(img)
    normalisasi_image_array = (image_array.astype(np.float32) / 127.0) - 1

    data[0] = normalisasi_image_array
    prediksi = model.predict(data)
    for i in prediksi:
        # pakai masker
        if i[0] > 0.7:
            text ="Memakai Masker"
        # tidak pakai masker
        if i[1] > 0.7:
            text ="Tidak Memeakai Masker"
            # tambahkan perintah API discord untuk menambahkan fitur notifikasi discord disini
            @client.event
            async def on_message(message):
                message.channel.send(message_di_discord)
            client.run(token_dsc)
        # Bounding Box
        img = cv2.resize(img,(500, 500))
        cv2.putText(img,text,(10,30),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(0,255,0),1)
    cv2.imshow('img',img)
    if cv2.waitkey(1) & 0xFF ==ord('q'):
        break

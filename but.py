from aiogram.types import ReplyKeyboardMarkup, KeyboardButton

btnHello = KeyboardButton("/Summer")
btnHello1 = KeyboardButton("/Winter")
greet_kb = ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True).add(btnHello).add(btnHello1)

button1 = KeyboardButton("/Summer")
button2 = KeyboardButton("/Winter")

markup2 = ReplyKeyboardMarkup().add(button1).add(button2)
markup = ReplyKeyboardMarkup().row(button1, button2 )

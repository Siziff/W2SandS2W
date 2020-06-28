from config import TOKEN
import logging
from aiogram import Bot, Dispatcher, executor, types
import but
from testtt import *
from model import *
from torchvision import transforms

logging.basicConfig(level=logging.INFO)

bot = Bot(token=TOKEN)
dp = Dispatcher(bot)

G_XtoY = CycleGenerator(conv_dim=3, n_res_blocks=6)                                 # Генератор из лета в зиму
G_XtoY.load_state_dict(torch.load("G_XtoY.pkl"), True)
G_YtoX = CycleGenerator(conv_dim=3, n_res_blocks=6)                                 # Генератор из зимы в лето
G_YtoX.load_state_dict(torch.load("G_YtoX.pkl"), True)

size = 256
transforms_ = [transforms.Resize(size, Image.BICUBIC),
               transforms.ToTensor(),
               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

trs = transforms.Compose(transforms_)


def transformS2W(root):
    global trs
    im = get_img(root)
    im = scale(im)
    im = im.cpu()
    output = G_XtoY(im)
    tensor_save_bgrimage(output.data[0], 'outW.jpg', False)


def transformW2S(root):
    global trs
    im = get_img(root)
    im = scale(im)
    im = im.cpu()
    output = G_YtoX(im)
    tensor_save_bgrimage(output.data[0], 'outS.jpg', False)


@dp.message_handler(commands=['Winter'])
async def process_help(message: types.Message):
    await bot.send_message(message.from_user.id, 'Скинь картинку чего-нибудь летнего, и зделаю из нее зиму ❄')

    @dp.message_handler(content_types=['photo'])
    async def process_help1(message: types.Message):
        await message.photo[-1].download('Winter.jpg')

        await message.answer(text='Значит зима... Сейчас посмотрим, что я смогу с этим сделать...')
        transformS2W('Winter.jpg')
        with open('outW.jpg', 'rb') as file:
            await message.answer_photo(file, caption='Вжух!')


@dp.message_handler(commands=['Summer'])
async def process_help(message: types.Message):
    await bot.send_message(message.from_user.id, 'Скинь картинку чегонибудь зимнего, и сделаю из нее лето 🔆')

    @dp.message_handler(content_types=['photo'])
    async def process_help2(message: types.Message):
        await message.photo[-1].download('Summer.jpg')
        await message.answer(text='Значит лето... Сейчас посмотрим, что я смогу с этим сделать...')
        transformW2S('Summer.jpg')
        with open('outS.jpg', 'rb') as file:
            await message.answer_photo(file, caption='Вжух!')


@dp.message_handler(commands=['start'])
async def process_hello(message: types.Message):
    await bot.send_message(message.from_user.id, 'Привет, эт магический бот который превратит '
                                                 'твои зимние фотографии в летиние или наоборот, как сочтешь нужным.'
                                                 'Выбери /Winter если хочешь увидеть зиму либо /Summer если лето. ',
                           reply_markup=but.markup2)


@dp.message_handler(content_types=types.ContentType.TEXT)
async def process_UNK(message: types.Message):
    text = message.text
    if text:
        await message.reply(text='Вот это сообщение не совсем понятно сейчас было, давай по сценарию')


@dp.message_handler(commands=['Help'])
async def process_help(message: types.Message):
    await bot.send_message(message.from_user.id, 'Не понимаю с чем тебе помочь, все же просто.'
                                                 'Напиши еще раз /start чтобы все с начала начать')


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)




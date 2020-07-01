import logging
from aiogram import Bot, Dispatcher, executor, types
from networkx.drawing.tests.test_pylab import plt
import but
from config import *
from testtt import *
from model import *
from torchvision import transforms
import PIL
import gc

logging.basicConfig(level=logging.INFO)


def imshow(inp, title=None, plt_ax=None):
    if plt_ax is None:
        plt_ax = plt.gca()
    if title is None and type(inp) is tuple:
        inp, title = inp
    if type(inp) is tuple:
        inp = inp[0]
    if type(inp) is PIL.Image:
        inp = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])(inp)
    if type(inp) is torch.Tensor:
        inp = transforms.Compose([
            transforms.ToPILImage()])
        inp = inp.cpu()
    inp = inp.transpose((1, 2, 0))
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt_ax.imshow(inp)
    if title is not None:
        plt_ax.set_title(title)


bot = Bot(token=TOKEN)
dp = Dispatcher(bot)

G_XtoY = CycleGenerator(conv_dim=3, n_res_blocks=6)  # Генератор из лета в зиму
G_XtoY.load_state_dict(torch.load("G_XtoY.pkl", map_location=torch.device('cpu')), False)
G_YtoX = CycleGenerator(conv_dim=3, n_res_blocks=6)  # Генератор из зимы в лето
G_YtoX.load_state_dict(torch.load("G_YtoX.pkl", map_location=torch.device('cpu')), False)


def transformW2S(root):
    im = get_img(root)[None, ...].cpu()
    im = scale(im)
    output = G_YtoX(im)
    output = to_data(output)
    output = Image.fromarray(output)
    output.save('outS.jpg')

    del im
    del output
    torch.cuda.empty_cache()
    gc.collect()


def transformS2W(root):
    im = get_img(root)[None, ...].cpu()
    im = scale(im)
    output = G_XtoY(im)
    output = to_data(output)
    output = Image.fromarray(output)
    output.save('outW.jpg')

    del im
    del output
    torch.cuda.empty_cache()
    gc.collect()


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

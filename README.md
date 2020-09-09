Style-TransferW2S and S2W-Telegram-Bot
=========================================

Разработка и обучение модели
-------------------------------
@Testlolkek112bot  - сам бот (Бесплтаная версия AWS закончилась, так что могу только по запросу с компьютера включить бота)

_}{очу заранее извиниться за названия файлов, они немного не соответсвуют их содержанию. И в момент создания названия телеграм-бота я не планировал что он будет не тестовый, однако все пошло не попалану и нестоый бот с глупым названием стал основным.._

За основу создания модели которую собрался обучать, брался [данный репазиторий](https://github.com/vikashChouhan/CycleGan_weather_Summer2Winter-PyTorch), большое спасибо автору. Тем не менее в нем были сделаны достаточно значительные изменения. Генератор и дискриминатор создавался более продвинутый (на мой взгляд) и за основу брались [вот этот код](https://github.com/aitorzip/PyTorch-CycleGAN/blob/master/models.py) . В них так же были сделаны незначительные изменения. 

Было интересно самому с самого начало обучить сеть, так что сеть обучалась в Google Colab около  12000 эпох около 10 часов, после этого колаб вылетал. Даже при создании новых аккаунтов колаб не смог повторить свой предыдущий успех и останавливал свое обучение на 1-2 тыс. эпохе, так что веса я брал из первой попытки обучения. Размер бачей был всего 4 с  num_workers=2  с разрешением картинки 172 на 172.

На конец обучения сеть выдавала достаочно хорошие результаты (наставник проекта так же это подтвердил). 
![screenshot of sample](https://sun1-94.userapi.com/y9qqfs8Gac219DfY0dc1yRxKaNYhljLcFoymWQ/GL9bU90aQAw.jpg)
![screenshot of sample](https://sun1-18.userapi.com/6Yd9HIYVdfdriEtL96LokhYhdgeqVfS2g9UjDg/dpDkfg8u8es.jpg)

 Получилась из одной и той же картинке на тесте СУПЕРлето и зима (даже вода замерзла).
 
 Создание и деплой бота 
-------------------------------
 
 Отлично! модель обучена, я вижу красивые картинки зимы и лета, пол работы сделано(как я думал). Далее создал проект в PyCharm, достал веса с генераторов, и тут начались муки. Как оказалось для формата файла весов .pkl тренерованного на GPU критически не подходит тот код что я написал. Картинки в бот отправлялись либо в негативе, либо черные картинки. Ошибки к счатью получилось исправить, спасибо переносу дедлайнов, но на это потребовалось больше всего времени. 
 Сам бот писался на фреймворке [aiogram](https://docs.aiogram.dev/en/latest/index.html) для получения асинхронности бота. 
 
 Далее деплоил бота на AWS, благодаря [этому туториалу](https://github.com/hse-aml/natural-language-processing/blob/master/AWS-tutorial.md) данный этап прошел без каких либо затруднений. 
 
 ![screenshot of sample](https://sun1-22.userapi.com/jaTCznwYqFnOToOGZcLNxHzYmQ7QZUFhAg91CA/CfrbmTyifNM.jpg)
 
  
 Заключение
-------------------------------

Как же я был рад первой нормальной обработанной картинке полученной от бота в ответ.

 ![screenshot of sample](https://sun1-28.userapi.com/HW7oilYkV0QL4GMarHDFwBToLWnVZCUP4PduOQ/aXs4d7vZkrQ.jpg)
 ![screenshot of sample](https://sun1-88.userapi.com/9eKySOn9lqA28oH6xhIcBXPhuYPbkRHImSN4Ng/UTV2hfEDmD4.jpg)
 
 Зима конечно не так классно как лето обрабатывается, однако даже на такой совершенно заснеженной картинке был частично убран снег и елки поменяли свой цвет. 
 
Однако сеть обучена, выправлены все косяки, асинхронный бот создан и задеплоен И ОН ДАЖЕ ПРАВИЛЬНО ОТВЕЧАЕТ !!! Так что батут работает :) Спасибо авторам курса [Deep Learning School](https://vk.com/dlschool_mipt) за лекции и задания.
 

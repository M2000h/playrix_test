Надо с чего-то начать... А то они догадаются... Так, Гена... Геннадий

# Playrix Test Task 💚

Алгоритм дообучения модели FLUX.1 на примере локаций Москвы в стиле [Simon Stålenhag](https://www.simonstalenhag.se/).

Действия описываются в хронологическом порядке, но в некоторых местах для удобства объеденины по смыслу.

Некоторые пункты могут быть немного упрощены (например часть инференсов, процесс подгонки параметров и т.д),
чтобы не растягивать и без того большое описание,
поэтому, если какой-то вывод кажется безосновательным, лучше уточнить напрямую.

## 📋 Содержание

1. [Подготовка](#%EF%B8%8F-подготовка)
2. [Network Storage](#-network-storage)
3. [Загрузка модели](#-загрузка-модели)
4. [Загрузка датасетов](#-загрузка-датасетов)
5. [Обучение](#Обучение)
5. [Pod](#-pod)
6. [Процесс](#-процесс)
7. [Flux Kontext](#-flux-kontext)
8. [Промптинг переобученной на стиль модели](#%EF%B8%8F-промптинг-переобученной-на-стиль-модели)
9. [Доубечение стилю и конкретной локации](#-доубечение-стилю-и-конкретной-локации)
10. [Сравнение](#%EF%B8%8F-сравнение)
11. [Опробованные методы](#-опробованные-методы)
12. [Nano Banana](#-nano-banana)
13. [GPT-Image](#-gpt-image)
14. [Итог сравнения](#-итог-сравнения)
15. [Что можно улучшить](#-что-можно-улучшить)

## ⚙️ Подготовка

Все действия выполняются на платформе [runpod.io](https://runpod.io/)

### 💾 Network Storage

Сначала создадим Network Storage для хранения данных между подами.

При настройке Network Storage желательно, чтобы соблюдались все условия:

1. [x] Доступна Global Networking
2. [x] Доступен S3 Compatible
3. [x] Доступны GPU Nvidia H100 или H200
4. [x] Объем памяти не менее 150ГБ

<img src="/images/storage.png" width="700px" />

Note: поды лучше создавать по кнопке из объекта хранилища, чтобы не забывать подключить их:

<img src="/images/storage1.png" width="700px" />

Далее при создании подов хранилище будет маунтится в папку `/workspace`.

### 🧪 Загрузка модели

Загрузим модель FLUX.1 в хранилище, чтобы избежать потери времени и ресурсов при повторном скачивании.

Для этого возьмем ЦПУ под с увеличенной RAM, т.к huggingface скачивает веса одним файлом, прогоняя через
оперативную память, и если ее не хватит — процесс прервется.

<img src="/images/cpu_pod_1.png" width="700px" />

Установим пакет huggingface:

```shell
pip install "huggingface_hub>=0.24.6"
```

Скачиваем веса:

```python
from huggingface_hub import snapshot_download
import shutil

LOCAL_DIR = "/workspace/models/flux-dev"
HF_TOKEN = "..."  # Токен huggingface можно получить через личный кабинет: https://huggingface.co/settings/tokens

shutil.rmtree(LOCAL_DIR, ignore_errors=True)

local_dir = snapshot_download(
    repo_id="black-forest-labs/FLUX.1-dev",
    token=HF_TOKEN,
    local_dir=LOCAL_DIR,
    local_dir_use_symlinks=False,
    resume_download=True,
    max_workers=2,  # Качаем из облака в облако, можно не мелочиться
    allow_patterns=[
        "*.json", "*config*", "*.txt", "*tokenizer*", "*.model", "*.vocab", "*.spiece",
        "*.safetensors", "*.bin"
    ],
)
print("Snapshot at:", local_dir)
```

После этого в папке `/workspace/models/flux-dev` должны появится файлы:

```
ae.safetensors  flux1-dev.safetensors  model_index.json  scheduler  
text_encoder  text_encoder_2  tokenizer  tokenizer_2  transformer  vae
```

### 📚 Загрузка датасетов

Далее можно использовать под с минимальным CPU и RAM.

Загружаем картинки художника в максимальном разрешении. Они доступны по ссылке с указанием порядкового номера:

```python
from tqdm import tqdm
import requests

for i in tqdm(range(100)):
    response = requests.get(f"https://www.simonstalenhag.se/4k/svema_{i}_big.jpg", stream=True)
    if not response.ok:
        continue
    with open(f'/workspace/dataset/origs/{i}.jpg', 'wb') as f:
        for chunk in response:
            f.write(chunk)
```

Так же опишем скрипт для скачивания картинок из интернета, он нам понадобится позже:

```python
import requests
import os

Kremlin = [  # название объекта
    # список url строк на .jpg картинки #
]

folder = "./dataset/Kremlin"  # папка с датасетом объекта
os.makedirs(folder, exist_ok=True)

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}  # Обход анти-бот систем

# Скачивание картинок
for i, url in enumerate(Kremlin, 1):
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        filename = f"{i}.jpg"
        filepath = os.path.join(folder, filename)

        with open(filepath, "wb") as f:
            f.write(response.content)

        print(f"Скачано: {filename}")

    except Exception as e:
        print(f"Ошибка при скачивании {url}: {e}")

print("Завершено!")
```

## 👨‍🏫 Обучение

### 📟 Pod

Для обучения лучше всего подходит GPU Nvidia H200, если ее нет, можно придерживаться принципа "мощнее — лучше".

Note: загрузка весов происходит через RAM, поэтому RAM лучше не брать меньше 96ГБ, иначе загрузка прервется.

<img src="/images/H200.png" width="700px" />

Выбираем последний образ ubuntu с pytorch, 1 CPU, Spot (прерываемый) под
(он сильно дешевле, но нужно поставить свечку за то, что на 999 итерации он не выключится),
сразу включаем SSH и Jupyter Notebook:

<img src="/images/H200_1.png" width="700px" />

После запуска нужно перейти в директорию `/workspace` и установить репозиторий `huggingface/diffusers`:

```shell
git clone https://github.com/huggingface/diffusers
cd diffusers/examples/dreambooth
pip install accelerate
accelerate config default
```

### 📝 Процесс

Изначально была предпринята попытка самостоятельно написать файл для обучения flux модели
с использованием Lora, однако были сложности с тем, чтобы подружить данные (например тензоры)
между гпу и цпу, так же ситуацию осложнял тот факт, что поды "одноразовые", т.е при выключении
удаляются, поэтому с каждым новым запуском приходилось заново устанавливать все библиотеки и решать конфликты версий
между ними.

Поэтому, чтобы не тратить время, был взят готовый файл из примеров библиотеки - `train_dreambooth_lora_flux.py`.

Для начала установим зависимости (они довольно туго подгонялись, поэтому лучше установить as-is как написано здесь):

```shell
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu121
pip install transformers accelerate safetensors datasets peft torchao tensorboard
pip uninstall -y flash-attn triton
pip install --no-cache-dir "git+https://github.com/huggingface/diffusers.git@main"
pip install --upgrade --no-cache-dir bitsandbytes tokenizers sentencepiece protobuf
```

Note: эти зависимости устанавливаются после каждого пересоздания пода, все предыдущие команды нужно выполнить только
один раз для записи в общее хранилище.

Пробуем запустить обучение:

```shell
export HF_MODEL_ID="/workspace/models/flux-dev"
export INSTANCE_DIR="/workspace/dataset/origs"
export OUTPUT_DIR="/workspace/weights/style"
export CUDA_VISIBLE_DEVICES=0

cd /workspace/diffusers/examples/dreambooth 

sudo accelerate launch train_dreambooth_lora_flux.py \
  --pretrained_model_name_or_path "$HF_MODEL_ID" \
  --instance_data_dir "$INSTANCE_DIR" \
  --instance_prompt "a painting in sksartist style" \
  --rank 64 \
  --lora_alpha 64 \
  --lora_dropout 0.0 \
  --train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 \
  --max_train_steps 1000 \
  --checkpointing_steps 200 \
  --resolution 1024 \
  --mixed_precision bf16 \
  --gradient_checkpointing \
  --use_8bit_adam \
  --report_to tensorboard \
  --output_dir "$OUTPUT_DIR" \
  --resume_from_checkpoint checkpoint-500
```

Веса сохранились в `/workspace/weights/style`, пробуем инференс:

```shell
from diffusers import FluxPipeline
import torch

pipe = FluxPipeline.from_pretrained("/workspace/models/flux-dev", torch_dtype=torch.bfloat16).to("cuda")

# Подключаем LoRA
pipe.load_lora_weights("/workspace/weights/style/checkpoint-1000")

prompt = "Moscow Red Square, in sksartist style"
image = pipe(prompt, num_inference_steps=25, guidance_scale=3.5, width=1024, height=1024).images[0]
image.save("test.png")
```

По запросу `Moscow Red Square, in sksartist style` получаем:

<img src="/images/kremlin1_.png" width="700px" />

На первый взгляд неплохо. Есть куда работать, но на Красную Площадь похоже, стиль художника сохранен (метрики качества
стиля подключим позже).

Пробуем дальше, `Moscow State University, in sksartist style`:

<img src="/images/MSU1.png" width="700px" />

Видимо, нейросеть не знает, как выглядит МГУ. (небольшой забавный факт — левое двух башенное здание на фоне скорее
всего результат обучение на двух башенной конструкции, которая периодически есть в артах художника)

Попробуем попроще, `Moscow, in sksartist style`:

<img src="/images/Moscow1.png" width="700px" />

Видимо, слишком просто. Пробуем снова `Moscow Red Square, in sksartist style`:

<img src="/images/kremlin2.png" width="700px" />

Можно еще чуть-чуть подкрутить промпт, но главное, что понятно, — нейросеть не знает, как выглядит
Москва за пределами красной площади.
В целом ожидаемо конечно, нейросети не обучают на каждом городе отдельно,
в них столько не влезет, максимум — крупные достопримечательности.

Далее три варианта:

1. Дообучить модель на нескольких локациях (например популярных местах), которые мы хотим нарисовать;
2. Совершенствовать промпт для того, чтобы картинка лучше сходилась с реальностью;
3. Взять готовую фотографию какой-то локации в городе, и прогнать ее через модель (image-to-image);

Попробуем все три варианта, начиная с первого.

Берем скрипт для скачивания изображений из интернета, описанный выше, скачиваем 20 фотографий кремля.
Далее берем обученную на стиль художника в 1000 итераций модель и дообучаем еще на 500 итераций:

```shell
# Та же команда только меняем директорию
export INSTANCE_DIR="/workspace/dataset/kremlin"
# Меняем описание
# --instance_prompt "Moscow Kremlin" \
# увеличиваем число итераций на 500
# --max_train_steps 1500 \
# указываем начать с 1000 итерации
# --resume_from_checkpoint checkpoint-1000
```

Посмотрим результат по запросу `Moscow Kremlin, in sksartist style`:

<p float="left">
  <img src="/images/kremlin3.png" width="40%" />
  <img src="/images/kremlin4.png" width="40%" />
</p>

Модель выучила, как выглядит кремль, но "забыла" стиль художника, ожидаемо.

Попробуем сделать два раздельных LoRA и создать их композиция на инференсе.

Но перед этим снова обучим модель с 0 на стиль художника,
но уже **в разрешении 1024 вместо 768 так же на 1000 итераций**,
чтобы улучшить мелкие детали. Посмотрим, что получилось:

`urban area, several residential buildings, a sidewalk with cars, two children on the sidewalk, in sksartist style`

<p float="left">
  <img src="/images/test1.png" width="40%" />
  <img src="/images/test2.png" width="40%" />
</p>

<p float="left">
  <img src="/images/test3.png" width="40%" />
  <img src="/images/test4.png" width="40%" />
</p>

Стоит признать, по вайбам уже похоже на Москву. Чуть-чуть подкрутим инференс:

`image = pipe(prompt, num_inference_steps=36, guidance_scale=3.5, width=1024, height=1024).images[0]`

<p float="left">
  <img src="/images/test5.png" width="40%" />
  <img src="/images/test6.png" width="40%" />
</p>

<p float="left">
  <img src="/images/test7.png" width="40%" />
  <img src="/images/test8.png" width="40%" />
</p>

По стилю и наполнению очень хорошо. Отметим, что нужно будет разобраться с мелками деталями, особенно с лицами.

Возвращаемся к обучению локациям. Запускаем кремль (20 фотографий) с параметрами:

```shell
  --rank 32 \
  --lora_alpha 64 \
  --lora_dropout 0.0 \
  --train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --learning_rate 5e-4 \
  --max_train_steps 800 \
  --checkpointing_steps 200 \
  --resolution 768
```

Далее есть несколько вариантов нарисовать картинку в заданном стиле:

### 🎨 Flux Kontext

Попробуем генерировать картинки с помощью перерисовки с обычных фотографий c промптом `in <sksartist> style`:

<img src="/images/city_flux_1.png" width="700px" />

<img src="/images/msu_flux_1.png" width="700px" />

Не попали в стиль. Снизим вес адаптера стиля с 1 до 0.5 guidance_scale с 3.5 до 2.5:

<img src="/images/city_flux_2.png" width="700px" />

<img src="/images/msu_flux_2.png" width="700px" />

Уже получше (подписать вставил сам flux kontext), но всё еще не дотягивает.
Возможно, модели FLUX.1-Redux-dev и FLUX.1-IP-Adapter будут работать лучше.

### ✍️ Промптинг переобученной на стиль модели

При хорошей работе с промптами, можно добиться хороших результатов пусть и без конкретных локаций,
хотя местами есть низкая детализация которая обуславливается ограниченностью ресурсов на обучение и инференс:

<img src="/images/moscow_6.png" width="700px" />

<img src="/images/moscow_1__.png" width="700px" />

<img src="/images/moscow_2.png" width="700px" />

<img src="/images/moscow_3.png" width="700px" />

<img src="/images/moscow_5.png" width="700px" />

### 📍 Доубечение стилю и конкретной локации

В качестве примера модель была дообучена на 20 фотографиях кремля:

<img src="/images/kremlin_res_1.png" width="700px" />

<img src="/images/kremlin_res_2.png" width="700px" />

<img src="/images/kremlin_res_3.png" width="700px" />

<img src="/images/kremlin_res_4.png" width="700px" />



## ⚖️ Сравнение

### ✅ Опробованные методы

В данном контексте идет речь об использование моделей на базе FLUX.

| Метод                                     | Стиль                                                                                             | Локации            | Сложность                                                                                                                    |
|-------------------------------------------|---------------------------------------------------------------------------------------------------|--------------------|------------------------------------------------------------------------------------------------------------------------------|
| Использовать только промпты, без обучения | Приемлемый, но есть предел схожести с референсом                                                  | низкая точность    | Простая, не требует дообучения                                                                                               |
| Дообучение стилю                          | Хороший                                                                                           | низкая точность    | Дообучить модель на стиль (1000 итераций, разрешение 1024x1024)                                                              |
| Дообучение стилю и локациям               | Хороший                                                                                           | высокая точность   | Дообучить модель на стиль (1000 итераций, разрешение 1024x1024) и каждую локацию отдельно (800 итераций, разрешение 768x768) |
| Дообучение стилю + image-to-image         | Плохой для модели FLUX.1-Kontext-dev, возможно для FLUX.1-Redux-dev FLUX.1-IP-Adapter будут лучше | точное копирование | Дообучить модель на стиль (1000 итераций, разрешение 1024x1024)                                                              |

Далее выбор модели зависит от требований ТЗ, в первую очередь от того что мы хотим рисовать: локации с точным совпадением, просто атмосферные улицы и т.д.

### 🍌 Nano Banana

При загрузке нескольких референсов на промпт `Moscow State University, sunrise`
сгенерировалась картинка:

<img src="/images/nanobanana.png" width="700px" />

### 🧠 GPT-Image

При загрузке нескольких референсов на промпт `нарисуй МГУ на рассвете в стиле прикрепленных фотографий`
сгенерировалась картинка:

<img src="/images/gpt.png" width="700px" />

### 🧾 Итог сравнения

Судя по детализации здания на картинках Nano Banana и GPT-Image, в их основе лежит
image-to-image плюс text-to-image, т.е совмещается сразу несколько способов.
Гораздо большие мощности для обучения и инференса увеличивают четкость детализации.
Но главное преимущество — поиск и подгрузка фотографий из базы/интернета, что позволяет строить качественный
результат за счет image-to-image.

## 🌱 Что можно улучшить

1. Улучшить промпты
2. Сделать аугментацию изображений
   (тем более что исходные картинки в 4К, есть куда обрезать);
3. Подтьюнить параметры обучения и инференса;
4. Добавить модели для четкой отрисовки текста;
5. У художника есть еще несколько коллекций изображений, их можно добавить в датасет;
6. Научить модель рисовать, имитируя рельеф холста;
7. Добавить оценку качества с помощью других моделей;

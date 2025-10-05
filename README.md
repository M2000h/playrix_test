Надо с чего-то начать... А то они догадаются... Так, Гена... Геннадий

# Playrix Test Task

Алгоритм дообучения модели FLUX.1 на примере локаций Москвы в стиле [Simon Stålenhag](https://www.simonstalenhag.se/).

Действия описываются в хронологическом порядке, но в некоторых местах для удобства объеденины по смыслу.

## Содержание

1. Подготовка
2. Обучение

## Подготовка

Все действия выполняются на платформе runpod.io

### Network Storage

Первым делом снимаем Network Storage, на нем будем хранить данные, и через него поды будут обмениться информацией.

Под текущую задачу хватило 150 гигабайт.

При выборе локации Network Storage желательно, чтобы соблюдались все условия:

1. [x] Global Networking
2. [x] S3 Compatible
3. [x] Доступа GPU Nvidia H100 или H200

<img src="/images/storage.png" width="700px" />

Note: поды лучше создавать по кнопке из объекта хранилища, чтобы не забыть его подключить:

<img src="/images/storage1.png" width="700px" />

### Загрузка модели

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
HF_TOKEN = "..."  # Токен можно получить через личный кабинет huggingface: https://huggingface.co/settings/tokens

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

### Загрузка датасетов

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

## Обучение

### Pod

Для обучения лучше всего подходит GPU Nvidia H200, если ее нет, можно придерживаться принципа — мощнее лучше.

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

### Процесс

Изначально была предпринята попытка самостоятельно написать файл для обучения flux модели
с использованием Lora, однако были сложности с тем, чтобы подружить данные (например тензоры)
между гпу и цпу, так же ситуацию осложнял тот факт, что под "одноразовый", т.е при выключении
удаляется, поэтому с каждым новым запуском приходилось заново устанавливать все библиотеки и решать конфликты версий
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

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to("cuda")

# Подключаем LoRA
pipe.load_lora_weights("/workspace/weights/style/checkpoint-1000")

prompt = "Moscow Red Square, in sksartist style"
image = pipe(prompt, num_inference_steps=25, guidance_scale=3.5, width=1024, height=1024).images[0]
image.save("test.png")
```

По запросу `Moscow Red Square, in sksartist style` получаем:

<img src="/images/kremlin1.png" width="700px" />

На первый взгляд неплохо. Есть куда работать, но что-то похожее на Красную Площадь есть. Стиль художника похож.

Пробуем дальше, `Moscow State University, in sksartist style`:

<img src="/images/MSU1.png" width="700px" />

Что ж, видимо, нейросеть не знает, как выглядит МГУ.

Попробуем попроще, `Moscow, in sksartist style`:

<img src="/images/Moscow1.png" width="700px" />

Может что-то не так? Пробуем снова `Moscow Red Square, in sksartist style`:

<img src="/images/kremlin2.png" width="700px" />

Диагноз понятен — нейросеть не знает, как выглядит Москва,
ее познания ограничивается Кремлем.
В целом ожидаемо конечно, нейросети не обучают на каждом городе отдельно,
в них столько не влезет, максимум — крупные достопримечательности.

Далее два варианта:

1. Совершенствовать промпт для того, чтобы картинка лучше сходилась с реальностью;
2. Дообучить модель на нескольких локациях, которые мы хотим нарисовать.

Первый вариант мне не понравился, потому что:

* Локации в Москве широко известны, поэтому несостыковки с реальностью будут бросаться в глаза;
* Хотелось бы иметь модель, которая генерирует "правильную" картинку без необходимости каждый раз писать большой промпт
  с описанием локации;
* Модель ограничена небольшими вычислительными ресурсами, поэтому будут сложности с большими промптами;

Поэтому выбираем дообучать на локациях. Первое, что приходит в голову:

* Кремль
* МГУ
* Moscow city

Берем скрипт для скачивания изображений, описанный выше, скачиваем 20 фотографий кремля.
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

<img src="/images/kremlin3.png" width="40%" />
<img src="/images/kremlin4.png" width="40%" />

Модель выучила как выглядит кремль, но "забыла" стиль художника, ожидаемо.
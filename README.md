## Installation Guide:

### Install poetry
```shell
pip install poetry==2.0
```

### Install environment
```shell
poetry install
```

## Usefull command:
### autoencoder:
```shell
python run.py -t -b ./config/vqgan/vqgan.yaml -n "vqgan_pre_post"
```
### LDM:
#### image_to_image:
```shell
python run.py -t -b ./config/ldm/image_2_image_noisy_src_free_guidance_2c_classifier.yaml -n "ldm"
```

### resume training:
```shell
python run.py -t -r "./logs/<model_dir>"
```

### test model:
```shell
python run.py -r  "./logs/<model_dir>" --trials=1 --checkpoint-test=""
```

### tensorboard:
```shell
python -m tensorboard.main --logdir="./logs/<model_dir>"
```


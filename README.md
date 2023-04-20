# Music generation project for the subject Deep Learning for Acoustic Signal Processing (DLAS)

**Disclaimer:** this project is meant to be used only with audio without copyright, or audio you have permission to use.

## Project structure
The source code is divided as follows:
- **data:** contains the Lightning data modules.
- **datasets:** contains PyTorch datasets.
- **models:** contains Lightning modules and PyTorch modules.
- **notebooks:** contains useful notebooks to visualize results.
- **preprocess_datasets:** contains scripts to preprocess data and extract features.
- **specvqgan:** contains a modified version of [SpecVQGAN repository](https://github.com/v-iashin/SpecVQGAN), to use the models inside this project.
- **training:** includes training scripts and configuration files. The configuration is setup using [hydra-core](https://hydra.cc) and `OmegaConf`, similarly to SpecVQGAN.

A modified version of `laion_clap` library is also added to work with `colossalai`, only needed for the model that computes the CLAP embeddings online.

## Training
The following steps are needed to train a GPT-2 with precomputed [CLAP](https://github.com/LAION-AI/CLAP) embeddings and [EnCodec](https://github.com/facebookresearch/encodec) codes as input:

0. Make sure CUDA is installed in your machine.
1. Precompute CLAP embeddings and EnCodec embeddings from the .wav files using the scripts in **preprocess_datasets**.
2. Build the Docker container running `docker build . -t circe` in the root directory of the project. The CUDA version of the container should be less or equal than the CUDA version in the host machine.
3. Run the training with a command similar to the following:
```
docker run --rm --runtime=nvidia --gpus all -v /home/user/Circe/models-circe-encodec:/Circe/models-circe-encodec -v /home/user/Circe/datasets:/Circe/datasets circe circe.training.training_circe ++trainer.default_root_dir=/Circe/models-circe-encodec/
```

In this example, the folders with the precomputed codes and CLAP embeds should be located in `/home/user/Circe/datasets`. If the CLAP embeddings are extracted online, the property `++model.path_pretrained_clap` should be included in `circe-gpt.yaml`.

Finally, in the case CLAP is not wanted and only EnCodec or SpecVQGAN codes are needed, the model `hf-gpt.yaml` should be used by modifying `config.yaml`.

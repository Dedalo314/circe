FROM nvidia/cuda:11.7.0-devel-ubuntu22.04
ENV TZ=Europe/Madrid
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt update && apt install -y python3-pip && apt install -y libsndfile1 libsndfile1-dev
ADD requirements.txt /Circe/requirements.txt
WORKDIR /Circe/
RUN pip install -r requirements.txt
ADD laion_clap /usr/lib/python3.8/laion_clap
ADD src/circe/data /Circe/src/circe/data
ADD src/circe/datasets /Circe/src/circe/datasets
ADD src/circe/models /Circe/src/circe/models
ADD src/circe/training /Circe/src/circe/training
ADD src/circe/utils /Circe/src/circe/utils
WORKDIR /Circe/src/
ENTRYPOINT ["python3", "-m"]

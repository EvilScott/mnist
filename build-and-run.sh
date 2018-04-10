docker run \
  -v $(pwd)/srv:/srv \
  -v $(pwd)/datasets:/root/.keras/datasets \
  --rm -it \
  gw000/keras:2.1.4-py3-tf-cpu python3 /srv/main.py

# Recommendor Systems

## Objective
This repo contains a simple implementation of collaberative filtering using Pytorch.

This dataset (ml-latest-small) describes 5-star rating and free-text tagging activity from [MovieLens](http://movielens.org), a movie recommendation service. It contains 100836 ratings and 3683 tag applications across 9742 movies.

## Training
The model is trained using movielens dataset containing 100836 samples which is further divided in to train and validation dataset.

The custom model architecture is small but it was able to provide good accuracy.

```python
RecSysModel(
  (user_embed): Embedding(610, 32)
  (movie_embed): Embedding(9724, 32)
  (out): Linear(in_features=64, out_features=1, bias=True)
)
```

## Install Dependencies

Please install the requirements before running the python script.

``` shell
pip install -r requirements.txt
```

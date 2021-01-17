# IMDBSentimentAnalysis
This project is my second project at my internship.

## IMDB movie review sentiment classification dataset
### load_data function

```python
tf.keras.datasets.imdb.load_data(
    path="imdb.npz",
    num_words=None,
    skip_top=0,
    maxlen=None,
    seed=113,
    start_char=1,
    oov_char=2,
    index_from=3,
    **kwargs
)
```

Loads the IMDB dataset.

This is a dataset of 25,000 movies reviews from IMDB, labeled by sentiment (positive/negative). Reviews have been preprocessed, and each review is encoded as a list of word indexes (integers). For convenience, words are indexed by overall frequency in the dataset, so that for instance the integer "3" encodes the 3rd most frequent word in the data. This allows for quick filtering operations such as: "only consider the top 10,000 most common words, but eliminate the top 20 most common words".

As a convention, "0" does not stand for a specific word, but instead is used to encode any unknown word.

#### Arguments

* <b>path: </b> where to cache the data (relative to ~/.keras/dataset).
* <b>num_words: </b> integer or None. Words are ranked by how often they occur (in the training set) and only the `num_words` most frequent words are kept. Any less frequent word will appear as `oov_char` value in the sequence data. If None, all words are kept. Defaults to None, so all words are kept.
* <b>skip_top: </b> skip the top N most frequently occurring words (which may not be informative). These words will appear as `oov_char` value in the dataset. Defaults to 0, so no words are skipped.
* <b>maxlen: </b> int or None. Maximum sequence length. Any longer sequence will be truncated. Defaults to None, which means no truncation.
* <b>seed: </b> int. Seed for reproducible data shuffling.
* <b>start_char: </b> int. The start of a sequence will be marked with this character. Defaults to 1 because 0 is usually the padding character.
* <b>oov_char: </b> int. The out-of-vocabulary character. Words that were cut out because of the `num_words` or `skip_top` limits will be replaced with this character.
* <b>index_from: </b> int. Index actual words with this index and higher.
* <b>**kwargs: </b> Used for backwards compatibility.

#### Returns

* <b>Tuple of Numpy arrays: </b> `(x_train, y_train), (x_test, y_test)`.
* <b>x_train, x_test: </b> lists of sequences, which are lists of indexes (integers). If the num_words argument was specific, the maximum possible index value is `num_words - 1`. If the `maxlen` argument was specified, the largest possible sequence length is `maxlen`.
* <b>y_train, y_test: </b> lists of integer labels (1 or 0).

#### Raises

* <b>ValueError: </b> in case `maxlen` is so low that no input sequence could be kept.

Note that the 'out of vocabulary' character is only used for words that were present in the training set but are not included because they're not making the `num_words` cut here. Words that were not seen in the training set but are in the test set have simply been skipped.

### get_word_index function

```python
tf.keras.datasets.imdb.get_word_index(path="imdb_word_index.json")
```

Retrieves a dict mapping words to their index in the IMDB dataset.

#### Arguments

* <b>path: </b> where to cache the data (relative to ~/.keras/dataset).

#### Returns

The word index dictionary. Keys are word strings, values are their index.

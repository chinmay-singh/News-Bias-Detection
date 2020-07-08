import json

"""
Reads files from data/train.txt or dev.txt
and appends the sentence level sentiment to each word

say sentiment of <sentence_sentiment> is predicted on sentences level in train.txt
output in train_senti.txt is

<token1>\t<label1>\t<1_sentence_sentiment>\n
<token2>\t<label2>\t<1_sentence_sentiment>\n
<token3>\t<label3>\t<1_sentence_sentiment>\n
<token4>\t<label4>\t<1_sentence_sentiment>\n
\n
<token1>\t<label1>\t<2_sentence_sentiment>\n
<token2>\t<label2>\t<2_sentence_sentiment>\n
<token3>\t<label3>\t<2_sentence_sentiment>\n
<token4>\t<label4>\t<2_sentence_sentiment>\n
...

"""

from allennlp.predictors.predictor import Predictor
from allennlp.predictors.predictor import Predictor
import allennlp_models.classification
from time import time
predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/sst-roberta-large-2020.06.08.tar.gz")
print("LOADED ROBERTA LARGE!")
predictor._model = predictor._model.cuda()

tic = time()
p = predictor.predict(sentence="a very well-made, funny and entertaining picture.")
p2 = predictor.predict(sentence="a very well-made, funny and entertaining picture it was.")
toc = time()
print(p['probs'][0])  # Print positive sentiment
print("one eval takes", (toc-tic)/2, "seconds")

lexi_path = "lexicon_NRC/NRC-VAD-Lexicon.txt"
fo = open(lexi_path, "r")
lines = list(map(lambda x: x.strip().split(), fo.readlines()[1:-1]))
token_to_values_map = {(" ".join(line[0:-3])).strip().lower(): line[-3:] for line in lines}
print(list(map(type, lines[0])))
print(list(token_to_values_map.keys())[:100])
fo.close()

def read_data(path):
    fo = open(path, "r")
    words, ids = json.load(fo)
    fo.close()
    assert len(words) == len(ids)

    data = {}
    done = 0
    total = len(words)
    for w, i in zip(words, ids):
        sentiment = predictor.predict(sentence=(" ".join(w)))['probs'][0]

        temp_lexi_attr = []
        temp_line = w
        idx = 0
        while idx < (len(temp_line) - 2):
            trigram = " ".join(temp_line[idx:idx+3]).strip().lower()
            if trigram in token_to_values_map.keys():
                temp_lexi_attr.append(token_to_values_map[trigram])
                temp_lexi_attr.append(token_to_values_map[trigram])
                temp_lexi_attr.append(token_to_values_map[trigram])
                idx += 3
            else:
                bigram = " ".join(temp_line[idx:idx+2]).strip().lower()
                if bigram in token_to_values_map.keys():
                    temp_lexi_attr.append(token_to_values_map[bigram])
                    temp_lexi_attr.append(token_to_values_map[bigram])
                    idx += 2
                else:
                    temp_lexi_attr.append(token_to_values_map.get((temp_line[idx]).lower(), ["0.5", "0.5", "0.5"]))
                    idx += 1

        if len(temp_line) > 2 and idx != len(temp_line) - 2:
            print(idx, len(temp_line), temp_line)

        if (len(temp_line)-idx) == 2:
            bigram = " ".join(temp_line[idx:idx+2]).strip().lower()
            if bigram in token_to_values_map.keys():
                temp_lexi_attr.append(token_to_values_map[bigram])
                temp_lexi_attr.append(token_to_values_map[bigram])
                idx += 2
            else:
                temp_lexi_attr.append(token_to_values_map.get((temp_line[idx]).lower(), ["0.5", "0.5", "0.5"]))
                idx += 1
                temp_lexi_attr.append(token_to_values_map.get((temp_line[idx]).lower(), ["0.5", "0.5", "0.5"]))
                idx += 1
        else:
            temp_lexi_attr.append(token_to_values_map.get((temp_line[idx]).lower(), ["0.5", "0.5", "0.5"]))
            idx += 1

        assert idx == len(temp_line)
        Lexical_Attributes = temp_lexi_attr
        if i not in data.keys():
            data[i] = []
        assert len(Lexical_Attributes) == len(w)
        data[i].append([w, sentiment, Lexical_Attributes])

        done += 1
        if done % 10 ==0:
            print("Done", done, "/", total)

    assert sum(list(map(len, data.values()))) == len(ids)

    return data

def save_data(data, save_path):
    fo = open(save_path, "w+")
    json.dump(data, fo)
    fo.close()

if __name__ == "__main__":
    train_path = "train_words.json"
    dev_path = "dev_words.json"
    test_path = "test_words.json"

    train_new_path = "train_senti.json"
    dev_new_path = "dev_senti.json"
    test_new_path = "test_senti.json"

    data = read_data(dev_path)
    print("\n=============\nRead Dev data, sentiment and lexicons performed.\n=============\n")
    save_data(data, dev_new_path)
    print("\n=============\nSaved Dev data with sentiment.\n=============\n")

    data = read_data(test_path)
    print("\n=============\nRead Test data, sentiment and lexicons performed.\n=============\n")
    save_data(data, test_new_path)
    print("\n=============\nSaved Test data with sentiment.\n=============\n")

    data = read_data(train_path)
    print("\n=============\nRead Training data, sentiment and lexicons performed.\n=============\n")
    save_data(data, train_new_path)
    print("\n=============\nSaved Training data with sentiment.\n=============\n")


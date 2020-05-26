import numpy as np
import torch
from torch.utils import data
import pathlib
from preprocess import make_dataset, make_bert_dataset, make_bert_testset
from transformers import BertTokenizer
from params import params

if params.bert:
    num_task = 1
    masking = 0
    hier = 0
elif params.joint:
    num_task = 2
    masking = 0
    hier = 0
elif params.granu:
    num_task = 2
    masking = 0
    hier = 1
elif params.mgn:
    num_task = 2
    masking = 1
    hier = 0

if params.sig:
    sig = 1
    rel = 0
elif params.rel:
    sig = 0
    rel = 1

input_size = 768
VOCAB, tag2idx, idx2tag = [], [], []

if num_task == 1:
    VOCAB.append(("<PAD>", "O", "Name_Calling,Labeling", "Repetition", "Slogans", "Appeal_to_fear-prejudice", "Doubt", "Exaggeration,Minimisation", "Flag-Waving", "Loaded_Language", "Reductio_ad_hitlerum", "Bandwagon", "Causal_Oversimplification", "Obfuscation,Intentional_Vagueness,Confusion", "Appeal_to_Authority", "Black-and-White_Fallacy", "Thought-terminating_Cliches", "Red_Herring", "Straw_Men", "Whataboutism"))

#sentence classification
if num_task == 2:
    VOCAB.append(("<PAD>", "O", "Name_Calling,Labeling", "Repetition", "Slogans", "Appeal_to_fear-prejudice", "Doubt", "Exaggeration,Minimisation", "Flag-Waving", "Loaded_Language", "Reductio_ad_hitlerum", "Bandwagon", "Causal_Oversimplification", "Obfuscation,Intentional_Vagueness,Confusion", "Appeal_to_Authority", "Black-and-White_Fallacy", "Thought-terminating_Cliches", "Red_Herring", "Straw_Men", "Whataboutism", "CD", "ST"))
    VOCAB.append(("Non-prop", "Prop"))

# for i in range(num_task):
#     tag2idx.append({tag:idx for idx, tag in enumerate(VOCAB[i])})
#     idx2tag.append({idx:tag for idx, tag in enumerate(VOCAB[i])})

for i in range(num_task):
    tag2idx.append({})
    idx2tag.append({})

if params.group_classes:
    BIOES =  ["B", "I", "E", "S"]

    i = 0
    for pre in BIOES:
        p = pre + "-"
        tag2idx[0][p + "Red_Herring"] = 2 + i  # 2+i is classify and delete for i = 0, 2, 4, 6, 
        tag2idx[0][p + "Name_Calling,Labeling"] = 2 + i
        tag2idx[0][p + "Reductio_ad_hitlerum"] = 2 + i

        # 3 is the Style Transfer Class
        tag2idx[0][p + "Obfuscation,Intentional_Vagueness,Confusion"] = 3 + i
        tag2idx[0][p + "Loaded_Language"] = 3 + i
        
        i+= 2

    tag2idx[0]["Slogans"] = 1  # 1 is the ignore class 'O'
    tag2idx[0]["Appeal_to_fear-prejudice"] = 1
    tag2idx[0]["Doubt"] = 1
    tag2idx[0]["Exaggeration,Minimisation"] = 1
    tag2idx[0]["Flag-Waving"] = 1
    tag2idx[0]["Bandwagon"] = 1
    tag2idx[0]["Causal_Oversimplification"] = 1
    tag2idx[0]["Appeal_to_Authority"] = 1
    tag2idx[0]["Black-and-White_Fallacy"] = 1
    tag2idx[0]["Thought-terminating_Cliches"] = 1
    tag2idx[0]["Straw_Men"] = 1
    tag2idx[0]["Whataboutism"] = 1
    tag2idx[0]["Repetition"] = 1 # Maybe Put to others class

    i = 0
    for pre in BIOES:
        p = pre + "-"

        tag2idx[0][p + "CD"] = 2 + i
        tag2idx[0][p + "ST"] = 3 + i

        idx2tag[0][2 + i] = p + "CD"
        idx2tag[0][3 + i] = p + "ST"
        
        i += 2

    tag2idx[0]["<PAD>"] = 0
    tag2idx[0]["O"] = 1

    idx2tag[0][1] = "O"
    idx2tag[0][0] = "<PAD>"
print(tag2idx, idx2tag)

tokenizer = BertTokenizer.from_pretrained(
    'bert-base-cased', do_lower_case=False)

def convert_to_BIOES(tags):
    BIOES_tags = []
    IGNORE_CLASS = ["Slogans",
                "Appeal_to_fear-prejudice",
                "Doubt", 
                "Exaggeration,Minimisation",
                "Flag-Waving",
                "Bandwagon",
                "Causal_Oversimplification",
                "Appeal_to_Authority",
                "Black-and-White_Fallacy",
                "Thought-terminating_Cliches",
                "Straw_Men",
                "Whataboutism",
                "Repetition"]

    i = 0
    while i < len(tags):
        tag = tags[i]
        if tag == "O" or tag in IGNORE_CLASS:
            BIOES_tags.append("O")
            i += 1
        else:
            current_tag = tag
            first=True
            num=0
            while tag == current_tag:
                if first:
                    first=False
                    num+=1
                    if (i+1) < len(tags) and tags[i+1] == current_tag:
                        BIOES_tags.append("B-" + current_tag)
                        i+=1
                    else:
                        BIOES_tags.append("S-" + current_tag)
                        i+=1
                        break
                else:
                    num+=1
                    BIOES_tags.append("I-" + current_tag)
                    i+=1

                if i >= len(tags):
                    break
                else:
                    tag = tags[i]
            
            if num > 1:
                BIOES_tags[i-1] = "E" + BIOES_tags[i-1][1:]


    assert len(BIOES_tags) == len(tags)

    return BIOES_tags

class PropDataset(data.Dataset):
    def __init__(self, fpath, IsTest=False):
        directory = fpath
        dataset = make_dataset(directory)
        if IsTest:
            words, tags, ids = make_bert_testset(dataset)
        else:
            words, tags, ids = make_bert_dataset(dataset)
        flat_words, flat_tags, flat_ids, changed_ids = [], [], [], []

        count = 0
        for article_w, article_t, article_id in zip(words, tags, ids):
            for sentence, tag, id in zip(article_w, article_t, article_id):
                # Convert into BIOES Tagging Scheme()
                bioes_tag = convert_to_BIOES(tag)
                # print(tag, bioes_tag)
                # print(tag2idx[0])
                # Seperated them from the groupings
                changed = [idx2tag[0][tag2idx[0][temp_tag]] for temp_tag in bioes_tag]

                # which were article wise to make a list of just sentences
                if set(changed) == {'O'}:
                    count += 1
                    continue
                else:
                    flat_words.append(sentence)
                    changed_ids.append(changed)

                    flat_tags.append(changed)

                    flat_ids.append(id)
        print("{} sentences dropped".format(count))
        print("sentence is {} \n tag is {} \n id is {} \n changed_ids is {}".format(
            flat_words[:2], flat_tags[:2], flat_ids[:2], changed_ids[:2]))

        sents, ids = [], []
        tags_li = [[] for _ in range(num_task)]

        if params.dummy_run:
            flat_words = [flat_words[0]]
            flat_tags = [flat_tags[0]]
            flat_ids = [flat_ids[0]]

        for word, tag, id in zip(flat_words, flat_tags, flat_ids):
            words = word
            tags = tag

            ids.append([id])
            sents.append(["[CLS]"] + words + ["[SEP]"])
            tmp_tags = []

            # We here are just making the tags dict, basically and adding tags for the Sep and start tokens

            if num_task != 2:
                for i in range(num_task):
                    tmp_tags.append(['O']*len(tags))
                    for j, tag in enumerate(tags):
                        if tag != 'O' and tag != "<PAD>":
                            tmp_tags[i][j] = tag
                    tags_li[i].append(["<PAD>"] + tmp_tags[i] + ["<PAD>"])
            elif num_task == 2:
                tmp_tags.append(['O']*len(tags))
                tmp_tags.append(['Non-prop'])
                for j, tag in enumerate(tags):
                    if tag != 'O' and tag in VOCAB[0]:
                        tmp_tags[0][j] = tag
                        tmp_tags[1] = ['Prop']
                for i in range(num_task):
                    tags_li[i].append(["<PAD>"] + tmp_tags[i] + ["<PAD>"])

        self.sents, self.ids, self.tags_li = sents, ids, tags_li
        assert len(sents) == len(ids) == len(tags_li[0])

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, idx):
        words = self.sents[idx]  # tokens, tags: string list
        ids = self.ids[idx]  # tokens, tags: string list
        tags = list(list(zip(*self.tags_li))[idx])
        x, is_heads = [], []  # list of ids
        y = [[] for _ in range(num_task)]  # list of lists of lists
        tt = [[] for _ in range(num_task)]  # list of lists of lists
        if num_task != 2:
            for w, *t in zip(words, *tags):
                tokens = tokenizer.tokenize(w) if w not in (
                    "[CLS]", "[SEP]") else [w]
                xx = tokenizer.convert_tokens_to_ids(tokens)

                is_head = [1] + [0]*(len(tokens) - 1)
                if len(xx) < len(is_head):
                    xx = xx + [100] * (len(is_head) - len(xx))

                t = [[t[i]] + [t[i]] * (len(tokens) - 1)
                     for i in range(num_task)]

                y_tmp = []
                for i in range(num_task):
                    y[i].extend([tag2idx[i][each] for each in t[i]])
                    tt[i].extend(t[i])

                x.extend(xx)
                is_heads.extend(is_head)

        elif masking or num_task == 2:
            for w, t in zip(words, tags[0]):
                tokens = tokenizer.tokenize(w) if w not in (
                    "[CLS]", "[SEP]") else [w]
                xx = tokenizer.convert_tokens_to_ids(tokens)

                is_head = [1] + [0]*(len(tokens) - 1)
                if len(xx) < len(is_head):
                    xx = xx + [100] * (len(is_head) - len(xx))

                t = [t] + [t] * (len(tokens) - 1)
                y[0].extend([tag2idx[0][each] for each in t])
                tt[0].extend(t)

                x.extend(xx)
                is_heads.extend(is_head)
            if tags[1][1] == 'Non-prop':
                y[1].extend([1, 0])
                tt[1].extend(['Non-prop'])
            elif tags[1][1] == 'Prop':
                y[1].extend([0, 1])
                tt[1].extend(['Prop'])

        seqlen = len(y[0])

        words = " ".join(ids + words)

        for i in range(num_task):
            tags[i] = " ".join(tags[i])

        att_mask = [1] * seqlen
        # print("####  WORDS #####")
        # print(words)
        # print("#### X #####")
        # print(x)
        # print("####IS HEADS#####")
        # print(is_heads)
        # print("#### ATTENTION MASK #####")
        # print(att_mask)
        # print("#### TAGS #####")
        # print(tags)
        # print("#### Y #####")
        # print(y)
        # print("##### SEQUENCE LENGTH ####")
        # print(seqlen)
        return words, x, is_heads, att_mask, tags, y, seqlen


def pad(batch):
    def f(x): return [sample[x] for sample in batch]
    words = f(0)
    is_heads = f(2)
    seqlen = f(-1)
    maxlen = 210

    def f(x, seqlen): return [
        sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch]  # 0: <pad>
    x = torch.LongTensor(f(1, maxlen))

    att_mask = f(-4, maxlen)
    y = []
    tags = []

    if num_task != 2:
        for i in range(num_task):
            y.append(torch.LongTensor(
                [sample[-2][i] + [0] * (maxlen-len(sample[-2][i])) for sample in batch]))
            tags.append([sample[-3][i] for sample in batch])
    else:
        y.append(torch.LongTensor(
            [sample[-2][0] + [0] * (maxlen-len(sample[-2][0])) for sample in batch]))
        y.append(torch.LongTensor([sample[-2][1] for sample in batch]))
        for i in range(num_task):
            tags.append([sample[-3][i] for sample in batch])

    return words, x, is_heads, att_mask, tags, y, seqlen


# path_data = '.'
# train_path = "/data/protechn_corpus_eval/train"
# text_path = "/*.tsv"
# label_path = "/*.txt"
# dev_path = "/data/protechn_corpus_eval/dev"

# getter = PropDataset(params.trainset)

# print("###################################")
# print(getter.__getitem__(0))

if __name__ == "__main__":
    o = convert_to_BIOES(['Loaded_Language', 'Loaded_Language', 'Loaded_Language', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'])
    print(o)

    o = convert_to_BIOES(['Flag-Waving', 'Flag-Waving', 'Flag-Waving', 'Flag-Waving', 'Flag-Waving', 'Flag-Waving', 'Flag-Waving', 'Flag-Waving', 'Flag-Waving', 'Flag-Waving'])
    print(o)
    
    o = convert_to_BIOES(['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'Exaggeration,Minimisation', 'Exaggeration,Minimisation', 'Exaggeration,Minimisation', 'Exaggeration,Minimisation', 'Exaggeration,Minimisation', 'Exaggeration,Minimisation', 'Exaggeration,Minimisation', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'Exaggeration,Minimisation', 'Exaggeration,Minimisation', 'Exaggeration,Minimisation', 'Exaggeration,Minimisation', 'Exaggeration,Minimisation', 'Exaggeration,Minimisation', 'Exaggeration,Minimisation', 'Exaggeration,Minimisation', 'Exaggeration,Minimisation', 'O', 'O', 'O', 'Exaggeration,Minimisation', 'Exaggeration,Minimisation', 'Exaggeration,Minimisation', 'Exaggeration,Minimisation', 'Exaggeration,Minimisation', 'Exaggeration,Minimisation', 'Exaggeration,Minimisation', 'Exaggeration,Minimisation', 'Exaggeration,Minimisation', 'Exaggeration,Minimisation'])
    print(o)


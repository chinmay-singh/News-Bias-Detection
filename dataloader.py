import numpy as np
import torch
from torch.utils import data
import pathlib
from preprocess import make_dataset, make_bert_dataset, make_bert_testset
from pytorch_pretrained_bert import BertTokenizer
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
    
input_size=768
VOCAB, tag2idx, idx2tag = [], [], []

if num_task == 1:

    VOCAB.append(("<PAD>", "O", "Name_Calling,Labeling", "Repetition", "Slogans", "Appeal_to_fear-prejudice", "Doubt"
                , "Exaggeration,Minimisation", "Flag-Waving", "Loaded_Language"
                , "Reductio_ad_hitlerum", "Bandwagon"
                , "Causal_Oversimplification", "Obfuscation,Intentional_Vagueness,Confusion", "Appeal_to_Authority", "Black-and-White_Fallacy"
                , "Thought-terminating_Cliches", "Red_Herring", "Straw_Men", "Whataboutism"))

#sentence classification
if num_task == 2:
    VOCAB.append(("<PAD>", "O", "Name_Calling,Labeling", "Repetition", "Slogans", "Appeal_to_fear-prejudice", "Doubt"
                , "Exaggeration,Minimisation", "Flag-Waving", "Loaded_Language"
                , "Reductio_ad_hitlerum", "Bandwagon"
                , "Causal_Oversimplification", "Obfuscation,Intentional_Vagueness,Confusion", "Appeal_to_Authority", "Black-and-White_Fallacy"
                , "Thought-terminating_Cliches", "Red_Herring", "Straw_Men", "Whataboutism"))
    VOCAB.append(("Non-prop", "Prop"))




for i in range(num_task):
    tag2idx.append({tag:idx for idx, tag in enumerate(VOCAB[i])})
    idx2tag.append({idx:tag for idx, tag in enumerate(VOCAB[i])})

if params.group_classes:
    
tag2idx[0]["Red_Herring"] = 2 #2 is classify and delete
    tag2idx[0]["Name_Calling,Labeling"] = 2
    tag2idx[0]["Reductio_ad_hitlerum"] = 2
    tag2idx[0]["Repetition"] = 2

    tag2idx[0]["Obfuscation,Intentional_Vagueness,Confusion"] = 3 ###3 is the Style Transfer Class
    tag2idx[0]["Loaded_Language"] = 3

    tag2idx[0]["Slogans"] = 1                                     #### 1 is the ignore class 'O'
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
    
    tag2idx[0]["CD"] = 2
    tag2idx[0]["O"] = 1    
    tag2idx[0]["ST"]= 3

    idx2tag[0][1] = "O"
    idx2tag[0][2] = "CD"
    idx2tag[0][3] = "ST"



tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)

class PropDataset(data.Dataset):
    def __init__(self, fpath, IsTest=False):

        directory = fpath
        dataset = make_dataset(directory)
        if IsTest:
            words, tags, ids = make_bert_testset(dataset)
        else:
            words, tags, ids = make_bert_dataset(dataset)
        flat_words, flat_tags, flat_ids = [], [], []
        for article_w, article_t, article_id in zip(words, tags, ids):
            for sentence, tag, id in zip(article_w, article_t, article_id):
                flat_words.append(sentence)             # We seperated the sentences and 
                                                        # Seperated them from the groupings
                                                        # which were article wise to make a list of just sentences
                flat_tags.append(tag)
                flat_ids.append(id)
        
        # print("sentence is {} \n tag is {} \n id is {} \n".format(flat_words[0],flat_tags[0],flat_ids[0]))


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
            
            #We here are just making the tags dict, basically and adding tags for the Sep and start tokens

            if num_task != 2:
                for i in range(num_task):
                    tmp_tags.append(['O']*len(tags))
                    for j, tag in enumerate(tags):
                        if tag != 'O' and tag in VOCAB[i]:
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
        words = self.sents[idx] # tokens, tags: string list
        ids = self.ids[idx] # tokens, tags: string list
        tags = list(list(zip(*self.tags_li))[idx])
        x, is_heads = [], [] # list of ids
        y = [[] for _ in range(num_task)] # list of lists of lists
        tt = [[] for _ in range(num_task)] # list of lists of lists
        if num_task != 2:
            for w, *t in zip(words, *tags):
                tokens = tokenizer.tokenize(w) if w not in ("[CLS]", "[SEP]") else [w]
                xx = tokenizer.convert_tokens_to_ids(tokens)
    
                is_head = [1] + [0]*(len(tokens) - 1)
                if len(xx) < len(is_head):
                    xx = xx + [100] * (len(is_head) - len(xx))
    
                t = [[t[i]] + [t[i]] * (len(tokens) - 1) for i in range(num_task)]

                y_tmp = []
                for i in range(num_task):
                    y[i].extend([tag2idx[i][each] for each in t[i]])
                    tt[i].extend(t[i])

                x.extend(xx)
                is_heads.extend(is_head)
    
        elif masking or num_task == 2:
            for w, t in zip(words, tags[0]):
                tokens = tokenizer.tokenize(w) if w not in ("[CLS]", "[SEP]") else [w]
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
            tags[i]= " ".join(tags[i]) 

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
    f = lambda x: [sample[x] for sample in batch]
    words = f(0)
    is_heads = f(2)
    seqlen = f(-1)
    maxlen = 210

    f = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch] # 0: <pad>
    x = torch.LongTensor(f(1, maxlen))

    att_mask = f(-4, maxlen)
    y = []
    tags = []

    if num_task !=2:
        for i in range(num_task):
            y.append(torch.LongTensor([sample[-2][i] + [0] * (maxlen-len(sample[-2][i])) for sample in batch]))
            tags.append([sample[-3][i] for sample in batch])
    else:
        y.append(torch.LongTensor([sample[-2][0] + [0] * (maxlen-len(sample[-2][0])) for sample in batch]))
        y.append(torch.LongTensor([sample[-2][1] for sample in batch]))
        for i in range(num_task):
            tags.append([sample[-3][i] for sample in batch])


    return words, x, is_heads, att_mask, tags, y, seqlen


path_data = '.'
train_path = "/data/protechn_corpus_eval/train"
text_path = "/*.tsv"
label_path = "/*.txt"
dev_path = "/data/protechn_corpus_eval/dev"

# print(pad(b))

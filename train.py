import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import os
from params import params
import numpy as np
from model import BertMultiTaskLearning

from transformers import BertConfig , AdamW, get_linear_schedule_with_warmup

from dataloader import PropDataset, pad, VOCAB, tokenizer, tag2idx, idx2tag, num_task, masking
import time
from early_stopping import EarlyStopping
import wandb
import sklearn

import torch
import gc

count = 0

def check_cuda():
    torch.cuda.empty_cache()
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(count)
                print(type(obj), obj.size())
                count+=1
        except:
            pass


timestr = time.strftime("%Y%m%d-%H%M%S")

if not params.run:
    params.run = timestr
if params.dummy_run:
    params.batch_size = 1

dev = params.device

def train(model, iterator, optimizer, criterion, binary_criterion):
    check_cuda()
    model.train()
    check_cuda()
    train_losses = []

    for k, batch in enumerate(iterator):
        words, x, is_heads, att_mask, tags, y, seqlens = batch
        check_cuda()

        att_mask = torch.Tensor(att_mask).to(params.device)
        check_cuda()

        optimizer.zero_grad()
        logits, _ = model(x, attention_mask=att_mask)
        loss = []

        if num_task == 2:
            for i in range(num_task):
                logits[i] = logits[i].view(-1, logits[i].shape[-1])

            y[0] = y[0].view(-1).to(dev)
            y[1] = y[1].float().to(dev)

            loss.append(criterion(logits[0], y[0]))
            loss.append(binary_criterion(logits[1], y[1]))
        else:
            y[0] = y[0].to(dev)
            if params.crf:
                loss.append(model.module.forward_alg_loss(logits[0], y[0], att_mask))
            else:
                loss.append(criterion(logits[0], y[0]))

        if num_task == 1:
            joint_loss = loss[0]
        elif num_task==2:
            joint_loss = params.alpha*loss[0] + (1-params.alpha)*loss[1]
        check_cuda()
        joint_loss.backward()

        optimizer.step()
        scheduler.step()

        train_losses.append(joint_loss.item())

        if k%100 == 0:
            print("step: {}, loss: {}".format(k,loss[0].item()))

    train_loss = np.average(train_losses)

    return train_loss

def eval(model, iterator, f, criterion, binary_criterion):
    model.eval()

    print("EVALUATION:\n\n")
    valid_losses = []

    Words, Is_heads = [], []
    Tags = [[] for _ in range(num_task)]
    Y = [[] for _ in range(num_task)]
    Y_hats = [[] for _ in range(num_task)]
    with torch.no_grad():
        for _ , batch in enumerate(iterator):
            words, x, is_heads, att_mask, tags, y, seqlens = batch
            # print(x, tags, y)
            att_mask = torch.Tensor(att_mask).to(params.device)
            bert_feats, _ = model(x, attention_mask=att_mask) # logits: (N, T, VOCAB), y: (N, T)

            y[0] = y[0].to(dev)
            loss = model.module.forward_alg_loss(bert_feats[0], y[0], att_mask)
            y_hats = [model.module._viterbi_decode(bert_feats[0], att_mask)[1]]

            valid_losses.append(loss.item())
            Words.extend(words)
            Is_heads.extend(is_heads)

            Tags[0].extend(tags[0])
            Y[0].extend(y[0].cpu().numpy().tolist())
            Y_hats[0].extend(y_hats[0]) 
    valid_loss = np.average(valid_losses) 
    print("+++++++++ VALID LOSS = ", valid_loss)

    valid_tags = ["<START>", "O", "CD", "ST", "<PAD>"]
    valid_tag2id = {"<START>": 0, "O": 1, "CD": 2, "ST": 3,"<PAD>": 4}
    id2valid_tag = {id_: valid_tag for valid_tag, id_ in valid_tag2id.items()}
    tags_to_valid = {
                        "B-ST": "ST",
                        "S-ST": "ST",
                        "I-ST": "ST",
                        "E-ST": "ST",
                        "B-CD": "CD",
                        "S-CD": "CD",
                        "I-CD": "CD",
                        "E-CD": "CD",
                        "O": "O",
                        "<START>": "<START>",
                        "<PAD>": "<PAD>"
                    }
    print("+++++++++++++++++++++++++++")
    confusion_mat = np.zeros((len(valid_tags), len(valid_tags)))
    if num_task == 1:
        Y_hats = [single_sequence.cpu().tolist() for single_sequence in Y_hats[0]]
        Y = Y[0]
        # print(Y_hats,"\n\n", Y)

        yhat_all = []
        y_all = []

        assert len(Y) == len(Y_hats)
        for seq_num in range(len(Y)):
            this_Yhat = Y_hats[seq_num]
            this_Y = Y[seq_num]
        
            yhat_all.extend([tags_to_valid[idx2tag[0][yhat]] for yhat in this_Yhat])
            y_all.extend([tags_to_valid[idx2tag[0][y]] for y in this_Y[:len(this_Yhat)]])

        # print(y_all, "\n", yhat_all, len(y_all))
        assert len(y_all) == len(yhat_all)

        # Confusion Matrix[i, j] where i = ground_truth_idx and j = predicted_idx
        for i in range(len(y_all)):
            confusion_mat[valid_tag2id[y_all[i]], valid_tag2id[yhat_all[i]]] += 1

        print(confusion_mat)

        group_report = {}
        if params.group_classes:
            for tag, idx in valid_tag2id.items():
                this_grp = {}
                this_grp["TP"] = confusion_mat[idx, idx]
                this_grp["FP"] = np.sum(confusion_mat[:, idx]) - this_grp["TP"]
                this_grp["FN"] = np.sum(confusion_mat[idx, :]) - this_grp["TP"]
                
                tp_plus_fp = this_grp["TP"] + this_grp["FP"]
                if tp_plus_fp == 0.:
                    this_grp["Prec"] = 0.
                else:
                    this_grp["Prec"] = this_grp["TP"] / tp_plus_fp

                tp_plus_fn = this_grp["TP"] + this_grp["FN"]
                if tp_plus_fn == 0.:
                    this_grp["Recall"] = 0.
                else:
                    this_grp["Recall"] = this_grp["TP"] / tp_plus_fn

                prec_plus_rec = this_grp["Prec"] + this_grp["Recall"]
                if prec_plus_rec == 0.:
                    this_grp["f1-score"] = 0.
                else:
                    this_grp["f1-score"] = 2 * this_grp["Prec"] * this_grp["Recall"] \
                                        / prec_plus_rec

                group_report[tag] = this_grp

    print(group_report)
    num_correct = confusion_mat[valid_tag2id["ST"], valid_tag2id["ST"]] + \
                    confusion_mat[valid_tag2id["CD"], valid_tag2id["CD"]]

    num_predicted = np.sum(confusion_mat[:, valid_tag2id["ST"]]) + \
                    np.sum(confusion_mat[:, valid_tag2id["CD"]])

    num_gold = np.sum(confusion_mat[valid_tag2id["ST"], :]) + \
                np.sum(confusion_mat[valid_tag2id["CD"], :])

    # print("num_predicted:{}".format(num_predicted))
    # print("num_correct:{}".format(num_correct))
    # print("num_gold:{}".format(num_gold))

    if num_predicted == 0.0:
        precision = 0.0
    else:
        precision = num_correct / num_predicted

    if num_predicted == 0.0:
        recall = 0.0
    else:
        recall = num_correct / num_gold

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2*precision*recall / (precision + recall)

    final = f + ".P%.4f_R%.4f_F1%.4f" %(precision, recall, f1)

    print("precision=%.4f"%precision)
    print("recall=%.4f"%recall)
    print("f1=%.4f"%f1)

    if not params.group_classes:
        return precision, recall, f1, valid_loss
    else:
        return precision, recall, f1, valid_loss, group_report

if __name__ == "__main__":
    if params.wandb:
        wandb.init(project="news_bias", name=params.run)

    model_bert = BertMultiTaskLearning.from_pretrained('bert-base-uncased')
    print("Detected", torch.cuda.device_count(), "GPUs!")
    model_bert = nn.DataParallel(model_bert)
    model_bert = model_bert.to(dev)

    train_dataset = PropDataset(params.trainset, False)
    eval_dataset = PropDataset(params.validset, True)

    train_iter = data.DataLoader(dataset=train_dataset,
                                 batch_size=params.batch_size,
                                 shuffle=True,
                                 num_workers=1,
                                 collate_fn=pad)

    eval_iter = data.DataLoader(dataset=eval_dataset,
                                batch_size=params.batch_size,
                                shuffle=False,
                                num_workers=1,
                                collate_fn=pad)

    # test_iter = data.DataLoader(dataset=test_dataset,
    #                             batch_size=params.batch_size,
    #                             shuffle=False,
    #                             num_workers=1,
    #                             collate_fn=pad)

    warmup_proportion = 0.1
    num_train_optimization_steps = int(len(train_dataset) / params.batch_size) * params.n_epochs

    param_optimizer = list(model_bert.named_parameters())
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr = params.lr, correct_bias=True )
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = int(warmup_proportion*num_train_optimization_steps), num_training_steps = num_train_optimization_steps )

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    binary_criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.Tensor([3932/14263]).to(dev))

    avg_train_losses = []
    avg_valid_losses = []

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=params.patience, verbose=True)

    for epoch in range(1, params.n_epochs+1):
        print("=========eval at epoch={epoch}=========")
        if not os.path.exists('checkpoints'):
            os.makedirs('checkpoints')
        if not os.path.exists('results'):
            os.makedirs('results')
        fname = os.path.join('checkpoints','epoch_{}_'.format(epoch)+params.run)
        spath = os.path.join('checkpoints','epoch_{}_'.format(epoch)+params.run+".pt")

        print("For epoch {} cached is {} allocated is {}".format(epoch, torch.cuda.memory_cached(0), torch.cuda.memory_allocated(0)))

        train_loss = train(model_bert, train_iter, optimizer, criterion, binary_criterion)

        avg_train_losses.append(train_loss.item())

        if not params.group_classes:
            precision, recall, f1, valid_loss = eval(
                model_bert, eval_iter, fname, criterion, binary_criterion)
            avg_valid_losses.append(valid_loss.item())

            if params.wandb:
                wandb.log({"Training Loss": train_loss.item(), "Validation Loss": valid_loss.item(
                ), "Precision": precision, "Recall": recall, "F1": f1})
        else:
            precision, recall, f1, valid_loss, group_report = eval(
                model_bert, eval_iter, fname, criterion, binary_criterion)
            avg_valid_losses.append(valid_loss.item())

            if params.wandb:
                wandb.log({"Training Loss": train_loss.item(), "Validation Loss": valid_loss.item(
                ), "Precision": precision, "Recall": recall, "F1": f1, "CD_F1": group_report["CD"]["f1-score"], "ST_F1": group_report["ST"]["f1-score"], "O_F1": group_report["O"]["f1-score"]})


        epoch_len = len(str(params.n_epochs))
        print_msg = (f'[{epoch:>{epoch_len}}/{params.n_epochs:>{epoch_len}}]     ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')
        print(print_msg)

        # early_stopping(-1*f1, model_bert, spath)

        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     break

    res = os.path.join('results', timestr)
    # load the last checkpoint with the best model
    model_bert.load_state_dict(torch.load(spath))


    # precision, recall, f1, test_loss = eval(
    #     model_bert, test_iter, res, criterion, binary_criterion)
    # print_msg = (f'test_precision: {precision:.5f} ' +
    #              f'test_recall: {recall:.5f} ' +
    #              f'test_f1: {f1:.5f}')
    # print(print_msg)

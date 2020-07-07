import torch
from params import params
import numpy as np
from sklearn.metrics import average_precision_score
import sklearn
from collections import Counter

lexi_ignore = (0.5 - params.lexical_ignore_range, 0.5 + params.lexical_ignore_range)
print(lexi_ignore)
def train(model, iterator, optimizer, scheduler, criterion):
    model.train()
    train_losses = []

    for k, batch in enumerate(iterator):
        if params.sentence_level:
            input_ids, y, att_mask, seq_lens = batch
        else:
            input_ids, y, att_mask, lexicon_sequence, sentiment, seq_len, y_for_loss, is_heads = batch

        optimizer.zero_grad()

        if params.sentence_level:
            outputs = model(input_ids, att_mask, y)
            loss = outputs[0]
            loss.mean().backward()

            optimizer.step()
            scheduler.step()
            train_losses.append(loss.mean().item())

            if k%100 == 0:
                print("step: {}, loss: {}".format(k,loss[0].item()))
        else:
            outputs, sentiment_preds, lexical_preds = model(input_ids, att_mask)

            label_loss = criterion['label_crit'](outputs.view(-1, outputs.size(-1)), y_for_loss.view(-1))

            senti_loss = criterion['sentiment_crit'](sentiment_preds, sentiment)

            lexi_loss = criterion['lexicon_crit'](lexical_preds , lexicon_sequence)

            lexical_mask = ~ ((lexicon_sequence >= lexi_ignore[0]) & (lexicon_sequence <= lexi_ignore[1]))
            lexi_loss = lexi_loss * lexical_mask

            lexi_loss = torch.sum(lexi_loss) / torch.sum(lexical_mask) # Average over non zero values

            loss = label_loss + (params.sentiment_loss_wt * senti_loss) + (params.lexical_loss_wt * lexi_loss)

            loss.backward()
            optimizer.step()
            scheduler.step()

            train_losses.append([loss.item(), label_loss.item(), params.sentiment_loss_wt * senti_loss.item(), 
                                params.lexical_loss_wt * lexi_loss.item()])
            if k % 100 == 0:
                print("step: {}, loss: {}".format(k, np.mean(np.asarray(train_losses), axis=0)))
    return np.mean(np.asarray(train_losses), axis=0)

def count_seqs(seq, O_idx=0):
    i = 0
    mod_T = 0 # |T| or |S|
    while i < len(seq):
        if seq[i] != O_idx:
            this_label = seq[i]
            mod_T += 1
            while i < len(seq) and this_label == seq[i]:
                i += 1
        else:
            i += 1

    return mod_T


# This metric is implementation from the 2019 paper by Goivanni et. al. of Fine-Grained Analysis of Propaganda in News Article
# Not used as per official code by propaganda.qcri.org
def paper_metric(Y, Y_hats, O_idx=0):    
    mod_T = count_seqs(Y)
    mod_S = count_seqs(Y_hats)

    C = []
    i = 0
    while i < len(Y_hats):
        if Y_hats[i] != O_idx:
            h = 0
            same = 0
            this_label = Y_hats[i]
            while i < len(Y_hats) and this_label == Y_hats[i]:
                if Y_hats[i] == Y[i]:
                    same += 1
                h += 1
                i += 1
            C.append(same/h)
        else:
            i += 1
    assert mod_S == len(C)

    if mod_S > 0:
        P_metric = sum(C)/len(C)
    else:
        P_metric = 0

    C = []
    i = 0
    while i < len(Y):
        if Y[i] != O_idx:
            h = 0
            same = 0
            this_label = Y[i]
            while i < len(Y) and this_label == Y[i]:
                if Y_hats[i] == Y[i]:
                    same += 1
                h += 1
                i += 1
            C.append(same/h)
        else:
            i += 1
    assert mod_T == len(C)
    # R_metric = sum(C)/len(C)

    if mod_T > 0:    
        R_metric = sum(C)/len(C)
    else:
       	R_metric = 0
    
    F_denom = P_metric + R_metric
    if F_denom == 0:
        F_metric = 0
    else:
        F_metric = (2 * P_metric * R_metric) / F_denom
  
    return P_metric, R_metric, F_metric


def metric(Y, Y_hats):
    num_predicted = len(Y_hats[Y_hats > 0])
    num_correct = (np.logical_and(Y==Y_hats, Y > 0)).astype(np.int).sum()
    num_gold = len(Y[Y > 0])

    if num_predicted == 0:
        precision = 0.0
    else:
        precision = num_correct / num_predicted
    
    if num_gold == 0:
        raise "Num golds should not be zero"
    else:
        recall = num_correct / num_gold

    if (precision + recall) == 0:
        f1 = 0.0
    else:
        f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1, num_predicted, num_correct, num_gold


def eval(model, iterator, criterion):
    model.eval()

    valid_losses = []
    Y = []
    Y_hats = []
    All_is_heads = []

    with torch.no_grad():
        for _, batch in enumerate(iterator):
            if params.sentence_level:
                input_ids, y, att_mask, seq_lens = batch

                outputs = model(input_ids, attention_mask=att_mask, labels=y)
                loss = outputs[0]
                logits = outputs[1]

                y_hats = logits.argmax(-1)
                valid_losses.append(loss.mean().item())

                Y.extend(y.cpu().numpy().tolist())
                Y_hats.extend(y_hats.cpu().numpy().tolist())
            else:
                input_ids, y, att_mask, lexicon_sequence, sentiment, seq_len, y_for_loss, is_heads = batch

                outputs, sentiment_preds, lexical_preds = model(input_ids, att_mask)

                label_loss = criterion['label_crit'](outputs.view(-1, outputs.size(-1)), y_for_loss.view(-1))

                senti_loss = criterion['sentiment_crit'](sentiment_preds, sentiment)

                lexi_loss = criterion['lexicon_crit'](lexical_preds , lexicon_sequence)
                lexical_mask = ~ ((lexicon_sequence >= lexi_ignore[0]) & (lexicon_sequence <= lexi_ignore[1]))
                lexi_loss = lexi_loss * lexical_mask
                lexi_loss = torch.sum(lexi_loss) / torch.sum(lexical_mask) # Average over non zero values

                loss = label_loss + (params.sentiment_loss_wt * senti_loss) + (params.lexical_loss_wt * lexi_loss)
                y_hats = outputs.argmax(-1).view(-1)
                valid_losses.append([loss.item(), label_loss.item(), params.sentiment_loss_wt * senti_loss.item(),
                                        params.lexical_loss_wt * lexi_loss.item()])
                All_is_heads.extend(is_heads.view(-1).cpu().tolist())
                Y.extend(y.view(-1).cpu().tolist())
                Y_hats.extend(y_hats.cpu().tolist())

    print(len(All_is_heads))
    assert len(All_is_heads) == len(Y)
    assert len(All_is_heads) == len(Y_hats)
    print(Counter(Y), Counter(Y_hats), Counter(All_is_heads))

    if params.sentence_level == False:
        new_Y, new_Y_hats = [], []

        for i in range(len(Y)):
            if All_is_heads[i] == True:
                new_Y.append(Y[i])
                new_Y_hats.append(Y_hats[i])
        assert len(new_Y) == len(new_Y_hats)

        Y = new_Y
        Y_hats = new_Y_hats

    assert len(Y) == len(Y_hats)
    assert len(Y) == sum(All_is_heads)
    print(Counter(Y), Counter(Y_hats))

    group_report = sklearn.metrics.classification_report(
                    Y, Y_hats, output_dict=True)

    Y = np.asarray(Y)
    Y_hats = np.asarray(Y_hats)

    if params.sentence_level == False:
        # print(Y[:100])
        # print(Y_hats[:100])
        precision, recall, f1, num_pred, num_correct, num_gold = metric(Y, Y_hats)
        print(metric(Y, Y), "||", metric(Y_hats, Y_hats))
        group_report["P Metric"] = precision
        group_report["R Metric"] = recall
        group_report["F1 Metric"] = f1
    else:
        precision = group_report['weighted avg']['precision']
        recall = group_report['weighted avg']['recall']
        f1 = group_report['weighted avg']['f1-score']

    print("precision=%.4f" % precision)
    print("recall=%.4f" % recall)
    print("f1=%.4f" % f1)
    print("num_pred=%.4f" % num_pred)
    print("num_correct=%.4f" % num_correct)
    print("num_gold=%.4f" % num_gold) 
    
    return precision, recall, f1, np.mean(np.asarray(valid_losses), axis=0), group_report

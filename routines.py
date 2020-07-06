import torch
from params import params
import numpy as np
from sklearn.metrics import average_precision_score
import sklearn

lexi_ignore = (0.5 - params.lexical_ignore_range, 0.5 + params.lexical_ignore_range)
print(lexi_ignore)
def train(model, iterator, optimizer, scheduler, criterion):
    model.train()
    train_losses = []

    for k, batch in enumerate(iterator):
        if params.sentence_level:
            input_ids, y, att_mask, seq_lens = batch
        else:
            input_ids, y, att_mask, lexicon_sequence, sentiment, seq_len = batch

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

            label_loss = criterion['label_crit'](outputs.view(-1, outputs.size(-1)), y.view(-1))

            senti_loss = criterion['sentiment_crit'](sentiment_preds, sentiment)

            lexi_loss = criterion['lexicon_crit'](lexical_preds , lexicon_sequence)

            lexical_mask = ~ ((lexicon_sequence >= lexi_ignore[0]) & (lexicon_sequence <= lexi_ignore[1]))
            lexi_loss = lexi_loss * lexical_mask

            lexi_loss = torch.sum(lexi_loss) / torch.sum(lexical_mask) # Average over non zero values

            loss = label_loss + (params.sentiment_loss_wt * senti_loss) + (params.lexical_loss_wt * lexi_loss)
            # print(loss, label_loss, senti_loss, lexi_loss)
            # # model.eval()
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_losses.append([loss.item(), label_loss.item(), senti_loss.item(), lexi_loss.item()])
            if k % 100 == 0:
                print("step: {}, loss: {}".format(k, np.mean(np.asarray(train_losses), axis=0)))
    return np.mean(np.asarray(train_losses), axis=0)

def eval(model, iterator, criterion):
    model.eval()

    valid_losses = []
    Y = [ ]
    Y_hats = [ ]

    with torch.no_grad():
        for _, batch in enumerate(iterator):
            if params.sentence_level:
                input_ids, y, att_mask, seq_lens = batch

                outputs = model(x, attention_mask=att_mask, labels=y)
                loss = outputs[0]
                logits = outputs[1]

                y_hats = logits.argmax(-1)
                valid_losses.append(loss.mean().item())

                Y.extend(y.cpu().numpy().tolist())
                Y_hats.extend(y_hats.cpu().numpy().tolist())
            else:
                input_ids, y, att_mask, lexicon_sequence, sentiment, seq_len = batch

                outputs, sentiment_preds, lexical_preds = model(input_ids, att_mask)

                label_loss = criterion['label_crit'](outputs.view(-1, outputs.size(-1)), y.view(-1))

                senti_loss = criterion['sentiment_crit'](sentiment_preds, sentiment)

                lexi_loss = criterion['lexicon_crit'](lexical_preds , lexicon_sequence)
                lexical_mask = ~ ((lexicon_sequence >= lexi_ignore[0]) & (lexicon_sequence <= lexi_ignore[1]))
                lexi_loss = lexi_loss * lexical_mask
                lexi_loss = torch.sum(lexi_loss) / torch.sum(lexical_mask) # Average over non zero values

                loss = label_loss + (params.sentiment_loss_wt * senti_loss) + (params.lexical_loss_wt * lexi_loss)

                y_hats = outputs.argmax(-1).view(-1)
                valid_losses.append([loss.item(), label_loss.item(), senti_loss.item(), lexi_loss.item()])
                Y.extend(y.view(-1).cpu().tolist())
                Y_hats.extend(y_hats.cpu().tolist())

    group_report = sklearn.metrics.classification_report(
                    Y, Y_hats, output_dict=True)

    precision = group_report['weighted avg']['precision']
    recall = group_report['weighted avg']['recall']
    f1 = group_report['weighted avg']['f1-score']
    
    print("precision=%.4f" % precision)
    print("recall=%.4f" % recall)
    print("f1=%.4f" % f1)
    
    return precision, recall, f1, np.mean(np.asarray(valid_losses), axis=0), group_report

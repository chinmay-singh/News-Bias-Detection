import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn.functional import relu, tanh, sigmoid
from transformers import BertModel, BertConfig
from transformers.modeling_bert import BertPreTrainedModel
from dataloader import num_task, VOCAB, hier, sig, rel
from params import params

dev = params.device

def log_sum_exp_1vec(vec):  # shape(1,m)
    max_score = vec[0, np.argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

def log_sum_exp_mat(log_M, axis=-1):  # shape(n,m)
    return torch.max(log_M, axis)[0]+torch.log(torch.exp(log_M-torch.max(log_M, axis)[0][:, None]).sum(axis))

def log_sum_exp_batch(log_Tensor, axis=-1): # shape (batch_size,n,m)
    return torch.max(log_Tensor, axis)[0]+torch.log(torch.exp(log_Tensor-torch.max(log_Tensor, axis)[0].view(log_Tensor.shape[0],-1,1)).sum(axis))

class BertMultiTaskLearning(BertPreTrainedModel):
    def __init__(self, config):
        super(BertMultiTaskLearning, self).__init__(config)
        # super(BertMultiTaskLearning, self)

        self.start_label_id = 11
        self.cls_id = 10
        self.stop_label_id = 0

        if params.crf:
            self.tagset_size = 12 # 12 tags including <CLS> <START> <PAD> in BIOES format
            self.num_labels = self.tagset_size
        
            self.transitions = nn.Parameter(
                    torch.randn(self.tagset_size, self.tagset_size))

            # Four statements enforce the constraints for transitions:
            # 1. Nothing goes to CLS tag <except> start
            self.transitions.data[self.cls_id, :] = -10000.0

            # 2. Start only transitions to CLS tag
            self.transitions.data[:, self.start_label_id] = -10000.0
            self.transitions.data[self.cls_id, self.start_label_id] = 0.0 # Start only goes to CLS

            # 3. Never go directly from cls to stop
            self.transitions.data[self.stop_label_id, self.cls_id] = -10000.0

            # 4. From the Stop tag, go nowhere except for stop tag itself
            self.transitions.data[self.stop_label_id+1:, self.stop_label_id] = -10000.0

            self.classifier = nn.ModuleList(
                [nn.Linear(config.hidden_size, self.tagset_size)])
        else: 
            self.classifier = nn.ModuleList(
                [nn.Linear(config.hidden_size, 4)])

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # self.classifier = nn.ModuleList([nn.Linear(config.hidden_size, len(VOCAB[i])) for i in range(num_task)])
        # self.apply(self._init_bert_weights)
        self.masking_gate = nn.Linear(2,1)

        if num_task == 2:
            self.merge_classifier_1 = nn.Linear(len(VOCAB[0]+len(VOCAB[1])), len(VOCAB[0]))
        
        self.init_weights()

    def _forward_alg(self, feats):
        '''
        this also called alpha-recursion or forward recursion, to calculate log_prob of all barX 
        '''
        
        # T = self.max_seq_length
        T = feats.shape[1]
        batch_size = feats.shape[0]
        
        # alpha_recursion,forward, alpha(zt)=p(zt,bar_x_1:t)
        log_alpha = torch.Tensor(batch_size, 1, self.num_labels).fill_(-10000.).to(params.device)
        # normal_alpha_0 : alpha[0]=Ot[0]*self.PIs
        # self.start_label has all of the score. it is log,0 is p=1
        log_alpha[:, 0, self.start_label_id] = 0
        
        # feats: sentances -> word embedding -> lstm -> MLP -> feats
        # feats is the probability of emission, feat.shape=(1,tag_size)
        for t in range(0, T):
            log_alpha = (log_sum_exp_batch(self.transitions + log_alpha, axis=-1) + feats[:, t]).unsqueeze(1)
            print(log_alpha)
        # log_prob of all barX
        log_prob_all_barX = log_sum_exp_batch(log_alpha + self.transitions[self.stop_label_id])
        return log_prob_all_barX

    def _score_sentence(self, feats, label_ids):
        '''
        Gives the score of a provided label sequence
        p(X=w1:t,Zt=tag1:t)=...p(Zt=tag_t|Zt-1=tag_t-1)p(xt|Zt=tag_t)...
        '''

        # T = self.max_seq_length
        T = feats.shape[1]
        batch_size = feats.shape[0]

        batch_transitions = self.transitions.expand(batch_size, self.num_labels, self.num_labels)
        batch_transitions = batch_transitions.flatten(1)

        score = torch.zeros((feats.shape[0],1)).to(params.device)
        # print("BATCH_TRANS_SIZE = ", batch_transitions.size())

        # the 0th node is start_label->start_word, the probability of them=1. so t begin with 1.
        prev_labels = torch.tensor([11]).expand(batch_size).to(params.device) # Initially it all starts with <START>

        for t in range(0, T):
            this_labels = label_ids[:, t]

            score = score + \
                batch_transitions.gather(-1, (this_labels*self.num_labels+prev_labels).view(-1,1)) \
                    + feats[:, t].gather(-1, this_labels.view(-1,1)).view(-1,1)
            prev_labels = this_labels

        return score

    def _viterbi_decode(self, feats):
        '''
        Max-Product Algorithm or viterbi algorithm, argmax(p(z_0:t|x_0:t))
        '''
        
        # T = self.max_seq_length
        T = feats.shape[1]
        batch_size = feats.shape[0]

        # batch_transitions=self.transitions.expand(batch_size,self.num_labels,self.num_labels)

        log_delta = torch.Tensor(batch_size, 1, self.num_labels).fill_(-10000.).to(params.device)
        log_delta[:, 0, self.start_label_id] = 0

        # psi is for the vaule of the last latent that make P(this_latent) maximum.
        psi = torch.zeros((batch_size, T, self.num_labels), dtype=torch.long).to(params.device)  # psi[0]=0000 useless
        for t in range(0, T):
            # delta[t][k]=max_z1:t-1( p(x1,x2,...,xt,z1,z2,...,zt-1,zt=k|theta) )
            # delta[t] is the max prob of the path from  z_t-1 to z_t[k]
            log_delta, psi[:, t] = torch.max(self.transitions + log_delta, -1)
            # psi[t][k]=argmax_z1:t-1( p(x1,x2,...,xt,z1,z2,...,zt-1,zt=k|theta) )
            # psi[t][k] is the path choosed from z_t-1 to z_t[k],the value is the z_state(is k) index of z_t-1
            log_delta = (log_delta + feats[:, t]).unsqueeze(1)

        # trace back
        path = torch.zeros((batch_size, T), dtype=torch.long).to(params.device)

        # max p(z1:t,all_x|theta)
        max_logLL_allz_allx, path[:, -1] = torch.max(log_delta.squeeze(), -1)

        for t in range(T-2, -1, -1):
            # choose the state of z_t according the state choosed of z_t+1.
            path[:, t] = psi[:, t+1].gather(-1,path[:, t+1].view(-1,1)).squeeze()

        return max_logLL_allz_allx, path

    def forward_alg_loss(self, bert_feats, labels, mask):
        forward_score = self._forward_alg(bert_feats)
        # print(forward_score)
        # p(X=w1:t,Zt=tag1:t)=...p(Zt=tag_t|Zt-1=tag_t-1)p(xt|Zt=tag_t)...
        gold_score = self._score_sentence(bert_feats, labels)
        # print(gold_score)
        # - log[ p(X=w1:t,Zt=tag1:t)/p(X=w1:t) ] = - log[ p(Zt=tag1:t|X=w1:t) ]
        return torch.mean(forward_score - gold_score)

    # def pred(self, bert_feats, input_ids):
        #     score, label_seq_ids = self._viterbi_decode(bert_feats)
        #     return score, label_seq_ids


    def forward(self, input_ids, token_type_ids= None, attention_mask= None, labels = None):
        input_ids = input_ids.to(dev)
        # token_type_ids =  token_type_ids.to("cuda")
        attention_mask = attention_mask.to(dev)

        output = self.bert(input_ids,token_type_ids= token_type_ids,attention_mask = attention_mask)
        sequence_output = self.dropout(output[0])
        pooled_output = self.dropout(output[1])

        if num_task == 1:
            logits = [self.classifier[i](sequence_output) for i in range(num_task)]

        elif num_task ==2:
            token_level = self.classifier[0](sequence_output)
            sen_level = self.classifier[1](pooled_output)

            if sig:
                gate = sigmoid(self.masking_gate(sen_level))
            else:
                gate = relu(self.masking_gate(sen_level))

            dup_gate = gate.unsqueeze(1).repeat(1, token_level.size()[1], token_level.size()[2])
            wei_token_level = torch.mul(dup_gate, token_level)

            logits = [wei_token_level, sen_level]

        elif num_task == 2 and hier:
            token_level = self.classifier[0](sequence_output)
            sen_level = self.classifier[1](pooled_output)
            dup_sen_level = sen_level.repeat(1, token_level.size()[1])
            dup_sen_level = dup_sen_level.view(sen_level.size()[0], -1, sen_level.size()[-1])
            logits = [self.merge_classifier_1(torch.cat((token_level, dup_sen_level), 2)), self.classifier[1](pooled_output)]

        elif num_task == 2:
            token_level = self.classifier[0](sequence_output)
            sen_level = self.classifier[1](pooled_output)
            logits = [token_level, sen_level]

        y_hats = [logits[i].argmax(-1) for i in range(num_task)]

        return logits, y_hats

    # def log_sum_exp(vec, batch, tagset_size):
        #     max_scores = torch.max(vec, 2)[0]
        #     max_scores_broadcast = max_scores.view(batch, 1, -1).expand(batch, 1, tagset_size)
        #     logz = torch.log(torch.sum(torch.exp(vec - max_scores_broadcast), axis=2))
        #     return max_scores + logz

        # def forward_alg_loss(self, feats, tags):
        #     return _forward_alg(11, self.init_alphas, self.transitions, feats, params.batch_size) - \
        #             score_sentence(self, feats, tags)

        # def decode():
        #     do_nothing()
        #     EROEROEORERE
        #    WRONG_INDENT

        # def _forward_alg(tagset_size, init_alphas, transitions, feats, batch):
        #     """Do the forward algorithm to compute the partition function."""
        #     permuted_feats = feats.permute(1, 0, 2)

        #     # Wrap in a variable so that we will get automatic backprop
        #     forward_var = init_alphas.repeat(batch, 1, 1)

        #     # Iterate through the sentence
        #     for feat in permuted_feats:
        #         alphas_t = []  # The forward tensors at this timestep

        #         for next_tag in range(tagset_size):
        #             # broadcast the emission score: it is the same regardless of the previous tag
        #             emit_score = feat[:, next_tag].view(batch, 1, -1).expand(batch, 1, tagset_size)

        #             # the ith entry of trans_score is the score of transitioning to next_tag from i
        #             trans_score = transitions[next_tag].view(1, -1)

        #             # The ith entry of next_tag_var is the value for the
        #             # edge (i -> next_tag) before we do log-sum-exp
        #             next_tag_var = forward_var + trans_score + emit_score

        #             # The forward variable for this tag is log-sum-exp of all the scores.
        #             alphas_t.append(log_sum_exp(next_tag_var, batch, tagset_size).view(batch, 1))

        #         concatted_alphas = torch.cat(alphas_t, 1).view(batch, 1, -1)
        #         # assert concatted_alphas.size() == forward_var.size()
                
        #         forward_var = concatted_alphas
            
        #     terminal_var = forward_var + transitions[0] # 0=STOP/PAD Tag
        #     alpha = log_sum_exp(terminal_var, batch, tagset_size)
        #     return alpha


if __name__ == "__main__":
    # torch.manual_seed(124)
    # t = torch.randn(1, 16, 12).to(params.device)#.expand(24, 160, 12)
    # # y = [[]]
    # # for i in range(10):
    # #     y[0].extend([10, 1, 1, 1, 1, 1, 1, 3, 5, 5, 5, 5, 7, 1, 1, 0])
    # # y = [[10, 1, 1, 1, 1, 1, 3, 3, 5, 5, 5, 5, 7, 1, 1, 0, 0, 0, 0]]
    # y = torch.tensor(y).to(params.device)
    # print("\n\n", y, y.size(), t.size())

    transitions = torch.zeros(12, 12)
    print(transitions)

    log_alpha = torch.Tensor(1, 1, 12).fill_(-10000.)#.to(params.device)
    log_alpha[:, 0, 11] = 0
    print(log_alpha)

    print(transitions + log_alpha)
    print(log_sum_exp_batch(transitions + log_alpha, axis=-1))

    # model_bert = BertMultiTaskLearning.from_pretrained('bert-base-uncased')
    # model_bert = model_bert.to(params.device)
    # # print(model_bert.transitions, model_bert.transitions.size())
    
    # # import torch.optim as optim
    # # optimizer = optim.SGD(model_bert.parameters(), lr=0.00001)
    # import time
    # start = time.time()
    # loss = model_bert.forward_alg_loss(t, y)
    # end = time.time()
    # print(end - start)
    # print(loss)
    # start = time.time()
    # loss.backward()
    # optimizer.step()
    # end = time.time()
    # print(end - start)

    # for i in range(500):         
    #     loss = model_bert.forward_alg_loss(model_bert.classifier[0](t), y)
    #     loss.backward()
    #     optimizer.step()
    #     _, seq = model_bert._viterbi_decode(model_bert.classifier[0](t))
    #     print(loss, seq, seq.size())


import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AutoModel
from prefix_encoder import PrefixEncoder

from torch import nn
import alignment

class LayerNorm(nn.Module):
    def __init__(self, input_dim, cond_dim=0, center=True, scale=True, epsilon=None, conditional=False,
                 hidden_units=None, hidden_activation='linear', hidden_initializer='xaiver', **kwargs):
        super(LayerNorm, self).__init__()
        """
        input_dim: inputs.shape[-1]
        cond_dim: cond.shape[-1]
        """
        self.center = center
        self.scale = scale
        self.conditional = conditional
        self.hidden_units = hidden_units
        self.hidden_initializer = hidden_initializer
        self.epsilon = epsilon or 1e-12
        self.input_dim = input_dim
        self.cond_dim = cond_dim

        if self.center:
            self.beta = nn.Parameter(torch.zeros(input_dim))
        if self.scale:
            self.gamma = nn.Parameter(torch.ones(input_dim))

        if self.conditional:
            if self.hidden_units is not None:
                self.hidden_dense = nn.Linear(in_features=self.cond_dim, out_features=self.hidden_units, bias=False)
            if self.center:
                self.beta_dense = nn.Linear(in_features=self.cond_dim, out_features=input_dim, bias=False)
            if self.scale:
                self.gamma_dense = nn.Linear(in_features=self.cond_dim, out_features=input_dim, bias=False)

        self.initialize_weights()

    def initialize_weights(self):

        if self.conditional:
            if self.hidden_units is not None:
                if self.hidden_initializer == 'normal':
                    torch.nn.init.normal(self.hidden_dense.weight)
                elif self.hidden_initializer == 'xavier':  # glorot_uniform
                    torch.nn.init.xavier_uniform_(self.hidden_dense.weight)

            if self.center:
                torch.nn.init.constant_(self.beta_dense.weight, 0)
            if self.scale:
                torch.nn.init.constant_(self.gamma_dense.weight, 0)

    def forward(self, inputs, cond=None):
        if self.conditional:
            if self.hidden_units is not None:
                cond = self.hidden_dense(cond)

            for _ in range(len(inputs.shape) - len(cond.shape)):
                cond = cond.unsqueeze(1)  # cond = K.expand_dims(cond, 1)

            if self.center:
                beta = self.beta_dense(cond) + self.beta
            if self.scale:
                gamma = self.gamma_dense(cond) + self.gamma
        else:
            if self.center:
                beta = self.beta
            if self.scale:
                gamma = self.gamma

        outputs = inputs
        if self.center:
            mean = torch.mean(outputs, dim=-1).unsqueeze(-1)
            outputs = outputs - mean
        if self.scale:
            variance = torch.mean(outputs ** 2, dim=-1).unsqueeze(-1)
            std = (variance + self.epsilon) ** 0.5
            outputs = outputs / std
            outputs = outputs * gamma
        if self.center:
            outputs = outputs + beta

        return outputs


class ConvolutionLayer(nn.Module):
    def __init__(self, input_size, channels, dilation, dropout=0.1): # 552, 128, kernel_size=(1, 1), stride=(1, 1))
        super(ConvolutionLayer, self).__init__()
        self.base = nn.Sequential(
            nn.Dropout2d(dropout),
            nn.Conv2d(input_size, channels, kernel_size=1),
            nn.GELU(),
        ) #

        self.convs = nn.ModuleList(
            [nn.Conv2d(channels, channels, kernel_size=3, groups=channels, dilation=d, padding=d) for d in dilation])
    def forward(self, x):
        # print(x.shape)
        x = x.permute(0, 3, 1, 2).contiguous()
        # x: 8 552 34 34
        x = self.base(x) # 8 128 51 51
        # print(x.shape)

        outputs = []
        for conv in self.convs:
            x = conv(x)
            x = F.gelu(x)
            outputs.append(x)
        outputs = torch.cat(outputs, dim=1)
        # print(outputs.shape) # 8 sq sq 128

        outputs = outputs.permute(0, 2, 3, 1).contiguous()
        # print(outputs.shape)

        return outputs


class Biaffine(nn.Module):
    def __init__(self, n_in, n_out=1, bias_x=True, bias_y=True):
        super(Biaffine, self).__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        weight = torch.zeros((n_out, n_in + int(bias_x), n_in + int(bias_y)))
        nn.init.xavier_normal_(weight)
        self.weight = nn.Parameter(weight, requires_grad=True)

    def extra_repr(self):
        s = f"n_in={self.n_in}, n_out={self.n_out}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"

        return s

    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        # [batch_size, n_out, seq_len, seq_len]
        s = torch.einsum('bxi,oij,byj->boxy', x, self.weight, y)
        # remove dim 1 if n_out == 1
        s = s.permute(0, 2, 3, 1)

        return s


class MLP(nn.Module):
    def __init__(self, n_in, n_out, dropout=0):
        super().__init__()

        # self.linear = nn.Linear(n_in , n_out)
        self.linear = nn.Linear(n_in, n_out)

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear(x)
        x = self.activation(x)
        return x


class CoPredictor(nn.Module):
    def __init__(self, cls_num, hid_size, biaffine_size, channels, ffnn_hid_size, dropout=0):
        super().__init__()
        self.mlp1 = MLP(n_in=hid_size, n_out=biaffine_size, dropout=dropout)
        self.mlp2 = MLP(n_in=hid_size, n_out=biaffine_size, dropout=dropout)
        self.biaffine = Biaffine(n_in=biaffine_size, n_out=cls_num, bias_x=True, bias_y=True)
        self.mlp_rel = MLP(channels, ffnn_hid_size, dropout=dropout)
        self.linear = nn.Linear(ffnn_hid_size, cls_num)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y, z):
        ent_sub = self.dropout(self.mlp1(x))
        ent_obj = self.dropout(self.mlp2(y))
        # o1 = self.biaffine(ent_sub, ent_obj)
        z = self.dropout(self.mlp_rel(z))
        o2 = self.linear(z)
        # return o1 + o2

        return o2
class dpcnn(nn.Module):
    def __init__(self,  input_size, channels,  dropout=0.1):
        super(dpcnn, self).__init__()
        self.dropout = dropout
        # 卷积核数量(channels数)
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.conv_region = nn.Conv2d(input_size, channels, 1)
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=3)
        # self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))  # top bottom
        # self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))  # bottom
        self.relu = nn.ReLU()

    def forward(self, x):
        # x [8, 34, 34, 552]]
        x = x.permute(0, 3, 1, 2).contiguous() # 8 552 sq sq
        # print(x.shape)
        # x = x.unsqueeze(1)  # [batch_size, 250, seq_len, 1]
        x = self.conv_region(x)  # [batch_size, 128, seq_len-3+1, 1]
        # print(x.shape)
        # x = self.padding1(x)  # [batch_size, 552, seq_len, 1]
        x = self.relu(x)
        x = self.conv(x)  # [batch_size, 552, seq_len-3+1, 1]

        # x = self.padding1(x)  # [batch_size, 128, seq_len, 1]8, 34, 34, 552]

        # print(x.shape)

        x = self.relu(x)
        return x.permute(0, 2, 3, 1)


import torch.nn.functional as F
class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        config.prefix_projection = False
        self.use_bert_last_4_layers = config.use_bert_last_4_layers
        self.bert = AutoModel.from_pretrained(config.bert_name, cache_dir="./cache/", output_hidden_states=True)
        # self.bert_dep = BertModel(config)
    # # ********************************
    #     from_pretrained = False
    #     if from_pretrained:
    #         self.classifier.load_state_dict(torch.load('model/checkpoint.pkl'))
    #
        for param in self.bert.parameters():
            param.requires_grad = True
        config.pre_seq_len = 13
        self.pre_seq_len = config.pre_seq_len
        self.n_layer = config.num_hidden_layers = 12
        self.n_head = config.num_attention_heads = 12
        config.hidden_size = 768
        # self.n_layer = config.num_hidden_layers = 24
        # self.n_head = config.num_attention_heads = 16
        # config.hidden_size = 1024

        self.n_embd = config.hidden_size // config.num_attention_heads
        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_encoder = PrefixEncoder(config)
        # ***************aggcn*******************
        # self._dep_tree = AGGCN.from_params(vocab=vocab, params=modules.pop("gat_tree"))
        # self._dep_tree = AGGCN.from_params()

        # ***************idcnn*******************


        # **********************************


        self.lstm_hid_size = config.lstm_hid_size
        self.conv_hid_size = config.conv_hid_size

        lstm_input_size = 0

        lstm_input_size += config.bert_hid_size

        self.dis_embs = nn.Embedding(20, config.dist_emb_size)
        self.reg_embs = nn.Embedding(3, config.type_emb_size)

        self.encoder = nn.LSTM(lstm_input_size, config.lstm_hid_size // 2, num_layers=1, batch_first=True,
                               bidirectional=True)
        conv_input_size = config.lstm_hid_size + config.dist_emb_size + config.type_emb_size # 552

        # 552 128 [1,2,3] dropout
        # self.convLayer = ConvolutionLayer(conv_input_size + 64, config.conv_hid_size + 64, config.dilation, config.conv_dropout)
        self.convLayer = ConvolutionLayer(conv_input_size, config.conv_hid_size, config.dilation, config.conv_dropout)

        #self.convLayer = dpcnn(conv_input_size + 64 , config.conv_hid_size + 64, config.conv_dropout)
        # self.convLayer = dpcnn(conv_input_size, config.conv_hid_size, config.conv_dropout)

        self.dropout = nn.Dropout(config.emb_dropout)
        self.predictor = CoPredictor(config.label_num, config.lstm_hid_size, config.biaffine_size,
                                     config.conv_hid_size * len(config.dilation), config.ffnn_hid_size,
                                     config.out_dropout)

        # self.cln = LayerNorm(config.lstm_hid_size + 64 , config.lstm_hid_size + 64 , conditional=True)
        self.cln = LayerNorm(config.lstm_hid_size , config.lstm_hid_size, conditional=True)




    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.bert.device)
        past_key_values = self.prefix_encoder(prefix_tokens)
        # bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            self.n_layer * 2,
            self.n_head,
            self.n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values
    def forward(self,
                bert_inputs,
                grid_mask2d,
                dist_inputs,
                pieces2word,
                sent_length,
                output_attentions = True,
                ):
        '''
        :param bert_inputs: [B, L']
        :param grid_mask2d: [B, L, L]
        :param dist_inputs: [B, L, L]
        :param pieces2word: [B, L, L']
        :param sent_length: [B]
        :return:
        '''

        batch_size = bert_inputs.shape[0]
        past_key_values = self.get_prompt(batch_size=batch_size)
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.bert.device)
        attention_mask = torch.cat((prefix_attention_mask, bert_inputs.ne(0).float()), dim=1)
        attention_mask_origin = bert_inputs.ne(0)



        bert_embs = self.bert(input_ids=bert_inputs,
                              attention_mask=attention_mask_origin.float(),
                              output_attentions=output_attentions

                              )


        attentions = bert_embs['attentions']  # 12个 [bs 12 52 63 ]
        # # attentions = attentions[3: 11]
        attentions = torch.stack(attentions, dim=0)  # [12, bs, 12, 52,63]
        attentions1 = torch.mean(attentions, -1)  # 对所有token求平均 [12，4,12,256]
        attentions2 = torch.mean(attentions, -2)
        attentions1 = attentions1.transpose(1, 0)  # [4,12,12,256]
        attentions1 = attentions1.reshape(attentions1.shape[0], -1, attentions1.shape[-1])  # [64,144,256]
        attentions2 = attentions2.transpose(1, 0)  # [4,12,12,256]
        attentions2 = attentions2.reshape(attentions2.shape[0], -1, attentions2.shape[-1])  # [64,144,256]



        if self.use_bert_last_4_layers:
            bert_embs = torch.stack(bert_embs[2][-4:], dim=-1).mean(-1)
        else:
            # bert_embs = bert_embs[0]  # [batch_size, seqence_length, 768]
            bert_embs = bert_embs['hidden_states'][11]  # ablation study [batch_size, seqence_length, 768]

        length = pieces2word.size(1)  # 48
        min_value = torch.min(bert_embs).item()
        # Max pooling word representations from pieces
        _bert_embs = bert_embs.unsqueeze(1).expand(-1, length, -1, -1) # [8,41,64,768]
        _bert_embs = torch.masked_fill(_bert_embs, pieces2word.eq(0).unsqueeze(-1), min_value)
        word_reps, _ = torch.max(_bert_embs, dim=2)
        word_reps = self.dropout(word_reps) # [8,41,768]
        word_reps_shape = word_reps.shape  # [8 51 768]

        packed_embs = pack_padded_sequence(word_reps, sent_length.cpu(), batch_first=True, enforce_sorted=False)

        # print(packed_embs)
        packed_outs, (hidden, _) = self.encoder(packed_embs)


        word_reps, _ = pad_packed_sequence(packed_outs, batch_first=True, total_length=sent_length.max()) # [8 51 512]


        cln = self.cln(word_reps.unsqueeze(2), word_reps)
        dis_emb = self.dis_embs(dist_inputs)
        tril_mask = torch.tril(grid_mask2d.clone().long())
        reg_inputs = tril_mask + grid_mask2d.clone().long()
        reg_emb = self.reg_embs(reg_inputs)

        conv_inputs = torch.cat([dis_emb, reg_emb, cln], dim=-1)
        conv_inputs = torch.masked_fill(conv_inputs, grid_mask2d.eq(0).unsqueeze(-1), 0.0) # 8 sq sq 552
        # print(conv_inputs.shape)
        conv_outputs = self.convLayer(conv_inputs) # 8 sq sq 128

        conv_outputs = torch.masked_fill(conv_outputs, grid_mask2d.eq(0).unsqueeze(-1), 0.0)
        outputs = self.predictor(word_reps, word_reps, conv_outputs)

        return outputs, attentions1, attentions2

        # return outputs


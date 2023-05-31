'''
This code is provided solely for the personal and private use of students
taking the CSC401H/2511H course at the University of Toronto. Copying for
purposes other than this use is expressly prohibited. All forms of
distribution of this code, including but not limited to public repositories on
GitHub, GitLab, Bitbucket, or any other online platform, whether as given or
with any changes, are expressly prohibited.

Authors: Sean Robertson, Jingcheng Niu, Zining Zhu, and Mohamed Abdall
Updated by: Raeid Saqur <raeidsaqur@cs.toronto.edu>

All of the files in this directory and all subdirectories are:
Copyright (c) 2023 University of Toronto
'''

'''Concrete implementations of abstract base classes.

You don't need anything more than what's been imported here
'''

import torch
from typing import Optional, Union, Tuple, Type, Set

from a2_abcs import EncoderBase, DecoderBase, EncoderDecoderBase

# All docstrings are omitted in this file for simplicity. So please read
# a2_abcs.py carefully so that you can have a solid understanding of the
# structure of the assignment.

class Encoder(EncoderBase):


    def init_submodules(self):
        # Hints:
        # 1. You must initialize the following submodules:
        #   self.rnn, self.embedding
        # 2. You will need the following object attributes:
        #   self.source_vocab_size, self.word_embedding_size,
        #   self.pad_id, self.dropout, self.cell_type,
        #   self.hidden_state_size, self.num_hidden_layers.
        # 3. cell_type will be one of: ['lstm', 'rnn']
        # 4. Relevant pytorch modules: torch.nn.{LSTM, RNN, Embedding}
        if self.cell_type == 'lstm':
            self.rnn = torch.nn.LSTM(self.word_embedding_size, self.hidden_state_size,
                                     num_layers=self.num_hidden_layers,
                                     dropout=self.dropout,
                                     bidirectional=True)
        elif self.cell_type == 'rnn':
            self.rnn = torch.nn.RNN(self.word_embedding_size, self.hidden_state_size,
                                    num_layers=self.num_hidden_layers,
                                    dropout=self.dropout,
                                    bidirectional=True)

        self.embedding = torch.nn.Embedding(self.source_vocab_size,
                                            self.word_embedding_size,
                                            padding_idx=self.pad_id)

    def forward_pass(
            self,
            source_x: torch.LongTensor,
            source_x_lens: torch.LongTensor,
            h_pad: float = 0.) -> torch.FloatTensor:
        # Recall:
        #   source_x is shape (S, B)
        #   source_x_lens is of shape (B,)
        #   h_pad is a float
        #
        # Hints:
        # 1. The structure of the encoder should be:
        #   input seq -> |embedding| -> embedded seq -> |rnn| -> seq hidden
        # 2. You will need to use the following methods:
        #   self.get_all_rnn_inputs, self.get_all_hidden_states
        all_rnn_inputs = self.get_all_rnn_inputs(source_x)
        all_hidden_states = self.get_all_hidden_states(all_rnn_inputs, source_x_lens, h_pad)
        return all_hidden_states

    def get_all_rnn_inputs(self, source_x: torch.LongTensor) -> torch.FloatTensor:
        # Recall:
        #   source_x is shape (S, B)
        #   x (output) is shape (S, B, I)
        return self.embedding(source_x)

    def get_all_hidden_states(
            self, 
            x: torch.FloatTensor,
            source_x_lens: torch.LongTensor,
            h_pad: float) -> torch.FloatTensor:
        # Recall:
        #   x is of shape (S, B, I)
        #   source_x_lens is of shape (B,)
        #   h_pad is a float
        #   h (output) is of shape (S, B, 2 * H)
        #
        # Hint:
        #   relevant pytorch modules:
        #   torch.nn.utils.rnn.{pad_packed,pack_padded}_sequence
        pack_padded = torch.nn.utils.rnn.pack_padded_sequence(x, source_x_lens, enforce_sorted=False)
        output, _ = self.rnn.forward(pack_padded)
        pad_packed, _ = torch.nn.utils.rnn.pad_packed_sequence(output, padding_value=h_pad)
        return pad_packed


class DecoderWithoutAttention(DecoderBase):
    '''A recurrent decoder without attention'''

    def init_submodules(self):
        # Hints:
        # 1. You must initialize the following submodules:
        #   self.embedding, self.cell, self.output_layer
        # 2. You will need the following object attributes:
        #   self.target_vocab_size, self.word_embedding_size, self.pad_id
        #   self.hidden_state_size, self.cell_type.
        # 3. cell_type will be one of: ['lstm', 'rnn']
        # 4. Relevant pytorch modules:
        #   torch.nn.{Embedding, Linear, LSTMCell, RNNCell}
        self.embedding = torch.nn.Embedding(num_embeddings=self.target_vocab_size,
                                            embedding_dim=self.word_embedding_size,
                                            padding_idx=self.pad_id)

        self.output_layer = torch.nn.Linear(in_features=self.hidden_state_size, out_features=self.target_vocab_size)

        if self.cell_type == 'lstm':
            self.cell = torch.nn.LSTMCell(self.word_embedding_size, self.hidden_state_size)

        elif self.cell_type == 'rnn':
            self.cell = torch.nn.RNNCell(self.word_embedding_size, self.hidden_state_size)


    def forward_pass(
            self,
                target_y_tm1: torch.LongTensor,
                htilde_tm1: Union[
                    torch.FloatTensor,
                    Tuple[torch.FloatTensor, torch.FloatTensor]],
                h: torch.FloatTensor,
                source_x_lens: torch.LongTensor) -> Tuple[
                    torch.FloatTensor, Union[
                        torch.FloatTensor,
                        Tuple[torch.FloatTensor, torch.FloatTensor]]]:
            # Recall:
            #   target_y_tm1 is of shape (B,)
            #   htilde_tm1 is of shape (B, 2 * H)
            #   h is of shape (S, B, 2 * H)
            #   source_x_lens is of shape (B,)
            #   logits_t (output) is of shape (B, V)
            #   htilde_t (output) is of same shape as htilde_tm1
            #
            # Hints:
            # 1. The structure of the encoder should be:
            #   encoded hidden -> |embedding| -> embedded hidden -> |rnn| ->
            #   decoded hidden -> |output layer| -> output logits
            # 2. You will need to use the following methods:
            #   self.get_current_rnn_input, self.get_current_hidden_state,
            #   self.get_current_logits
            # 3. You can assume that htilde_tm1 is not empty. I.e., the hidden state
            #   is either initialized, or t > 1.
            # 4. The output of an LSTM cell is a tuple (h, c), but a GRU cell or an
            #   RNN cell will only output h.
        xtilde_t = self.get_current_rnn_input(target_y_tm1, htilde_tm1, h, source_x_lens)

        htilde_t = self.get_current_hidden_state(xtilde_t, htilde_tm1)

        if self.cell_type == 'lstm':
            logits_t = self.get_current_logits(htilde_t[0])
        else:
            logits_t = self.get_current_logits(htilde_t)

        return logits_t, htilde_t


    def get_first_hidden_state(
            self,
            h: torch.FloatTensor,
            source_x_lens: torch.LongTensor) -> torch.FloatTensor:
        # Recall:
        #   h is of shape (S, B, 2 * H)
        #   source_x_lens is of shape (B,)
        #   htilde_tm1 (output) is of shape (B, 2 * H)
        #
        # Hint:
        # 1. Ensure it is derived from encoder hidden state that has processed
        # the entire sequence in each direction. You will need to:
        # - Populate indices [0: self.hidden_state_size // 2] with the hidden
        #   states of the encoder's forward direction at the highest index in
        #   time *before padding*
        # - Populate indices [self.hidden_state_size//2:self.hidden_state_size]
        #   with the hidden states of the encoder's backward direction at time
        #   t=0
        # 2. Relevant pytorch function: torch.cat
        forward_obs = h[source_x_lens - 1, torch.arange(source_x_lens.shape[0]), 0: self.hidden_state_size // 2]
        backward_obs = h[0, :, self.hidden_state_size // 2:self.hidden_state_size]
        concat_entire = torch.cat([forward_obs, backward_obs], dim=1)
        return concat_entire

    def get_current_rnn_input(
            self,
            target_y_tm1: torch.LongTensor,
            htilde_tm1: Union[
                torch.FloatTensor,
                Tuple[torch.FloatTensor, torch.FloatTensor]],
            h: torch.FloatTensor,
            source_x_lens: torch.LongTensor) -> torch.FloatTensor:
        # Recall:
        #   target_y_tm1 is of shape (B,)
        #   htilde_tm1 is of shape (B, 2 * H) or a tuple of two of those (LSTM)
        #   h is of shape (S, B, 2 * H)
        #   source_x_lens is of shape (B,)
        #   xtilde_t (output) is of shape (B, Itilde)
        return self.embedding(target_y_tm1)

    def get_current_hidden_state(
            self,
            xtilde_t: torch.FloatTensor,
            htilde_tm1: Union[
                torch.FloatTensor,
                Tuple[torch.FloatTensor, torch.FloatTensor]]) -> Union[
                    torch.FloatTensor,
                    Tuple[torch.FloatTensor, torch.FloatTensor]]:
        # Recall:
        #   xtilde_t is of shape (B, Itilde)
        #   htilde_tm1 is of shape (B, 2 * H) or a tuple of two of those (LSTM)
        #   htilde_t (output) is of same shape as htilde_tm1
        if self.cell_type == 'lstm':
            htilde_tm1 = (htilde_tm1[0], htilde_tm1[1])

        return self.cell(xtilde_t, htilde_tm1)

    def get_current_logits(
            self,
            htilde_t: torch.FloatTensor) -> torch.FloatTensor:
        # Recall:
        #   htilde_t is of shape (B, 2 * H), even for LSTM (cell state discarded)
        #   logits_t (output) is of shape (B, V)

        return self.output_layer.forward(htilde_t)

class DecoderWithAttention(DecoderWithoutAttention):
    '''A decoder, this time with attention

    Inherits from DecoderWithoutAttention to avoid repeated code.
    '''

    def init_submodules(self):
        # Hints:
        # 1. Same as the case without attention, you must initialize the
        #   following submodules: self.embedding, self.cell, self.output_layer
        # 2. You will need the following object attributes:
        #   self.target_vocab_size, self.word_embedding_size, self.pad_id
        #   self.hidden_state_size, self.cell_type.
        # 3. cell_type will be one of: ['lstm', 'rnn']
        # 4. Relevant pytorch modules:
        #   torch.nn.{Embedding, Linear, LSTMCell, RNNCell, GRUCell}
        # 5. The implementation of this function should be different from
        #   DecoderWithoutAttention.init_submodules.
        self.embedding = torch.nn.Embedding(num_embeddings=self.target_vocab_size,
                                            embedding_dim=self.word_embedding_size,
                                            padding_idx=self.pad_id)

        self.ff = torch.nn.Linear(in_features=self.hidden_state_size, out_features=self.target_vocab_size)

        decoder_input_size = self.word_embedding_size + self.hidden_state_size

        if self.cell_type == 'lstm':
            self.cell = torch.nn.LSTMCell(decoder_input_size, self.hidden_state_size)

        elif self.cell_type == 'rnn':
            self.cell = torch.nn.RNNCell(decoder_input_size, self.hidden_state_size)

    def get_first_hidden_state(
            self,
            h: torch.FloatTensor,
            source_x_lens: torch.LongTensor) -> torch.FloatTensor:
        # Hint: For this time, the hidden states should be initialized to zeros.
        return torch.zeros_like(h[0]).to(h.device)

    def get_current_rnn_input(
            self,
            target_y_tm1: torch.LongTensor,
            htilde_tm1: Union[
                torch.FloatTensor,
                Tuple[torch.FloatTensor, torch.FloatTensor]],
            h: torch.FloatTensor,
            source_x_lens: torch.LongTensor) -> torch.FloatTensor:
        # Hint: Use attend() for c_t
        h_t = self.embedding(target_y_tm1)
        c_t = self.attend(htilde_tm1, h, source_x_lens)

        return torch.cat([h_t, c_t], dim=1)

    def attend(
            self,
            htilde_t: Union[
                torch.FloatTensor,
                Tuple[torch.FloatTensor, torch.FloatTensor]],
            h: torch.FloatTensor,
            source_x_lens: torch.LongTensor) -> torch.FloatTensor:
        '''The attention mechanism. Calculate the context vector c_t.

        Parameters
        ----------
        htilde_t : torch.FloatTensor or tuple
            Like `htilde_tm1` (either a float tensor or a pair of float
            tensors), but matching the current hidden state.
        h : torch.FloatTensor
            A float tensor of shape ``(S, B, self.hidden_state_size)`` of
            hidden states of the encoder. ``h[s, b, i]`` is the
            ``i``-th index of the encoder RNN's last hidden state at time ``s``
            of the ``b``-th sequence in the batch. The states of the
            encoder have been right-padded such that ``h[source_x_lens[b]:, b]``
            should all be ignored.
        source_x_lens : torch.LongTensor
            An integer tensor of shape ``(B,)`` corresponding to the lengths
            of the encoded source sentences.

        Returns
        -------
        c_t : torch.FloatTensor
            A float tensor of shape ``(B, self.hidden_state_size)``. The
            context vector c_t is the product of weights alpha_t and h.

        Hint: Use get_attention_weights() to calculate alpha_t.
        '''
        alpha_t = self.get_attention_weights(htilde_t, h, source_x_lens)
        alpha_t = alpha_t.unsqueeze(2)
        return (alpha_t * h).sum(dim=0)

    def get_attention_weights(
            self,
            htilde_t: Union[
                torch.FloatTensor,
                Tuple[torch.FloatTensor, torch.FloatTensor]],
            h: torch.FloatTensor,
            source_x_lens: torch.LongTensor) -> torch.FloatTensor:
        # DO NOT MODIFY! Calculates attention weights, ensuring padded terms
        # in h have weight 0 and no gradient. You have to implement
        # get_attention_scores()
        # alpha_t (output) is of shape (S, B)
        a_t = self.get_attention_scores(htilde_t, h)
        pad_mask = torch.arange(h.shape[0], device=h.device)
        pad_mask = pad_mask.unsqueeze(-1) >= source_x_lens.to(h.device)  # (S, B)
        a_t = a_t.masked_fill(pad_mask, -float('inf'))
        return torch.nn.functional.softmax(a_t, 0)

    def get_attention_scores(
            self,
            htilde_t: Union[
                torch.FloatTensor,
                Tuple[torch.FloatTensor, torch.FloatTensor]],
            h: torch.FloatTensor) -> torch.FloatTensor:
        # Recall:
        #   htilde_t is of shape (B, 2 * H)
        #   h is of shape (S, B, 2 * H)
        #   a_t (output) is of shape (S, B)
        #
        # Hint:
        # Relevant pytorch function: torch.nn.functional.cosine_similarity
        if self.cell_type == "lstm":
            htilde_t = htilde_t[0].unsqueeze(0)
        else:
            htilde_t = htilde_t.unsqueeze(0)
        return torch.nn.functional.cosine_similarity(htilde_t, h, dim=2)

class DecoderWithMultiHeadAttention(DecoderWithAttention):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.W is not None, 'initialize W!'
        assert self.Wtilde is not None, 'initialize Wtilde!'
        assert self.Q is not None, 'initialize Q!'

    def init_submodules(self):
        super().init_submodules()  # Do not change this line

        # Hints:
        # 1. The above line should ensure self.output_layer, self.embedding, self.cell are
        #    initialized
        # 2. You need to initialize the following submodules:
        #       self.W, self.Wtilde, self.Q
        # 3. You will need the following object attributes:
        #       self.hidden_state_size
        # 4. self.W, self.Wtilde, and self.Q should process all heads at once. They
        #    should not be lists!
        # 5. You do *NOT* need self.heads at this point
        # 6. Relevant pytorch module: torch.nn.Linear (note: set bias=False!)
        self.W = torch.nn.Linear(in_features=self.hidden_state_size,
                                 out_features=self.hidden_state_size, bias=False)
        self.Wtilde = torch.nn.Linear(in_features=self.hidden_state_size,
                                      out_features=self.hidden_state_size, bias=False)
        self.Q = torch.nn.Linear(in_features=self.hidden_state_size,
                                 out_features=self.hidden_state_size, bias=False)

    def attend(
            self,
            htilde_t: Union[
                torch.FloatTensor,
                Tuple[torch.FloatTensor, torch.FloatTensor]],
            h: torch.FloatTensor,
            source_x_lens: torch.LongTensor) -> torch.FloatTensor:
        # Hints:
        # 1. You can use super().attend to call for the regular attention
        #   function.
        # 2. Relevant pytorch function:
        #   tensor().view, tensor().repeat_interleave
        # 3. Fun fact:
        #   tensor([1,2,3,4]).repeat(2) will output tensor([1,2,3,4,1,2,3,4]).
        #   tensor([1,2,3,4]).repeat_interleave(2) will output
        #   tensor([1,1,2,2,3,3,4,4]), just like numpy.repeat.
        # 4. You *WILL* need self.heads at this point
        htilde_twt = self.Wtilde(htilde_t)
        htilde_tw = self.W(h)

        htilde_twt = htilde_twt.view(htilde_t.shape[0] * self.heads, self.hidden_state_size // self.heads)
        htilde_tw = htilde_tw.view(h.shape[0] * self.heads, h.shape[1], self.hidden_state_size // self.heads)

        repearted_x_lens = source_x_lens.repeat_interleave(repeats=self.heads)

        c_t = super().attend(htilde_twt, htilde_tw, repearted_x_lens)
        return self.Q(c_t.view(htilde_t.shape[0], htilde_t.shape[1]))

class EncoderDecoder(EncoderDecoderBase):

    def init_submodules(
            self,
            encoder_class: Type[EncoderBase],
            decoder_class: Type[DecoderBase]):
        # Hints:
        # 1. You must initialize the following submodules:
        #   self.encoder, self.decoder
        # 2. encoder_class and decoder_class inherit from EncoderBase and
        #   DecoderBase, respectively.
        # 3. You will need the following object attributes:
        #   self.source_vocab_size, self.source_pad_id,
        #   self.word_embedding_size, self.encoder_num_hidden_layers,
        #   self.encoder_hidden_size, self.encoder_dropout, self.cell_type,
        #   self.target_vocab_size, self.target_eos, self.heads
        # 4. Recall that self.target_eos doubles as the decoder pad id since we
        #   never need an embedding for it.
        self.encoder = encoder_class(self.source_vocab_size, self.source_pad_id,
                                    self.word_embedding_size, self.encoder_num_hidden_layers,
                                    self.encoder_hidden_size, self.encoder_dropout, self.cell_type)

        self.decoder = decoder_class(self.target_vocab_size, self.target_eos,
                                     self.word_embedding_size, self.encoder_hidden_size * 2,
                                     self.cell_type, self.heads)

        self.encoder.init_submodules()
        self.decoder.init_submodules()


    def translate(self, input_sentence):
        # This method translates the input sentence from the model's source
        # language to the target language.
        # 1. Tokenize the input sentence.
        # 2. Compute the length of the input sentence.
        # 3. Feed the tokenized sentence into the model.
        # 4. Decode the output of the sentence into a string.

        # Hints:
        # 1. You will need the following methods/attributs from the dataset.
        # Consult :class:`HansardEmptyDataset` for a description of parameters
        # and attributes.
        #   self.dataset.tokenize(input_sentence)
        #       This function tokenizes the input sentence.  For example:
        #       >>> self.dataset.tokenize('This is a sentence.')
        #       ['this', 'is', 'a', 'sentence']
        #   self.dataset.source_word2id
        #       A dictionary that maps tokens to ids for the source language.
        #       For example: `self.dataset.source_word2id['francophone'] -> 5127`
        #   self.dataset.source_unk
        #       The speical token for unknown input tokens.  Any token in the
        #       input sentence that isn't present in the source vocabulary should
        #       be converted to this special token.
        #   self.dataset.target_id2word
        #       A dictionary that maps ids to tokens for the target language.
        #       For example: `self.dataset.source_word2id[6123] -> 'anglophone'`
        # 
        # 2. Relevant pytorch function:
        #   tensor().view, tensor().repeat_interleave
        # Tokenize the input sentence.

        tokenized_sentence = self.dataset.tokenize(input_sentence)

        source_ids = [self.dataset.source_word2id.get(token, self.dataset.source_unk)
                      for token in tokenized_sentence]
        source_tensor = torch.tensor(source_ids, dtype=torch.long).unsqueeze(1)
        with torch.no_grad():
            output = self.model(source_tensor)
        output_ids = output.argmax(dim=2).squeeze().tolist()
        output_tokens = [self.dataset.target_id2word[output_id] for output_id in output_ids]
        output_tokens = [token for token in output_tokens if token not in self.dataset.special_tokens]
        return ' '.join(output_tokens)

    def get_logits_for_teacher_forcing(
            self,
            h: torch.FloatTensor,
            source_x_lens: torch.LongTensor,
            target_y: torch.LongTensor) -> torch.FloatTensor:
        # Recall:
        #   h is of shape (S, B, 2 * H)
        #   source_x_lens is of shape (B,)
        #   target_y is of shape (T, B)
        #   logits (output) is of shape (T - 1, B, Vo)
        #
        # Hints:
        # 1. Relevant pytorch modules: torch.{zero_like, stack}
        # 2. Recall an LSTM's cell state is always initialized to zero.
        # 3. Note logits sequence dimension is one shorter than target_y (why?)
        logits = []
        hidden = None
        for t in range(target_y.shape[0] - 1):
            logits.append(self.decoder.forward_pass(target_y[t], hidden, h, source_x_lens)[0])
        logits_new = logits[:]
        logits = torch.stack(logits_new, dim=0)

        return logits

    def update_beam(
            self,
            htilde_t: torch.FloatTensor,
            b_tm1_1: torch.LongTensor,
            logpb_tm1: torch.FloatTensor,
            logpy_t: torch.FloatTensor) -> Tuple[
                torch.FloatTensor, torch.LongTensor, torch.FloatTensor]:
        # perform the operations within the psuedo-code's loop in the
        # assignment.
        # You do not need to worry about which paths have finished, but DO NOT
        # re-normalize logpy_t.
        #
        # Recall
        #   htilde_t is of shape (B, K, 2 * H) or a tuple of two of those (LSTM)
        #   logpb_tm1 is of shape (B, K)
        #   b_tm1_1 is of shape (t, B, K)

        ## Output order:
        #   logpb_t (first output) is of shape (B, K)
        #   b_t_0 (second output) is of shape (B, K, 2 * H) or a tuple of two of
        #      those (LSTM)
        #   b_t_1 (third output) is of shape (t + 1, B, K)
        # Hints:
        # 1. Relevant pytorch modules:
        #   torch.{flatten, topk, unsqueeze, expand_as, gather, cat}
        # 2. If you flatten a two-dimensional array of shape z of (X, Y),
        #   then the element z[a, b] maps to z'[a*Y + b]
        M, K, V = logpy_t.size()

        pkss = logpb_tm1.unsqueeze(-1)
        pkss = pkss + logpy_t

        req_dim = pkss.shape[0]
        req_shape = (req_dim, -1)

        final_p = pkss.view(req_shape)
        # TODO: Look up how to get the best k from final_p
        logpb_t, max_idxs = torch.topk(input=final_p,
                                       k=self.beam_width,
                                       dim=-1,
                                       sorted=True,
                                       largest=True)
        divided_idxs = max_idxs // V
        residual_idxs = max_idxs % V

        b_tm1_1 = b_tm1_1.gather(index=divided_idxs.unsqueeze(0).expand_as(b_tm1_1),
                                 dim=2)
        lst = []
        if self.cell_type == "lstm":
            lst.append(htilde_t[0])
            lst.append(htilde_t[1])
        else:
            lst.append(htilde_t)
        outs = []
        for index in range(len(lst)):
            element = lst[index].gather(index=divided_idxs.unsqueeze(-1).expand_as(lst[index]),
                                        dim=1)
            outs.append(element)
        if len(outs) == 1:
            b_t_0 = outs[0]
        else:
            b_t_0 = tuple(outs)
        req_shape = (1, M, K)
        residual_idxs = residual_idxs.view(req_shape)
        b_t_1 = torch.cat([b_tm1_1, residual_idxs], dim=0)
        return logpb_t, b_t_0, b_t_1

import dynet as dy
import utils

EMBEDDING_SIZE = 75

"""This file contains various encoders than can encode a string word to a real vector"""

class EmbeddingEncoder(object):

	def __init__(self, inp_size, model, W2I):

		self.model = model
		self.W2I = W2I
		self.E = model.add_lookup_parameters((inp_size, EMBEDDING_SIZE))

	def encode(self, w):

		word_encoded = self.W2I[w] if w in self.W2I else self.W2I["<unk>"]
		#print "word_encoded:", word_encoded
		word_encoded = dy.lookup(self.E, word_encoded)
		return word_encoded


class LSTMEncoder(object):

	def __init__(self, inp_size, model, CHAR2I):

		self.model = model
		self.CHAR2I = CHAR2I

		self.E2 = model.add_lookup_parameters((inp_size, EMBEDDING_SIZE))
		self.builder = dy.LSTMBuilder(16, EMBEDDING_SIZE, EMBEDDING_SIZE, self.model)


	def encode(self, w):

		s = self.builder.initial_state()
		assert w!=""
		encoded = [self.CHAR2I[c] if c in self.CHAR2I else self.CHAR2I["<unk>"] for c in w]
		embedded = [dy.lookup(self.E2, char) for char in encoded]

		lstm_out = s.transduce(embedded)[-1]
		assert lstm_out is not None
		return lstm_out

class SubwordEncoder(object):

	def __init__(self, inp_size, model, W2I, S2I, P2I):

		self.model = model
		self.W2I = W2I
		self.S2I = S2I
		self.P2I = P2I
		self.i = 0


		self.E = model.add_lookup_parameters((inp_size, EMBEDDING_SIZE))
		self.E_pre = self.model.add_lookup_parameters((len(P2I), EMBEDDING_SIZE))
               	self.E_suf = self.model.add_lookup_parameters((len(S2I), EMBEDDING_SIZE))
		self.W = model.add_parameters((EMBEDDING_SIZE, EMBEDDING_SIZE*3))

	def encode(self, w):
		      self.i+=1
                      pre3, suf3 = w[:3], w[-3:]
                      pre3_idx =  self.P2I[pre3] if pre3 in self.P2I else self.P2I["<unk>"]			
                      suf3_idx =  self.S2I[suf3] if suf3 in self.S2I else self.S2I["<unk>"]

                      pre2, suf2 = w[:2], w[-2:]
                      pre2_idx =  self.P2I[pre2] if pre2 in self.P2I else self.P2I["<unk>"]			
                      suf2_idx =  self.S2I[suf2] if suf2 in self.S2I else self.S2I["<unk>"]

		      word_encoded = self.W2I[w] if w in self.W2I else self.W2I["<unk>"]

                      word_e = dy.lookup(self.E, word_encoded)
	              pre2_e  = dy.lookup(self.E_pre, pre2_idx)
                      suf2_e = dy.lookup(self.E_suf, suf2_idx)
	              pre3_e  = dy.lookup(self.E_pre, pre3_idx)
                      suf3_e = dy.lookup(self.E_suf, suf3_idx)

		      W = dy.parameter(self.W)

		      return W * dy.concatenate([word_e, pre2_e+pre3_e, suf2_e+suf3_e])


class ComplexEncoder(object):

	def __init__(self, inp_size, model, W2I, CHAR2I):

		self.embedding_encoder = EmbeddingEncoder(inp_size, model, W2I)
		self.LSTM_encoder = LSTMEncoder(len(CHAR2I), model, CHAR2I)

		self.W = model.add_parameters((EMBEDDING_SIZE, EMBEDDING_SIZE*2))

	def encode(self, w):

		lstm_encoding = self.LSTM_encoder.encode(w)
		embedding_encoding = self.embedding_encoder.encode(w)
		W = dy.parameter(self.W)

		return W * dy.concatenate([embedding_encoding, lstm_encoding])




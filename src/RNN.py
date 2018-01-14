import dynet as dy
import numpy as np
import random
import time

import dynet_config
dynet_config.set_gpu()
#from googletrans import Translator

NUM_LAYERS = 1
EMBEDDING_SIZE = 75


ABSO = {"<NR_HK>": "absolutive pl3", "<NR_HU>": "absolutive sg3", "<NR_HI>": "absolutive ??",
"<NR_GU>": "absolutive 1pl", "<NR_NI>":"absolutive 1sg", "<NR_ZU>": "absolutive 2sg", "<NR_ZK>": "absolutive 2pl", "None": "None"}

ERG = {"<NK_HU>": "ergative sg3", "<NK_HK>": "ergative pl3", "<NK_HI>": "ergative ??",
"<NK_GU>": "ergative 1pl", "<NK_NI>": "ergative 1sg", "<NK_ZU>": "ergative 2sg", "<NK_ZK>": "ergative 2pl", "None": "None"}

DAT = {"<NI_HU>": "dative sg3", "<NI_HK>": "dative pl3", "<NI_HI>": "dative ??",
"<NI_GU>": "dative 1pl", "<NI_NI>": "dative 1sg", "<NI_ZU>": "dative 2sg", "<NI_ZK>": "dative 2pl", "None": "None"}

class RNN(object):

	def __init__(self, in_size, hid_size, (a_out, e_out, d_out), dataGenerator, I2A, I2E, I2D, I2W, model, encoder):

		self.in_size = in_size
		self.hid_size = hid_size
		self.a_out = a_out
		self.e_out = e_out
		self.d_out = d_out

		self.I2A = I2A
		self.I2E = I2E
		self.I2D = I2D
		self.I2W = I2W

		self.generator = dataGenerator
		self.model = model
		self.encoder = encoder
		self.attention_fwd, self.attention_bwd = None, None
		self.create_model()
		

	def create_model(self):

                """add parameters to the model, that consists of a biLSTM layer(s),
		and 3 softmax layers (for dative (d), ergative (e) and absolutive (a) agreement prediction).

		W_attention1, W_attention2 - attention matrices
		W_ha, W_he, W_hd - hidden-output matrices for absolutive, ergative and dative.
		W_hh, W_hh2 - hidden-hidden matrices
		fwdLSTM, bwdLSTM - builders for bidirectional lstm layers
		"""

		self.W_attention1 = self.model.add_parameters((16, EMBEDDING_SIZE+self.hid_size))
		self.W_attention2 = self.model.add_parameters((1, 16))

		hid = 64

		self.W_ha = self.model.add_parameters((self.a_out, hid))
		self.W_he = self.model.add_parameters((self.e_out, hid))
		self.W_hd = self.model.add_parameters((self.d_out, hid))
		self.W_hh = self.model.add_parameters((hid, 2*self.hid_size))
		self.W_hh2 = self.model.add_parameters((hid, hid))

		self.fwdLSTM = dy.LSTMBuilder(NUM_LAYERS, EMBEDDING_SIZE, self.hid_size, self.model)
		self.bwdLSTM = dy.LSTMBuilder(NUM_LAYERS, EMBEDDING_SIZE, self.hid_size, self.model)
                self.trainer = dy.AdamTrainer(self.model)
        

	def _attend(self, encoded_sent, states):

                """computes attention weights over the lsmt states
		- encoded_sent is the encoding of the input sentence
		- states is a list of lsmt states (one for each encoded word)
		
		returns: weights, an array of attention weights

		"""

		assert len(encoded_sent)==len(states)

		W_attention1 = dy.parameter(self.W_attention1)
		W_attention2 = dy.parameter(self.W_attention2)

		# pass the concatenation of the words & the states through a hidden layer, then softmax

		h = [dy.rectify(W_attention1*dy.concatenate([w,s])) for w,s in zip(encoded_sent,states)]
		weights = dy.concatenate([W_attention2*h_elem for h_elem in h])
		weights =  dy.softmax(weights)
	

		assert len(weights.npvalue())==len(encoded_sent)

		return weights


        def _predict(self, sentence, training=True, dropout_rate = 0.1):	

                """predict the agreement of the subject, object and indirect object.

		- sentence - a list of sentence words (as strings)
		
		returns: a_pred, e_pred, d_pred - absolutive, ergative and dative agreements

		"""
		#prepare parameters

		W_ha = dy.parameter(self.W_ha)
		W_he = dy.parameter(self.W_he)
		W_hd = dy.parameter(self.W_hd)

		W_hh = dy.parameter(self.W_hh)
		W_hh2 = dy.parameter(self.W_hh2)

		s_fwd = self.fwdLSTM.initial_state()
		s_bwd = self.bwdLSTM.initial_state()

		verb_index = sentence.index("<verb>")

		# encode sentence & pass through biLstm

		encoded = [self.encoder.encode(w) for w in sentence]
		if training: encoded = [dy.dropout(e,dropout_rate) for e in encoded]
 
		seq_upto_verb, seq_upto_verb_rev = encoded[:verb_index+1], encoded[verb_index:][::-1]
		output_fwd = s_fwd.transduce(seq_upto_verb)
		output_bwd = s_bwd.transduce(seq_upto_verb_rev)

		# attend over bilstm states

		weights_fwd = self._attend(seq_upto_verb, output_fwd)
		weights_bwd = self._attend(seq_upto_verb_rev, output_bwd)
		self.attention_fwd, self.attention_bwd = weights_fwd, weights_bwd
		
		fwd_weighted = dy.esum([o*w for o,w in zip(output_fwd, weights_fwd)])
		bwd_weighted = dy.esum([o*w for o,w in zip(output_bwd, weights_bwd)])

		#output is a concatenation of weighted forward and backward states

		output = dy.concatenate([fwd_weighted, bwd_weighted])

		h = dy.rectify(W_hh * output)
		#h = dy.rectify(W_hh2 * h)

		# predict absolutive, ergative and dative agreements.

		a_pred = dy.softmax(W_ha * h)
		e_pred = dy.softmax(W_he * h)
		d_pred = dy.softmax(W_hd * h)

		return (a_pred, e_pred, d_pred)

	def encode(self, sentence):

		"""encode the sentence words with the encoder"""

		return [self.encoder.encode(w) for w in sentence]

        def train(self, epochs=30):

	  n = self.generator.get_train_size()
	  print "size of training set: ", n
          print "training..."

	  iteration = 0
	  good_a, bad_a, good_e, bad_e, good_d, bad_d = 1., 1., 1., 1., 1., 1.
	  losses = []

	  for i, batch in enumerate(self.generator.generate(is_training=True)):

		dy.renew_cg()
		for j, training_example in enumerate(batch):


			iteration+=1

			#stopping criteria
	
                        if iteration > epochs*n: 
				#print "Calcualting accuracy on test set. This may take some time"
				#self.test(Mode.TEST)
				return

			# report progress. 

			if iteration%n == 0:

				iteration = 0
				print "EPOCH {} / {}".format(iteration/n, epochs)


			# prepare input & predict
	
			sent, (e_true,a_true,d_true), data_sample = training_example

			a_pred, e_pred, d_pred = self._predict(sent, training=True)

			# collect loss & errors

			loss = -dy.log(a_pred[a_true]) + -dy.log(e_pred[e_true]) + -dy.log(d_pred[d_true])
			a_pred, e_pred, d_pred = np.argmax(a_pred.npvalue()),  np.argmax(e_pred.npvalue()),  np.argmax(d_pred.npvalue())

			if a_pred==a_true:
				good_a+=1
			else:
				bad_a+=1
			if e_pred==e_true:
				good_e+=1
			else:
				bad_e+=1

			losses.append(loss)

		# backprop

		loss_sum = dy.esum(losses)
		loss_sum.backward()
		self.trainer.update()
		losses = []

		# check dev set accuracy

		if i%(3500) == 0:
			print "iteration {} / {}".format(iteration, n)
			print "Calculating accuracy on dev set."
			self.test()
			print "train accuracy: a: {}; e: {}.".format((good_a/(good_a+bad_a)), good_e/(good_e+bad_e))
			good_a, bad_a, good_e, bad_e = 1., 1., 1., 1.


        def test(self, train_set=False):

	   good_e,bad_e, good_a, bad_a, good_d, bad_d = 0., 0., 0, 0., 0., 0.

	   n = self.generator.get_dev_size()
	   iteration = 0
	   for i, batch in enumerate(self.generator.generate(is_training=False)):
		dy.renew_cg()

		for j, training_example in enumerate(batch):

			iteration+=1
			sent, (e_true, a_true, d_true), data_sample = training_example


			a_hat, e_hat, d_hat = self._predict(sent, training=False)
			#loss = -dy.log(a_hat[a_true]) + -dy.log(e_hat[e_true])# + -dy.log(d_hat[d_true])

			a_pred, e_pred, d_pred = np.argmax(a_hat.npvalue()),  np.argmax(e_hat.npvalue()),  np.argmax(d_hat.npvalue())

			if iteration%1000==0:
				print "{}/{}".format(iteration, n)
				#print "predicted: {}, {}".format(ABSO[self.I2A[a_pred]], ERG[self.I2E[e_pred]])#, self.D2I[d_pred]
				print "predicted: {}, {}, {}".format(ABSO[self.I2A[a_pred]], ERG[self.I2E[e_pred]], DAT[self.I2D[d_pred]])
				print "true: {}, {}, {}".format(ABSO[self.I2A[a_true]], ERG[self.I2E[e_true]], DAT[self.I2D[d_true]])
				print "success:", a_true==a_pred and e_true==e_pred and d_true==d_pred

				verb_index = sent.index("<verb>")
				print "verb index:", verb_index
				print "verb output:", data_sample['verb_output']
				print "orig sentence:", data_sample['orig_sentence']
				print "sentence as string: ", " ".join(sent)
				
				att = self.attention_fwd.npvalue()
				att_fwd = [round(a,3) for a in att]
				att_bwd = [round(a,3) for a in self.attention_bwd.npvalue()]
				attention_and_words_fwd = [(w,a) for w, a in zip(sent[:verb_index+1],att_fwd)]
				most_attended_fwd = sorted(attention_and_words_fwd, key = lambda pair: -pair[1])
				attention_and_words_bwd = [(w,a) for w, a in zip(sent[verb_index:][::-1],att_bwd)]
				most_attended_bwd = sorted(attention_and_words_bwd, key = lambda pair: -pair[1])
				print "attention forwards:"
				print attention_and_words_fwd
				print "====="
				print most_attended_fwd
				print "================="
				print "attention backwards:"
				print attention_and_words_bwd
				print "====="
				print most_attended_bwd
				print "============================="

			if a_pred==a_true:
				good_a+=1
			else:
				bad_a+=1

			if e_pred==e_true:
				good_e+=1
			else:
				bad_e+=1

			if d_pred==d_true:
				good_d+=1
			else:
				bad_d+=1
	

			
			#if d_pred==d_true:
			#	good_d+=1
			#else:
			#	bad_d+=1


			if iteration > n:

	   			print "accuracy: e: {}; a: {}; d: {}; total: {}".format(good_a/(good_a+bad_a), good_e/(good_e+bad_e),good_d/(good_d+bad_d), (good_a+good_e+good_d)/(good_a+good_e+bad_a+bad_e+good_d+bad_d))
				return

	





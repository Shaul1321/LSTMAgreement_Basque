import dynet as dy
from utils import *
from p1_generator import *
#from RNN import *
from RNN import *

from Encoder import *


if __name__ == '__main__':

	dg = P1Generator(utils.SENTENCES, utils.WORDS, utils.D2I, utils.A2I, utils.E2I)
	in_size, a_out_size, d_out_size, e_out_size = len(WORDS), len(A2I), len(D2I), len(E2I)
	model = dy.Model()

	#encoder = ComplexEncoder(in_size, model, W2I, C2I)
	#encoder = LSTMEncoder(len(C2I), model, C2I)
	#encoder = EmbeddingEncoder(in_size, model, W2I)
	encoder = SubwordEncoder(in_size, model, utils.W2I, utils.S2I, utils.P2I)
	rnn = RNN(in_size, 64, (a_out_size, e_out_size, d_out_size), dg,  utils.I2A, utils.I2E, utils.I2D, utils.I2W, model, encoder)

	rnn.train()





from DataGenerator_batch import *
import random


class P1Generator(DataGenerator):

 def __init__(self, data_dicts, voc, D2I, A2I, E2I):
     DataGenerator.__init__(self, data_dicts, voc, D2I, A2I, E2I)

 
 def create_example(self, data_sample, prev_data, next_data):
        """
        creates a training example - the sentence without the verb, alongisde its agreements

 	returns:
        	sent_without_verb, (ergative_encoded, absolutive_encoded, dative_encoded) 

    	sent_without_verb: the sentence, as a list of string words, with the vebr ommited
	(ergative_encoded, absolutive_encoded, dative_encoded) - encoding of the coorresponding 	agreements.
	
        """

	verb_index = int(data_sample['verb_index'])
	verb_output = data_sample['verb_output']
	dative = [d for d in self.D2I if d in verb_output]
	absolutive = [a for a in self.A2I if a in verb_output]
	ergative = [e for e in self.E2I if e in verb_output]	
	
	assert len(dative)<=1
	assert len(absolutive)<=1
	assert len(ergative)<=1

	ergative_encoded = self.E2I["None"] if len(ergative) == 0 else self.E2I[ergative[0]] 
	absolutive_encoded = self.A2I["None"] if len(absolutive)==0 else self.A2I[absolutive[0]]
	dative_encoded = self.D2I["None"] if len(dative) == 0 else self.D2I[dative[0]]
	sent = data_sample['orig_sentence'].split(" ")
	prev = prev_data['orig_sentence'].split(" ")
	next = next_data['orig_sentence'].split(" ")
	verb = sent[verb_index]

	sent_without_verb = sent[:verb_index]+["<verb>"]+sent[verb_index+1:]
	sent_without_verb = ["<begin>"] + [w for w in sent_without_verb if w!=""] + ["<end>"]

	#return sent_without_verb, (ergative, absolutive, dative)
	return sent_without_verb, (ergative_encoded, absolutive_encoded, dative_encoded) 



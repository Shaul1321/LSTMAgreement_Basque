import csv

"""
<NI_HU>	Agreement (dative, 3sg)
<NI_HK>	Agreement (dative, 3pl)
<NI_GU>	Agreement (dative, 1pl)
<NI_HI>	Agreement (dative, ??)
<NI_NI>	Agreement (dative, 1sg)
<NI_ZU>	Agreement (dative, 2sg)
<NI_ZK>	Agreement (dative, 2pl)

<NR_HK>	Agreement (absolutive, pl3)
<NR_HU>	Agreement (absolutive, sg3)
<NR_HI>	Agreement (absolutive, ??)
<NR_GU>	Agreement (absolutive, 1pl)
<NR_NI>	Agreement (absolutive, 1sg)
<NR_ZU>	Agreement (absolutive, 2sg)
<NR_ZK>	Agreement (absolutive, 2pl)


<NK_HU>	Agreement (ergative, sg3)
<NK_HK>	Agreement (ergative, pl3)
<NK_HI>	Agreement (ergative, ??)
<NK_GU>	Agreement (ergative, 1pl)
<NK_NI>	Agreement (ergative, 1sg)
<NK_ZU>	Agreement (ergative, 2sg)
<NK_ZK>	Agreement (ergative, 2pl)
"""

dative = set(["<NI_HU>", "<NI_HK>","<NI_HI>", "<NI_GU>", "<NI_NI>", "<NI_ZU>", "<NI_ZK>", "None"])
absolutive = set(["<NR_HU>", "<NR_HK>","<NR_HI>", "<NR_GU>", "<NR_NI>", "<NR_ZU>", "<NR_ZK>", "None"])
ergative = set(["<NK_HU>", "<NK_HK>", "<NK_HI>", "<NK_GU>", "<NK_NI>", "<NK_ZU>", "<NK_ZK>", "None"])

all_combs = set()
for d in dative:
	for a in absolutive:
		for e in ergative:
			all_combs.add((d,a,e))

ALL2I = {agreement:index for index, agreement in enumerate(sorted(all_combs))}
D2I = {agreement:index for index, agreement in enumerate(sorted(dative))}
A2I = {agreement:index for index, agreement in enumerate(sorted(absolutive))}
E2I = {agreement:index for index, agreement in enumerate(sorted(ergative))}

I2D = {index:agreement for index, agreement in enumerate(sorted(dative))}
I2A = {index:agreement for index, agreement in enumerate(sorted(absolutive))}
I2E = {index:agreement for index, agreement in enumerate(sorted(ergative))}
I2ALL = {index:agreement for index, agreement in enumerate(sorted(all_combs))}

SENTENCES = []
WORDS = set()
labels = ["orig_sentence", "output", "verb_output", "verb_index"]

dative_count = 0

# these words are associated with bot-generated sentences
bad_words = ["familiak", "pertsonak", "pertsona", "inaktiboetatik", "inaktiboak", "erretiraturik", "etxetan", "etxebizitza","etxek", "udalerri", "metropolitanoan", "diagrama", "etxeak", "apartamentuak", "eskola", "kilometrotako", "kilometroko", "biztanleriak", "komertzialetatik", "publikoetatik","probintzian", "ospitale", "konderrian", "kokatua", "kilometro", "osasunekipamendu", "zentsuaren", "biztanle", "publikoak", "komertzioetatik", "administratiboki"]

count = 0.
total = 0.

from collections import Counter
fc = Counter()


with open("data_present_all.csv", "r") as f:
	reader = csv.reader(f)
	for row in reader:
		#print len(row)
		#print row
		total+=1
		orig_sentence, output, verb_output, verb_index = row
		bad_words_occures = len([w for w in bad_words if w in orig_sentence])>0

		if bad_words_occures: 
			count+=1
			continue
		sentence_words = orig_sentence.split(" ")
		
		sent_dictionary = {}
		for i, val in enumerate(row):
			sent_dictionary[labels[i]] = val

		SENTENCES.append(sent_dictionary)

		for w in sentence_words:
			#WORDS.add(w)
			fc.update([w])
		verb_output =  SENTENCES[-1]["verb_output"]

		existing_dative = [d for d in dative if d in verb_output] 
		existing_absolutive = [d for d in absolutive if d in verb_output] 
		existing_ergative = [d for d in ergative if d in verb_output] 
		if len(existing_dative)!=0:
			dative_count+=1

voc_size = 15000
WORDS = set([w for w,c in fc.most_common(voc_size)])

WORDS.add("<unk>")
WORDS.add("<verb>")
WORDS.add("<begin>")
WORDS.add("<end>")
LETTERS = set([char for w in WORDS for char in w])
LETTERS.add("<unk>")

W2I = {w:i for i, w in enumerate(sorted(WORDS))}
I2W = {i:w for i, w in enumerate(sorted(WORDS))}
C2I = {c:i for i, c in enumerate(sorted(LETTERS))}

prefixes, suffixes = set(), set()


for word in WORDS:
 		i = min(3, len(word))
		for j in range(1,i+1):
 			prefixes.add(word[:j])
			suffixes.add(word[-j:])

prefixes.add("<unk>")
suffixes.add("<unk>")

P2I = {l:i for i,l in enumerate(sorted(prefixes))   }
S2I = {l:i for i,l in enumerate(sorted(suffixes))   }



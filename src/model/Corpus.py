import numpy as np

import json
from ..structure.seg_sentence import SegSentence,Sentence,PosSentence

from collections import defaultdict,OrderedDict
class Corpus(object):

    UNK = '<UNK>'
    START = '<s>'
    STOP = '</s>'

    @staticmethod
    def vocab_init(fname):

        sentences = Sentence.load_sentence_file(fname,'joint')

        unigrams_freq = defaultdict(int)
        bigrams_freq = defaultdict(int)
        label_freq = defaultdict(int)
        tag_freq = defaultdict(int)
        for i,sentence in enumerate(sentences):
            pre_unigram = Corpus.START
            for (unigram,tag,label) in sentence:
                unigrams_freq[unigram] += 1
                bigrams_freq [pre_unigram+unigram] += 1
                bigrams_freq[unigram + pre_unigram] +=1
                pre_unigram = unigram
                label_freq[label] += 1
                if tag != 'NotStart':
                    tag_freq[tag] += 1
            bigrams_freq[pre_unigram+Corpus.STOP] += 1
            bigrams_freq[Corpus.STOP+pre_unigram] += 1

        bigrams = [Corpus.UNK] + sorted(bigrams_freq)
        unigrams = [
            Corpus.START,
            Corpus.STOP,
            Corpus.UNK,
        ] + sorted(unigrams_freq)

        bdict = OrderedDict((b,i) for (i,b) in enumerate(bigrams))
        udict = OrderedDict((u,i) for (i,u) in enumerate(unigrams))

        labels = sorted(label_freq)
        ldict = OrderedDict((l,i) for (i,l) in enumerate(labels))

        tags = sorted(tag_freq)
        tdict = OrderedDict((t,i) for (i,t) in enumerate(tags))

        return {
            'origin_sentences':sentences,
            'bdict':bdict,
            'bigrams_freq':bigrams_freq,
            'unigrams_freq':unigrams_freq,
            'udict':udict,
            'ldict':ldict,
            'tdict':tdict,
            'bigrams':bigrams,
            'unigrams':unigrams,
        }

    def __init__(self,vocabfile):
        if vocabfile is not None:
            data = Corpus.vocab_init(vocabfile)

            self.origin_sentences = data['origin_sentences']
            self.bdict = data['bdict']
            self.udict = data['udict']
            self.tdict = data['tdict']
            self.bigrams_freq = data['bigrams_freq']
            self.unigrams_freq = data['unigrams_freq']
            self.bigrams = data['bigrams']
            self.unigrams = data['unigrams']
            self.ldict = data['ldict']

            self.bigrams_freq_list = []
            for word in self.bdict.keys():
                if word in self.bigrams_freq:
                    self.bigrams_freq_list.append(self.bigrams_freq[word])
                else:
                    self.bigrams_freq_list.append(0)

            self.unigrams_freq_list = []
            for word in self.udict.keys():
                if word in self.unigrams_freq:
                    self.unigrams_freq_list.append(self.unigrams_freq[word])
                else:
                    self.unigrams_freq_list.append(0)


    @staticmethod
    def from_dict(data):
        new = Corpus(None)
        new.bdict = data['bdict']
        new.udict = data['udict']
        new.ldict = data['ldict']
        new.tdict = data['tdict']
        new.bigrams = data['bigrams']
        new.unigrams = data['unigrams']
        new.bigrams_freq_list = data['bigrams_freq_list']
        new.unigrams_freq_list = data['unigrams_freq_list']

        return new

    def as_dict(self):
        return {
        'bdict':self.bdict,
        'udict':self.udict,
        'ldict':self.ldict,
        'tdict':self.tdict,
        'bigrams':self.bigrams,
        'unigrams':self.unigrams,
        'bigrams_freq_list':self.bigrams_freq_list,
        'unigrams_freq_list':self.unigrams_freq_list,

    }

    def save_json(self,filename):
        with open (filename,'w') as fh:
            json.dump(self.as_dict(),fh,encoding='utf-8')

    @staticmethod
    def load_json(filename):
        with open (filename) as fh:
            data = json.load(fh,object_pairs_hook=OrderedDict,encoding='utf-8')
        return Corpus.from_dict(data)

    def total_bigrams(self):
        return len(self.bdict)

    def total_unigrams(self):
        return len(self.udict)

    def total_segs(self):
        return len(self.ldict)

    def total_tags(self):
        return 1 + len(self.tdict)

    def seg_span_num(self):
        return 4

    def tag_span_num(self):
        return 1

    def s_action_index(self,action):
        if action == 'NA':
            return 0
        elif action == 'AP':
            return 1

    def s_action(self,action_index):
        if action_index == 0:
            return 'NA'
        else:
            return 'AP'

    def t_action_index(self,action):
        if action == 'none':
            return 0
        else:
            tag_index = self.tdict.get(action[4:])
            if tag_index is not  None:
                return tag_index
            else:
                return 0

    def t_action(self,action_index):
        if action_index == 0 :
            return 'none'
        else:
            return 'tag-' + self.tdict.keys()[action_index-1]

    def sentence_sequence(self,sentence):

        fwd_bi = sentence.fwd_bigram()
        uni = sentence.unigram()
        fwd_bigram_id = [
            self.bdict[b.decode('utf-8')]
            if b.decode('utf-8') in self.bdict else self.bdict[Corpus.UNK]
            for b in fwd_bi
        ]
        unigram_id = [
            self.udict[u.decode('utf-8')]
            if u.decode('utf-8') in self.udict else self.udict[Corpus.UNK]
            for u in uni
         ]
        fwd_bi = np.array(fwd_bigram_id).astype('int32')
        u = np.array(unigram_id).astype('int32')

        return fwd_bi,u

    def gold_data(self,sentence):
        sentence = Sentence(sentence)
        fwd_bigrams,unigrams = self.sentence_sequence(sentence)

        #features = Segmenter.training_data(seg_sentence)

        return {
            'sentence':sentence,
            'fwd_bigrams': fwd_bigrams,
            'unigrams': unigrams,
        }
        
    def gold_data_from_file(self,fname):

        sentences = Sentence.load_sentence_file(fname,'joint')
        result = []
        for sentence in sentences:
            sentence_data = self.gold_data(sentence)
            result.append(sentence_data)
        return result

class PosCorpus(object):
    UNK= '<UNK>'
    START = '<S>'
    STOP  = '</S>'

    @staticmethod
    def vocab_init(fname):

        sentences = Sentence.load_sentence_file(fname,'pos-tagging')

        words_freq = defaultdict(int)
        tag_freq = defaultdict(int)

        for s in sentences:
            sentence = PosSentence.ct2wt(s)
            for (w,t) in sentence:
                words_freq[w] += 1
                tag_freq[t] += 1

        words =  [PosCorpus.UNK] + sorted(words_freq)
        wdict = OrderedDict((w,i) for (i,w) in enumerate(words))

        tag_freq.pop('<S>')
        tag_freq.pop('</S>')
        tags = sorted(tag_freq)
        tdict = OrderedDict((t,i) for (i,t) in enumerate(tags))

        return {
            'origin_sentences':sentences,
            'wdict':wdict,
            'words':words,
            'words_freq':words_freq,
            'tdict':tdict,
            'tags':tags,
            'tag_freq':tag_freq
        }

    def __init__(self,vocabfile):
        if vocabfile is not None:
            data = PosCorpus.vocab_init(vocabfile)

            self.origin_sentences = data['origin_sentences']
            self.wdict = data['wdict']
            self.tdict = data['tdict']
            self.words = data['words']
            self.tags = data['tags']
            self.words_freq = data['words_freq']

            self.words_freq_list = []
            for w in self.wdict.keys():
                if w in self.words_freq:
                    self.words_freq_list.append(self.words_freq[w])
                else:
                    self.words_freq_list.append(0)

    @staticmethod
    def from_dict(data):
        new = PosCorpus(None)
        new.wdict = data['wdict']
        new.tdict = data['tdict']
        new.words = data['words']
        new.tags = data['tags']
        new.words_freq_list = data['words_freq_list']

        return new

    def as_dict(self):
        return{
            'wdict':self.wdict,
            'tdict':self.tdict,
            'words':self.words,
            'tags':self.tags,
            'words_freq_list':self.words_freq_list
        }

    def save_json(self,filename):
        with open(filename,'w') as fh:
            json.dump(self.as_dict(),fh,encoding='utf-8')

    @staticmethod
    def load_json(filename):
        with open (filename) as fh:
            data = json.load(fh,object_pairs_hook=OrderedDict,encoding='utf-8'
                             )
        return PosCorpus.from_dict(data)

    def total_words(self):
        return len(self.wdict)

    def total_tags(self):
        return len(self.tdict)

    def total_span_nums(self):
        return 1

    def tag_index(self,action):
        return self.tdict.get(action)

    def tag_action(self,index):
        return self.tdict.keys()[index]

    def sentence_sequence(self,pos_sentence):
        words = pos_sentence.words
        words_id = [
            self.wdict[w.decode('utf-8')]
            if w.decode('utf-8') in self.wdict else self.wdict[PosCorpus.UNK]
            for w in words
        ]

        fwd_w = np.array(words_id).astype('int32')

        return fwd_w

    def gold_data(self,sentence):
        pos_sentence = PosSentence(sentence)
        words = self.sentence_sequence(pos_sentence)

        return {
            'posSentence':pos_sentence,
            'words':words
        }

    def gold_data_from_file(self,fname):

        sentences = Sentence.load_sentence_file(fname,'pos-tagging')
        result = []
        for sentence in sentences:
            sentence_data = self.gold_data(sentence)
            result.append(sentence_data)
        return result
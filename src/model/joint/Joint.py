from __future__ import print_function

import time
import numpy as np
import dynet
import os
import sys


from ...structure.seg_sentence import Sentence
from ...structure.seg_sentence import SegSentence
from ...structure.seg_sentence import Accuracy
from ...structure.seg_sentence import FScore


class Joint(object):
    def __init__(self,n0,sentence):
        self.stack = []
        self.chars = []
        self.n0 = n0
        self.i = 1
        self.labels = []
        self.gold_sentence = sentence.sentence
        self.curword = None
        self.words = []
        self.o = []

    def s_features(self):
        lefts = []
        rights = []

        lefts.append(1)
        if len(self.stack) < 1:
            rights.append(0)
        else:
            s0_left = self.stack[-1][0]+1
            rights.append(s0_left-1)

        if len(self.stack) < 1:
            lefts.append(1)
            rights.append(0)
        else:
            s0_left = self.stack[-1][0] + 1
            lefts.append(s0_left)
            s0_right = self.stack[-1][1]+1
            rights.append(s0_right)

        lefts.append(self.i)
        rights.append(self.i)

        lefts.append(self.i+1)
        rights.append(self.n0)

        return tuple(lefts),tuple(rights)

    def can_append(self):
        return not (self.i == 1)

    def no_append(self):
        assert len(self.stack) > 0
        if len(self.stack) > 0:
            self.curword = self.stack[-1]
            self.stack.pop()

        self.stack.append((self.i,self.i))
        self.labels.append('NA')
        self.i += 1

    def wrap_result(self):
        return SegSentence([('S','NA')]+zip(self.chars,self.labels)+[('/S','NA')])


    def s_oracle(self):
        return self.gold_sentence[self.i][2]

    def take_action(self,action):
        if action == 'NA':
            self.no_append()
        elif action == 'AP':
            self.append()
        elif action.startswith('tag-'):
            self.tag(action[4:])

    def append(self):
        left,right = self.stack.pop()
        self.stack.append((left,right+1))
        self.i +=1
        self.labels.append('AP')

    def tag(self,tag_name):
        self.words.append((self.curword,tag_name))

    def t_oracle(self,sentence):
        if self.curword == None:
            return None
        else:
            left,right = self.curword
            tag_name = sentence.pos_sentence.span_tags(left,right)
            if tag_name == None:
                return 'none'
            else:
                return 'tag-' + tag_name

    def t_features(self):
        lefts = []
        rights = []
        if self.curword is None:
            for c in self.chars:
                print (c)
        left,right = self.curword
        lefts.append(left)
        rights.append(right)

        return tuple(lefts),tuple(rights)

    @staticmethod
    def training_data(fm,sentence):
        s_features = []
        t_features = []
        n0 = sentence.n0
        state = Joint(n0,sentence)

        for step in range(state.n0):
            if not state.can_append():
                state.no_append()
                continue
            else:
                action = state.s_oracle()
                features = state.s_features()
                state.take_action(action)
                s_features.append((features,fm.s_action_index(action)))

            if action == 'NA':
                action = state.t_oracle(sentence)
                features = state.t_features()
                state.take_action(action)
                t_features.append((features,fm.t_action_index(action)))

        return (s_features,t_features)

    def at_begin(self):
        return self.i == 1

    def begin(self):
        self.labels.append('NA')
        self.stack.append((1,1))
        self.i += 1

    @staticmethod
    def exploration(data,fm,network,alpha=1.0,beta=0):
        dynet.renew_cg()
        network.prep_params()

        s_data = {}
        t_data ={}

        sentence = data['sentence']

        n0 = sentence.n0
        state = Joint(n0,sentence)
        state.chars = [c for (c,l) in sentence.seg_sentence[1:-1]]
        # for c in state.chars :
        #     print (c)

        fwd_bigrams = data['fwd_bigrams']
        unigrams = data['unigrams']
        fwd,back = network.evaluate_recurrent(fwd_bigrams,unigrams,test=True)

        for step in range(state.n0):
            s_features =  state.s_features()

            if state.at_begin():
                state.begin()
                continue
            if not state.can_append():
                action = 'NA'
                correct_action = 'NA'
            else:
                correct_action = state.s_oracle()

                r=  np.random.random()
                if r < beta:
                    action = correct_action
                    state.o.append(1)
                else:
                    state.o.append(0)
                    left,right = s_features
                    scores = network.evaluate_segs(
                        fwd,
                        back,
                        left,
                        right,
                        test=True
                    ).npvalue()

                    # exp = np.exp(scores * alpha)
                    # softmax = exp / exp.sum()
                    # r = np.random.random()
                    #
                    # if r <= softmax[0]:
                    #     action = 'AP'
                    # else:
                    #     action = 'NA'
                    action_index = np.argmax(scores)
                    action = fm.s_action(action_index)

            s_data[s_features] = fm.s_action_index(correct_action)
            state.take_action(action)

            if action == 'NA' :
                t_features = state.t_features()
                correct_action = state.t_oracle(sentence)

                t_data[t_features] = fm.t_action_index(correct_action)

                r = np.random.random()
                if r < beta:
                    action = correct_action
                else:
                    left,right = t_features
                    scores = network.evaluate_tags(
                        fwd,
                        back,
                        left,
                        right,
                        test=True
                    ).npvalue()

                    t_action_index= np.argmax(scores)
                    action = fm.t_action(t_action_index)

                state.take_action(action)

        predicted = state.wrap_result()
        accuracy = predicted.compare(sentence)

        #d_pred = Joint.joint(sentence,fm,network)
        #d_accuracy = d_pred.compare(sentence)
        #print (d_accuracy)

#assert (d_accuracy == accuracy)

        example = {
            'fwd_bigrams':fwd_bigrams,
            'unigrams':unigrams,
            's_data':s_data,
            't_data':t_data
        }

        return example,accuracy

    def finished(self):
        return  (self.i == self)

    # def append(self):
    #     left,right = self.stack.pop()
    #     self.stack.append((left,right+1))
    #     self.i += 1
    #     self.labels.append('AP')


    @staticmethod
    def joint(sentence, fm, network):

        dynet.renew_cg()
        network.prep_params()

        n0 = sentence.n0
        state = Joint(n0, sentence)
        state.chars = [c for (c,l) in sentence.seg_sentence[1:-1]]


        b, u = fm.sentence_sequence(sentence)

        fwd, back = network.evaluate_recurrent(b, u, test=True)

        for step in range(state.n0):
            if not state.can_append():
                state.begin()
                continue
            else:
                left, right = state.s_features()
                scores = network.evaluate_segs(
                    fwd,
                    back,
                    left,
                    right,
                    test=True
                ).npvalue()
                action_index = np.argmax(scores)
                action = fm.s_action(action_index)

            state.take_action(action)

            if action == 'NA':
                left, right = state.t_features()
                scores = network.evaluate_tags(
                    fwd,
                    back,
                    left,
                    right,
                    test=True
                ).npvalue()

                t_action_index = np.argmax(scores)
                action = fm.t_action(t_action_index)

                state.take_action(action)

            # if not state.finished():
            #     raise RuntimeError('Bad ending state')

        predicted = state.wrap_result()
        return predicted

    @staticmethod
    def evaluate_corpus(sentences,fm,network):
        fscore = FScore()
        for sentence in sentences:
            sentence = Sentence(sentence)
            predicted = Joint.joint(sentence,fm,network)
            local_fscore = predicted.compare(sentence)
            fscore += local_fscore

        return fscore

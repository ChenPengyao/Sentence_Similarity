#!/usr/bin/env python3
# coding: utf-8
# File: Similarity_output.py
# Author: @CPY
# Date: 18-11-30
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import distance
from sklearn.feature_extraction.text import CountVectorizer
from scipy.linalg import norm
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim
import jieba
jieba.load_userdict("c:\\Users\\pactera\\Desktop\\whitelists.txt")
def stopwordslist():
    stopwords = [line.strip() for line in open('c:\\Users\\pactera\\Desktop\\中文停用词表.txt',encoding='UTF-8').readlines()]
    return stopwords
stopwords = stopwordslist()
import jieba.posseg as pseg
from collections import Counter
import difflib
import math
import scipy as sp
import Levenshtein
from numpy import dot
from numpy.linalg import norm
import scipy
from sklearn.metrics import precision_recall_fscore_support as score
cosine = lambda a, b: dot(a, b)/(norm(a)*norm(b))
def sigmoid(x):
	return 1.0 / (1.0 + np.exp(-x))


#Jaccard Index
def jaccard_similarity(s1, s2):
	if (s1 == s2):
		return 1.0
	def add_space(s):
		return ' '.join(list(s))

	# 将字中间加入空格
	s1, s2 = add_space(s1), add_space(s2)
	# 转化为TF矩阵
	cv = CountVectorizer(tokenizer=lambda s: s.split())
	corpus = [s1, s2]
	vectors = cv.fit_transform(corpus).toarray()
	# 求交集
	numerator = np.sum(np.min(vectors, axis=0))
	# 求并集
	denominator = np.sum(np.max(vectors, axis=0))
	#计算杰卡德系数
	return 1.0 * numerator / denominator

#TF-IDF
def tfidf_similarity(s1, s2):
	if(s1 == s2):
		return 1.0
	def add_space(s):
		return ' '.join(list(s))
	# 将字中间加入空格
	s1, s2 = add_space(s1), add_space(s2)
	# 转化为TF矩阵
	cv = TfidfVectorizer(tokenizer=lambda s: s.split())
	corpus = [s1, s2]
	vectors = cv.fit_transform(corpus).toarray()
	#print(vectors)
	# 计算TFIDF系数
	return np.dot(vectors[0], vectors[1]) / (norm(vectors[0]) * norm(vectors[1]))

#diff
def diff_dist(s1,s2):
	if s1 == s2:
		return 1.0
	diff_result = difflib.SequenceMatcher(None, s1, s2).ratio()
	return diff_result

#Word2Vec

model_file = 'c:\\Users\\pactera\\Desktop\\self_embedding.bin'
model = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=True)

def compute_ngrams(word, min_n, max_n):
	#BOW, EOW = ('<', '>')  # Used by FastText to attach to all words as prefix and suffix
	extended_word = word
	ngrams = []
	for ngram_length in range(min_n, min(len(extended_word), max_n) + 1):
		for i in range(0, len(extended_word) - ngram_length + 1):
			ngrams.append(extended_word[i:i + ngram_length])
	return list(set(ngrams))

def random_vector(word,size=120):
	random_state = np.random.RandomState(seed=(hash(word) % (2 ** 32 - 1)))
	return random_state.uniform(low=-10.0, high=10.0, size=(size,))

def wordVec(word,wv_from_text,min_n = 1, max_n = 3):
	'''
	ngrams_single/ngrams_more,主要是为了当出现oov的情况下,最好先不考虑单字词向量
	'''
	# 确认词向量维度
	word_size = wv_from_text.wv.syn0[0].shape[0]
	# 计算word的ngrams词组
	ngrams = compute_ngrams(word, min_n=min_n, max_n=max_n)
	#如果在词典之中，直接返回词向量
	if word in wv_from_text.wv.vocab.keys():
		return wv_from_text[word]
	else:
		return random_vector(word)
		# 不在词典的情况下
		word_vec = np.zeros(word_size, dtype=np.float32)
		ngrams_found = 0
		ngrams_single = [ng for ng in ngrams if len(ng) == 1]
		ngrams_more = [ng for ng in ngrams if len(ng) > 1]
		# 先只接受2个单词长度以上的词向量
		for ngram in ngrams_more:
			if ngram in wv_from_text.wv.vocab.keys():
				word_vec += wv_from_text[ngram]
				ngrams_found += 1
				#print(ngram)
		# 如果，没有匹配到，那么最后是考虑单个词向量
		if ngrams_found == 0:
			for ngram in ngrams_single:
				word_vec += wv_from_text[ngram]
				ngrams_found += 1
		if word_vec.any():
			return word_vec / max(1, ngrams_found)
		else:
			raise KeyError('all ngrams for word %s absent from model' % word)

def sentence_vector(s):
	words = jieba.lcut(s)
	v = np.zeros(120)
	for word in words:
		if word not in stopwords:
			v += wordVec(word,model,min_n = 1, max_n = 3)
	v /= len(words)
	return v

def vector_similarity(s1, s2):
	if(s1 == s2):
		return 1.0
	v1=sentence_vector(s1)
	v2=sentence_vector(s2)
	#print(v1, v2)
	return np.dot(v1, v2) / (norm(v1) * norm(v2))

def Euclidean(s1, s2):
	vec1 = sentence_vector(s1)
	vec2 = sentence_vector(s2)
	npvec1, npvec2 = np.array(vec1), np.array(vec2)
	return math.sqrt(((npvec1-npvec2)**2).sum())
# euclidean,欧式距离算法，传入参数为两个向量，返回值为欧式距离

def Manhattan(s1, s2):
	vec1 = sentence_vector(s1)
	vec2 = sentence_vector(s2)
	npvec1, npvec2 = np.array(vec1), np.array(vec2)
	return np.abs(npvec1-npvec2).sum()

def Chebyshev(s1, s2):
	vec1 = sentence_vector(s1)
	vec2 = sentence_vector(s2)
	npvec1, npvec2 = np.array(vec1), np.array(vec2)
	return max(np.abs(npvec1-npvec2))


#计算特征和类的平均值
def calcMean(x,y):
	sum_x = sum(x)
	sum_y = sum(y)
	n = len(x)
	x_mean = float(sum_x+0.0)/n
	y_mean = float(sum_y+0.0)/n
	return x_mean,y_mean

#计算Pearson系数
def calcPearson(s1,s2):
	x = sentence_vector(s1)
	y = sentence_vector(s2)
	x_mean,y_mean = calcMean(x,y)	#计算x,y向量平均值
	n = len(x)
	sumTop = 0.0
	x_pow = 0.0
	y_pow = 0.0
	for i in range(n):
		sumTop += (x[i]-x_mean)*(y[i]-y_mean)
	for i in range(n):
		x_pow += math.pow(x[i]-x_mean,2)
	for i in range(n):
		y_pow += math.pow(y[i]-y_mean,2)
	sumBottom = math.sqrt(x_pow*y_pow)
	p = sumTop/sumBottom
	return p


## encoding=utf-8
## CPY

##软投票的集成方法
def soft_voting(df):
	'''df有两列，一列是prelabel，一列是weighted'''
	res = Counter(df['prelabel'])
	#print(res['0'])
	means = pd.DataFrame(df['weighted'].groupby([df['prelabel']]).mean())
	means['prelabel']=means.index
	means['freq']=res.values()
	List1 = means['weighted']
	List2 = means['freq']
	means['res'] = np.multiply(np.array(List1),np.array(List2)).tolist()
	res=means['prelabel'].loc[means['res']==means['res'].max()][0]
	return res
  
  
#相似度输出函数
def func_max_similarity(test,train):
	"""
	作为pd.apply的调用函数对test做相似度输出，
	输入一个series，输出的是经过选择后的最佳的预测结果。
	:param x:
	:return:
	"""
	train_copy=train.copy()

	'''文字计算'''
	train_copy['jaccard_S'] = train_copy['Question'].apply(jaccard_similarity, **{'s2': test[0]})
	#train_copy['tfidf_S'] = train_copy['Question'].apply(tfidf_similarity, **{'s2': test[0]})
	#train_copy['diff_S'] = train_copy['Question'].apply(diff_dist, **{'s2': test[0]})
	'''向量计算'''
	#train_copy['vector_S'] = train_copy['Question'].apply(vector_similarity, **{'s2': test[0]})
	#train_copy['Eucl_D'] = train_copy['Question'].apply(Euclidean, **{'s2': test[0]})
	#train_copy['Man_D'] = train_copy['Question'].apply(Manhattan, **{'s2': test[0]})
	#train_copy['Cheb_D'] = train_copy['Question'].apply(Chebyshev, **{'s2': test[0]})
	#train_copy['calc_S'] = train_copy['Question'].apply(calcPearson, **{'s2': test[0]})


	label_a=[]
	train_copy=pd.DataFrame(train_copy)
	jm=train_copy['jaccard_S'].max()
	#tim=train_copy['tfidf_S'].max()
	#diffm = train_copy['diff_S'].max()
	#vecm=train_copy['vector_S'].max()
	#euclm=train_copy['Eucl_D'].min()
	#manm=train_copy['Man_D'].min()
	#chebm = train_copy['Cheb_D'].min()
	#calcm = train_copy['calc_S'].max()

	label_a.append(train_copy['Label'].loc[train_copy['jaccard_S']==jm])
	#label_a.append(train_copy['Label'].loc[train_copy['tfidf_S']==tim])
	#label_a.append(train_copy['Label'].loc[train_copy['diff_S'] == diffm])
	#label_a.append(train_copy['Label'].loc[train_copy['vector_S']==vecm])
	#label_a.append(train_copy['Label'].loc[train_copy['Eucl_D'] == euclm])
	#label_a.append(train_copy['Label'].loc[train_copy['Man_D'] == manm])
	#label_a.append(train_copy['Label'].loc[train_copy['Cheb_D'] == chebm])
	#label_a.append(train_copy['Label'].loc[train_copy['calc_S'] == calcm])
	#print(label_a)
	#print(train_copy['label'].loc[train_copy['jaccard_S']==0.2])


	label_b=[]
	label_b.append(Counter(label_a[0]).most_common(1)[0][0])
	#label_b.append(Counter(label_a[1]).most_common(1)[0][0])
	#label_b.append(Counter(label_a[2]).most_common(1)[0][0])
	#label_b.append(Counter(label_a[3]).most_common(1)[0][0])
	#label_b.append(Counter(label_a[4]).most_common(1)[0][0])
	#label_b.append(Counter(label_a[5]).most_common(1)[0][0])
	#label_b.append(Counter(label_a[6]).most_common(1)[0][0])
	#label_b.append(Counter(label_a[7]).most_common(1)[0][0])
	#print(Counter(label_a[3]).most_common(1))
	#print(Counter(label_b).most_common(1)[0][0])

	#hard voting
	test[2]=Counter(label_b).most_common(1)[0][0]

	#soft voting
	data = {"prelabel": label_b,"weighted": [0.8611]}
	#0.8611, 0.8472, 0.8472,0.8472,0.8333, 0.8472, 0.8333, 0.8194,0.8333
	ddf = pd.DataFrame(data)
	test[3]=soft_voting(ddf)
	#print("Finished once!")
	del jm
	#del tim
	#del diffm
	#del vecm
	#del euclm
	#del manm
	#del chebm
	#del calcm
	#del train_copy
	#del label_a
	#del label_b
	return test

def Accuracy_Func(df):

	correct1 = df[df['Label'] == df['pred_label_hard']]
	accuracy1 = round(len(correct1) / len(df) * 100, 2)
	correct2 = df[df['Label'] == df['pred_label_soft']]
	accuracy2 = round(len(correct2) / len(df) * 100, 2)
	print("The hard accuracy is " + str(accuracy1) + '%!')
	#print("The soft accuracy is " + str(accuracy2) + '%!')
	print(len(df))
	del accuracy1,accuracy2,correct2,correct1

	predicted = np.asarray(df['pred_label_hard'])
	y_test = np.asarray(df['Label'])
	precision, recall, fscore, support = score(y_test, predicted)
	print('precision: ' + str(round(precision.mean(), 4)))
	print('recall: ' + str(round(recall.mean(), 4)))
	#print('fscore: {}'.format(fscore))
	#print('support: {}'.format(support))



def sentence_classification(text,train):
	train_maxtrix=train.as_matrix(columns=None)
	train=pd.DataFrame(train_maxtrix,columns=['Question','Label'])
	df = pd.DataFrame(columns=['Question'])
	df.loc[0]=text
	df['label']='0'
	df['pred_label']='NaN'
	df.apply(func_max_similarity, axis=1,**{'train': train})
	key=df['pred_label'][0]
	#print(key)
	return key

#获取key
def get_key (dict, value):
	return [k for k, v in dict.items() if v == value]

#Output
def output(text,df):
	a = str(sentence_classification(text, df))
	b = get_key(dict_QA, a)
	print("Q: " + text + '\n' + "A: " + b[0]+'\n')

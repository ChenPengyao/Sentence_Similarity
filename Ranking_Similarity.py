#encoding=utf-8

from Various_Similarity import *


def pre_data_func_top(test,train,top_num,num):
	train_copy=train.copy()

	train_copy['tfidf_S'] = train_copy['Question'].apply(tfidf_similarity, **{'s2': test[0]})
	train_copy['Cheb_D'] = train_copy['Question'].apply(Chebyshev, **{'s2': test[0]})
	tim = sorted(train_copy['tfidf_S'])
	chebm = sorted(train_copy['Cheb_D'])
	train_copy_2=pd.DataFrame(train_copy.iloc[0:1])
	str_1=[]
	for i in range(num):
		str_1.append(np.asarray(train_copy['Question'].loc[train_copy['tfidf_S'] == tim[-(i + 1)]])[0])
		str_1.append(np.asarray(train_copy['Question'].loc[train_copy['Cheb_D'] == chebm[i]])[0])
	for j in range(len(str_1)):
		train_copy_2=pd.concat([train_copy_2,train_copy[train_copy['Question'].isin([str(str_1[j])])]])
	train_copy_2=train_copy_2.drop_duplicates()

	'''文字计算'''
	train_copy_2['jaccard_S'] = train_copy_2['Question'].apply(jaccard_similarity, **{'s2': test[0]})

	train_copy_2['diff_S'] = train_copy_2['Question'].apply(diff_dist, **{'s2': test[0]})
	'''向量计算'''
	train_copy_2['vector_S'] = train_copy_2['Question'].apply(vector_similarity, **{'s2':test[0]})
	train_copy_2['Eucl_D'] = train_copy_2['Question'].apply(Euclidean, **{'s2':test[0]})
	train_copy_2['Man_D'] = train_copy_2['Question'].apply(Manhattan, **{'s2': test[0]})

	train_copy_2['calc_S'] = train_copy_2['Question'].apply(calcPearson, **{'s2':test[0]})
	train_copy_2=pd.DataFrame(train_copy_2)

	jm = sorted(train_copy_2['jaccard_S'])
	diffm = sorted(train_copy_2['diff_S'])
	vecm = sorted(train_copy_2['vector_S'])
	euclm = sorted(train_copy_2['Eucl_D'])
	manm = sorted(train_copy_2['Man_D'])
	calcm = sorted(train_copy_2['calc_S'])

	for i in range(top_num):
		label_a=[]
		label_a.append(train_copy_2['Label'].loc[train_copy_2['jaccard_S']==jm[-(i+1)]])
		label_a.append(train_copy_2['Label'].loc[train_copy_2['tfidf_S']==tim[-(i+1)]])
		label_a.append(train_copy_2['Label'].loc[train_copy_2['diff_S'] == diffm[-(i+1)]])
		label_a.append(train_copy_2['Label'].loc[train_copy_2['vector_S']==vecm[-(i+1)]])
		label_a.append(train_copy_2['Label'].loc[train_copy_2['Eucl_D'] == euclm[i]])
		label_a.append(train_copy_2['Label'].loc[train_copy_2['Man_D'] == manm[i]])
		label_a.append(train_copy_2['Label'].loc[train_copy_2['Cheb_D'] == chebm[i]])
		label_a.append(train_copy_2['Label'].loc[train_copy_2['calc_S'] == calcm[-(i+1)]])
		label_b = []
		label_b.append(Counter(label_a[0]).most_common(1)[0][0])
		label_b.append(Counter(label_a[1]).most_common(1)[0][0])
		label_b.append(Counter(label_a[2]).most_common(1)[0][0])
		label_b.append(Counter(label_a[3]).most_common(1)[0][0])
		label_b.append(Counter(label_a[4]).most_common(1)[0][0])
		label_b.append(Counter(label_a[5]).most_common(1)[0][0])
		label_b.append(Counter(label_a[6]).most_common(1)[0][0])
		label_b.append(Counter(label_a[7]).most_common(1)[0][0])

		test[(2+i)] = Counter(label_b).most_common(1)[0][0]

	del jm
	del tim
	del diffm
	del vecm
	del euclm
	del manm
	del chebm
	del calcm
	del train_copy
	del str_1
	del train_copy_2
	print("Finished once!")
	return test

def Accuracy_Func_1(df):
	for i in range(df.shape[1]-2):
		str_1=str('Top'+str(i+1))
		correct1 = df[df['Label'] == df[str_1]]
		accuracy1 = round(len(correct1) / len(df) * 100, 2)
		predicted = np.asarray(df[str_1])
		y_test = np.asarray(df['Label'])

		precision, recall, fscore, support = score(y_test, predicted)
		print(str_1+'  Accuracy: '+str(accuracy1))

		print(str_1+'  precision: '+str(round(precision.mean(),4)))
		print(str_1+'  recall: '+str(round(recall.mean(),4)))
		#print('fscore: {}'.format(fscore))
		#print('support: {}'.format(support))

if __name__ == '__main__':
	print("Strated!")
	df, dict_QA = read_corpus_QA1v1(r"E:\Pactera\Project\Rob\data\corpus\图谱语料\1w的QA.xlsx")
	trainS = df.iloc[range(0, len(df), 2), :]
	testS = df.iloc[range(1, len(df), 2), :]

	trainS = trainS.reset_index(drop=True)
	testS = testS.reset_index(drop=True)

	for i in range(len(trainS)):
		if (trainS['Question'][i] == ''):
			trainS = trainS.drop(i)
		elif (trainS['Question'][i] == 'nan'):
			trainS = trainS.drop(i)
	for i in range(len(testS)):
		if (testS['Question'][i] == ''):
			testS = testS.drop(i)
		elif (testS['Question'][i] == 'nan'):
			testS = testS.drop(i)

	trainS = trainS.reset_index(drop=True)
	testS = testS.reset_index(drop=True)

	for i in range(10):
		testS['Top' + str(i + 1)] = 'NaN'

	#print(testS.iloc[0:10].apply(pre_data_func_top,axis=1,**{'train': trainS},**{'top_num':10},**{'num':50}))

	testS=testS.apply(pre_data_func_top,axis=1,**{'train': trainS},**{'top_num':10},**{'num':50})
	testS.to_excel(r'E:\Pactera\Project\Rob\data\corpus\图谱语料\标注建模数据_top10rank_36.xlsx',encoding='utf-8',header=True,index=False)

	Accuracy_Func_1(testS)
'''
	test_data1=test_data.rename(columns={"Label_x": "Label"})
	train_data1=train_data.rename(columns={"Label_x": "Label"})
	test_data2=test_data1.iloc[:, 1:12].astype('int64')
	train_data2=train_data1.iloc[:, 1:12].astype('int64')
	Accuracy_Func_1(train_data2)
'''









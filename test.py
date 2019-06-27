# -*- coding: utf-8 -*-
  
import numpy
from sklearn import metrics
from ssklearn.svm import LinearSVC
from ssklearn.naive_bayes import MultinomialNB
from sklearn import linear_model
from ssklearn.datasets import load_iris
from ssklearn.cross_validation import train_test_split
from ssklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn import cross_validation
from sklearn import preprocessing
#import iris_data
  
def load_data():
  iris = load_iris()
  x, y = iris.data, iris.target
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)
  return x_train,y_train,x_test,y_test
  
def train_clf3(train_data, train_tags):
  clf = LinearSVC(C=1100.0)#default with 'rbf' 
  clf.fit(train_data,train_tags)
  return clf
  
def train_clf(train_data, train_tags):
  clf = MultinomialNB(alpha=0.01)
  print (numpy.asarray(train_tags))
  clf.fit(train_data, numpy.asarray(train_tags))
  return clf
  
def evaluate(actual, pred):
  m_precision = metrics.precision_score(actual, pred)
  m_recall = metrics.recall_score(actual, pred)
  print ('precision:{0:.3f}'.format(m_precision))
  print ('recall:{0:0.3f}'.format(m_recall))
  print ('f1-score:{0:.8f}'.format(metrics.f1_score(actual,pred)));
  
x_train,y_train,x_test,y_test = load_data()
  
clf = train_clf(x_train, y_train)
  
pred = clf.predict(x_test)
evaluate(numpy.asarray(y_test), pred)
print metrics.classification_report(y_test, pred)
  
  
#使用自定义数据
# coding: utf-8
  
import numpy
from sklearn import metrics
from ssklearn.feature_extraction.text import HashingVectorizer
from ssklearn.feature_extraction.text import TfidfVectorizer
from ssklearn.naive_bayes import MultinomialNB
from ssklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from ssklearn.neighbors import KNeighborsClassifier
from ssklearn.svm import SVC
from ssklearn.svm import LinearSVC
import codecs
from ssklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn import linear_model
  
train_corpus = [
   '我们 我们 好孩子 认证 。 就是',
   '我们 好孩子 认证 。 中国',
   '我们 好孩子 认证 。 孤独',
   '我们 好孩子 认证 。',
 ]
  
test_corpus = [
   '我 菲律宾 韩国',
   '我们 好孩子 认证 。 中国',
 ]
  
def input_data(train_file, test_file):
  train_words = []
  train_tags = []
  test_words = []
  test_tags = []
  f1 = codecs.open(train_file,'r','utf-8','ignore')
  for line in f1:
    tks = line.split(':', 1)
    word_list = tks[1]
    word_array = word_list[1:(len(word_list)-3)].split(", ")
    train_words.append(" ".join(word_array))
    train_tags.append(tks[0])
  f2 = codecs.open(test_file,'r','utf-8','ignore')
  for line in f2:
    tks = line.split(':', 1)
    word_list = tks[1]
    word_array = word_list[1:(len(word_list)-3)].split(", ")
    test_words.append(" ".join(word_array))
    test_tags.append(tks[0])
  return train_words, train_tags, test_words, test_tags
  
  
def vectorize(train_words, test_words):
  #v = HashingVectorizer(n_features=25000, non_negative=True)
  v = HashingVectorizer(non_negative=True)
  #v = CountVectorizer(min_df=1)
  train_data = v.fit_transform(train_words)
  test_data = v.fit_transform(test_words)
  return train_data, test_data
  
def vectorize1(train_words, test_words):
  tv = TfidfVectorizer(sublinear_tf = False,use_idf=True);
  train_data = tv.fit_transform(train_words);
  tv2 = TfidfVectorizer(vocabulary = tv.vocabulary_);
  test_data = tv2.fit_transform(test_words);
  return train_data, test_data
   
def vectorize2(train_words, test_words):
  count_v1= CountVectorizer(stop_words = 'english', max_df = 0.5); 
  counts_train = count_v1.fit_transform(train_words); 
    
  count_v2 = CountVectorizer(vocabulary=count_v1.vocabulary_);
  counts_test = count_v2.fit_transform(test_words);
    
  tfidftransformer = TfidfTransformer();
    
  train_data = tfidftransformer.fit(counts_train).transform(counts_train); 
  test_data = tfidftransformer.fit(counts_test).transform(counts_test);
  return train_data, test_data
  
def evaluate(actual, pred):
  m_precision = metrics.precision_score(actual, pred)
  m_recall = metrics.recall_score(actual, pred)
  print 'precision:{0:.3f}'.format(m_precision)
  print 'recall:{0:0.3f}'.format(m_recall)
  print 'f1-score:{0:.8f}'.format(metrics.f1_score(actual,pred));
  
  
def train_clf(train_data, train_tags):
  clf = MultinomialNB(alpha=0.01)
  clf.fit(train_data, numpy.asarray(train_tags))
  return clf
  
  
def train_clf1(train_data, train_tags):
  #KNN Classifier
  clf = KNeighborsClassifier()#default with k=5 
  clf.fit(train_data, numpy.asarray(train_tags)) 
  return clf
  
def train_clf2(train_data, train_tags):
  clf = linear_model.LogisticRegression(C=1e5) 
  clf.fit(train_data,train_tags)
  return clf
  
def train_clf3(train_data, train_tags):
  clf = LinearSVC(C=1100.0)#default with 'rbf' 
  clf.fit(train_data,train_tags)
  return clf
  
def train_clf4(train_data, train_tags):
  """
  随机森林，不可使用稀疏矩阵
  """
  clf = RandomForestClassifier(n_estimators=10)
  clf.fit(train_data.todense(),train_tags)
  return clf
  
#使用codecs逐行读取
def codecs_read_label_line(filename):
  label_list=[]
  f = codecs.open(filename,'r','utf-8','ignore')
  line = f.readline()
  while line:
    #label_list.append(line[0:len(line)-2])
    label_list.append(line[0:len(line)-1])
    line = f.readline()
  f.close()
  return label_list
  
def save_test_features(test_url, test_label):
  test_feature_list = codecs_read_label_line('test.dat')
  fw = open('test_labeded.dat',"w+")
   
  for (url,label) in zip(test_feature_list,test_label):
    fw.write(url+'\t'+label)
    fw.write('\n')
  fw.close()
  
def main():
  train_file = u'..\\file\\py_train.txt'
  test_file = u'..\\file\\py_test.txt'
  train_words, train_tags, test_words, test_tags = input_data(train_file, test_file)
  #print len(train_words), len(train_tags), len(test_words), len(test_words), 
   
  train_data, test_data = vectorize1(train_words, test_words)
  print type(train_data)
  print train_data.shape
  print test_data.shape
  print test_data[0].shape
  print numpy.asarray(test_data[0])
   
  clf = train_clf3(train_data, train_tags)
   
  scores = cross_validation.cross_val_score(
  clf, train_data, train_tags, cv=5, scoring="f1_weighted")
  print scores
  
  #predicted = cross_validation.cross_val_predict(clf, train_data,train_tags, cv=5)  
  '''
   
  '''
  pred = clf.predict(test_data)
  error_list=[]
  for (true_tag,predict_tag) in zip(test_tags,pred):
    if true_tag != predict_tag:
      print true_tag,predict_tag
      error_list.append(true_tag+' '+predict_tag)
  print len(error_list)
  evaluate(numpy.asarray(test_tags), pred)
  '''
  #输出打标签结果
  test_feature_list = codecs_read_label_line('test.dat')
  save_test_features(test_feature_list, pred)
  '''
   
  
if __name__ == '__main__':
  main()
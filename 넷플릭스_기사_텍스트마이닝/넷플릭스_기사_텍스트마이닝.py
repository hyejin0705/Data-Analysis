# -*- coding: utf-8 -*-
"""
Created on Mon May 23 10:34:57 2022

@author: C
"""


#텍스트마이닝_감성분석과 토픽분석


# # 1부. 감성 분류 모델 구축

# ## 1. 데이터 수집

# #### 깃허브에서 데이터 파일 다운로드 : https://github.com/e9t/nsmc 

# ## 2. 데이터 준비 및 탐색

# ### 2-1) 훈련용 데이터 준비


#%% (1) 훈련용 데이터 파일 로드

#warning 메시지 표시 안함
import warnings
warnings.filterwarnings(action = 'ignore')


import pandas as pd

df = pd.read_csv('data/nlp.csv')
# nsmc_train_df.head()


# #### (2) 데이터의 정보 확인

df.info()


# #### (4) 타겟 컬럼 label 확인 (0: 부정감성,   1: 긍정감성)

df['label'].value_counts()
'''
0    7067
1    7067
'''


# #### 

import re

# r'[^(not) ㄱ-ㅣ(자음, 모음) 가-힣(단어)]+'
df['document'] = \
    df['document'].apply(lambda x : re.sub(r'[^ ㄱ-ㅣ가-힣]+', " ", x))


df['document'].head()


#%% 훈련용 테스트용 데이터 분리

x_data = df['document']

y_data = df['label']

# ### 2-2) 평가용 데이터 준비

# #### (1) 평가용 데이터 파일 로드

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = \
    train_test_split(x_data, y_data,
                     test_size = 0.2, # test_size : 분리하는 비율
                     random_state = 0, # random_state : 데이터 고정하는 값
                     stratify = y_data)
                     # stratify : y데이터를 고르게 가져오는 거


# #### (4) 타겟 컬럼 label 확인 (0: 부정감성, 1: 긍정감성)

print(y_train.value_counts())
'''
1    5654
0    5653
'''


#%%

from konlpy.tag import Okt

okt = Okt()


def okt_tokenizer(text):
    tokens = okt.morphs(text)
    return tokens


# #### (2) TF-IDF 기반 피처 벡터 생성

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(tokenizer = okt_tokenizer,
                        # ngram_range : 단어 묶어서 분석(최소, 최대)
                        ngram_range=(1,2), 
                        
                        # min_df : 최소 빈도수 단어 제거
                        min_df = 3, max_df = 0.7)

tfidf.fit(x_train)

train_tfidf = tfidf.transform(x_train)
test_tfidf = tfidf.transform(x_test)


#%% 로지스틱 모델 

from sklearn.linear_model import LogisticRegression

SA_lr = LogisticRegression(random_state = 0)



# #### (2) 로지스틱 하이퍼파라미터 찾기

#%% 하이퍼파라미터 찾기
from sklearn.model_selection import GridSearchCV

params = {'C': [1, 3, 3.5, 4, 4.5, 5]}

# crtl + i = 명령어 도움말
SA_lr_grid_cv = GridSearchCV(SA_lr,  # 모델
                             param_grid = params,  # 파라미터 사전
                             cv = 3, 
                             scoring = 'accuracy',  # 점수 계산 방식
                             verbose = 1)  



SA_lr_grid_cv.fit(train_tfidf, y_train)


print(SA_lr_grid_cv.best_params_, round(SA_lr_grid_cv.best_score_, 4))


SA_lr_best = SA_lr_grid_cv.best_estimator_



#%% 검증
# print(SA_lr_best.score(train_tfidf, y_train))

from sklearn.metrics import accuracy_score, roc_auc_score

predict = SA_lr_best.predict(test_tfidf)

print(accuracy_score(y_test, predict))
print(roc_auc_score(y_test, predict))  


pred_test_LR= SA_lr_best.predict(test_tfidf)
from sklearn.metrics import classification_report
cfreport_test_LR = classification_report(y_test, pred_test_LR)
print(cfreport_test_LR)


#%% naive_bayes 모델
from sklearn.metrics import accuracy_score  # 정확도 계산
from sklearn.naive_bayes import MultinomialNB  
naive = MultinomialNB()

#%% 하이퍼파라미터 찾기
params = {'alpha': [0.5, 1.0, 1.5, 2.0]}
naive_grid_cv = GridSearchCV(naive,  # 모델
                             param_grid = params,  # 파라미터 사전
                             cv = 3, 
                             scoring = 'accuracy',  # 점수 계산 방식
                             verbose = 1)  


# #### (3) 최적 분석 모델 훈련
naive_grid_cv.fit(train_tfidf, y_train)


# 최적 파라미터의 best 모델 저장
naive_best = naive_grid_cv.best_estimator_



# print(naive_best.score(train_tfidf, y_train))
#%% 검증 

from sklearn.metrics import accuracy_score, roc_auc_score

predict = naive_best.predict(test_tfidf)

print(accuracy_score(y_test, predict))  # 0.8275

print(roc_auc_score(y_test, predict))   # 0.7948




pred_test_rbf = naive_best.predict(test_tfidf)
from sklearn.metrics import classification_report
cfreport_test_rbf = classification_report(y_test, pred_test_rbf)
print(cfreport_test_rbf)




#%% LGBM 모델

from lightgbm import LGBMClassifier


lgb = LGBMClassifier(random_state = 123, n_jobs = -1)
    

#%%  (2) LGBM 회귀의  best 하이퍼파라미터 찾기


# 모델 생성의 파라미터
from sklearn.model_selection import GridSearchCV

grid_param = {'max_depth' : [1, 3, 5, 7],
              'n_estimators' : [100, 300, 500],
              'learning_rate' : [0.05, 0.1, 1, 10],
              'subsample' : [0.5, 1]}


lgb_grid = GridSearchCV(estimator = lgb,
                           param_grid = grid_param,
                           cv = 3,
                           scoring = 'accuracy')

fit_param = {'early_stopping_rounds' : 50, 
              'eval_metric' : 'error', 
              'eval_set' : [[test_tfidf, y_test]]}


lgb_grid.fit(train_tfidf, y_train, **fit_param)
lgb_grid.best_estimator_
lgb_best = lgb_grid.best_estimator_


#%%

# 검증
print(lgb_grid.best_params_,   
      round(lgb_grid.best_score_, 4))   # 0.7654

# {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 500, 'subsample': 0.5}

# 최적 파라미터의 best 모델 저장
lgb_best = lgb_grid.best_estimator_

predict = lgb_best.predict(test_tfidf) 


print(accuracy_score(y_test, predict)) 
print(roc_auc_score(y_test, predict)) 
                                  

pred_test_lgb = lgb_best.predict(test_tfidf)
from sklearn.metrics import classification_report
cfreport_test_lgb = classification_report(y_test, pred_test_lgb)
print(cfreport_test_lgb)


#%% SVM 모델

from sklearn.svm import SVC

model_rbf = SVC(kernel='rbf', random_state = 123)

#%% SVM 하이퍼파라미터 찾기
param_grid =[{'C' : [0.01, 0.1, 1, 3,7],
             'gamma' : [0.05,0.01, 0.1, 1, 3]}]


from sklearn.model_selection import GridSearchCV
grid_search_RBF = GridSearchCV(model_rbf, param_grid, cv = 3)
grid_search_RBF.fit(train_tfidf, y_train )


print(grid_search_RBF.best_params_)
# {'C': 3, 'gamma': 1}
print(grid_search_RBF.best_score_)
# 0.843813566817016


#%%검증 

svm_best = grid_search_RBF.best_estimator_
predict = svm_best.predict(test_tfidf)

from sklearn.metrics import accuracy_score,roc_auc_score
print(accuracy_score(y_test, predict))  
print(roc_auc_score(y_test, predict))  

from sklearn.metrics import accuracy_score
pred_test_rbf = svm_best.predict(test_tfidf)
from sklearn.metrics import classification_report
cfreport_test_rbf = classification_report(y_test, pred_test_rbf)
print(cfreport_test_rbf)

#%% RandomForestClassifier모델

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
RC = RandomForestClassifier(random_state=123)

#%% 하이퍼파라미터 찾기
# parameter 들을 dictionary 형태로 설정
parameters = {'max_depth':[1,2,3], 'min_samples_split':[2,3]}


# param_grid의 하이퍼 파라미터들을 3개의 train, test set fold 로 나누어서 테스트 수행 설정.  
### refit=True 가 default 임. True이면 가장 좋은 파라미터 설정으로 재 학습 시킴.  
grid_search_RC= GridSearchCV(RC, param_grid=parameters, cv=3, refit=True)
                          
                          
# param_grid의 하이퍼 파라미터들을 순차적으로 학습/평가
grid_search_RC.fit(train_tfidf, y_train)
#%% 검증
                          

print(grid_search_RC.best_params_)
print(grid_search_RC.best_score_)



rf_best = grid_search_RC.best_estimator_
predict = rf_best.predict(test_tfidf)

from sklearn.metrics import accuracy_score,roc_auc_score
print(accuracy_score(y_test, predict))  
print(roc_auc_score(y_test, predict))  


from sklearn.metrics import accuracy_score
pred_test_RC = rf_best.predict(test_tfidf)
from sklearn.metrics import classification_report
cfreport_test_RC = classification_report(y_test, pred_test_RC)
print(cfreport_test_RC)


#%% 모델 성능 비교 그래프 그리기

models = [SA_lr_best, naive_best, lgb_best, svm_best, rf_best]

model_df = pd.DataFrame(columns = ['Logistic', 'naive_bayes', 'LGBM', 'SVM', 'RandomForest'],
                        index = ['train', 'test', 'roc'])

for idx, model in enumerate(models):
    train = model.score(train_tfidf, y_train)
    test = model.score(test_tfidf, y_test)
    
    predict = model.predict(test_tfidf)
    roc = roc_auc_score(y_test, predict)
    
    model_df.iloc[:, idx] = [train, test, roc]
    
print(model_df)


import matplotlib.pyplot as plt

plt.rc('font', family='Malgun Gothic')

model_df.plot()

plt.title('감성분석 모델 성능 비교')
plt.xlabel('성능지표')
plt.ylabel('')


#%% 뉴스기사 가져오기

import json

file_name = '넷플릭스_naver_news'



df = pd.read_json(file_name +'.json', orient = 'str')
# orient='str' 지정


# #### (2) 분석할 컬럼을 추출하여 데이터 프레임에 저장



data_df = df[['title', 'description']]  # 데이터프레임


# data_df['description'].nunique() # 899   # 1000 (원본, 중복 있음.)


# # # description 열에서 중복 제거 (99행에는 중복이 없음.)
# data_df.drop_duplicates(subset = ['description'], inplace = True)

print(data_df.head())


#%%  (3) 한글 이외 문자 제거


import re
data_df['title'] = data_df['title'].apply(lambda x : re.sub(r'[^ ㄱ-ㅣ가-힣]+', " ", x))
data_df['description'] = data_df['description'].apply(lambda x : re.sub(r'[^ ㄱ-ㅣ가-힣]+', " ", x))

print(data_df.head())  #작업 확인용 출력


# ## 3. 감성 분석 수행

# ### 3-1) 'title'에 대한 감성 분석

# 1) 분석할 데이터의 피처 벡터화 ---<< title >> 분석
data_title_tfidf = tfidf.transform(data_df['title'])



# 2) 최적 파라미터 학습모델에 적용하여 감성 분석
data_title_predict = svm_best.predict(data_title_tfidf)


# 3) 감성 분석 결과값을 데이터 프레임에 저장
data_df['title_label'] = data_title_predict


#%%  3-2) 'description' 에 대한 감성 분석


# 1) 분석할 데이터의 피처 벡터화 ---<< description >> 분석
data_description_tfidf = tfidf.transform(data_df['description'])


# 2) 최적 파라미터 학습모델에 적용하여 감성 분석
data_description_predict = svm_best.predict(data_description_tfidf)


# 3) 감성 분석 결과값을 데이터 프레임에 저장
data_df['description_label'] = data_description_predict



data_df.head()

print(data_df['title_label'].value_counts())
print(data_df['description_label'].value_counts())


#%%
# ### 4-2) 결과 저장 : 긍정과 부정을 분리하여 CSV 파일 저장

columns_name = ['title','title_label','description','description_label']

# 빈 데이터프레임 생성
NEG_data_df = pd.DataFrame(columns = columns_name)
POS_data_df = pd.DataFrame(columns = columns_name)


for i, data in data_df.iterrows(): 
    title = data["title"] 
    description = data["description"] 
    t_label = data["title_label"] 
    d_label = data["description_label"] 
    
    if d_label == 0: # 부정 감성 샘플만 추출
        NEG_data_df = NEG_data_df.append(pd.DataFrame([[title, t_label, description, d_label]],
                                                      columns = columns_name), 
                                         ignore_index = True)
    
    else : # 긍정 감성 샘플만 추출
        POS_data_df = POS_data_df.append(pd.DataFrame([[title, t_label, description, d_label]],columns=columns_name),ignore_index=True)

# print(data) 
        
 
# 파일에 저장.
NEG_data_df.to_csv('data/'+file_name+'_NES.csv', encoding='euc-kr') 
POS_data_df.to_csv('data/'+file_name+'_POS.csv', encoding='euc-kr') 


len(NEG_data_df), len(POS_data_df)


#%% # ### 4-3)  감성 분석 결과 시각화 : 바 차트

# #### (1) 명사만 추출하여 정리하기

# #### - 긍정 감성의 데이터에서 명사만 추출하여 정리 

POS_description = POS_data_df['description']


POS_description_noun_tk = []

for d in POS_description:
    POS_description_noun_tk.append(okt.nouns(d)) #형태소가 명사인 것만 추출


print(POS_description_noun_tk)  # 작업 확인용 출력



POS_description_noun_join = []

for d in POS_description_noun_tk:
    # 길이가 1인 토큰은 제외
    d2 = [w for w in d if len(w) > 1]
    # 리스트(d)에서 단어(w) 하나씩 가져와서  if 단어 길이가 2글자 이상인 단어만 추출
    
    POS_description_noun_join.append(" ".join(d2)) 
    # 토큰을 연결(join)하여 하나의 리스트 구성


print(POS_description_noun_join)  # 작업 확인용 출력


#%%

# #### - 부정 감성의 데이터에서 명사만 추출하여 정리 

NEG_description = NEG_data_df['description']

NEG_description_noun_tk = []
NEG_description_noun_join = []

for d in NEG_description:
    NEG_description_noun_tk.append(okt.nouns(d)) # 형태소가 명사인 것만 추출
    
for d in NEG_description_noun_tk:
    d2 = [w for w in d if len(w) > 1]  #길이가 1인 토큰은 제외
    NEG_description_noun_join.append(" ".join(d2)) # 토큰을 연결(join)하여 리스트 구성


print(NEG_description_noun_join)  #작업 확인용 출력


#%%
# #### (2) dtm 구성 : 단어 벡터 값을 내림차순으로 정렬

# #### - 긍정 감성 데이터에 대한 dtm 구성, dtm을 이용하여 단어사전 구성 후 내림차순 정렬


POS_tfidf = TfidfVectorizer(tokenizer = okt_tokenizer, min_df=2 )
POS_dtm = POS_tfidf.fit_transform(POS_description_noun_join)


POS_vocab = dict() 

for idx, word in enumerate(POS_tfidf.get_feature_names()):
    POS_vocab[word] = POS_dtm.getcol(idx).sum()
    
POS_words = sorted(POS_vocab.items(), key=lambda x: x[1], reverse=True)
          # 정렬,  사전.items() : 키값, value 
          # key(정렬 기준) : x[1] (value값으로 정렬)
          # reverse(정렬방법) : True(내림차순), False(오름차순)
           
          
POS_words  #작업 확인용 출력


#%%  - 부정 감성 데이터의 dtm 구성, dtm을 이용하여 단어사전 구성 후 내림차순 정렬

NEG_tfidf = TfidfVectorizer(tokenizer = okt_tokenizer, min_df=2 )
NEG_dtm = NEG_tfidf.fit_transform(NEG_description_noun_join)


NEG_vocab = dict() 

for idx, word in enumerate(NEG_tfidf.get_feature_names()):
    NEG_vocab[word] = NEG_dtm.getcol(idx).sum()
    
NEG_words = sorted(NEG_vocab.items(), key=lambda x: x[1], reverse=True)

NEG_words   #작업 확인용 출력


#%%
# #### (3) 단어사전의 상위 단어로 바 차트 그리기

import matplotlib.pyplot as plt

plt.rc('font', family='Malgun Gothic')

max = 15  # 바 차트에 나타낼 단어의 수 

plt.figure(figsize = (15, 5))


plt.subplot(1, 2, 1)
plt.bar(range(max), [i[1] for i in POS_words[:max]], color="blue")
plt.title("긍정 뉴스의 단어 상위 %d개" %max, fontsize=15)
plt.xlabel("단어", fontsize=12)
plt.ylabel("TF-IDF의 합", fontsize=12)
plt.xticks(range(max), [i[0] for i in POS_words[:max]], rotation=70)
plt.ylim([0, 70]) 


plt.subplot(1, 2, 2)
plt.bar(range(max), [i[1] for i in NEG_words[:max]], color="red")
plt.title("부정 뉴스의 단어 상위 %d개" %max, fontsize=15)
plt.xlabel("단어", fontsize=12)
plt.ylabel("TF-IDF의 합", fontsize=12)
plt.xticks(range(max), [i[0] for i in NEG_words[:max]], rotation=70)
plt.ylim([0, 70])


#%%  3부. 토픽모델링 : LDA 기반 토픽 모델링

# ## 1. 데이터 준비 

# ### 1-1) 'description' 컬럼 추출


description = data_df['description']


# ### 1-2) 형태소 토큰화 : 명사만 추출


description_noun_tk = []

for d in description:
    description_noun_tk.append(okt.nouns(d)) #형태소가 명사인 것만 추출


description_noun_tk2 = []

for d in description_noun_tk:
    item = [i for i in d if len(i) > 1]  #토큰의 길이가 1보다 큰 것만 추출
    description_noun_tk2.append(item)


print(description_noun_tk2)


#%%  2. LDA 토픽 모델 구축

# ### 2-1) LDA 모델의 입력 벡터 생성 


# 최초 한번만 설치
#pip install gensim 


import gensim
import gensim.corpora as corpora


# #### (1) 단어 사전 생성
dictionary = corpora.Dictionary(description_noun_tk2)


print(dictionary[1])  #작업 확인용 출력


# #### (2) 단어와 출현빈도(count)의 코퍼스 생성
corpus = [dictionary.doc2bow(word) for word in description_noun_tk2]


print(corpus) #작업 확인용 출력


#%%  2-2) LDA 모델 생성 및 훈련 

k = 3  #토픽의 개수 설

lda_model = gensim.models.ldamulticore.LdaMulticore(corpus, 
                                                    iterations = 12, 
                                                    num_topics = k, 
                                                    id2word = dictionary, 
                                                    passes = 1, workers = 10,
                                                    random_state = 50)


#%%  3. LDA 토픽 분석 결과 시각화

# ### 3-1) 토픽 분석 결과 확인

print(lda_model.print_topics(num_topics = k, num_words = 15))


'''
[(0, '0.090*"넷플릭스" + 0.009*"시리즈" + 0.008*"공개" + 0.007*"제작" + 0.007*"한국" + 0.007*"글로벌" + 0.007*"서비스" + 0.006*"안나라수마나라" + 0.006*"영화" + 0.006*"투자" + 0.006*"지난" + 0.006*"세계" + 0.006*"온라인" + 0.006*"콘텐츠" + 0.005*"배우"'), 
 (1, '0.076*"넷플릭스" + 0.009*"투자" + 0.008*"자회사" + 0.008*"영화" + 0.008*"시리즈" + 0.007*"서비스" + 0.007*"분기" + 0.007*"감소" + 0.007*"최근" + 0.007*"글로벌" + 0.007*"동영상" + 0.007*"공개" + 0.006*"온라인" + 0.006*"국내" + 0.006*"가입자"'), 
 (2, '0.096*"넷플릭스" + 0.012*"시리즈" + 0.012*"공개" + 0.011*"서비스" + 0.008*"동영상" + 0.007*"가입자" + 0.007*"영화" + 0.007*"미국" + 0.006*"현지" + 0.006*"안나라수마나라" + 0.006*"세계" + 0.006*"글로벌" + 0.006*"온라인" + 0.006*"한국" + 0.006*"콘텐츠"')]
'''



#%% 토픽 분석 결과 데이터프레임으로 저장

topics = pd.DataFrame()

for num in range(3):
    topic = lda_model.show_topic(num, 15)
    print(topic)
    
    left = pd.DataFrame(topic)[[0]].sort_values(0)
    topics = pd.concat([topics, left], axis = 1)

print('-' * 30)

topics.columns = [0, 1, 2]

print(topics)


'''
0   넷플릭스  넷플릭스     넷플릭스
1    시리즈   서비스      시리즈
2     공개   글로벌       공개
3     투자   온라인      서비스
4     영화    공개       한국
5    서비스    영화  안나라수마나라
6     미국   동영상      글로벌
7    동영상   가입자      콘텐츠
8     감소    세계       투자
9    온라인   시리즈       세계
10    현지    분기       영화
11   콘텐츠    미국      가입자
12   가입자    시간       국내
13    제작   자회사       분기
14    지난    현지      동영상
'''


#%%  3-2) 토픽 분석 결과 시각화 : pyLDAvis

#최초 한번만 설치
#conda install -c anaconda gensim
#물어보면 y

# pip install pyLDAvis 설치

import pyLDAvis.gensim_models

lda_vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)

pyLDAvis.display(lda_vis)


pyLDAvis.save_html(lda_vis, 'data/'+file_name+"_vis.html")



#%% LDA 모델 평가(검증)

# 토픽 의미론적 일관성(Coherence)
# 단어 간의 유사도를 계산하여 해당 주제가 의미론적으로 일치하는 단어들끼리 모여있는지 파악

from gensim.models.coherencemodel import CoherenceModel

cm = CoherenceModel(model=lda_model, corpus = corpus, 
                    coherence='u_mass')

print(cm.get_coherence())  # -3.537459780644412
        
        
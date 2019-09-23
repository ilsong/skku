import numpy as np
from konlpy.tag import Mecab
from collections import Counter
from gensim.corpora import Dictionary
from sklearn.cluster import KMeans
from sklearn.feature_extraction import DictVectorizer
from soyclustering import SphericalKMeans, proportion_keywords
from scipy.sparse import csr_matrix

mecab = Mecab()
f = open("body.txt", 'r', encoding='utf8')
rawbody = f.read()
body = rawbody.split("\n")
f = open("subject.txt", 'r', encoding='utf8')
rawsubject = f.read()
subject = rawsubject.split("\n")
f = open("title.txt", 'r', encoding='utf8')
rawtitle = f.read()
title = rawtitle.split("\n")
f.close()

error_idx = []
print("Removing Error Rows")
for i in range(len(body)):
    if body[i] == "" or title[i] == "" or subject[i] == "":
        error_idx.append(i)

error_idx = list(set(error_idx))
error_idx.sort(reverse=True)
for idx in error_idx:
    del body[idx], subject[idx], title[idx]
print("Removing Error Rows Complete")

load = True
if load == False:
    titleWeight = 3
    topN = 10
    topNNouns = []
    print("Count Top %s Nouns from Article, Title Weight: %d" % (str(topN), titleWeight))
    for i in range(len(body)):
        try:
            bodyNouns = mecab.nouns(body[i])
            titleNouns = mecab.nouns(title[i])
            topNNoun = dict(Counter(bodyNouns + titleNouns*titleWeight).most_common(topN))
            topNNouns.append(topNNoun)
        except Exception as e:
            print("Error on %d, %s" % (i, str(e)))
else:
    f = open("topNNouns.txt", 'r', encoding='utf8')
    topNNouns = eval(f.read())
    f.close()
    topN = None

dict_vectorizer = DictVectorizer(sparse=False)
train_x = dict_vectorizer.fit_transform(topNNouns)
print("top %s Nouns shape: %s" % (str(topN), str(train_x.shape)))
train_x = csr_matrix(train_x)
print("Count Top %s Nouns from Article Complete" % str(topN))


print("Clustering Using KMeans..")
k = 6
inertias = []
print("Clustering %d-Means" % k)
spherical_kmeans = SphericalKMeans(n_clusters=k, init='similar_cut', verbose=1, max_iter=100)
labels = spherical_kmeans.fit_predict(train_x)
inertias.append(spherical_kmeans.inertia_)
labels_each_cluster = []
for i in range(k):
    labels_each_cluster.append(np.where(labels == i)[0])
print()
print("Clustering Complete")

#for i in range(3, 20):
#    print("%d cluster inertia: %f" % (i, inertias[i-3]))
#print(inertias)

keywords = proportion_keywords(
        spherical_kmeans.cluster_centers_,
        labels,
        index2word = dict_vectorizer.feature_names_,
        topk=100,
        candidates_topk=200
)

for cluster_idx, keyword in enumerate(keywords):
    keyword = ' '.join([w for w,_ in keyword])
    print('cluster#%d keywords: %s' % (cluster_idx, keyword))
    cluster_size = len(labels_each_cluster[cluster_idx])
    print("cluster node num: %d" % cluster_size)
    for title_idx in np.random.choice(labels_each_cluster[cluster_idx], min(cluster_size, 5), replace=False):
        print("title #%d: %s" % (title_idx, title[title_idx]))
    print()

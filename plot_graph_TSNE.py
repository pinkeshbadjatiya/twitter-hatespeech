import gensim
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.manifold import TSNE
import pdb
import codecs


words = ['mohammed', 'murderer', 'pedophile', 'religion', 'terrorism', 'islamic', 'muslim']

def load_initial_emb():
    initial_emb = gensim.models.Word2Vec.load_word2vec_format("/home/pinkesh/DATASETS/glove-twitter/GENSIM.glove.twitter.27B.200d.txt")
    return initial_emb

def load_final_emb():
    reverse_vocab = codecs.open3("reverse_vocab.json", 'r', encoding="utf-8").readlines()
    reverse_vocab = json.loads("".join(reverse_vocab))
    reverse_vocab['0'] = "<UNK>"

    final_emb = {}
    for i, emb in enumerate(np.load("embedding.npy")):
        final_emb[reverse_vocab[str(i)].encode("utf-8")] = emb
    return final_emb

def get_transform(initial_emb, final_emb):
    vec = []
    for w in words:
        vec.append(initial_emb[w])
    for w in words:
        vec.append(final_emb[w])
    
    X = np.array(vec)
    print X.shape
    
    model = TSNE(n_components=2, random_state=0)
    out = model.fit_transform(X)
    
    print out
    print "Will plot now!"
    return out



# Initial are original
# Next are final
def plot(out):
    A = out[:7,:]
    B = out[7:,:]
    area=150
    padding=0.0001
    xmin, xmax = min(out[:, 0]), max(out[:, 0])
    ymin, ymax = min(out[:, 1]), max(out[:, 1])
    
    fig, ax = plt.subplots()
    
    for (color, label, data) in [('red', 'GloVe', A), ('green', 'FastText+GloVe+Dyn', B)]:
        ax.scatter(data[:,0], data[:,1], c=color, s=area, label=label,
                                      alpha=0.3, edgecolors='none')
        for (row, word) in zip(data, words):
            ax.annotate(word, xy=(row[0], row[1]), xytext=(row[0], row[1]),)
    
    plt.axis([xmin-padding,xmax+padding,ymin-padding,ymax+padding])
    plt.legend()
    plt.grid(True)
    
    plt.show()


if __name__=="__main__":
    ini = load_initial_emb()
    fin = load_final_emb()
    out = get_transform(ini, fin)
    plot(out)

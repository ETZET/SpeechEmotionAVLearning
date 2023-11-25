import sys, os
import matplotlib.pyplot as plt
sys.path.append('module/umap')
import umap
import numpy as np
import pandas as pd
import pickle

# emotional anchors
def ld_mead():
    # ['A', 'C', 'D', 'F', 'H', 'N', 'S', 'U']
    ld = np.array([[-0.43,0.67],[-0.8,0.2],[-0.6,0.35],[-0.64,0.6],[0.76,0.48],[0,0],[-0.63,-0.27],[0, 0.6]] )
    return np.array(ld)
def ld_iemocap():
    # ["ANG","HAP","NEU","SAD"]
    ld = np.array([[-0.51,0.59],[0.81,0.51],[0,0],[-0.63,-0.27]])
    return np.array(ld)
def ld_iemocap_partial5():
    # ["ANG",'dis',"HAP","NEU","SAD"]
    ld = np.array([[-0.51,0.59],[-0.6,0.35],[0.81,0.51],[0,0],[-0.63,-0.27]])
    return np.array(ld)
def ld_iemocap_partial():
    # ['ang', 'dis', 'exc', 'fea', 'fru', 'hap', 'neu', 'sad', 'sur']
    return np.array([[-0.51,0.59],[-0.6,0.35], [0.62, 0.75],[-0.64,0.6],[-0.64, 0.52], [0.81,0.51], [0,0], [-0.63,-0.27],[0, 0.6]])
def ld_iemocap_full():
    # ['ang', 'dis', 'exc', 'fea', 'fru', 'hap', 'neu', 'oth', 'sad', 'sur', 'xxx']
    return np.array([[-0.51,0.59],[-0.6,0.35], [0.62, 0.75],[-0.64,0.6],[-0.64, 0.52], [0.81,0.51], [0,0], [0,0], [-0.63,-0.27],[0, 0.6],[0,0]])
def ld_emodb():
    # ['angry', 'boredom', 'disgust', 'fear', 'happy', 'neutral', 'sad']
    return np.array([[-0.43,0.67],[-0.65,-0.62],[-0.6,0.35],[-0.64,0.6],[0.76,0.48],[0,0],[-0.63,-0.27]])

# utils 
def load_data(name):
    if name=="iemocap":
        with open("./data/embeddings_spk.pickle", 'rb') as f:
            data = pickle.load(f)
        init_global = ld_iemocap()
    elif name=="iemocap_all":
        with open("./data/embeddings_spk_all.pickle", 'rb') as f:
            data = pickle.load(f)
        init_global = ld_iemocap_full() 
    elif name=="iemocap_partial":
        with open("dump/iemocap_partial/embeddings.pickle", 'rb') as f:
            data = pickle.load(f)
        init_global = ld_iemocap_partial() 
    elif name=="iemocap_partial5":
        with open("dump/iemocap_partial5/embeddings.pickle", 'rb') as f:
            data = pickle.load(f)
        init_global = ld_iemocap_partial5() 
    elif name=="iemocap_partial4":
        with open("dump/iemocap_partial4/embeddings.pickle", 'rb') as f:
            data = pickle.load(f)
        init_global = ld_iemocap() 
    elif name=="mead":
        with open("/dataHDD/ezhou12/MEAD/embeddings.pickle",'rb') as f:
            data = pickle.load(f)
        init_global = ld_mead()
    elif name=="emodb":
        with open("/dataHDD/ezhou12/EMODB/embeddings.pickle",'rb') as f:
            data = pickle.load(f)
        init_global = ld_emodb()
    
    embeddings, label = data['embeddings'], data['emotion']
    return {"embedding":embeddings, "label": label, "init_global": init_global, "data":data}

# Anchored Dimensionality Reduction
class AVLearner:
    def __init__(self,
                n_neighbors=20,
                n_epochs=150,
                negative_sample_rate=10,
                min_dist=0.1,
                learning_rate=0.01,
                target_weight=0.1,
                repulsion_strength=0.1) -> None:
        self.reducer = umap.UMAP(n_neighbors=n_neighbors,n_epochs=n_epochs,\
                        negative_sample_rate=negative_sample_rate,min_dist=min_dist,\
                        learning_rate=learning_rate, target_metric="categorical", \
                        target_metric_kwds = {}, target_weight=target_weight,
                        repulsion_strength=repulsion_strength, spread=1.0)
    
    def fit(self, embedding, labels, anchor_mappings):
        init = self.reducer.set_custom_intialization(embedding, labels, anchor_mappings)
        self.reducer.fit(embedding, labels)
    
    def transform(self, embedding):
        return self.reducer.tranform(embedding)
    
    def fit_transform(self, embedding, labels, anchor_mappings):
        init = self.reducer.set_custom_intialization(embedding, labels, anchor_mappings)
        return self.reducer.fit_transform(embedding, labels)

def train_inference(data):
    # data
    init_global = data['init_global']
    data = data['data']
    train_idx, test_idx = data['status'] != 'test', data['status'] == 'test'
    embedding, label = np.array(data['embeddings'])[train_idx], np.array(data['emotion'])[train_idx]
    test_embedding, test_label = np.array(data['embeddings'])[test_idx], np.array(data['pred_emotion'])[test_idx]

    reducer = AVLearner()
    train_y = reducer.fit_transform(embedding, label, init_global)
    test_y = reducer.transform(test_embedding) # not using the test portion of categorical labels

    return train_y, test_y


if __name__ == "__main__":
    data = load_data('iemocap')
    train_y, test_y = train_inference(data)
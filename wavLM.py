from transformers import AutoProcessor, WavLMModel, Wav2Vec2FeatureExtractor, set_seed
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from tqdm import tqdm
import random
import os
import wandb
import argparse
import audmetric
from sklearn.metrics import balanced_accuracy_score, recall_score
from torch.utils.data import DataLoader
from datasets import concatenate_datasets
# from data_prep.datagen import create_data_dictionary, create_dataset_MEAD, create_dataset

device = torch.device("cuda:0")

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-large')
def my_collate(batch):
    audios, emotions = [], []
    for data in batch:
        au, emo = data['audio'], data['emotion']
        audios.append(au['array'])
        emotions.append(emo)
    audios = feature_extractor(audios, sampling_rate=16000, return_tensors="pt", padding=True).to(device)
    return {"audio": audios,"emotion": emotions}


class EmotionClassifier(nn.Module):
    def __init__(self, layer_num, emb_dim, num_labels, hidden_dim=100):
        super().__init__()
        self.layer_num = layer_num
        self.emb_dim = emb_dim

        self.weights = nn.Parameter(torch.randn(layer_num))
        self.proj = nn.Linear(emb_dim,hidden_dim)
        self.out = nn.Linear(hidden_dim, num_labels)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.xavier_uniform_(self.out.weight)
    
    def forward(self, feature, feature_lens):

        # weighted sum of the features
        stacked_feature = torch.stack(feature,dim=0)
        _, *origin_shape = stacked_feature.shape
        stacked_feature = stacked_feature.view(self.layer_num, -1)
        norm_weights = F.softmax(self.weights, dim=-1)
        weighted_feature = (norm_weights.unsqueeze(-1) * stacked_feature).sum(dim=0)
        weighted_feature = weighted_feature.view(*origin_shape)

        # average pooling
        agg_vec_list = []
        for i in range(len(weighted_feature)):
            agg_vec = torch.mean(weighted_feature[i][:feature_lens[i]], dim=0)
            agg_vec_list.append(agg_vec)

        avg_emb = torch.stack(agg_vec_list)

        # classifier
        final_emb = self.proj(avg_emb)
        pred = self.out(final_emb)
        return pred, final_emb

class Trainer():
    def __init__(self, config):
        self.config = config
        device = config.device

        with open(config.data,"rb") as f:
            dataset = pickle.load(f)

        
        train_data, val_data, test_data = dataset['train'], dataset['val'], dataset['test']

        self.train_data, self.val_data, self.test_data = train_data, val_data, test_data
        self.num_labels = config.num_labels
        self.train_dataloader = DataLoader(train_data, batch_size=config.batch_size,collate_fn = my_collate)
        self.val_dataloader = DataLoader(val_data, batch_size=config.batch_size,collate_fn = my_collate)
        self.test_dataloader = DataLoader(test_data, batch_size=config.batch_size, collate_fn=my_collate)

        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-large')
        
        self.wavlm = WavLMModel.from_pretrained("microsoft/wavlm-large").to(device)
        wavlm_param = self.get_upstream_param()
        self.downsample_rate = wavlm_param['downsample_rate']

        # move it to four gpu
        print(f"Using {config.num_gpus} gpu to train")
        if config.num_gpus > 1:
            self.wavlm = nn.DataParallel(self.wavlm, device_ids=list(range(config.num_gpus)))

        self.clf = EmotionClassifier(layer_num=wavlm_param['layer_num'], emb_dim=wavlm_param['emb_dim'],num_labels=config.num_labels)
        if self.config.load_path!='':
            self.load_model()
        self.clf.to(device)
        self.opt = torch.optim.Adam(self.clf.parameters(),lr=config.lr,weight_decay=config.reg_lr)
        self.best_accuracy = 0.0

        self.loss = nn.CrossEntropyLoss()

        self.write = config.write
        if args.use_wandb:
            wandb_save_path = "/dataHDD/ezhou12/dump"
            wandb.init(project=args.wandb_name,config=args, dir=wandb_save_path)
        
    
    def get_upstream_param(self):
        paired_wavs = torch.randn(16000).reshape(1,16000).to(self.wavlm.device)
        with torch.no_grad():
            outputs = self.wavlm(paired_wavs,output_hidden_states=True)
        downsample_rate = round(
                max(len(wav) for wav in paired_wavs) / outputs.extract_features.size(1) )
        layer_num = len(outputs.hidden_states)
        emb_dim = outputs.last_hidden_state.size(2)
        return {'downsample_rate': downsample_rate, "layer_num":layer_num, "emb_dim": emb_dim}
    
    def get_feature_seq_length(self, wav_attention_mask):
        """
        compute the actual sequence length for extracted features

        arguments:
        features: Tensor of (B x T x D) extracted features by the wavlm model
        wav_attention_mask: attention mask of the original original wav foam 

        returns:
        list of ints 
        """
        actual_wav_length = wav_attention_mask.sum(dim=1).cpu().numpy()
        feature_lens = [round(wav_length/self.downsample_rate) for wav_length in actual_wav_length]
        return feature_lens

    def train_pass(self, epoch, is_training=True):
        config = self.config
        opt = self.opt
        if is_training:
            dataloader = self.train_dataloader
            status = 'TRAIN'
        else:
            dataloader = self.val_dataloader
            status = 'EVAL'

        # freeze upstream wavlm
        for p in self.wavlm.parameters():
            p.requires_grad = False
        
        running_loss = 0.0
        running_corrects = 0
        for i,batch in enumerate(dataloader):
            opt.zero_grad()

            input = batch['audio']
            # upstream inference
            outputs = self.wavlm(**input,output_hidden_states=True)
            # downstream
            hiddens = outputs.hidden_states
            feature_length = self.get_feature_seq_length(input['attention_mask'])
            pred, final_emb = self.clf(hiddens, feature_length)
            pred = F.softmax(pred,dim=1)

            label = torch.tensor(batch['emotion'],device=device)
            # label_onehot = F.one_hot(label,self.num_labels).float()
            loss = self.loss(pred,label)

            if is_training:
                loss.backward()
                opt.step()
            
            # statistics 
            running_loss += loss.item()
            running_corrects += sum(pred.argmax(1).cpu().numpy() == label.cpu().numpy())
            if i % 100 ==0:
                print(f"Epoch {epoch}, batch {i}: loss: {loss.item()}")
        
        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = running_corrects / len(dataloader.dataset)

        print('Epoch: {:d} {} Loss: {:.4f} Acc: {:.4f}'.format(epoch, status, epoch_loss, epoch_acc))
                
        #logging
        if self.config.use_wandb:
            wandb.log({f"{status} Loss": epoch_loss, f"{status} accuracy": epoch_acc, "epoch": epoch})
        
        # model checking
        if epoch % 10 == 0:
            self.save_model(epoch,f"epoch_{epoch}")
        self.save_model(epoch)
        if not is_training and epoch_acc > self.best_accuracy:
            self.save_model(epoch,"best")
            self.best_accuracy = epoch_acc
    
    def train(self):
        for epoch in range(self.config.num_epochs):
            self.train_pass(epoch)
            if epoch % 10 == 0:
                with torch.no_grad():
                    self.train_pass(epoch,is_training=False)

    def eval(self):
        opt = self.opt
        dataloader = self.val_dataloader
        status = 'EVAL'

        # freeze upstream wavlm
        for p in self.wavlm.parameters():
            p.requires_grad = False
        
        predictions = []
        gts = []
        for i,batch in enumerate(dataloader):

            input = batch['audio']
            # upstream inference
            outputs = self.wavlm(**input,output_hidden_states=True)
            # downstream
            hiddens = outputs.hidden_states
            feature_length = self.get_feature_seq_length(input['attention_mask'])
            pred, final_emb = self.clf(hiddens, feature_length)
            pred = F.softmax(pred,dim=1)

            label = torch.tensor(batch['emotion'],device=device)

            predictions.append(pred.argmax(1).cpu().numpy())
            gts.append(label.cpu().numpy())

        predictions = np.concatenate(predictions)
        gts = np.concatenate(gts)

        # compute metric
        acc = balanced_accuracy_score(gts, predictions)
        war = recall_score(gts, predictions,average='weighted')
        uar = audmetric.unweighted_average_recall(gts, predictions)

        print(f"accuracy: {acc}, weighted recall: {war}, Unweighted Recall: {uar}")
            
        
    
    def inference(self):
        # merge train test dataset
        dataset = concatenate_datasets([self.train_data,self.val_data, self.test_data])
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, collate_fn=my_collate)

        # freeze upstream wavlm
        for p in self.wavlm.parameters():
            p.requires_grad = False
        self.wavlm.eval()
        self.clf.eval()

        embeddings = []
        predictions = []

        for batch in tqdm(dataloader):

            input = batch['audio']
            # upstream inference
            with torch.no_grad():
                outputs = self.wavlm(**input,output_hidden_states=True)
                # downstream
                hiddens = outputs.hidden_states
                feature_length = self.get_feature_seq_length(input['attention_mask'])
                pred, final_emb = self.clf(hiddens, feature_length)
                pred = F.softmax(pred,dim=1)
            predictions.append(pred.argmax(1).cpu().numpy())
            embeddings += [final_emb.detach().cpu().numpy()]
        
        predictions = np.concatenate(predictions)
        embeddings = np.concatenate(embeddings)
        status = ["train"] * len(self.train_data) + ["val"] * len(self.val_data) + ["test"] * len(self.test_data)
        status = np.array(status)
        save_iemocap_partial(self.config.save_path,embeddings,status,predictions,dataset)
        # save_msp_partial(embeddings,status,dataset)
        # save_msp(embeddings,dataset)
        # save_mead(embeddings,dataset)
        # save_emodb(embeddings,dataset)
        # save_cremad(embeddings,dataset)
    
    def save_model(self,epoch,type='last_epoch'):
        if self.write:
            save_name = "model_{}.pth".format(type)
            torch.save({'model_state_dict':self.clf.state_dict(),
                        'optimizer_state_dict': self.opt.state_dict(),
                        'epoch': epoch,}, 
                        os.path.join(self.config.save_path,save_name))
    
    def load_model(self):
        # load the saved model
        checkpoint = torch.load(self.config.load_path)
        self.clf.load_state_dict(checkpoint['model_state_dict'])

def load_cremad():
    with open("/dataHDD/ezhou12/CREMA-D/audio_dataset.pickle","rb") as f:
        dataset = pickle.load(f)
    return dataset
def save_cremad(embeddings,dataset):
    save_data = {"embeddings":embeddings,"emotion":dataset['emotion'],"speaker": dataset['speaker'], "utterance": dataset['utterance']}
    with open("/dataHDD/ezhou12/CREMA-D/embeddings.pickle","wb") as f:
        pickle.dump(save_data,f)
def load_msp():
    with open("/dataHDD/ezhou12/MSP/audio_full_dataset.pickle","rb") as f:
        dataset = pickle.load(f)
    return dataset
def load_mead():
    with open("/dataHDD/ezhou12/MEAD/audio_dataset.pickle", "rb") as f:
        dataset = pickle.load(f)
    return dataset
def load_emodb():
    with open("/dataHDD/ezhou12/EMODB/audio_dataset.pickle", "rb") as f:
        dataset = pickle.load(f)
    return dataset
def save_emodb(embeddings,dataset):
    save_data = {"embeddings":embeddings,"emotion":dataset['emotion'],"speaker": dataset['speaker'], "utterance": dataset['utterance']}
    with open("/dataHDD/ezhou12/EMODB/embeddings.pickle","wb") as f:
        pickle.dump(save_data,f)
def save_mead(embeddings,dataset):
    save_data = {"embeddings":embeddings,"emotion":dataset['emotion'],"intensity":dataset['intensity'], "speaker": dataset['speaker'], "index": dataset['index']}
    with open("/dataHDD/ezhou12/MEAD/embeddings.pickle","wb") as f:
        pickle.dump(save_data,f) 
def save_msp(embeddings,dataset):
    save_data = {"embeddings":embeddings,"emotion":dataset['emotion'],"V": dataset['V'], "A": dataset['A'],"D":dataset['D']}
    with open("/dataHDD/ezhou12/MSP/embeddings.pickle","wb") as f:
        pickle.dump(save_data,f) 
def save_msp_partial(embeddings, status,dataset):
    save_data = {"embeddings":embeddings,"emotion":dataset['emotion'],"V": dataset['V'], "A": dataset['A'],"D":dataset['D'], "status":status}
    with open("/home/enting/Documents/EmoDR/data/msp-podcast/embeddings.pickle","wb") as f:
        pickle.dump(save_data,f) 
def load_iemocap():
    with open("/dataHDD/ezhou12/IEMOCAP/IEMOCAP_full_release/audio_full_dataset.pickle","rb") as f:
        dataset = pickle.load(f)
    return dataset
def save_iemocap(embeddings,status,dataset):
    save_data = {"embeddings":embeddings,"emotion":dataset['emotion'],"V": dataset['V'], "A": dataset['A'],"D":dataset['D'],"status":status}
    with open("/home/enting/Documents/EmoDR/data/embeddings_spk.pickle","wb") as f:
        pickle.dump(save_data,f)  
def save_iemocap_binary(embeddings,status,dataset):
    save_data = {"embeddings":embeddings,"emotion":dataset['emotion'],"V": dataset['V'], "A": dataset['A'],"D":dataset['D'],"status":status}
    with open("/home/enting/Documents/EmoDR/data/embeddings_spk_binary.pickle","wb") as f:
        pickle.dump(save_data,f)     
def save_iemocap_full(embeddings,status,dataset):
    save_data = {"embeddings":embeddings,"emotion":dataset['emotion'],"V": dataset['V'], "A": dataset['A'],"D":dataset['D'],"status":status}
    with open("/home/enting/Documents/EmoDR/data/embeddings_spk_full.pickle","wb") as f:
        pickle.dump(save_data,f)    
def save_iemocap_partial(dump_path,embeddings,status,pred,dataset):
    save_data = {"embeddings":embeddings,"emotion":dataset['emotion'], "pred_emotion": pred,
                 "V": dataset['V'], "A": dataset['A'],"D":dataset['D'],"status":status}
    with open(os.path.join(dump_path, "embeddings.pickle"),"wb") as f:
        pickle.dump(save_data,f)


if __name__ == "__main__":
    # reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    set_seed(42)

    # parse training argument
    parser = argparse.ArgumentParser(description='Train MLP for multiclass classification')

    # Add arguments
    parser.add_argument('--name',type=str, default='tmp', help="folder name to store the model file")
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate for optimizer')
    parser.add_argument('--reg-lr', type=float, default=1e-6, help='learning rate for optimizer')
    parser.add_argument('--num-epochs', type=int, default=10, help='number of epochs for training')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size for training')

    parser.add_argument('--data', type=str, default='/dataHDD/ezhou12/CREMA-D/audio_dataset.pickle', help='path to training data')
    parser.add_argument('--num-labels', type=int, default=6, help='number of categories in data')
    parser.add_argument('--save-path', type=str, default='/dataHDD/ezhou12/dump/', help='path to save trained model')
    # parser.add_argument('--load-path', type=str, default='/dataHDD/ezhou12/dump/crema/model_epoch_20.pth', help='path to load pretrained model')
    parser.add_argument('--load-path', type=str, default='', help='path to load pretrained model') 

    parser.add_argument('--device', type=str, default='cuda:0', help='running device')
    parser.add_argument('--num-gpus', type=int, default=4, help='number of gpu to train on')

    parser.add_argument('--use-wandb', action='store_true')
    parser.add_argument('--wandb-name', type=str, default='tmp', help='wandb name')
    parser.add_argument('--write', action='store_true')
    parser.add_argument('--mode', type=str, default='eval', help='running mode: train/eval/inference')

    args = parser.parse_args() 
    args.save_path = os.path.join(args.save_path, args.name)

    if not os.path.exists(args.save_path):
        print(f"save path {args.save_path} not exist, creating...")
        os.mkdir(args.save_path) 

    trainer = Trainer(args)
   
    if args.mode=='train':
        trainer.train()  
    elif args.mode=='eval':
        trainer.eval()
    else:
        trainer.inference()                                     




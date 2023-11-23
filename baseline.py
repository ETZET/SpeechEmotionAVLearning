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
    audios, arousal, valence = [], [], []
    for data in batch:
        au, a, v = data['audio'], data['A'], data['V']
        audios.append(au['array'])
        arousal.append(a)
        valence.append(v)
    audios = feature_extractor(audios, sampling_rate=16000, return_tensors="pt", padding=True).to(device)
    return {"audio": audios,"arousal": arousal, "valence": valence}

def CCC_loss(x, y):
    cov = torch.cov(torch.stack((x,y), dim=0), correction = 0)[0][1]
    ccc = 1.0 - (2.0 * cov) / (x.var(correction=0) + y.var(correction=0) + (x.mean() - y.mean())**2)
    return ccc

def CCC_loss_np(x, y):
    cov = np.cov(x,y)[0][1]
    ccc = 1.0 - (2.0 * cov) / (np.var(x) + np.var(y) + (np.mean(x) - np.mean(y))**2)
    return ccc


class EmotionClassifier(nn.Module):
    def __init__(self, layer_num, emb_dim, num_labels, hidden_dim=100):
        super().__init__()
        self.layer_num = layer_num
        self.emb_dim = emb_dim

        self.weights = nn.Parameter(torch.randn(layer_num))
        self.proj = nn.Linear(emb_dim,hidden_dim)
        self.a_out = nn.Linear(hidden_dim, 1)
        self.v_out = nn.Linear(hidden_dim, 1)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.xavier_uniform_(self.a_out.weight)
        nn.init.xavier_uniform_(self.v_out.weight)
    
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

        a_pred = self.a_out(final_emb)
        v_pred = self.v_out(final_emb)

        return a_pred, v_pred

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
        self.best_ccc = float('-inf')

        self.loss = CCC_loss

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
        alpha, beta = config.alpha, config.beta
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
        
        a_running_loss = 0.0
        v_running_loss = 0.0

        for i,batch in enumerate(dataloader):

            input = batch['audio']
            # upstream inference
            outputs = self.wavlm(**input,output_hidden_states=True)
            # downstream
            hiddens = outputs.hidden_states
            feature_length = self.get_feature_seq_length(input['attention_mask'])
            a_pred, v_pred = self.clf(hiddens, feature_length)

            a_pred, v_pred = a_pred.squeeze(dim=1), v_pred.squeeze(dim=1)
            a_gt, v_gt = torch.tensor(batch['arousal'], dtype=torch.float32, device=device), torch.tensor(batch['valence'], dtype=torch.float32, device=device) 
            a_loss, v_loss = self.loss(a_pred, a_gt), self.loss(v_pred, v_gt)
            loss = alpha * a_loss + beta * v_loss

            if is_training:
                loss.backward()
                if (i + 1) % self.config.accumulation_steps == 0 or (i + 1 == len(dataloader)):
                    opt.step()
                    opt.zero_grad()
            
            # statistics 
            a_running_loss += a_loss.item()
            v_running_loss += v_loss.item()

            if i % 100 ==0:
                print(f"Epoch {epoch}, batch {i}: a_loss: {a_loss.item()}, v_loss: {v_loss.item()}")
        
        a_epoch_loss = a_running_loss / len(dataloader.dataset)
        v_epoch_loss = v_running_loss / len(dataloader.dataset)

        print('Epoch: {:d} {} A Loss: {:.4f}, V Loss: {:.4f}'.format(epoch, status, a_epoch_loss, v_epoch_loss))
                
        #logging
        if self.config.use_wandb:
            wandb.log({f"{status} A Loss": a_epoch_loss, f"{status} V Loss": v_epoch_loss, "epoch": epoch})
        
        # model checking
        if epoch % 10 == 0:
            self.save_model(epoch,f"epoch_{epoch}")
        self.save_model(epoch)
    
    def train(self):
        for epoch in range(self.config.num_epochs):
            self.train_pass(epoch)
            if epoch % 2 == 0:
                with torch.no_grad():
                    accc, vccc = self.eval()
                    print('Epoch: {:d} {} A CCC: {:.4f}, V CCC: {:.4f}'.format(epoch, 'EVAL', accc, vccc))
                    if accc + vccc > self.best_ccc:
                        self.best_ccc = accc + vccc
                        self.save_model(epoch, 'best')


    def eval(self):
        opt = self.opt
        dataloader = self.val_dataloader
        status = 'EVAL'

        # freeze upstream wavlm
        for p in self.wavlm.parameters():
            p.requires_grad = False
        
        v_predictions, a_predictions= [], []
        v_gts, a_gts = [], []
        a_running_loss = 0.0
        v_running_loss = 0.0

        for batch in dataloader:
            input = batch['audio']
            # upstream inference
            outputs = self.wavlm(**input,output_hidden_states=True)
            # downstream
            hiddens = outputs.hidden_states
            feature_length = self.get_feature_seq_length(input['attention_mask'])
            a_pred, v_pred = self.clf(hiddens, feature_length)
            a_pred, v_pred = a_pred.squeeze(dim=1), v_pred.squeeze(dim=1)

            a_gt, v_gt = torch.tensor(batch['arousal'], dtype=torch.float32, device=device), torch.tensor(batch['valence'], dtype=torch.float32, device=device) 
            a_loss, v_loss = self.loss(a_pred, a_gt), self.loss(v_pred, v_gt)

            v_predictions.append(v_pred.detach().cpu().numpy())
            a_predictions.append(a_pred.detach().cpu().numpy())
            v_gts.append(v_gt.cpu().numpy())
            a_gts.append(a_gt.cpu().numpy())
            a_running_loss += a_loss.item()
            v_running_loss += v_loss.item()

        a_predictions, v_predictions = np.concatenate(a_predictions), np.concatenate(v_predictions)
        a_gts, v_gts = np.concatenate(a_gts), np.concatenate(v_gts)

        # compute metric
        acc = audmetric.concordance_cc(a_gts, a_predictions)
        vcc = audmetric.concordance_cc(v_gts, v_predictions)

        return acc, vcc

    def test(self):
        opt = self.opt
        dataloader = self.test_dataloader

        # freeze upstream wavlm
        for p in self.wavlm.parameters():
            p.requires_grad = False
        
        v_predictions, a_predictions= [], []
        v_gts, a_gts = [], []
        a_running_loss = 0.0
        v_running_loss = 0.0

        for batch in dataloader:

            input = batch['audio']
            # upstream inference
            outputs = self.wavlm(**input,output_hidden_states=True)
            # downstream
            hiddens = outputs.hidden_states
            feature_length = self.get_feature_seq_length(input['attention_mask'])
            a_pred, v_pred = self.clf(hiddens, feature_length)
            a_pred, v_pred = a_pred.squeeze(dim=1), v_pred.squeeze(dim=1)

            a_gt, v_gt = torch.tensor(batch['arousal'], dtype=torch.float32, device=device), torch.tensor(batch['valence'], dtype=torch.float32, device=device) 
            a_loss, v_loss = self.loss(a_pred, a_gt), self.loss(v_pred, v_gt)

            v_predictions.append(v_pred.detach().cpu().numpy())
            a_predictions.append(a_pred.detach().cpu().numpy())
            v_gts.append(v_gt.cpu().numpy())
            a_gts.append(a_gt.cpu().numpy())
            a_running_loss += a_loss.item()
            v_running_loss += v_loss.item()

        a_predictions, v_predictions = np.concatenate(a_predictions), np.concatenate(v_predictions)
        a_gts, v_gts = np.concatenate(a_gts), np.concatenate(v_gts)

        # compute metric
        acc = audmetric.concordance_cc(a_gts, a_predictions)
        vcc = audmetric.concordance_cc(v_gts, v_predictions)
        amae = audmetric.mean_absolute_error(a_gts, a_predictions)
        vmae = audmetric.mean_absolute_error(v_gts, v_predictions)

        a_epoch_loss = a_running_loss / len(dataloader.dataset)
        v_epoch_loss = v_running_loss / len(dataloader.dataset)

        a_epoch_loss = CCC_loss_np(a_predictions, a_gts)
        v_epoch_loss = CCC_loss_np(v_predictions, v_gts)

        print(f"acc: {acc}, vcc: {vcc}, amae: {amae}, vmae: {vmae}, aloss: {a_epoch_loss}, vloss: {v_epoch_loss}")
            
        
    
    def inference(self):

        # merge train test dataset
        dataset = concatenate_datasets([self.train_data,self.val_data])
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, collate_fn=my_collate)

        # freeze upstream wavlm
        for p in self.wavlm.parameters():
            p.requires_grad = False
        self.wavlm.eval()
        self.clf.eval()

        v_predictions, a_predictions= [], []
        v_gts, a_gts = [], []

        for batch in tqdm(dataloader):

            input = batch['audio']
            # upstream inference
            outputs = self.wavlm(**input,output_hidden_states=True)
            # downstream
            hiddens = outputs.hidden_states
            feature_length = self.get_feature_seq_length(input['attention_mask'])
            a_pred, v_pred = self.clf(hiddens, feature_length)
            a_pred, v_pred = a_pred.squeeze(dim=1), v_pred.squeeze(dim=1)

            a_gt, v_gt = torch.tensor(batch['arousal'], dtype=torch.float32, device=device), torch.tensor(batch['valence'], dtype=torch.float32, device=device) 

            v_predictions.append(v_pred.detach().cpu().numpy())
            a_predictions.append(a_pred.detach().cpu().numpy())
            v_gts.append(v_gt.cpu().numpy())
            a_gts.append(a_gt.cpu().numpy())

        a_predictions, v_predictions = np.concatenate(a_predictions), np.concatenate(v_predictions)
        a_gts, v_gts = np.concatenate(a_gts), np.concatenate(v_gts)

        # compute metric
        acc = audmetric.concordance_cc(a_gts, a_predictions)
        vcc = audmetric.concordance_cc(v_gts, v_predictions)
        amae = audmetric.mean_absolute_error(a_gts, a_predictions)
        vmae = audmetric.mean_absolute_error(v_gts, v_predictions)

        a_epoch_loss = CCC_loss_np(a_predictions, a_gts)
        v_epoch_loss = CCC_loss_np(v_predictions, v_gts)


        print(f"acc: {acc}, vcc: {vcc}, amae: {amae}, vmae: {vmae}, aloss: {a_epoch_loss}, vloss: {v_epoch_loss}")

        status = ["train"] * len(self.train_data) + ["val"] * len(self.val_data)
        status = np.array(status)

        # do something
        save_data = {"predicted_V":v_predictions,"predicted_A":a_predictions,"emotion":dataset['emotion'],"V": dataset['V'], "A": dataset['A'],"D":dataset['D'], "status":status}
        with open(os.path.join(self.config.save_path, "prediction.pickle"),"wb") as f:
            pickle.dump(save_data,f) 

    
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
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate for optimizer')
    parser.add_argument('--reg-lr', type=float, default=0, help='learning rate for optimizer')
    parser.add_argument('--alpha', type=float, default=1.00, help='weight for arousal loss')
    parser.add_argument('--beta', type=float, default=1.00, help='weight for valence loss')
    parser.add_argument('--num-epochs', type=int, default=10, help='number of epochs for training')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size for training')
    parser.add_argument('--accumulation-steps', type=int, default=1, help='accumulation step')

    parser.add_argument('--data', type=str, default='/dataHDD/ezhou12/CREMA-D/audio_dataset.pickle', help='path to training data')
    parser.add_argument('--num-labels', type=int, default=6, help='number of categories in data')
    parser.add_argument('--save-path', type=str, default='/dataHDD/ezhou12/dump/', help='path to save trained model')
    # parser.add_argument('--load-path', type=str, default='/dataHDD/ezhou12/dump/crema/model_epoch_20.pth', help='path to load pretrained model')
    parser.add_argument('--load-path', type=str, default='', help='path to load pretrained model') 

    parser.add_argument('--device', type=str, default='cuda:0', help='running device')
    parser.add_argument('--num-gpus', type=int, default=1, help='number of gpu to train on')

    parser.add_argument('--use-wandb', action='store_true')
    parser.add_argument('--wandb-name', type=str, default='tmp', help='wandb name')
    parser.add_argument('--write', action='store_true')
    parser.add_argument('--mode', type=str, default='eval', help='running mode: train/eval/inference')

    args = parser.parse_args() 
    args.save_path = os.path.join(args.save_path, args.name)

    # debug
    args.mode = 'train'
    args.load_path = "/home/enting/Documents/EmoDR/dump/iemocap_baseline_partial/model_last_epoch.pth"
    args.save_path = "/home/enting/Documents/EmoDR/dump"
    args.data = "/home/enting/Documents/EmoDR/data/IEMOCAP_full_release/audio_partial_train_dataset.pickle"
    args.batch_size = 8

    if not os.path.exists(args.save_path):
        print(f"save path {args.save_path} not exist, creating...")
        os.mkdir(args.save_path) 

    trainer = Trainer(args)
   
    if args.mode=='train':
        trainer.train()  
    elif args.mode=='eval':
        acc, vcc = trainer.eval()
        print(acc, vcc)
    elif args.mode=='test':
        trainer.test()
    else:
        trainer.inference()                                     




from datasets import Dataset, Audio
from glob import glob
import librosa
import soundfile as sf
import pickle
import os
import warnings
from tqdm import tqdm 
import json
import numpy as np
warnings.simplefilter('ignore')

#######################
#       MEAD
#######################
    
def create_data_dictionary_MEAD(data_path):
    audios, speakers, emotions, intensities,indices = [], [], [], [], []
    for file in tqdm(glob(data_path+"/*.wav")):
        # this preprocessing is specified for MEAD video dataset
        file_arr = os.path.basename(file)[:-4].split("_")
        speaker, emo, intensity, index = file_arr[0], file_arr[1], file_arr[2], file_arr[3]
        audios.append(file)
        speakers.append(speaker)
        emotions.append(emo)
        intensities.append(intensity)
        indices.append(index)
    return {"audio":audios, "speaker": speakers, "emotion": emotions, "intensity": intensities, "index": indices}

def create_dataset_MEAD(data_dict):
    dataset = Dataset.from_dict(data_dict).cast_column("audio", Audio())
    dataset = dataset.class_encode_column('emotion')
    return 

def MEAD_train_test_split(dataset):
    # Get the speaker indexes 
    preprocessed_dir = "/dataHDD/ezhou12/MEAD/preprocessed"
    speakear_indexes = []
    for folder in glob(preprocessed_dir+"/*"):
        folder_name = os.path.basename(folder)
        speakear_indexes.append(folder_name)

    speakear_indexes.sort()

    train_spk,test_spk = train_test_split(speakear_indexes, test_size=0.4, random_state=0)
    train = dataset.filter(lambda row: row['speaker'] in train_spk)
    test = dataset.filter(lambda row: row['speaker'] in test_spk)

    return {'train':train, 'test': test}

def save_MEAD():
    data_path = "/dataHDD/ezhou12/MEAD/wav"
    data_dict = create_data_dictionary_MEAD(data_path)
    dataset = create_dataset_MEAD(data_dict)
    with open("/dataHDD/ezhou12/MEAD/audio_dataset.pickle","wb") as f:
        pickle.dump(dataset,f)

    train_dataset = train_test_split(dataset)
    with open("/dataHDD/ezhou12/MEAD/audio_train_dataset.pickle","wb") as f:
        pickle.dump(train_dataset,f)

#######################
#       EMODB
#######################

def create_data_dictionary_EMODB(data_path):
    files, speakers, emotions, uttrs = [], [], [], []
    mapping = {"W":"angry","L":"boredom","E":"disgust","A":"fear","F":"happy","T":"sad", "N": "neutral"}
    for file in glob(data_path+"/*.wav"):
        file_arr = list(os.path.basename(file)[:-4])
        speaker, uttr, emo = file_arr[0:2], file_arr[2:5], file_arr[5]
        files.append(file)
        speakers.append("".join(speaker))
        emotions.append("".join(emo))
        uttrs.append("".join(uttr))
    emotions = [mapping[emo] for emo in emotions]
    return {"audio":files, "speaker": speakers, "emotion": emotions, "utterance":uttrs}

def create_dataset_EMODB(data_dict):
    dataset = Dataset.from_dict(data_dict).cast_column("audio", Audio())
    dataset = dataset.class_encode_column('emotion')
    return dataset

def EMODB_train_test_split(dataset):
    train = dataset.filter(lambda row: row['speaker'] !='15' and row['speaker']!='16')
    test = dataset.filter(lambda row: row['speaker'] =='15' and row['speaker']=='16')
    return {'train':train, 'test': test}

def save_EMODB():
    data_path = "/dataHDD/ezhou12/EMODB/wav"
    data_dict = create_data_dictionary_EMODB(data_path)
    dataset = create_dataset_EMODB(data_dict)
    with open("/dataHDD/ezhou12/EMODB/audio_dataset.pickle","wb") as f:
        pickle.dump(dataset,f)

    train_dataset = EMODB_train_test_split(dataset)
    with open("/dataHDD/ezhou12/EMODB/audio_train_dataset.pickle","wb") as f:
        pickle.dump(train_dataset,f)

#######################
#       Utils
#######################

def train_test_split(dataset):
    return dataset.train_test_split(test_size=0.2,stratify_by_column='emotion',shuffle=True,seed=42)

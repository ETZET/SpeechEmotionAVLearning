# this code adapted from https://github.com/s3prl/s3prl/blob/master/s3prl/downstream/emotion/IEMOCAP_preprocess.py

import os
from os.path import basename, splitext, join as path_join
import re
import numpy as np
import pickle
from librosa.util import find_files
from datasets import Dataset, Audio

LABEL_DIR_PATH = 'dialog/EmoEvaluation'
WAV_DIR_PATH = 'sentences/wav'

def get_wav_paths(data_dirs):
    wav_paths = find_files(data_dirs)
    wav_dict = {}
    for wav_path in wav_paths:
        wav_name = splitext(basename(wav_path))[0]
        start = wav_path.find('Session')
        wav_path = wav_path[start:]
        wav_dict[wav_name] = wav_path

    return wav_dict


def preprocess(data_dirs, paths):
    files, session, emotions, valence, arousal, dominance = [], [], [], [], [], []
    for path in paths:
        wav_paths = get_wav_paths(path_join(data_dirs, path, WAV_DIR_PATH))
        label_dir = path_join(data_dirs, path, LABEL_DIR_PATH)
        label_paths = list(os.listdir(label_dir))
        label_paths = [label_path for label_path in label_paths
                       if splitext(label_path)[1] == '.txt']
        for label_path in label_paths:
            # escape hidden files:
            if label_path.startswith("."):
                continue
            with open(path_join(label_dir, label_path)) as f:
                print(path_join(label_dir, label_path))
                for line in f:
                    if line[0] != '[':
                        continue
                    line = re.split('[\t\n]', line)
                    line = list(filter(None, line))
                    if line[2] not in ['neu', 'hap', 'ang', 'sad', 'exc']:
                        continue
                    if line[1] not in wav_paths:
                        continue
                    # append vad value
                    vad = line[3].strip('[]').split(', ')
                    vad = [float(e) for e in vad]
                    files.append(path_join(data_dirs,wav_paths[line[1]]))
                    session.append(path)
                    emotions.append(line[2].replace('exc', 'hap'))
                    valence.append(vad[0])
                    arousal.append(vad[1])
                    dominance.append(vad[2])

    # normalize the values
    valence, arousal, dominance = np.array(valence), np.array(arousal), np.array(dominance)
    valence = 2 * (valence - min(valence)) / (max(valence) - min(valence)) - 1
    arousal = 2 * (arousal - min(arousal)) / (max(arousal) - min(arousal)) - 1
    dominance = 2 * (dominance - min(dominance)) / (max(dominance) - min(dominance)) - 1
    valence, arousal, dominance = list(valence), list(arousal), list(dominance)

    data_dict = {"audio":files, "emotion": emotions, "session":session, "V":valence, "A":arousal, "D":dominance}
    dataset = Dataset.from_dict(data_dict).cast_column("audio", Audio())
    dataset = dataset.class_encode_column('emotion')
    return dataset 

def partial_preprocess(data_dirs, paths):
    files, session, emotions, valence, arousal, dominance = [], [], [], [], [], []
    for path in paths:
        wav_paths = get_wav_paths(path_join(data_dirs, path, WAV_DIR_PATH))
        label_dir = path_join(data_dirs, path, LABEL_DIR_PATH)
        label_paths = list(os.listdir(label_dir))
        label_paths = [label_path for label_path in label_paths
                       if splitext(label_path)[1] == '.txt']
        for label_path in label_paths:
            # escape hidden files:
            if label_path.startswith("."):
                continue
            with open(path_join(label_dir, label_path)) as f:
                print(path_join(label_dir, label_path))
                for line in f:
                    if line[0] != '[':
                        continue
                    line = re.split('[\t\n]', line)
                    line = list(filter(None, line))
                    if line[2] in ['oth', 'xxx']:
                        continue
                    if line[1] not in wav_paths:
                        continue
                    # append vad value
                    vad = line[3].strip('[]').split(', ')
                    vad = [float(e) for e in vad]
                    files.append(path_join(data_dirs,wav_paths[line[1]]))
                    session.append(path)
                    emotions.append(line[2])
                    valence.append(vad[0])
                    arousal.append(vad[1])
                    dominance.append(vad[2])

    valence, arousal, dominance = np.array(valence), np.array(arousal), np.array(dominance)
    valence = 2 * (valence - min(valence)) / (max(valence) - min(valence)) - 1
    arousal = 2 * (arousal - min(arousal)) / (max(arousal) - min(arousal)) - 1
    dominance = 2 * (dominance - min(dominance)) / (max(dominance) - min(dominance)) - 1
    valence, arousal, dominance = list(valence), list(arousal), list(dominance)

    data_dict = {"audio":files, "emotion": emotions, "session":session, "V":valence, "A":arousal, "D":dominance}

    dataset = Dataset.from_dict(data_dict).cast_column("audio", Audio())
    dataset = dataset.class_encode_column('emotion')
    return dataset 

def partial4_preprocess(data_dirs, paths):
    files, session, emotions, valence, arousal, dominance = [], [], [], [], [], []
    for path in paths:
        wav_paths = get_wav_paths(path_join(data_dirs, path, WAV_DIR_PATH))
        label_dir = path_join(data_dirs, path, LABEL_DIR_PATH)
        label_paths = list(os.listdir(label_dir))
        label_paths = [label_path for label_path in label_paths
                       if splitext(label_path)[1] == '.txt']
        for label_path in label_paths:
            # escape hidden files:
            if label_path.startswith("."):
                continue
            with open(path_join(label_dir, label_path)) as f:
                print(path_join(label_dir, label_path))
                for line in f:
                    if line[0] != '[':
                        continue
                    line = re.split('[\t\n]', line)
                    line = list(filter(None, line))
                    if line[2] not in ['ang', 'hap', 'neu', 'sad']:
                        continue
                    if line[1] not in wav_paths:
                        continue
                    # append vad value
                    vad = line[3].strip('[]').split(', ')
                    vad = [float(e) for e in vad]
                    files.append(path_join(data_dirs,wav_paths[line[1]]))
                    session.append(path)
                    emotions.append(line[2])
                    valence.append(vad[0])
                    arousal.append(vad[1])
                    dominance.append(vad[2])

    valence, arousal, dominance = np.array(valence), np.array(arousal), np.array(dominance)
    valence = 2 * (valence - min(valence)) / (max(valence) - min(valence)) - 1
    arousal = 2 * (arousal - min(arousal)) / (max(arousal) - min(arousal)) - 1
    dominance = 2 * (dominance - min(dominance)) / (max(dominance) - min(dominance)) - 1
    valence, arousal, dominance = list(valence), list(arousal), list(dominance)

    data_dict = {"audio":files, "emotion": emotions, "session":session, "V":valence, "A":arousal, "D":dominance}

    dataset = Dataset.from_dict(data_dict).cast_column("audio", Audio())
    dataset = dataset.class_encode_column('emotion')
    return dataset 

def partial5_preprocess(data_dirs, paths):
    files, session, emotions, valence, arousal, dominance = [], [], [], [], [], []
    for path in paths:
        wav_paths = get_wav_paths(path_join(data_dirs, path, WAV_DIR_PATH))
        label_dir = path_join(data_dirs, path, LABEL_DIR_PATH)
        label_paths = list(os.listdir(label_dir))
        label_paths = [label_path for label_path in label_paths
                       if splitext(label_path)[1] == '.txt']
        for label_path in label_paths:
            # escape hidden files:
            if label_path.startswith("."):
                continue
            with open(path_join(label_dir, label_path)) as f:
                print(path_join(label_dir, label_path))
                for line in f:
                    if line[0] != '[':
                        continue
                    line = re.split('[\t\n]', line)
                    line = list(filter(None, line))
                    if line[2] not in ['ang', 'dis', 'hap', 'neu', 'sad']:
                        continue
                    if line[1] not in wav_paths:
                        continue
                    # append vad value
                    vad = line[3].strip('[]').split(', ')
                    vad = [float(e) for e in vad]
                    files.append(path_join(data_dirs,wav_paths[line[1]]))
                    session.append(path)
                    emotions.append(line[2])
                    valence.append(vad[0])
                    arousal.append(vad[1])
                    dominance.append(vad[2])

    valence, arousal, dominance = np.array(valence), np.array(arousal), np.array(dominance)
    valence = 2 * (valence - min(valence)) / (max(valence) - min(valence)) - 1
    arousal = 2 * (arousal - min(arousal)) / (max(arousal) - min(arousal)) - 1
    dominance = 2 * (dominance - min(dominance)) / (max(dominance) - min(dominance)) - 1
    valence, arousal, dominance = list(valence), list(arousal), list(dominance)

    data_dict = {"audio":files, "emotion": emotions, "session":session, "V":valence, "A":arousal, "D":dominance}

    dataset = Dataset.from_dict(data_dict).cast_column("audio", Audio())
    dataset = dataset.class_encode_column('emotion')
    return dataset 

def all_preprocess(data_dirs, paths):
    files, session, emotions, valence, arousal, dominance = [], [], [], [], [], []
    for path in paths:
        wav_paths = get_wav_paths(path_join(data_dirs, path, WAV_DIR_PATH))
        label_dir = path_join(data_dirs, path, LABEL_DIR_PATH)
        label_paths = list(os.listdir(label_dir))
        label_paths = [label_path for label_path in label_paths
                       if splitext(label_path)[1] == '.txt']
        for label_path in label_paths:
            # escape hidden files:
            if label_path.startswith("."):
                continue
            with open(path_join(label_dir, label_path)) as f:
                print(path_join(label_dir, label_path))
                for line in f:
                    if line[0] != '[':
                        continue
                    line = re.split('[\t\n]', line)
                    line = list(filter(None, line))
                    if line[1] not in wav_paths:
                        continue
                    # append vad value
                    vad = line[3].strip('[]').split(', ')
                    vad = [float(e) for e in vad]
                    files.append(path_join(data_dirs,wav_paths[line[1]]))
                    session.append(path)
                    emotions.append(line[2])
                    valence.append(vad[0])
                    arousal.append(vad[1])
                    dominance.append(vad[2])

    valence, arousal, dominance = np.array(valence), np.array(arousal), np.array(dominance)
    valence = 2 * (valence - min(valence)) / (max(valence) - min(valence)) - 1
    arousal = 2 * (arousal - min(arousal)) / (max(arousal) - min(arousal)) - 1
    dominance = 2 * (dominance - min(dominance)) / (max(dominance) - min(dominance)) - 1
    valence, arousal, dominance = list(valence), list(arousal), list(dominance)

    data_dict = {"audio":files, "emotion": emotions, "session":session, "V":valence, "A":arousal, "D":dominance}

    dataset = Dataset.from_dict(data_dict).cast_column("audio", Audio())
    dataset = dataset.class_encode_column('emotion')
    return dataset 

def binary_preprocess(data_dirs, paths):
    files, session, emotions, valence, arousal, dominance = [], [], [], [], [], []
    for path in paths:
        wav_paths = get_wav_paths(path_join(data_dirs, path, WAV_DIR_PATH))
        label_dir = path_join(data_dirs, path, LABEL_DIR_PATH)
        label_paths = list(os.listdir(label_dir))
        label_paths = [label_path for label_path in label_paths
                       if splitext(label_path)[1] == '.txt']
        for label_path in label_paths:
            # escape hidden files:
            if label_path.startswith("."):
                continue
            with open(path_join(label_dir, label_path)) as f:
                print(path_join(label_dir, label_path))
                for line in f:
                    if line[0] != '[':
                        continue
                    line = re.split('[\t\n]', line)
                    line = list(filter(None, line))
                    if line[2] not in ['hap','sad']:
                        continue
                    if line[1] not in wav_paths:
                        continue
                    # append vad value
                    vad = line[3].strip('[]').split(', ')
                    vad = [float(e) for e in vad]
                    files.append(path_join(data_dirs,wav_paths[line[1]]))
                    session.append(path)
                    emotions.append(line[2])
                    valence.append(vad[0])
                    arousal.append(vad[1])
                    dominance.append(vad[2])

    valence, arousal, dominance = np.array(valence), np.array(arousal), np.array(dominance)
    valence = 2 * (valence - min(valence)) / (max(valence) - min(valence)) - 1
    arousal = 2 * (arousal - min(arousal)) / (max(arousal) - min(arousal)) - 1
    dominance = 2 * (dominance - min(dominance)) / (max(dominance) - min(dominance)) - 1
    valence, arousal, dominance = list(valence), list(arousal), list(dominance)

    data_dict = {"audio":files, "emotion": emotions, "session":session, "V":valence, "A":arousal, "D":dominance}

    dataset = Dataset.from_dict(data_dict).cast_column("audio", Audio())
    dataset = dataset.class_encode_column('emotion')
    return dataset 

def binary_main(data_dir):
    """Main function."""
    paths = list(os.listdir(data_dir))
    paths = [path for path in paths if path[:7] == 'Session']
    paths.sort()
    out_dir = os.path.join(data_dir, 'meta_data')
    os.makedirs(out_dir, exist_ok=True)
    dataset = binary_preprocess(data_dir,paths)
    train_dataset = SI_train_test_spilt(dataset)

    with open(path_join(data_dir,"audio_binary_full_dataset.pickle"), "wb") as f:
        pickle.dump(dataset, f)

    with open(path_join(data_dir,"audio_binary_train_dataset.pickle"), "wb") as f:
        pickle.dump(train_dataset, f)

def partial_main(data_dir):
    """Main function."""
    paths = list(os.listdir(data_dir))
    paths = [path for path in paths if path[:7] == 'Session']
    paths.sort()
    out_dir = os.path.join(data_dir, 'meta_data')
    os.makedirs(out_dir, exist_ok=True)
    dataset = partial_preprocess(data_dir,paths)
    train_dataset = SI_train_test_spilt(dataset)

    with open(path_join(data_dir,"audio_partial_full_dataset.pickle"), "wb") as f:
        pickle.dump(dataset, f)

    with open(path_join(data_dir,"audio_partial_train_dataset.pickle"), "wb") as f:
        pickle.dump(train_dataset, f)

def all_main(data_dir):
    """Main function."""
    paths = list(os.listdir(data_dir))
    paths = [path for path in paths if path[:7] == 'Session']
    paths.sort()
    out_dir = os.path.join(data_dir, 'meta_data')
    os.makedirs(out_dir, exist_ok=True)
    dataset = all_preprocess(data_dir,paths)
    train_dataset = SI_train_test_spilt(dataset)

    with open(path_join(data_dir,"audio_all_full_dataset.pickle"), "wb") as f:
        pickle.dump(dataset, f)

    with open(path_join(data_dir,"audio_all_train_dataset.pickle"), "wb") as f:
        pickle.dump(train_dataset, f)

def partial5_main(data_dir):
    """Main function."""
    paths = list(os.listdir(data_dir))
    paths = [path for path in paths if path[:7] == 'Session']
    paths.sort()
    out_dir = os.path.join(data_dir, 'meta_data')
    os.makedirs(out_dir, exist_ok=True)
    dataset = partial5_preprocess(data_dir,paths)
    train_dataset = SI_train_test_spilt(dataset)

    with open(path_join(data_dir,"audio_partial5_full_dataset.pickle"), "wb") as f:
        pickle.dump(dataset, f)

    with open(path_join(data_dir,"audio_partial5_train_dataset.pickle"), "wb") as f:
        pickle.dump(train_dataset, f)

def partial4_main(data_dir):
    """Main function."""
    paths = list(os.listdir(data_dir))
    paths = [path for path in paths if path[:7] == 'Session']
    paths.sort()
    out_dir = os.path.join(data_dir, 'meta_data')
    os.makedirs(out_dir, exist_ok=True)
    dataset = partial4_preprocess(data_dir,paths)
    train_dataset = SI_train_test_spilt(dataset)

    with open(path_join(data_dir,"audio_partial4_full_dataset.pickle"), "wb") as f:
        pickle.dump(dataset, f)

    with open(path_join(data_dir,"audio_partial4_train_dataset.pickle"), "wb") as f:
        pickle.dump(train_dataset, f)

def SI_train_test_spilt(dataset):
    # leave the fifth session out
    train = dataset.filter(lambda row: row['session'] !='Session5' and row['session']!='Session4')
    val = dataset.filter(lambda row: row['session'] == 'Session4')
    test = dataset.filter(lambda row: row['session'] =='Session5')
    return {'train':train, 'val': val, 'test': test}


def main(data_dir):
    """Main function."""
    paths = list(os.listdir(data_dir))
    paths = [path for path in paths if path[:7] == 'Session']
    paths.sort()
    out_dir = os.path.join(data_dir, 'meta_data')
    os.makedirs(out_dir, exist_ok=True)
    dataset = preprocess(data_dir,paths)
    train_dataset = SI_train_test_spilt(dataset)

    with open(path_join(data_dir,"audio_full_dataset.pickle"), "wb") as f:
        pickle.dump(dataset, f)

    with open(path_join(data_dir,"audio_train_dataset.pickle"), "wb") as f:
        pickle.dump(train_dataset, f)


if __name__ == "__main__":
    path = "/home/enting/Documents/EmoDR/data/IEMOCAP_full_release"
    partial4_main(path)
    partial5_main(path)
    partial_main(path)

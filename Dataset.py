import os
import io
import numpy as np
import pandas as pd
from tqdm import tqdm
from random import shuffle, randrange
from torch import tensor, float32, save, load
from torch.utils.data import Dataset, random_split
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
import inspect
import csv

def prepare_HCPRest_timeseries(atlas='schaefer400_sub19'):
    prefix = f"[{inspect.getframeinfo(inspect.currentframe()).function}]"
    timeseries_dir = os.path.join('/u4/surprise/YAD_STAGIN', 'data', 'timeseries')
    sessions = ['REST1', 'REST2']
    phase_encodings = ['RL', 'LR']
    #source_dir = f'/u4/HCP/mean_TS/{atlas.split("_")[0]}-yeo17/Atlas_ROIs.2'
    source_dir=f'/content/drive/My drive/SIG-VAE-GIN-and-fMRI/SIG-VAE-GIN-and-fMRI/fMRI'
    for session in sessions:   ## takes ~1 hr
        for phase in phase_encodings:
            file_list = [file for file in os.listdir(source_dir) if session in file and phase in file ]
            print(f"{session} {phase} is {len(file_list)}")
            timeseries_dict = {}
            for subject in tqdm(file_list, ncols=60):
                id = subject.split('.')[0]
                timeseries = pd.read_csv(os.path.join(source_dir, subject), header=None).to_numpy()  # assumes timesries shape [node x time]
                timeseries_dict[id] = timeseries
            timeseries_file = f'HCP_{atlas}_{session}_{phase}.pth'
            timeseries_path = os.path.join(timeseries_dir, timeseries_file)
            save(timeseries_dict, timeseries_path)
            print(f"{prefix} {timeseries_file} is saved.")
    return

class DatasetHCPRest(Dataset):
    def __init__(self, atlas='schaefer400_sub19', target_feature='Gender', k_fold=None, session='REST1', phase_encoding='LR'):
        prefix = f'[{type(self).__name__}.{inspect.getframeinfo(inspect.currentframe()).function}]'
        super().__init__()
        # argparsing
        self.session = session
        self.phase_encoding = phase_encoding
        self.atlas = atlas

        # setting file path
        base_dir='/content/drive/MyDrive/SIG-VAE-GIN-and-fMRI/SIG-VAE-GIN-and-fMRI'
        source_dir='/content/drive/MyDrive/SIG-VAE-GIN-and-fMRI/SIG-VAE-GIN-and-fMRI/fMRI'
        #base_dir=f'/SIG-VAE-GIN-and-fMRI/SIG-VAE-GIN-and-fMRI'
        #source_dir=f'/SIG-VAE-GIN-and-fMRI/SIG-VAE-GIN-and-fMRI/fMRI'
        label_path = os.path.join(base_dir, 'fMRI', 'HCP_behavior_data.csv')
        #timeseries_dir = os.path.join(base_dir,'data', 'timeseries')
        timeseries_dir=os.path.join(base_dir+'/fMRI')
        timeseries_file = f'HCP_{session}_{phase_encoding}_{atlas}.pth'
        timeseries_path = os.path.join(timeseries_dir, timeseries_file)
        '''
        if not os.path.exists(timeseries_path):    # no cached file --> caching
            #file_list = [file for file in os.listdir(source_dir) if file.endswith(f"{session}_{phase_encoding}.419.csv")]
            file_list = [file for file in os.listdir(source_dir) if file.endswith(f"{session}_{phase_encoding}_{atlas}.pth")]
            print(file_list)
            print(f"{prefix} {session} {phase_encoding} is {len(file_list)}")
            #timeseries_dict={}
            print("$$$$$$$")
            for filename in tqdm(file_list, ncols=10):
                id=filename.split('.')[0]
                timeseries_load=load(os.path.join(source_dir, filename))
            save(timeseries_load, timeseries_path)
                
            print(f"{prefix} {timeseries_file} is saved.")
        '''
        '''
            timeseries_dict = {}
            for filename in tqdm(file_list, ncols=10):
                id = filename.split('.')[0]
                delimiter=','
                #timeseries_dict[id] = pd.read_csv(os.path.join(source_dir, filename), delimiter=delimiter, encoding='ISO-8859-1', header=None).to_numpy() 
                
                f=open(os.path.join(source_dir, filename), 'rb')
                #reader=csv.reader(x.replace('\0', '') for x in f)
                #reader=csv.reader(f)
                with io.TextIOWrapper(f, encoding='latin-1') as text_file:
                    reader = csv.reader((x.replace('\0', '') for x in text_file), delimiter=',')
                    csv_list=[]
                    for i in reader:
                        csv_list.append(i)
                    f.close()
                timeseries_dict[id]=pd.DataFrame(csv_list)
                # assumes timesries shape [node x time]
            print(timeseries_dict)
            save(timeseries_dict, timeseries_path)
            self.timeseries_dict = timeseries_dict
            print(f"{prefix} {timeseries_file} is saved.")
        '''
        # loading a cached timeseries files
        print(timeseries_path)
        self.timeseries_dict = load(timeseries_path)
        print(f"{prefix} {timeseries_file} is loaded.")
        #print(self.timeseries_dict)
        self.num_nodes, self.num_timepoints = list(self.timeseries_dict.values())[0].shape
        self.full_subject_list = list(self.timeseries_dict.keys())
        if k_fold is None:
            self.subject_list = self.full_subject_list
            self.k = 1
        else:
            self.k_fold = StratifiedKFold(k_fold, shuffle=True, random_state=0) if k_fold is not None else None
            self.k = None

        # loading the label file
        behavior_df = pd.read_csv(label_path, encoding='CP949').set_index('Subject')
        behavior_df.index = behavior_df.index.astype('str')
        labels_series = behavior_df[target_feature]

        # encoding the labels to integer
        le = LabelEncoder()
        self.labels = le.fit_transform(labels_series).tolist()
        self.labels_dict = { id:label for id,label in zip(labels_series.index.tolist(), self.labels) }
        self.class_names = le.classes_.tolist()
        self.num_classes = len(self.class_names)
        self.full_label_list = [self.labels_dict[subject] for subject in self.full_subject_list]
        print(f"{prefix} Done.")

    def __len__(self):
        return len(self.subject_list) if self.k is not None else len(self.full_subject_list)

    def __getitem__(self, idx):
        subject = self.subject_list[idx]
        timeseries = self.timeseries_dict[subject]
        timeseries = (timeseries - np.mean(timeseries, axis=0, keepdims=True)) / np.std(timeseries, axis=0, keepdims=True)
        label = self.labels_dict[subject]
        return {'id': subject, 'timeseries': tensor(timeseries, dtype=float32), 'label': label}


    def split(self):
        if self.k_fold is None:
            self.subject_list = self.full_subject_list
            train_idx, test_idx = train_test_split(list(range(len(self.subject_list))), test_size=0.2, random_state=0)  
            train_subjects = [ self.full_subject_list[i] for i in train_idx ]
            test_subjects = [ self.full_subject_list[i] for i in test_idx ]       
            self.k = 1
        else:
            split_fold = list(self.k_fold.split(self.full_subject_list, self.full_label_list))
            train_subjects, test_subjects = {}, {}
            for k in range(len(split_fold)):
                train_subjects[k] = [ self.full_subject_list[i] for i in split_fold[k][0] ]
                test_subjects[k] = [ self.full_subject_list[i] for i in split_fold[k][1] ]
        return train_subjects, test_subjects

    def set_fold(self, fold, train):
        assert self.k_fold is not None
        self.k = fold
        train_idx, test_idx = list(self.k_fold.split(self.full_subject_list, self.full_label_list))[fold]
        if train: shuffle(train_idx)
        self.subject_list = [self.full_subject_list[idx] for idx in train_idx] if train else [self.full_subject_list[idx] for idx in test_idx]
        #shuffle(subject_list)
        #self.subject_list = subject_list
        #return train_idx, test_idx ######## check that differed after new instantiation

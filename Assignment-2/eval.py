import json
import sklearn.metrics as metrics
import numpy as np
import argparse
class Evaluator():

    def __init__(self, gold_file):
        self.gold_data = self.get_data(gold_file)
        self.possible_labels = ['O','B-Species', 'S-Species', 'S-Biological_Molecule', 'B-Chemical_Compound', 'B-Biological_Molecule', 'I-Species', 'I-Biological_Molecule', 'E-Species', 'E-Chemical_Compound', 'E-Biological_Molecule', 'I-Chemical_Compound', 'S-Chemical_Compound']
        self.possible_labels.remove('O')

    def get_data(self, file_path):
        with open(file_path,"r")as fread:
            data = fread.readlines()

        tags = []
        for i,d in enumerate(data):
            d = d.replace("\n",'')
            if d == '':
                continue
            
            tag = d.split("\t")[-1]
            tags.append(tag)
        
        return tags

    def evaluate_predictions(self, pred_data):
        f1_micro = metrics.f1_score(self.gold_data, pred_data, average="micro", labels=self.possible_labels)
        f1_macro = metrics.f1_score(self.gold_data, pred_data, average="macro", labels=self.possible_labels)
        return f1_micro, f1_macro

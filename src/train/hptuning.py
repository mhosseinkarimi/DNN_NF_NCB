import contextlib
import json
import os
from collections import OrderedDict

import numpy as np

from src.model.dnn import FCDNN
from src.utils.losses import circular_mae, rmse


class HPModelSelection:
    def __init__(self, params, input_size, output_size):
        self.params = params
        self.input_size = input_size
        self.output_size = output_size
        self.model_dict = dict(zip([f"model_{i}" for i in range(1, len(params['structures'])+1)], params['structures']))
        self.num_layers = len(params['structures'][0])
        
        # create log directory
        if not os.path.exists(params['directory']+'/'+params['name']):
            os.makedirs(params['directory']+'/'+params['name'])
    
    def print_summary(self, bests_phase, bests_mag):
        # sorting loss values
        sorted_bests_phase = OrderedDict(sorted(bests_phase.items(), key=lambda x:x[1]["val"]))
        sorted_bests_mag = OrderedDict(sorted(bests_mag.items(), key=lambda x:x[1]["val"]))
        with open(self.params['directory']+'/'+self.params['name']+'/bests.txt', 'w') as f:
            with contextlib.redirect_stdout(f):
                print(f"Phase Estimator Model Selection Results for {self.num_layers} layers")
                print("--------------------------------------------------------------") 
                for model_name in sorted_bests_phase.keys():
                    print(f"{model_name} \t Train: {sorted_bests_phase[model_name]['train']} \t Validation: {sorted_bests_phase[model_name]['val']}")
                print(f"Magnitude Estimator Model Selection Results for {self.num_layers} layers")
                print("--------------------------------------------------------------") 
                for model_name in sorted_bests_mag.keys():
                    print(f"{model_name} \t Train: {sorted_bests_mag[model_name]['train']} \t Validation: {sorted_bests_mag[model_name]['val']}")
                
    def run(self, train_data, train_w_phase, train_w_mag, val_data, val_w_phase, val_w_mag):
        bests_phase = {}
        bests_mag = {}
        for model_name, model_structure in self.model_dict.items():
            # initialize the best value disctionary for each model
            bests_phase[model_name] = {'train': [], 'val': []}
            bests_mag[model_name] = {'train': [], 'val': []}
            # training the models max_trial times
            for trial in range(1, self.params['max_trials']+1):
                print(f"-------------------Trial {trial}-------------------")
                # phase estimator model
                model = FCDNN(num_layers=self.num_layers, 
                            units=model_structure, 
                            input_shape=self.input_size, 
                            output_dim=self.output_size, 
                            dropout=self.params['dropout'], 
                            loss=circular_mae)
                model.summary()
                losses = model.train(
                    train_data, train_w_phase, val_data, val_w_phase,
                    epochs=self.params['num_epochs'], batch_size=self.params['batch_size'], lr=self.params['learning_rate'], 
                    lr_scheduler=None, device=self.params['device'], grad_clip=self.params['grad_clip'])
                arg_best_phase = np.argmin(losses['val'])
                bests_phase[model_name]['train'].append(losses['train_infer'][arg_best_phase])
                bests_phase[model_name]['val'].append(losses['val'][arg_best_phase])
                
                # magnitude estimator model
                model = FCDNN(num_layers=self.num_layers, 
                            units=model_structure, 
                            input_shape=self.input_size, 
                            output_dim=self.output_size, 
                            dropout=self.params['dropout'], 
                            loss=rmse)
                model.summary()
                losses = model.train(
                    train_data, train_w_mag, val_data, val_w_mag,
                    epochs=self.params['num_epochs'], batch_size=self.params['batch_size'], lr=self.params['learning_rate'], 
                    lr_scheduler=None, device=self.params['device'], grad_clip=self.params['grad_clip'])
                arg_best_mag = np.argmin(losses['val'])
                bests_mag[model_name]['train'].append(losses['train_infer'][arg_best_mag])
                bests_mag[model_name]['val'].append(losses['val'][arg_best_mag])
            # take the average of trials
            bests_phase[model_name]['train'] = np.mean(bests_phase[model_name]['train'])    
            bests_phase[model_name]['val'] = np.mean(bests_phase[model_name]['val'])
            
            bests_mag[model_name]['train'] = np.mean(bests_mag[model_name]['train'])    
            bests_mag[model_name]['val'] = np.mean(bests_mag[model_name]['val'])
            
        
        # save the logs
        with open(self.params['directory']+'/'+self.params['name']+'/bests_phase.json', 'w') as f:
            json.dump(bests_phase, f)
        with open(self.params['directory']+'/'+self.params['name']+'/bests_mag.json', 'w') as f:
            json.dump(bests_mag, f)
        
        self.print_summary(bests_phase, bests_mag)
            

    

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import torch
import torch.nn as nn
from Main_model import GCN
from argparse import ArgumentParser
import torch.optim as optim
import os
import pickle

class Model:
  def __init__(self,Graph,Data,index,Sub,Ab):
    self.parser = ArgumentParser()
    self.parser.add_argument("--hidden_size_1", type=int, default=330, help="Size of first GCN hidden weights")
    self.parser.add_argument("--hidden_size_2", type=int, default=130, help="Size of second GCN hidden weights")
    self.parser.add_argument("--num_classes", type=int, default=66, help="Number of prediction classes")
    self.parser.add_argument("--test_ratio", type=float, default=0.1, help="Ratio of test to training nodes")
    self.parser.add_argument("--num_epochs", type=int, default=3300, help="No of epochs")
    self.parser.add_argument("--lr", type=float, default=0.011, help="learning rate")
    self.parser.add_argument("--model_no", type=int, default=0, help="Model ID")
    self.args = self.parser.parse_args()
    self.Submission = Sub
    self.graph = Graph
    self.data = Data
    self.ind =  index
    self.AB = Ab 
    self.L = []
    

    self.f, self.X, self.A_hat, self.selected, self.labels_selected, self.labels_not_selected, self.test_idxs = self.load_datasets(self.args) 
    self.net = GCN(self.X.shape[1], self.A_hat, self.args)
    self.criterion = nn.CrossEntropyLoss()
    print(self.net.parameters)
    print(self.args.lr)
    self.optimizer = optim.Adam(self.net.parameters(),lr=self.args.lr)
    self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[1000,2000,3000,4000,5000,6000], gamma=0.50)
    self.start_epoch, self.best_pred = self.load_state(self.net, self.optimizer, self.scheduler, model_no=self.args.model_no, load_best=True)
    self.losses_per_epoch, self.evaluation_untrained = self.load_results(model_no=self.args.model_no)  
    self.net.train()
    self.evaluation_trained = []  
    for e in range(self.start_epoch, self.args.num_epochs):
        self.optimizer.zero_grad()
        self.output = self.net(self.f)
        self.loss = self.criterion(self.output[self.selected], torch.tensor(self.labels_selected).long())
        self.losses_per_epoch.append(self.loss.item())
        self.loss.backward()
        self.optimizer.step()
        if e % 50 == 0:
            self.net.eval()
            with torch.no_grad():
                self.pred_labels = self.net(self.f)
                self.trained_accuracy = self.evaluate(self.output[self.selected], self.labels_selected); self.untrained_accuracy = self.evaluate(self.pred_labels[self.test_idxs], self.labels_not_selected)
            self.evaluation_trained.append((e, self.trained_accuracy)); self.evaluation_untrained.append((e, self.untrained_accuracy))
            print("[Epoch %d]: Evaluation accuracy of trained nodes: %.7f" % (e, self.trained_accuracy))
            print("[Epoch %d]: Evaluation accuracy of test nodes: %.7f" % (e, self.untrained_accuracy))
            print("Labels of trained nodes: \n", self.output[self.selected].max(1)[1])
            print("Labels of test nodes: \n", self.output[self.test_idxs].max(1)[1])
            self.Lab = self.output[self.test_idxs].max(1)[1]
            self.L = self.Lab
            self.net.train()
            if self.trained_accuracy > self.best_pred:
                self.best_pred = self.trained_accuracy
                torch.save({
                    'epoch': e + 1,\
                    'state_dict': self.net.state_dict(),\
                    'best_acc': self.trained_accuracy,\
                    'optimizer' : self.optimizer.state_dict(),\
                    'scheduler' : self.scheduler.state_dict(),\
                }, os.path.join("D:/Kenya/" ,\
                    "test_model_best_%d.pth.tar" % self.args.model_no))
        if (e % 250) == 0:
            self.save_as_pickle("test_losses_per_epoch_%d.pkl" % self.args.model_no, self.losses_per_epoch)
            self.save_as_pickle("test_accuracy_per_epoch_%d.pkl" % self.args.model_no, self.evaluation_untrained)
            torch.save({
                    'epoch': e + 1,\
                    'state_dict': self.net.state_dict(),\
                    'best_acc': self.trained_accuracy,\
                    'optimizer' : self.optimizer.state_dict(),\
                    'scheduler' : self.scheduler.state_dict(),\
                }, os.path.join("D:/Kenya/",\
                    "test_checkpoint_%d.pth.tar" % self.args.model_no))
        self.scheduler.step() 
    for i in range(len(self.Submission)):
      ch = self.AB[self.L[i]]
      print(i,",",ch)
      self.Submission[ch][i] = 99
    for i in range(len(self.AB)):
      self.Submission[self.AB[i]] = self.Submission[self.AB[i]]/100
    print(self.Submission)
    self.Submission.to_csv('D:/Kenya/SampleSubmission.csv')
    

  def load_pickle(self,filename):
    completeName = os.path.join("D:/Kenya/",\
                                filename)
    with open(completeName, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data
  
  def save_as_pickle(self,filename, data):
    completeName = os.path.join("D:/Kenya/",\
                                filename)
    with open(completeName, 'wb') as output:
        pickle.dump(data, output)


  def load_datasets(self,args):
    A = nx.to_numpy_matrix(self.graph, weight="weight") 
    A = A + np.eye(self.graph.number_of_nodes())
    degrees = []
    for d in self.graph.degree(weight=None):
        if d == 0:
            degrees.append(0)
        else:
            degrees.append(d[1]**(-0.5))
    degrees = np.diag(degrees)
    X = np.eye(self.graph.number_of_nodes())
    A_hat = degrees@A@degrees
    f = X
    test_idxs = self.ind
    selected = []
    for i in range(len(self.data)):
        if i not in test_idxs:
            selected.append(i)
    f_selected = f[selected]; f_selected = torch.from_numpy(f_selected).float()
    labels_selected = [l for idx, l in enumerate(self.data['label']) if idx in selected]
    f_not_selected = f[test_idxs]; f_not_selected = torch.from_numpy(f_not_selected).float()
    labels_not_selected = [l for idx, l in enumerate(self.data['label']) if idx not in selected]
    f = torch.from_numpy(f).float()
    return f, X, A_hat, selected, labels_selected, labels_not_selected, test_idxs


  def load_state(self,net, optimizer, scheduler, model_no=0, load_best=False):
    base_path = "D:/Kenya/"
    checkpoint_path = os.path.join(base_path,"test_checkpoint_%d.pth.tar" % model_no)
    best_path = os.path.join(base_path,"test_model_best_%d.pth.tar" % model_no)
    start_epoch, best_pred, checkpoint = 0, 0, None
    if (load_best == True) and os.path.isfile(best_path):
        checkpoint = torch.load(best_path)
    elif os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
    if checkpoint != None:
        start_epoch = checkpoint['epoch']
        best_pred = checkpoint['best_acc']
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])    
    return start_epoch, best_pred 
  
  def load_results(self,model_no=0):
    losses_path = "D:/Kenya/test_losses_per_epoch_%d.pkl" % model_no
    accuracy_path = "D:/Kenya/test_accuracy_per_epoch_%d.pkl" % model_no
    if os.path.isfile(losses_path) and os.path.isfile(accuracy_path):
        losses_per_epoch = self.load_pickle("test_losses_per_epoch_%d.pkl" % model_no)
        accuracy_per_epoch = self.load_pickle("test_accuracy_per_epoch_%d.pkl" % model_no)
    else:
        losses_per_epoch, accuracy_per_epoch = [], []
    return losses_per_epoch, accuracy_per_epoch
 
  def evaluate(self,output, labels_e):
    _, labels = output.max(1); labels = labels.numpy()
    return sum([(e-1) for e in labels_e] == labels)/len(labels)     
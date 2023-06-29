from pyopp import cSimpleModule, cMessage, simTime
import numpy as np
from Dataset import Dataset

import pandas as pd 
from dataMessage import dataMessage
from WeightsMessage import WeightsMessage
import utility as util
from evaluate import evaluate_model
import random
from Dataset import Dataset
from collections import defaultdict
import sys 
import wandb
import os 
import time
from scipy import stats, optimize



util.reset_random_seeds()

start_time = time.time()

epsilon = 0.1
sync_ = 1
name_ = "FedAvg Vanilla Attacked-ML with DP (Model Evaluation)" #"FedAvg Vanilla Attacked-FS (Full Models)"   
dataset_name = 'ml-100k' #foursquareNYC 
topK = 20

epochs = 2
dataset = Dataset(dataset_name) 

train ,testRatings, testNegatives, trainNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives, dataset.trainNegatives
testRatings = testRatings[:1000]
testNegatives = testNegatives[:1000]
topK_clustering = 5

clipping = 2
noise_mult = 4
delta=1e-6
steps = 2




# get for every user its positives ratings which will be sent from the server to the nodes
def get_user_vector(train,user = 0):
    positive_instances = []
    for (u,i) in train.keys():
        if u == user:
            positive_instances.append(i)
        if u  > user :
            break

    return positive_instances

def zero():
    return 0

def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return float(len(s1.intersection(s2)) / len(s1.union(s2)))

def cdf(data, metric, sync=sync_, topK=topK):
    data_size = len(data)

    # Set bins edges
    data_set = sorted(set(data))
    bins = np.append(data_set, data_set[-1] + 1)

    # Use the histogram function to bin the data
    counts, bin_edges = np.histogram(data, bins=bins, density=False)

    counts = counts.astype(float) / data_size

    # Find the cdf
    cdf = np.cumsum(counts)
    idx = np.arange(cdf.shape[0])
    data = [[x, y] for (x, y) in zip(bin_edges[idx], cdf[idx])]
    if sync:
        if topK == None:
            table = wandb.Table(data=data, columns=[metric, "CDF"])
            wandb.log({metric + " CDF ": wandb.plot.line(table, metric, "CDF", stroke="dash",
                                                         title=metric + " last round cumulative distribution")})

        else:
            table = wandb.Table(data=data, columns=[metric + "@" + str(topK), "CDF"])
            wandb.log(
                {metric + "@" + str(topK) + " CDF": wandb.plot.line(table, metric, "CDF", stroke="dash",
                                                                               title=metric + " last round cumulative distribution")})

def get_individual_set(user, ratings, negatives):
    personal_Ratings = []
    personal_Negatives = []

    for i in range(len(ratings)):
        idx = ratings[i][0]
        if idx == user:
            personal_Ratings.append(ratings[i].copy())
            personal_Negatives.append(negatives[i].copy())
        elif idx > user:
            break

    return personal_Ratings, personal_Negatives


def get_training_as_list(train):
    trainingList = []
    for (u, i) in train.keys():
        trainingList.append([u, i])
    return trainingList


def compute_mu_uniform(epoch, noise_multi, q):
    """Compute mu from uniform subsampling."""
    t = epoch / q
    c = q * np.sqrt(t) 
    return np.sqrt(2) * c * np.sqrt(
        np.exp(noise_multi**(-2)) * stats.norm.cdf(1.5 / noise_multi) +
        3 * stats.norm.cdf(-0.5 / noise_multi) - 2)

def delta_eps_mu(eps, mu):
    """Compute dual between mu-GDP and (epsilon, delta)-DP."""
    
    return stats.norm.cdf(-eps / mu + mu / 2) - np.exp(eps) * stats.norm.cdf(-eps / mu - mu / 2)

def eps_from_mu(mu, delta):
    """Compute epsilon from mu given delta via inverse dual."""

    def f(x):
        """Reversely solve dual by matching delta."""
        return delta_eps_mu(x, mu) - delta

    return optimize.root_scalar(f, bracket=[0, 500], method='brentq').root

def compute_eps_uniform(epoch, noise_multi, q, delta):
    """Compute epsilon given delta from inverse dual of uniform subsampling."""
    
    return eps_from_mu(
        compute_mu_uniform(epoch, noise_multi, q), delta)



class Server(cSimpleModule):
    def initialize(self):
        # server initializes number of rounds, number of participants
        # creates the global model
        self.number_rounds = 500
        self.init_round = self.number_rounds
        self.current_round = 0
        self.best_perf = [0,0,0]
        self.message_round = cMessage('message_round')
        self.message_averaging = cMessage('StartAveraging')
        self.global_weights = []
        self.total_samples = []  # a list of positive rating of every user
        self.all_participants = [i for i in range(self.gateSize('sl'))]
        self.oldparticipants = []
        self.round_participants = dict()
        self.round_participants_number = int(100 *  len(self.all_participants) / 100)
        self.num_items =  1682 #38333   TODO: automate this train.shape[1]   
        self.num_users =  train.shape[0]     
        self.model = util.get_model(self.num_items,self.num_users)
        self.best_model = self.model.get_weights()
        # num samples stands for the number of weights received during a training round
        # will be used to know when the server can start the averaging (enough models have been received)
        self.num_samples = 0 
        self.hrs = []
        self.ndcgs = []
        self.global_hrs = []
        self.global_ndcgs = []
        self.global_rounds = []
        self.attack_rounds = []
        self.attack_results = []
        self.att_accs = []
        self.epsilons = []
        self.privacy_budget = 0 

        self.training_ratings = defaultdict(list)
        self.training_negatives = defaultdict(list)
        # get all users' training sets to use them in the attack 
        for u in self.all_participants:
                self.training_ratings[u], self.training_negatives[u] = get_individual_set(u, get_training_as_list(train), trainNegatives)

        # get ground truth
        self.users_true_topK = self.groundTruth_TopKItemsLiked(topK = topK_clustering)
        
        if self.getName() == 'server':
            # diffuse preparation phase messages which contains the data for every user
            self.diffuse_message('PreparationPhase')
             
              
    def finish(self):
        pass
        
    def handleMessage(self, msg):
        if msg.isSelfMessage():
            # if this self message is received, it means the server got enough models to do the averaging
            if msg.getName() == 'StartAveraging':
                self.fedAvg_Att()
                # self.fedAvg()
                    
            # if this self message is received, server can start first round 
            else: 
                self.diffuse_message('Round', True)
                self.global_weights = []   
            
       
        else:
            if msg.getName() == 'Node_weights' :
            # append any weights received from a node in addition to its number of positives ratings which
            # can be used in averaging even tho it was apparently a solution with a basic averaging that gave the best results 
            
                self.global_weights.append(msg.weights)
                self.total_samples.append(msg.positives_nums)
                self.round_participants[msg.id_user] = len(self.global_weights) - 1
                self.num_samples = self.num_samples + 1
                if self.num_samples == self.round_participants_number:
                        self.num_samples = 0
                        self.number_rounds = self.number_rounds - 1
                        self.current_round = self.current_round + 1
                        self.scheduleAt(simTime(),self.message_averaging)
                        
            else: # Node_performance
                self.hrs.append(msg.hr)
                self.ndcgs.append(msg.ndcg) 
                if (len(self.hrs) == len(self.all_participants)):
                      self.average_lhr = sum(self.hrs)/len(self.hrs)
                    self.average_lndcg = sum(self.ndcgs)/len(self.ndcgs)
                    print('Final Average Local HR =',self.average_lhr)
                    print('Final Average Local NDCG =',self.average_lndcg)
                    print('Final Privacy Budget=',self.privacy_budget)
                    sys.stdout.flush()
                    self.synch_wandb_cloud()
                    
            
            self.delete(msg)


    def diffuse_message(self, type, sample = False):
        participants = self.sampling(self.round_participants_number)  if sample else self.all_participants
        self.round_clients = participants
        if type == 'PreparationPhase':
            for i in range(len(participants)):
                msg = dataMessage(type)
                msg.user_ratings = np.array(get_user_vector(train,i))
                msg.num_items = self.num_items
                msg.num_users = self.num_users
                msg.testRatings = testRatings
                msg.testNegatives = testNegatives
                msg.id_user = i
                self.send(msg, 'sl$o',i)
            self.scheduleAt(simTime(),self.message_round)
        
        else:
            for p in participants:
                weights = WeightsMessage(type)
                weights.weights = self.model.get_weights()
                self.send(weights,'sl$o',p)
                       
            
        
       
    
    def fedAvg_Att(self, based_on = "Users_Embeddings"):
            print('in fed att function')
            sys.stdout.flush()
            if self.current_round < 5:  
                # Uncomment for DP-SGD moment accounting
                #self.privacy_budget = compute_eps_uniform(self.current_round
                                                                  , noise_mult, 1, 1e-6)
                #self.epsilons.append(self.privacy_budget)        
                if based_on == 'Users_Embeddings':
                    users_topk = defaultdict(list)
                    for u in self.round_participants.keys(): # changed all participants to round participants; 
                        for v in self.round_participants.keys(): # also here
                                if u != v:
                                    model_v = self.global_weights[self.round_participants[v]]
                                    _, ndcg_uv = self.evaluate_on_train_full(model_v, u, v)
                                    users_topk[u].append((v, ndcg_uv))
                        users_topk[u].sort(key = lambda x: x[1], reverse = True)
                        self.attack_results.append([self.current_round, u, users_topk[u]])
                        users_topk[u] = [x[0] for x in users_topk[u]][:topK_clustering]
                    self.att_accs.append(self.compute_metrics(self.users_true_topK, users_topk))
                    self.attack_rounds.append(self.current_round)
                else:
                    users_topk = defaultdict(list)
                    for u in self.round_participants.keys(): # changed all participants to round participants; 
                        for v in self.round_participants.keys(): # also here
                                if u != v:
                                    model_v = self.global_weights[self.round_participants[v]]
                                    model_u = self.global_weights[self.round_participants[u]]
                                    _, ndcg_uv = self.evaluate_on_train_iembeddings(model_v, model_u, u)
                                    users_topk[u].append((v, ndcg_uv))
                        users_topk[u].sort(key = lambda x: x[1], reverse = True)
                        users_topk[u] = [x[0] for x in users_topk[u]][:topK_clustering]
                    self.att_accs.append(self.compute_metrics(self.users_true_topK, users_topk))
                    self.attack_rounds.append(self.current_round)
        
            self.fedAvg()


    def fedAvg(self):
            # having a lot of positives ratings already have a bigger dataset and thus will make bigger gradient steps
            # giving him even more weights can drag the model to a local minimum that suits only this kind of users)
            sum_ = sum(self.total_samples)
            for j in range(len(self.global_weights)):
                for i in range(len(self.global_weights[j])):  
                    self.global_weights[j][i] = self.global_weights[j][i]  * self.total_samples[j] / sum_
                
            # summing and then combining in one entity of weights
            new_weights = self.global_weights[0].copy()
            for i in range(1,len(self.global_weights)):
                new_weights = [ np.add(x,y) for x, y in zip(self.global_weights[i], new_weights)]
  
            # new_weights[0] = self.model.get_weights()[0].copy()
            self.model.set_weights(new_weights)
            self.global_weights = []
            self.total_samples = []
           
            self.evaluate()
          

    def evaluate(self):
       
        if self.number_rounds % 10 == 0 and self.number_rounds > 0:
            evaluation_threads = 1   #mp.cpu_count() 
            (hits, ndcgs) = evaluate_model(self.model, testRatings, testNegatives, topK, evaluation_threads)
            hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()

            if(hr > self.best_perf[0]):
                self.best_perf = [hr,ndcg]
                self.best_model = self.model.get_weights()
            
            self.global_hrs.append(hr)
            self.global_ndcgs.append(ndcg)
            self.global_rounds.append(self.current_round)
            print("Round : ",self.number_rounds)
            print('Temporary HR =  ',hr)
            print('Temporary NDCG =  ',ndcg)
            sys.stdout.flush()
            
        if self.number_rounds > 0:
            self.diffuse_message('Round', True)
        
        else:
            self.diffuse_message('FinalRound', False)
            print("Round : ",self.number_rounds)
            print('Final HR =  ',self.best_perf[0])
            print('Final NDCG =  ',self.best_perf[1])                
            self.model.set_weights(self.best_model)          
            sys.stdout.flush() 
            # For saving the model  
            # self.model.save_weights("FL"+dataset_name+"_"+str(topK)+"_Model.h5", overwrite=True)

    # at attack based on what the combination of the attacker (user embeddings etc) and the items' embeddings of the victim can predict
    def evaluate_on_train_iembeddings(self, model_v, model_u, user):          
        local_items_embeddings = model_u[0].copy()
        model_u[0] = model_v[0]
        evaluation_threads = 1 #mp.cpu_count()
        local = self.model.get_weights().copy()
        self.model.set_weights(model_u)
        (hits, ndcgs) = evaluate_model(self.model, self.training_ratings[user], self.training_negatives[user], topK, evaluation_threads)               
        hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
        self.model.set_weights(local)
        model_u[0] = local_items_embeddings
        return hr, ndcg


    # an attack based on what the model of the victim can predict; an inference attack somehow
    def evaluate_on_train_full(self, model, user, user2):
        v_ratings = self.training_ratings[user].copy()
        for i in range(len(v_ratings)):
                v_ratings[i][0] = user2
          
        local = self.model.get_weights().copy()
        self.model.set_weights(model)
        evaluation_threads = mp.cpu_count() 
        sys.stdout.flush()
        (hits, ndcgs) = evaluate_model(self.model, v_ratings, self.training_negatives[user], topK, evaluation_threads)               
        hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
        self.model.set_weights(local)
        return hr, ndcg
    

    def compute_metrics(self, ground_truthK, users_topK):
        accs = []
        for i in range(len(users_topK)):
            accs.append(len(set(ground_truthK[i]) & set(users_topK[i])) / topK_clustering)
        
        average_acc = sum(accs) / len(users_topK)
        print("Round ", self.current_round)
        print("Average accuracy of attack", average_acc)
        sys.stdout.flush()
        return accs


    def sampling(self, num_samples):
        if num_samples > self.gateSize('sl'):
            raise Exception("ERROR : size of sampling set is bigger than total samples of clients") 
        else:
            size =  self.gateSize('sl')
            participants = []
            # we can decide not take into account all participants
            participants = random.sample(range(size),num_samples)

        return participants


    def groundTruth_TopKItemsLiked(self, topK = topK_clustering):
        users = []
        users_topk = defaultdict(list)
        for u in range(len(self.all_participants)):
            users.append(get_user_vector(train, u))

        for u in range(len(self.all_participants)):
            for v in range(len(self.all_participants)):
                if u != v:
                    users_topk[u].append((v, jaccard_similarity(users[u], users[v])))
            users_topk[u].sort(key=lambda x: x[1], reverse=True)
            users_topk[u] = [x[0] for x in users_topk[u]][:topK]

        return users_topk


    def synch_wandb_cloud(self):
        # pass
        if not sync_:
            return
        wandb_config = {
        "Dataset": dataset_name,
        "Implementation": "Tensorflow",
        "Rounds": self.init_round,
        "Nodes": len(self.all_participants),
        "Learning_rate": 0.01,
        "Epochs": 2,
        "TopK": topK,
        "TopK_Clustering": topK_clustering,
        "Attacker id": "all",
        "Delta" : 10e-6
        }

        os.environ["WANDB_APepsilonsI_KEY"] = "anonymous"
        os.environ["WANDB_MODE"] = "offline"
        os.environ["WANDB_START_METHOD"] = "fork"
        wandb.init(project="FederatedLearningStandardSetting", entity="anonymous", name=name_, config=wandb_config)

        if sync_ :
            for i in range(len(self.global_hrs)):
                wandb.log({"Average GHR": self.global_hrs[i], "Average GNDCG": self.global_ndcgs[i], 
                           "Round ": self.global_rounds[i]})
            max_avg = 0
            idx = -1
            for i in range(len(self.att_accs)):
                avg_acc = sum(self.att_accs[i]) / len(self.att_accs[i])
                if avg_acc > max_avg:
                    idx = i
                    max_avg = avg_acc
                wandb.log({"Average Attack accuracy": avg_acc,
                        "round": self.attack_rounds[i], "Epsilon": self.epsilons[i]})


            wandb.log({"Average LHR": self.average_lhr, "Average LNDCG": self.average_lndcg})
            cdf(self.hrs, "Local HR")
            cdf(self.ndcgs, "Local NDCG")
            cdf(self.att_accs[idx], "Attack Accuracy", topK_clustering)
            wandb.finish()

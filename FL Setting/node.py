from pyopp import cSimpleModule
import numpy as np
from tensorflow.keras.optimizers import Adam
from WeightsMessage import WeightsMessage
from evaluate import evaluate_model
import utility as util
import sys
from numpy import linalg as LA
from sklearn.preprocessing import normalize

util.reset_random_seeds()


topK = 20
epochs = 1
dataset_name = "ml-100k" # "foursquareNYC" #GowallaNYC    
num_items = 1682 #38333  #10978     

def get_user_test_set(testRatings,testNegatives,user):
    personal_testRatings = []
    personal_testNegatives = []

    for i in range(len(testRatings)):  
        idx = testRatings[i][0]
        if idx == user:
            personal_testRatings.append(testRatings[i])
            personal_testNegatives.append(testNegatives[i])
        elif idx > user:
            break
        
    return personal_testRatings,personal_testNegatives


class Node(cSimpleModule):
    def initialize(self):
        self.positives_nums = 0
        self.vector = np.empty(0) # positive rated items
        self.labels = np.empty(0) # binary array for labeling items
        self.item_input = np.empty(0) # vector + negative items
        self.weightmsg = WeightsMessage("Node_weights")
        self.perfmsg = WeightsMessage("Node_performance")
        self.first = 1

        
    def handleMessage(self, msg):
        # every node receives data from the server and creates its own model 
        if msg.getName() == 'PreparationPhase': 
            self.vector = msg.user_ratings
            self.id_user = msg.id_user
            self.num_items = msg.num_items
            self.num_users = msg.num_users
            self.testRatings, self.testNegatives = get_user_test_set(msg.testRatings, msg.testNegatives, self.id_user) 
            self.model = util.get_model(self.num_items,self.num_users) # giving the size of items as par
            self.model.compile(optimizer=Adam(learning_rate=0.01), loss='binary_crossentropy')
        
        else: # Per Round global model message 
            self.item_input, self.labels, self.user_input = self.my_dataset()
            self.model.set_weights(msg.weights)
            self.update()

            
            if(msg.getName() != "FinalRound"):
                self.weightmsg.weights = self.model.get_weights()
                self.weightmsg.positives_nums = self.positives_nums            
                self.weightmsg.id_user = self.id_user
                self.send(self.weightmsg, 'nl$o', 0)
            else:
                lhr, lndcg = self.evaluate_model(20)
                print("Node :", self.id_user)
                print("Local HR20: ", lhr) 
                print("Local NDCG20 :", lndcg)
                self.perfmsg.hr = lhr
                self.perfmsg.ndcg = lndcg
                self.perfmsg.id_user = self.id_user
                self.send(self.perfmsg, 'nl$o', 0)

        
        self.delete(msg)
                
          
    def finish(self):
        pass
    
    def update(self):
        hist = self.model.fit([self.user_input, self.item_input], #input
                        np.array(self.labels), # labels 
                        batch_size = len(self.labels), epochs=epochs, verbose=2)  
        print("Node : ",self.getIndex())
        sys.stdout.flush()        
        return hist


    def evaluate_model(self, topK = topK):
        evaluation_threads = 1
        (hits, ndcgs) = evaluate_model(self.model, self.testRatings, self.testNegatives, topK, evaluation_threads)     
        hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
        return hr, ndcg


    def my_dataset(self,num_negatives = 4):
        item_input = []
        labels = []
        user_input = []
        self.positives_nums = 0
        for i in self.vector:
            item_input.append(i)
            labels.append(1)
            user_input.append(self.id_user)
            self.positives_nums = self.positives_nums + 1
            for i in range(num_negatives):
                j = np.random.randint(self.num_items)
                while j in self.vector:
                    j = np.random.randint(self.num_items)
                user_input.append(self.id_user)
                item_input.append(j)
                labels.append(0)            
        return np.array(item_input), np.array(labels), np.array(user_input)




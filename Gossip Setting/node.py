import numpy as np
from pyopp import cSimpleModule, cMessage, simTime
from keras.optimizers import Adam, SGD
from Dataset import Dataset
from WeightsMessage import WeightsMessage
import utility as util
import random
import sys
from evaluate import evaluate_model
from scipy.spatial.distance import cosine
from numpy import linalg as LA
from sklearn.preprocessing import normalize



topK = 20
dataset_name = "ml-100k" #foursquareNYC    GowallaNYC #TODO AUTOMATE CHOICE OF DATASET

if dataset_name == "foursquareNYC":
    num_items =   38333 #  10978  
elif dataset_name =="ml-100k ":
    num_items =   1682
else: # GowallaNYC
    num_items =   10978 

dataset = Dataset(dataset_name)
train ,testRatings, testNegatives, trainNegatives, \
validationRatings, validationNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives, \
    dataset.trainNegatives, dataset.validationRatings, dataset.validationNegatives

# testRatings = testRatings[:1000] #  2453 1000
# testNegatives= testNegatives[:1000]

number_peers = 3


def get_user_vector(train,user = 0):
    positive_instances = []    
    for (u,i) in train.keys():
        if u == user:
            positive_instances.append(i)
        if u  > user :
            break

    return positive_instances

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

def jaccard_similarity(list1, list2):
        s1 = set(list1)
        s2 = set(list2)
        return float(len(s1.intersection(s2)) / len(s1.union(s2)))


def cosine_similarity(list1, list2):
        return 1 - cosine(list1,list2)
        # return 1 - 0

def mykey(el1, el2):
        # bigger frequency
        if el1[1][1] > el2[1][1]:
            return 1
            # better similarity
        elif el1[1][1] == el2[1][1] and el1[1][0] > el2[1][0]:
            return 1
        
        return -1

def get_training_as_list(train):
    trainingList = []
    for (u, i) in train.keys():
        trainingList.append([u, i])
    return trainingList

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


class Node(cSimpleModule):
    def initialize(self):
        # initialization phase in which number of rounds, model age, data is read, model is created, connecter peers list is created
        self.rounds =  200  
        self.vector = np.empty(0)
        self.labels = np.empty(0)
        self.item_input = np.empty(0)
        self.age = 1
        self.alpha = 0.4
        self.num_items = num_items #train.shape[1] TO DO automate this, doesn't work since the validation data has been added because one item is present there and not in training
        self.num_users = train.shape[0]  
        self.id_user = self.getIndex()  
        self.period = np.random.exponential(0.1)

        self.vector = get_user_vector(train,self.id_user)
        self.trainRatings, self.trainNegatives = get_individual_set(self.id_user, get_training_as_list(train), trainNegatives)
        self.testRatings, self.testNegatives = get_user_test_set(testRatings,testNegatives,self.id_user)
        self.validationRatings, self.validationNegatives = get_user_test_set(validationRatings,validationNegatives,self.id_user)
        self.best_hr = 0.0
        self.best_ndcg = 0.0
        self.best_model = []

        self.positives_nums = len(self.vector)

        self.item_input, self.labels, self.user_input = self.my_dataset()
        self.model = util.get_model(self.num_items,self.num_users) # each node initialize its own model 

        self.model.compile(optimizer=Adam(lr=0.01), loss='binary_crossentropy')
        
        self.period_message = cMessage('period_message')
        self.update()
        self.peers =  []
        
        self.neighbours = dict()
        self.colliders = []
        self.peer_sampling()
        self.performances = {}
        self.scheduleAt(simTime() + self.period,self.period_message)
        

    def handleMessage(self, msg):
        # periodic self sent message used as a timer by each node to diffuse its model 
        if msg.getName() == 'attackers_list':
            self.colliders = msg.list_attackers.copy()
            if self.id_user in self.colliders:
                self.colliders.remove(self.id_user)
            self.delete(msg)
        elif msg.getName() == 'period_message':
            if self.rounds > 0 :

                if self.rounds % 10 == 0 or self.rounds == 1:
                    if self.rounds % 10 == 0:
                        lhr, lndcg = self.evaluate_local_model(False,False)   
                    else:
                        if(len(self.best_model) > 0):
                            self.model.set_weights(self.best_model)
                        lhr, lndcg = self.evaluate_local_model(False,False)
                
                    print('node : ',self.id_user)
                    print('Local HR =  ', lhr)
                    print('Local NDCG =  ',lndcg)
                    print('Round left = ', self.rounds)
                    sys.stdout.flush()
                    self.diffuse_to_server(lhr, lndcg)

             
                self.diffuse_to_peer()
                # self.broadcast()
                if self.rounds % 10 == 0:
                    # for Rand-Gossip
                    self.peer_sampling()
                    # for Pers-Gossip
                    # self.peer_sampling_enhanced()
               

                self.rounds = self.rounds - 1
                self.scheduleAt(simTime() + self.period,self.period_message)
            
            elif self.rounds == 0: # if number of rounds has been acheived, we can evaluate the model both locally
                    lhr, lndcg = self.evaluate_local_model(False, False)                    
                    print('node : ',self.id_user)
                    print('Local HR =  ', lhr)
                    print('Local NDCG =  ',lndcg)
                    print('Round left = ', self.rounds)
                    sys.stdout.flush() 
                    self.diffuse_to_server(lhr, lndcg)


                
             
        elif msg.getName() == 'Model':
            self.find_profiles(msg)
		
	    # For colluding setting
            # for id in self.colliders:
            #     if msg.id != id:
            #         new_message = WeightsMessage("Model_Collider")
            #         new_message.weights = msg.weights.copy()
            #         new_message.age = msg.age       
            #         new_message.samples = msg.samples 
            #         new_message.id = msg.id
            #         self.diffuse_to_specific_peer(id, new_message)

	    # TODO: switch case for different aggregation functions
            # for Rand-Gossip Aggregation
            dt = self.FullAvg(msg)
            # for Rand-Gossip share less Aggregation
            # dt = self.FullAvg_items_only(msg)
            # for Pers-Gossip Aggregation
            # dt = self.DKL_mergeJ(msg)
            # for Pers-Gossip share less Aggregation 
            # dt = self.DKL_mergeJ_Items_only(msg)
          
          
            # ## added the pull possibility
            # if msg.type == "pull":
            #     self.diffuse_to_specific_peer(msg.id)

            self.delete(msg)

        elif msg.getName() == 'Model_Collider':
            self.find_profiles(msg)
            self.delete(msg)


    def finish(self):
        pass
    

    def find_profiles(self, msg, based_on_items_only = False):
        if based_on_items_only:
            # just evaluating the items embeddings
            local_items_embeddings = self.model.get_layer('item_embedding').get_weights()
            self.model.get_layer('item_embedding').set_weights([msg.weights[0]])
            _, ndcg = self.evaluate_on_train_items()
            
            self.neighbours[msg.id] = (ndcg, 1)

            self.model.get_layer('item_embedding').set_weights(local_items_embeddings)
        else:
            local_model = self.model.get_weights().copy()
            self.model.set_weights(msg.weights)
            _, ndcg = self.evaluate_on_train(user = msg.id)
            self.neighbours[msg.id] = (ndcg, 1)
            self.model.set_weights(local_model)
                    
    def evaluate_on_train_items(self):
        evaluation_threads = 1 #mp.cpu_count()
        (hits, ndcgs) = evaluate_model(self.model, self.trainRatings, self.trainNegatives, topK, evaluation_threads)               
        hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
        return hr, ndcg
    
           
    def evaluate_on_train(self, user):
        v_ratings = self.trainRatings.copy()
        for i in range(len(v_ratings)):
                v_ratings[i][0] = user

        evaluation_threads = 1 #mp.cpu_count()
        (hits, ndcgs) = evaluate_model(self.model, v_ratings, self.trainNegatives, topK, evaluation_threads)               
        hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
        return hr, ndcg

    def evaluate_local_model(self,all_dataset = False, validation=True, topK = topK):
        evaluation_threads = 1 #mp.cpu_count()
        if not all_dataset:
            if validation :
                (hits, ndcgs) = evaluate_model(self.model, self.validationRatings, self.validationNegatives, topK, evaluation_threads)               
            else:
                (hits, ndcgs) = evaluate_model(self.model, self.testRatings, self.testNegatives, topK, evaluation_threads)
        else:
            (hits, ndcgs) = evaluate_model(self.model, testRatings, testNegatives, topK, evaluation_threads)
        hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
        return hr, ndcg

    def get_model(self):
        # return self.model.get_layer("item_embedding").get_weights().copy()
        return self.model.get_weights().copy()

    def get_attack_model(self):
        return self.attack_model
       
    def set_model(self, weights):
        # self.model.get_layer("item_embedding").set_weights(weights)
        self.model.set_weights(weights)

    def get_gate(self, peer):
        idx = self.getIndex()
        if peer < idx:
            return peer
        else:
            return peer - 1 

    def peer_sampling(self):
        size = self.gateSize("no") - 1
        old_peers = self.peers.copy()
        self.peers = []
        for _ in range(number_peers):
            p = random.randint(0,size - 1)
            while(p in self.peers or p in old_peers or p == self.id_user):
                p = random.randint(0,size - 1)
            self.peers.append(p)



    def peer_sampling_enhanced(self):       
        size = self.gateSize("no") - 1 
        self.peers = []
        exploitation_peers = int(number_peers * (1 - self.alpha))
        self.performances = sorted(self.performances.items(), key= lambda x: x[1] ,reverse=True)  
        keys = [x[0] for x in self.performances]
        i = 0

        while i < exploitation_peers and i < len(keys):
            self.peers.append(keys[i])
            i += 1
                
        self.performances = {}

        exploration_peers = number_peers - i

        for _ in range(exploration_peers):
            p = random.randint(0,size - 1)
            while(p in self.peers or p == self.id_user):
                p = random.randint(0,size - 1)
            self.peers.append(p)


    def diffuse_to_specific_peer(self, id, msg = None):

        if msg is not None:
            self.send(msg, 'no$o', self.get_gate(id))
        else:        
            weights = WeightsMessage('Model')
            weights.weights = self.get_model()
            weights.age = self.age       
            weights.samples = self.positives_nums 
            weights.type = "push"
            weights.id = self.id_user
            self.send(weights, 'no$o', self.get_gate(id))
    
    def broadcast (self):
        for p in self.peers:
            weights = WeightsMessage('Model')
            weights.weights = self.get_model()
            weights.age = self.age       
            weights.samples = self.positives_nums 
            weights.id = self.id_user
            weights.type = "pull"
            self.send(weights, 'no$o',self.get_gate(p))

    # select random peers and send its model weights and its age to it  
    def diffuse_to_peer(self,nb_peers = 3, type = "pull"): 
        peers = self.peers.copy()
        for _ in range(nb_peers):
            peer = random.choice(peers)
            weights = WeightsMessage('Model')
            weights.weights = self.get_model()
            weights.age = self.age       
            weights.samples = self.positives_nums 
            weights.id = self.id_user
            weights.type = type
            self.send(weights, 'no$o',self.get_gate(peer))
            peers.remove(peer)
                

    def diffuse_to_server(self,hr,ndcg):
        if self.rounds != 1:
            weights = WeightsMessage('Performance')
        else:
            weights = WeightsMessage('FinalPerformance')
            weights.model = self.get_model()
            weights.vector = self.vector


        neighbours = list(self.neighbours.items())
        neighbours.sort(key= lambda x:x[1][0], reverse = True)
        weights.cluster_found = [x[0] for x in neighbours]
        weights.user_id = self.id_user
        weights.round = self.rounds
        weights.hit_ratio = hr
        weights.ndcg = ndcg
        self.send(weights, 'nl$o',0)

    
    def update(self, epochs = 2, batch_size = None):
        batch_size = len(self.labels) if batch_size == None else batch_size
        hist = self.model.fit([self.user_input, self.item_input], #input
                        np.array(self.labels), # labels 
                        batch_size= batch_size, nb_epoch=epochs, verbose=0, shuffle=True) 
   
        self.age = self.age + 1
        

        print("Node : ",self.getIndex())
        print("Rounds Left : ",self.rounds)
        sys.stdout.flush()
        
        
    def merge(self,message_weights):
        weights = message_weights.weights
        local_weights = self.get_model()
        local_weights[:] =  [ (a * self.age + b * message_weights.age) / (self.age + message_weights.age) for a,b in zip(local_weights,weights)]
        self.age = max(self.age,message_weights.age)
        self.set_model(local_weights)

        self.update()
        self.item_input, self.labels, self.user_input = self.my_dataset()

        return 0

        
    def simple_merge(self,weights):
        local = self.get_model()
        local[:] =  [ (a + b) / 2 for a,b in zip(local,weights)]
        self.set_model(local)
 
    def FullAvg(self, message_weights):
        weights = message_weights.weights
        local_weights = self.get_model()
        local_weights [:] = [(self.positives_nums * a + message_weights.samples * b) / (message_weights.samples + self.positives_nums) for a,b in zip(local_weights, weights)]
        self.set_model(local_weights)
        self.update()
        self.item_input, self.labels, self.user_input = self.my_dataset()

        return 0

    def FullAvg_items_only(self, message_weights):
        weights = message_weights.weights
        local_weights = self.get_model()
        local_user_embedding = local_weights[1][self.id_user]
        local_weights [:] = [(self.positives_nums * a + message_weights.samples * b) / (message_weights.samples + self.positives_nums) for a,b in zip(local_weights, weights)]
        local_weights[1][self.id_user] = local_user_embedding
        self.set_model(local_weights)
        self.update()
        self.item_input, self.labels, self.user_input = self.my_dataset()

        return 0
    

    def DKL_mergeJ_Items_only(self, message_weights):
        if len(self.validationRatings) < 2:
            self.simple_merge(message_weights.weights)
            self.item_input, self.labels, self.user_input = self.my_dataset()
            self.update()
            return 0
         
        local = self.get_model()
        local_user_embedding = local[1][self.id_user]
        ndcgs = []
        lhr,lndcg =self.evaluate_local_model()
        ndcgs.append(lhr * lndcg)


        message_weights.weights[1][self.id_user] = local_user_embedding
        
        self.set_model(message_weights.weights)
        hr, ndcg = self.evaluate_local_model()
        self.performances[message_weights.id] = hr * ndcg
        ndcgs.append(hr * ndcg)

        ndcg_total = sum(ndcgs)
        
        if(ndcg_total) == 0:
            self.set_model(local)
            return 0

        norm = [ (float(i))/ndcg_total for i in ndcgs]
            

        local[:] = [w * norm[0] for w in local]
        message_weights.weights[:] = [w * norm[1] for w in message_weights.weights]
        
        local[:] = [a + b for a,b in zip(local,message_weights.weights)]
        local[1][self.id_user] = local_user_embedding
        self.set_model(local)
        self.item_input, self.labels, self.user_input = self.my_dataset()
        self.update()
        
        return 0
    
    def DKL_mergeJ(self,message_weights): 
        
        if len(self.validationRatings) < 2:
            self.simple_merge(message_weights.weights)
            self.item_input, self.labels, self.user_input = self.my_dataset()
            self.update()
            return 0
         
        local = self.get_model()
        ndcgs = []
        lhr,lndcg =self.evaluate_local_model()
        ndcgs.append(lhr * lndcg)
        self.set_model(message_weights.weights)
        hr, ndcg = self.evaluate_local_model()
        self.performances[message_weights.id] = hr * ndcg
        ndcgs.append(hr * ndcg)

        ndcg_total = sum(ndcgs)
        
        if(ndcg_total) == 0:
            self.set_model(local)
            return 0

        norm = [ (float(i))/ndcg_total for i in ndcgs]
            

        local[:] = [w * norm[0] for w in local]
        message_weights.weights[:] = [w * norm[1] for w in message_weights.weights]
        
        local[:] = [a + b for a,b in zip(local,message_weights.weights)]
        self.set_model(local)
        self.item_input, self.labels, self.user_input = self.my_dataset()
        self.update()
        
        return 0

    def my_dataset(self,num_negatives = 4):
        item_input = []
        labels = []
        user_input = []
        for i in self.vector:
            item_input.append(i)
            labels.append(1)
            user_input.append(self.id_user)
            for i in range(num_negatives):
                j = np.random.randint(self.num_items)
                while j in self.vector:
                    j = np.random.randint(self.num_items)
                user_input.append(self.id_user)
                item_input.append(j)
                labels.append(0)            
        return np.array(item_input), np.array(labels), np.array(user_input)

   
    

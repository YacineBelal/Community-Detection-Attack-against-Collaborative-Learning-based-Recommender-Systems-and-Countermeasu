import random
from pyopp import cSimpleModule
import sys
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.spatial.distance import cosine, euclidean
from collections import defaultdict
from Dataset import Dataset
import numpy as np
import wandb
import os
import matplotlib.pyplot as plt
import pandas as pd 

sync_ = 1
a2a = False
# nodes communicate with all nodes that they interacted with, in the last round, in order to get fresh models
name_ = "FedAvg-ML (Model Evaluation)"  # "Pepper Model_Age_Based Attacked" #  "Pepper Attacked"
dataset_ = "ml-100k"  # GowallaNYC foursquareNYC   
topK = 20
topK_clustering = 5
clustersK = 10
ground_truth_type = "topk"  # None
dataset = Dataset(dataset_)


def get_user_vector(user):
    positive_instances = []

    for (u, i) in dataset.trainMatrix.keys():
        if u == user:
            positive_instances.append(i)
        if u > user:
            break

    return positive_instances


def get_distribution_by_genre(vector):
    infos = []
    with open("u.item", 'r', encoding="ISO-8859-1") as info:
        line = info.readline()
        while line and line != '':
            arr = line.split("|")
            temp = arr[-19:]
            infos.append(temp)
            line = info.readline()

    dist = [0 for _ in range(19)]
    for item in vector:
        for i in range(len(dist)):
            dist[i] += int(infos[item][i])

    return dist


def kulczynski2_similarity_bags(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    a = len(s1.intersection(s2))
    b = len(s1 - s2)
    c = len(s2 - s1)

    return float((a / (a + b) + a / (a + c)) / 2)


def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return float(len(s1.intersection(s2)) / len(s1.union(s2)))


def indic(data):
    max = np.max(data, axis=1)
    std = np.std(data, axis=1)
    return max, std


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


if sync_:
    wandb_config = {
        "Dataset": dataset_,
        "Implementation": "TensorFlow",
        "Rounds": 250,
        "Nodes": 146,
        "Learning_rate": 0.01,
        "Epochs": 2,
        "Batch_size": "Full",
        "TopK": topK,
        "TopK_Clustering": topK_clustering,
        "Attacker id": "all",
        "Distance computed on": "both user and items embeddings",
        "Pull": False,
        "Epsilon": np.inf,
        "Delta": np.inf,
        "Number of clusters": clustersK,
    }

    os.environ["WANDB_API_KEY"] = "anonymous"
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_START_METHOD"] = "fork"
    # os.environ["WANDB_SILENT"] = "true"
    wandb.init(project="anonymous", entity="anonymous", name=name_, config=wandb_config)


def cosine_similarity(list1, list2):
    return 1 - cosine(list1, list2)


# hypergeometric law!
# a elements respect the criteria; N population size, echantillon;
def RandomBoundAccuracy(N, a=topK_clustering, n=topK_clustering):
    p = a / N
    Expected_value = n * p
    return Expected_value / n


class Server(cSimpleModule):

    def initialize(self):
        self.all_participants = [i for i in range(self.gateSize('sl'))]
        self.num_participants = len(self.all_participants)
        self.hit_ratios = defaultdict(list)
        self.ndcgs = defaultdict(list)
        self.cluster_found = defaultdict(list)
        self.att_acc = defaultdict(list)
        self.att_acc_bound = defaultdict(list)
        self.att_recall = defaultdict(list)
        self.att_random_bound = defaultdict(list)
        self.models = dict()
        self.vectors = dict()
        self.clusters = self.groundTruth_TopKItemsLiked()
        self.attack_results = []


    def handleMessage(self, msg):

        self.hit_ratios[msg.round].append(msg.hit_ratio)  # hit ratio ~ recall for recsys
        self.ndcgs[msg.round].append(msg.ndcg)  # ~ accuracy
        self.cluster_found[msg.user_id].append(msg.cluster_found)
        self.attack_results.append([msg.round, msg.user_id, msg.cluster_found])
        # get models to compute topK with last models; trying to figure out if temporality has an effect on topK quality of attack
        if msg.getName() == 'FinalPerformance':
            self.models[msg.user_id] = msg.model
            self.vectors[msg.user_id] = msg.vector

        self.delete(msg)

    def finish(self):
       global wandb
        if a2a == True:
            refreshed_usertopK = defaultdict(list)
            for u in range(self.num_participants):
                local_model = self.models[u]
                for v in self.cluster_found[u][len(self.cluster_found[u]) - 1]:
                    similarity = 0
                    for i in range(2):  # user and items embeddings
                        local_embeddings = local_model[i]
                        received_embeddings = self.models[v][i]
                        for j in self.vectors[u]:
                            if i == 0:
                                similarity += euclidean(local_embeddings[j], received_embeddings[j])
                            if i == 1:
                                similarity /= len(self.vectors[u])
                                similarity = similarity / 2 + euclidean(local_embeddings[u], received_embeddings[v]) / 2
                            break
                    refreshed_usertopK[u].append((v, similarity))

                refreshed_usertopK[u].sort(key=lambda x: x[1])
                self.cluster_found[u][len(self.cluster_found[u]) - 1] = [x[0] for x in refreshed_usertopK[u]]

        nb_rounds = max(self.hit_ratios.keys())
        rand_acc = RandomBoundAccuracy(N=self.num_participants)
        idx_round = 0
        for round in self.hit_ratios.keys():
            avg_acc = 0
            avg_acc_bound = 0
            avg_recall = 0
            avg_random_bound = 0
            # print("round = ", round)
            avg_hr = sum(self.hit_ratios[round]) / self.num_participants
            print("Average Test HR = ", avg_hr)
            avg_ndcg = sum(self.ndcgs[round]) / self.num_participants
            # print("Average Test NDCG = ",avg_ndcg)
            sys.stdout.flush()
            if ground_truth_type == "kmeans":
                for attacker in self.cluster_found.keys():
                    acc, recall = self.Accuracy_Clustering_Attack(self.clusters, attacker, idx_round)
                    self.att_acc[round].append(acc)
                    self.att_recall[round].append(recall)
                    avg_acc += acc
                    avg_recall += recall
                idx_round += 1
                avg_acc = avg_acc / self.num_participants
                avg_recall = avg_recall / self.num_participants

            elif ground_truth_type == "topk":
                for attacker in self.cluster_found.keys():
                    acc, acc_bound, rand_bound = self.Accuracy_Topk_Attack(self.clusters, attacker, idx_round)
                    self.att_acc[round].append(acc)
                    self.att_acc_bound[round].append(acc_bound)
                    self.att_random_bound[round].append(rand_bound)
                    avg_acc += acc
                    avg_acc_bound += acc_bound
                    avg_random_bound += rand_bound

                idx_round += 1
                avg_acc = avg_acc / self.num_participants
                avg_acc_bound = avg_acc_bound / self.num_participants
                avg_random_bound = avg_random_bound / self.num_participants


            if sync_ and ground_truth_type == "topk":
                wandb.log({"Average HR": avg_hr, "Average NDCG": avg_ndcg, "Average Attack Acc": avg_acc,
                           "Average Attack Acc Bound": avg_acc_bound,
                           "Average Random Bound": rand_acc, "Average Crossed Random Bound": avg_random_bound,
                           "Round ": nb_rounds - round})
                if round == 0:
                    wandb.log({"Final Average HR": avg_hr, "Final Average NDCG": avg_ndcg,
                               "Final Average Attack Acc": avg_acc, "Final Average Attack Acc Bound": avg_acc_bound})

            elif sync_ and ground_truth_type == "kmeans":
                wandb.log({"Average HR": avg_hr, "Average NDCG": avg_ndcg, "Average Attack Precision": avg_acc,
                           "Average Attack Recall": avg_recall,
                           "Round ": nb_rounds - round})
                if round == 0:
                    wandb.log({"Final Average HR": avg_hr, "Final Average NDCG": avg_ndcg,
                               "Final Attack Average Precision": avg_acc, "Final Attack Average Recall": avg_recall})

        if sync_ and ground_truth_type == "topk":
            cdf(self.hit_ratios[0], "Local HR")
            cdf(self.ndcgs[0], "Local NDCG")
            cdf(self.att_acc[0], "Attack Acc", topK_clustering)
            cdf(self.att_acc_bound[0], "Attack Acc bound", topK_clustering)
            wandb.finish()
            topks = [5,10,15,20]
            idx_round -= 1
            att_accs = []
            for topk in topks:
                self.clusters = self.groundTruth_TopKItemsLiked(topK = topk)
                accs = []
                for u in range(self.num_participants):
                    acc, _, _ = self.Accuracy_Topk_Attack(self.clusters, u, idx_round, topK=topk)
                    accs.append(acc) 
                att_accs.append(accs.copy())
            fig, ax = plt.subplots()
            ax.set_title('Attack accuracy distribution w.r.t different Topk Values')
            ax.boxplot(att_accs)
            ax.set_xticklabels(['5','10', '15','20'])
            fig.savefig("accuracy@topk.png")                       

       
        elif sync_ and ground_truth_type == "kmeans":
            cdf(self.hit_ratios[0], "Local HR")
            cdf(self.ndcgs[0], "Local NDCG")
            cdf(self.att_acc[0], "Attack Precision", topK_clustering)
            cdf(self.att_recall[0], "Attack Recall", topK_clustering)
            wandb.finish()

    def generateRandomCluster(self):
        return random.sample(range(self.num_participants), topK_clustering)

    def generateRandomBoundAccuracy(self, users_topk):
        rand_bound = []
        rand_accuracies = []
        for u in range(self.num_participants):
            rand_bound.append(self.generateRandomCluster())
            found_and_relevant = set(users_topk[u]) & set(rand_bound[u])
            acc = len(found_and_relevant) / len(users_topk[u])
            rand_accuracies.append(acc)

        return rand_accuracies
        
    def groundTruth_TopKItemsLiked_UnionSourceDestination_kulczynski2(self, topK = topK_clustering):
        users = []
        users_topk = defaultdict(list)
        for u in range(len(self.all_participants)):
            users.append(get_user_vector(u))


        for u in range(len(self.all_participants)):
            for v in range(len(self.all_participants)):
                if u != v:
                    users_topk[u].append((v, kulczynski2_similarity_bags([x[0] for x in users[u]],[x[0] for x in users[v]]) * kulczynski2_similarity_bags([x[1] for x in users[u]],[x[1] for x in users[v]])))
           
            users_topk[u].sort(key=lambda x: x[1], reverse=True)
            # print("user ",u," ground truth : ", users_topk[u][:topK])
            # sys.stdout.flush()
            users_topk[u] = [x[0] for x in users_topk[u]][:topK]
        
        return users_topk
    
    def groundTruth_TopKItemsLiked(self, topK = topK_clustering):
        users = []
        users_topk = defaultdict(list)
        for u in range(len(self.all_participants)):
            users.append(get_user_vector(u))

        for u in range(len(self.all_participants)):
            for v in range(len(self.all_participants)):
                if u != v:
                    users_topk[u].append((v, jaccard_similarity(users[u], users[v])))
            users_topk[u].sort(key=lambda x: x[1], reverse=True)
            # print("User ", u, " has true cluster@K :", users_topk[u])
            # sys.stdout.flush()
            users_topk[u] = [x[0] for x in users_topk[u]][:topK]

        return users_topk

    def groundTruth_TopK(self, topK = topK_clustering):
        users = []
        for u in range(len(self.all_participants)):
            vector = get_user_vector(u)
            dist = get_distribution_by_genre(vector)
            s = sum(dist)
            dist = [x / s for x in dist]
            users.append(dist)

        # scaler = StandardScaler(with_mean=False)
        # users = scaler.fit_transform(users) 
        users_topk = defaultdict(list)
        for u1 in range(len(self.all_participants)):
            for u2 in range(len(self.all_participants)):
                if u1 != u2:
                    users_topk[u1].append((u2, cosine_similarity(users[u1], users[u2])))

        for u in range(len(self.all_participants)):
            neighbours = list(users_topk[u])
            neighbours.sort(key=lambda x: x[1], reverse=True)
            users_topk[u] = [x[0] for x in neighbours]
            users_topk[u] = users_topk[u][:topK]

        return users_topk

    def groundTruth_Clustering(self):
        users = []
        for u in range(len(self.all_participants)):
            vector = get_user_vector(u)
            users.append(get_distribution_by_genre(vector))

        users = np.array(users)
        # scaler = StandardScaler(with_mean=False)
        # users = scaler.fit_transform(users)

        model = KMeans(n_clusters=clustersK, random_state=1245).fit(users)
        _labels = list(model.labels_)

        self.silhouette_avg = silhouette_score(users, _labels)
        print("For n_clusters = ", clustersK, "Average silhouette score is ", self.silhouette_avg)
        sys.stdout.flush()
        sample_silhouette_values = silhouette_samples(users, _labels)

        if sync_:
            wandb.log({"Average silhouette score": self.silhouette_avg})
            cdf(sample_silhouette_values, "Silhouette score value", topK=None)

        return _labels

    def Accuracy_Topk_Attack(self, users_topk, attacker_id, idx, topK = topK_clustering):
        interacted_with_fair_recall = []

        for u in users_topk[attacker_id]:
            if u in self.cluster_found[attacker_id][len(self.cluster_found[attacker_id]) - 1]:
                interacted_with_fair_recall.append(u)

        found_and_relevant = set(users_topk[attacker_id]) & set(self.cluster_found[attacker_id][idx][:topK])

        # size = len(self.cluster_found[attacker_id][idx])
        acc = len(found_and_relevant) / len(users_topk[attacker_id])

        if len(interacted_with_fair_recall) == 0:
            acc_bound = 0
        else:
            acc_bound = len(interacted_with_fair_recall) / len(users_topk[attacker_id])

        N = len(self.cluster_found[attacker_id][len(self.cluster_found[attacker_id]) - 1])
        if N == 0:
            random_acc = 0
        else:
            random_acc = RandomBoundAccuracy(
                N=len(self.cluster_found[attacker_id][len(self.cluster_found[attacker_id]) - 1]),
                a=len(interacted_with_fair_recall))

        return acc, acc_bound, random_acc

    def Accuracy_Clustering_Attack(self, clusters, attacker_id, idx):
        if idx > len(self.cluster_found[attacker_id]):
            return 1, 0

        cluster_user = []
        for u, c in enumerate(clusters):
            if c == clusters[attacker_id] and u != attacker_id:
                cluster_user.append(u)

        interacted_with_fair_recall = []
        for u in cluster_user:
            if u in cluster_user and u in self.cluster_found[attacker_id][idx]:
                interacted_with_fair_recall.append(u)

        found_and_relevant = set(cluster_user) & set(self.cluster_found[attacker_id][idx][:topK_clustering])

        acc = len(found_and_relevant) / len(self.cluster_found[attacker_id][idx][:topK_clustering])
        if len(cluster_user) == 0:
            recall = 1
        else:
            recall = len(found_and_relevant) / len(cluster_user)

        if len(interacted_with_fair_recall) == 0:
            recall_bound = 1
        else:
            recall_bound = len(found_and_relevant) / len(interacted_with_fair_recall)

        return acc, recall, recall_bound

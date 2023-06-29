import pickle
import re
from collections import defaultdict
from configparser import ConfigParser

def get_density_per_user():

    with open("pinterest-dense.train.rating") as f:
        lines = f.readlines()
        items = {}
        users = {}
        for l in lines:
            arr = l.split("\t")
            item = int(arr[1])
            user = int(arr[0])
            if item not in items:
                items[item] = 1
            else:
                items[item] += 1  
    
            if user not in users:
                users[user] = []
            users[user].append(item)

        user_density = {}
        for user in users:
            density = 0
            for item in users[user]:
                # print("item =",item)
                # print("rated by =",items[item])
                # print("density =",1 / items[item])
                density += items[item] / 4531
            user_density[user] = density/len(users[user])                
            
        
        return user_density

def averagePerf(logs = "FedFast-100k-20.txt"):
    with open(logs,"r") as f:
        nodes_performance = [] #defaultdict(list)
        nodes_performance_meta = [] #defaultdict(list)
        for line in f:
            
                # nodes_performance[re.findall("[0-9]+",prec_line.strip())[0]] = float(nextline.group(1)) if float(nextline.group(1)) > nodes_performance.get(re.findall("[0-9]+",prec_line.strip())[0],-1) else  nodes_performance[re.findall("[0-9]+",prec_line.strip())[0]]
            if nextline := re.match("^Local HR:.*([0-9]\.[0-9]+)",line):
                nodes_performance.append(float(nextline.group(1)))


    return (sum(nodes_performance) / len(nodes_performance))


def averagePerf(logs = "FedFast-foursquare20.txt", output="FedFast_HR20.txt", output1="FedFast_NDCG20.txt"):
    with open(logs,"r") as f:
        with open(output,"w") as out, open(output1,"w") as out1:
            nodes_performance = [] #defaultdict(list)
            nodes_performance_meta = [] #defaultdict(list)
            for line in f:
                
                if  nextline := re.match("^Local HR20:.*([0-9]\.[0-9]+)",line):
                    nodes_performance.append(float(nextline.group(1)))
                    out.write(nextline.group(1)+"\n")
                
                elif nextline := re.match("^Local NDCG20 :.*([0-9]\.[0-9]+)",line):
                    out1.write(nextline.group(1)+"\n")
                    


    return (sum(nodes_performance) / len(nodes_performance))


def overhead(cost_file_name="overhead.ini"):
    config_object = ConfigParser()
    config_object.read(cost_file_name)
    performance = config_object["performance"]
    min_perf_local_update = 100
    max_perf_local_update = 0
    num_perf = 0
    for i in range(100):
        if "average_local_update"+str(i) in performance:
            min_perf_local_update = float(performance["total_local_update"+str(i)]) if float(performance["total_local_update"+str(i)]) < min_perf_local_update else min_perf_local_update  
            max_perf_local_update = float(performance["total_local_update"+str(i)]) if float(performance["total_local_update"+str(i)]) > max_perf_local_update else max_perf_local_update  
        
            num_perf += 1

    print("Num participants :",num_perf)
    print("Min Update Time =",min_perf_local_update)
    print("Max Update Time =",max_perf_local_update)

# overhead()

     


print(averagePerf())

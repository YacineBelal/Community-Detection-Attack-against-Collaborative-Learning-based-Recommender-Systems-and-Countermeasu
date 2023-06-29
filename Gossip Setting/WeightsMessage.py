from pyopp import cMessage
import numpy as np


class WeightsMessage(cMessage):
    def __init__(self, name, kind = 0):
        cMessage.__init__(self,name, kind)
        self.weights = []
        self.age = 0
        self.cluster_found = None 
    

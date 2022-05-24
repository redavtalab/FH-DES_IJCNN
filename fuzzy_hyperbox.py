# -*- coding: utf-8 -*-

import numpy as np


class Hyperbox:
    def __init__(self,v=0,w=0,classifier=0,theta=.1):
        self.Min = v
        self.Max = w
        self.clsr = classifier
        self.Center = (v+w)/2
        #self.wCenter =(v+w)/2
        self.theta = theta
        #self.samples=[]
        #self.add_sample(v)
                
    def expand(self,x):
        self.Min = np.minimum(self.Min,x)
        self.Max = np.maximum(self.Max,x)
        self.Center = (self.Min + self.Max)/2
        #self.add_sample(x)
        
    def is_expandable(self,x):
        candidV = np.minimum(self.Min, x)
        candidW = np.maximum(self.Max, x) 
#        print(self.theta)
        return all( (candidW-candidV) < self.theta )
        
            
    def membership(self,x):

        disvec = np.abs(self.Center - x)
        halfsize = (self.Max - self.Min) / 2
        d = disvec - halfsize;
        m = np.linalg.norm(d[d>0])
        m = m / np.sqrt(len(x))  # adapting with high dimensional problems
        m= 1-m
        m = np.power(m,4)
        return m


        # d = np.linalg.norm(x-self.Center)
        #po= d/np.sqrt(len(x)-1)  # adapting with high dimensional problems
        #m = np.power(0.05,po)

#         t = np.sqrt(len(x))/3
#         m = (1-d/t)
#         if m<0:
#             m=0
#
 #        m = 1 - np.sqrt(np.sum((x-self.wCenter)**2))

        #y = 2
        #m = np.inf
        #ndimension = np.size(x)
        #for n in range(ndimension):
        #    m = np.minimum(m, np.minimum(1- self.f(x[n] - self.Max[n] , y) , 1-self.f(self.Min[n] - x[n],y)))
#       m = 1-m
#
#        return m
   
    #def add_sample(self,x):
    #    self.samples.append(x)
#        self.wCenter = ((len(self.samples)-1) * self.wCenter + x) / len(self.samples)
        
    def f(self, r,y):
        if r*y >1 :
            return 1
        if r*y >=0 :
            return r*y
        return 0
#    

   
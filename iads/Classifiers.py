# -*- coding: utf-8 -*-

"""
Package: iads
Fichier: Classifiers.py
Année: semestre 2 - 2018-2019, Sorbonne Université
"""

# Import de packages externes
import numpy as np
import pandas as pd
import random

# ---------------------------
class Classifier:
    """ Classe pour représenter un classifieur
        Attention: cette classe est une classe abstraite, elle ne peut pas être
        instanciée.
    """
 
     #TODO: A Compléter

    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
            Hypothèse : input_dimension > 0
        """
        raise NotImplementedError("Please Implement this method")
        
    def predict(self, x):
        """ rend la prediction sur x (-1 ou +1)
        """
        raise NotImplementedError("Please Implement this method")

    def train(self, labeledSet):
        """ Permet d'entrainer le modele sur l'ensemble donné
        """
        
        raise NotImplementedError("Please Implement this method")
    
    def accuracy(self, dataset):
        """ Permet de calculer la qualité du système 
        """
        total = 0
        j = 0
        for i in range(len(dataset.x)):
            score = self.predict(dataset.getX(i))
            total += 1
            if(score == dataset.getY(i)):
                j +=1
        return j/total

class ClassifierLineaireRandom(Classifier):
    """ Classe pour représenter un classifieur linéaire aléatoire
        Cette classe hérite de la classe Classifier
    """
    
    #TODO: A Compléter
    
    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
            Hypothèse : input_dimension > 0
        """
        self.input_dimension=input_dimension
        self.w=np.random.randn(2)
        
 
        
    def predict(self, x):
        """ rend la prediction sur x (-1 ou +1)
        """
        if np.dot(self.w,x)<0:
            return -1
        return 1
        raise NotImplementedError("Please Implement this method")

    def train(self, labeledSet):
        """ Permet d'entrainer le modele sur l'ensemble donné
        """   
        print("Pas d'apprentissage pour ce classifieur")     
     
    
# ---------------------------
class ClassifierKNN(Classifier):
    """ Classe pour représenter un classifieur par K plus proches voisins.
        Cette classe hérite de la classe Classifier
    """

    #TODO: A Compléter
 
    def __init__(self, input_dimension, k):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - k (int) : nombre de voisins à considérer
            Hypothèse : input_dimension > 0
        """
        self.input_dimension=input_dimension
        self.k=k
        
        
    def predict(self, x):
        """ rend la prediction sur x (-1 ou +1)
        """
        n=[]
        for i in range(self.d.size()):
            n.append(np.linalg.norm(x-self.d.getX(i)))
        res=0
        ind=np.argsort(n)
        for i in range(self.k):
            res+=self.d.getY(ind[i])
        if (res*1.0>=0):
            return +1
        return -1
        

    def train(self, labeledSet):
        """ Permet d'entrainer le modele sur l'ensemble donné
        """ 
        self.d=labeledSet       
        

# ---------------------------
class ClassifierPerceptronKernel(Classifier):
    
    def __init__(self,dimension_kernel,learning_rate,nb_Iterations,kernel):
        """ Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - learning_rate :
            Hypothèse : input_dimension > 0
        """
        self.dimension_kernel=dimension_kernel
        self.n=learning_rate
        self.nb_Iterations=nb_Iterations
        self.kernel = kernel
        self.w=np.random.randn(dimension_kernel)
        
        
    def train(self, labeledSet):
        self.labeledSet = labeledSet
        i = 0 
        while (i < self.nb_Iterations):
            index_aleatoire = random.randint(0,labeledSet.size()-1)
            if(labeledSet.getY(index_aleatoire)*self.predict(labeledSet.getX(index_aleatoire)) < 0 ):
                    self.w = self.w+self.n*labeledSet.getY(index_aleatoire)*self.kernel.transform(labeledSet.getX(index_aleatoire))
            i +=1
    
    def predict(self, x):
        z = np.dot(self.kernel.transform(x), self.w)
        if z > 0:
            return +1
        else:
            return -1



class ClassfierPerceptron(Classifier):
    """ Perceptron de Rosenblatt
    """
    def __init__(self,input_dimension,learning_rate):
        """ Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - learning_rate :
            Hypothèse : input_dimension > 0
        """
        self.input_dimension = input_dimension
        self.w=np.random.randn(input_dimension)*10*learning_rate
        self.n=learning_rate
    

    def predict(self,x):
        """ rend la prediction sur x (-1 ou +1)
        """
        
        z= np.dot(x,self.w)
        if z>0:
            return +1
        else:
            return -1

    
    def train(self,labeledSet,nb_iter):
        """ Permet d'entrainer le modele sur l'ensemble donné
        """
        self.labeledSet = labeledSet
    
        for i in range (nb_iter):
            index_aleatoire = random.randint(0,labeledSet.size()-1)
            if(labeledSet.getY(index_aleatoire)*self.predict(labeledSet.getX(index_aleatoire)) < 0 ): # Yifw(x) < 0 x est mal classé
                    self.w = self.w+self.n*labeledSet.getX(index_aleatoire)*labeledSet.getY(index_aleatoire)
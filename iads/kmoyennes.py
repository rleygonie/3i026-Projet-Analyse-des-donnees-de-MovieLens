# -*- coding: utf-8 -*-

"""
Package: iads
Fichier: kmoyennes.py
Année: semestre 2 - 2018-2019, Sorbonne Université
"""

# ---------------------------
# Fonctions pour les k-moyennes

# Importations nécessaires pour l'ensemble des fonctions de ce fichier:

import numpy as np
import pandas as pd
from datetime import datetime as dt
import matplotlib.pyplot as plt 
import random as rd
import math

# ---------------------------
# Dans ce qui suit, remplacer la ligne "raise.." par les instructions Python
# demandées.
# ---------------------------

# Normalisation des données :

# ************************* Recopier ici la fonction normalisation()
def normalisation(DF):
    """ DataFrame -> DataFrame
        rend le dataframe obtenu par normalisation des données selon 
             la méthode vue en cours 8.
    """
    return (DF - DF.min())/(DF.max()-DF.min())

# -------
# Fonctions distances

# ************************* Recopier ici la fonction dist_vect()
def dist_vect(S1, S2):
    """ Series**2 -> float
        rend la valeur de la distance euclidienne entre les 2 vecteurs
    """
    return np.sqrt(np.sum((S1-S2)**2))

# -------
# Calculs de centroïdes :
# ************************* Recopier ici la fonction centroide()
def centroide(DF):
    """ DataFrame -> DataFrame
        Hypothèse: len(M) > 0
        rend le centroïde des exemples contenus dans M
    """
    return pd.DataFrame(DF.mean()).T

def dist_vect1(S1,S2):
    S1 = S1.as_matrix()
    S2 = S2.as_matrix().reshape(2)
    print(S1.shape, S2.shape)
    return np.sqrt(np.sum((S1-S2)**2,axis=1))
    
    
# -------
# Inertie des clusters :
# ************************* Recopier ici la fonction inertie_cluster()
def inertie_cluster(DF):
    """ DataFrame -> float
        DF: DataFrame qui représente un cluster
        L'inertie est la somme (au carré) des distances des points au centroide.
    """
    return np.sum(dist_vect1(DF,centroide(DF))**2)


# -------
# Algorithmes des K-means :
# ************************* Recopier ici la fonction initialisation()
def initialisation(K,DF):
    """ int * DataFrame -> DataFrame
        K : entier >1 et <=n (le nombre d'exemples de DF)
        DF: DataFrame contenant n exemples
    """
    K_dataFrame.append(pd(columns=list(DF.columns))
    
    K_dataFrame.append(DF)
    for n in range(0,K):
        i=rd.randint(0,len(DF)-1)
        x = DF.iloc[i]
        print(i)
        print(K_dataFrame)
        K_dataFrame.iloc[i]=x
    return K_dataFrame


# -------
# ************************* Recopier ici la fonction plus_proche()
def plus_proche(exemple,DF):
    """ Series * DataFrame -> int
        exemple : Series contenant un exemple
        DF : DataFrame contenant les K centres
    """
    Min = 1000
    ind=-1000
    for i in range(0,len(DF)):
        minTempo=dist_vect(DF.iloc[i],exemple)
        if (Min>minTempo):
            Min=minTempo
            ind=i
    return ind

# -------
# ************************* Recopier ici la fonction affecte_cluster()
def affecte_cluster(DF,centroide):
    """ DataFrame * DataFrame -> dict[int,list[int]]
        Base: DataFrame contenant la base d'apprentissage
        Centres : DataFrame contenant des centroides
    """
    matrice=dict()
    for i in range(len(centroide)):
        matrice[i]=[]
    for j in range(len(DF)):
        c=plus_proche(DF.loc[j],centroide)
        matrice[c].append(j)
    return matrice

# -------
# ************************* Recopier ici la fonction nouveaux_centroides()
def nouveaux_centroides(DF,matrice):
    """ DataFrame * dict[int,list[int]] -> DataFrame
        Base : DataFrame contenant la base d'apprentissage
        U : Dictionnaire d'affectation
    """
    df=pd.DataFrame(columns=list(DF.columns))
    for cluster, exemple in matrice.items():
        moyenne=0
        for i in range (len(exemple)):
            moyenne=moyenne+DF.iloc[exemple[i]]
        df.loc[cluster]=1.0*moyenne/len(exemple)
    return df

# -------
# ************************* Recopier ici la fonction inertie_globale()
def inertie_globale(DF, matrice):
    """ DataFrame * dict[int,list[int]] -> float
        Base : DataFrame pour la base d'apprentissage
        U : Dictionnaire d'affectation
    """
    nouveaux=nouveaux_centroides(DF,matrice)
    inertie=0
    
    for cluster, exemple in matrice.items():
        df=pd.DataFrame(columns=list(DF.columns))
        df=DF.iloc[exemple]
        inertie+=inertie_cluster(df)
    return inertie
# -------
# ************************* Recopier ici la fonction kmoyennes()
def kmoyennes(K, DF, epsilon, iter_max):
    """ int * DataFrame * float * int -> tuple(DataFrame, dict[int,list[int]])
        K : entier > 1 (nombre de clusters)
        DF : DataFrame pour la base d'apprentissage
        epsilon : réel >0
        iter_max : entier >1
    """
    centroid=initialisation(K,DF)
    inertie=inertie_cluster(DF)
    
    for i in range(0,iter_max):
        matrice=affecte_cluster(DF,centroid)
        Nouvelle_inertie=inertie_globale(DF,matrice)
        inertie=Nouvelle_inertie
        centroid=nouveaux_centroides(DF,matrice)
        
    return centroid, matrice
# -------
# Affichage :
# ************************* Recopier ici la fonction affiche_resultat()
def affiche_resultat(df, centroid, da):
    
    cmap = plt.cm.get_cmap("hsv", len(centroid)+1)

    for k, liste in da.items():
        
        df_cluster = pd.DataFrame(columns=list(df.columns))

        for i in range(len(liste)):
            df_cluster.loc[i]=df.iloc[liste[i]]
        
        plt.scatter(df_cluster['X'],df_cluster['Y'],color=cmap(k))
    
    plt.scatter(centroid['X'],centroid['Y'],color='r',marker='x')
# -------

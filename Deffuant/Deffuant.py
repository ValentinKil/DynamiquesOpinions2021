# -*- coding: utf-8 -*-
# Introduction des Library
import numpy as np
import networkx as nx
from numba import njit,jit
from tqdm import tqdm


''' On introduit ici toutes les fonctions dont on aura besoins pour étudier le modèle de Deffuant, le graphs représentants les conexions entre agents peut être n'importe quel graphe non orienté issus de la bibliothéque Networkx et l'espace d'opinion considéré est l'intervalle [0,1]'''

# Fonction de simulation du modèle de Deffuant

def Etape(G,L,d,mu,nbu=100):

    '''Fonction qui réalise nbu étapes du modèle de Deffuant consistant chacune à choisir aléatoirement une arrête du graph et a mettre à jour l'opinion des agents à ses extrémités.

    Argument :
    G   : graphe Networkx
    L   : np.array des opinions (la taille doit être égale au nombre d'agent et chaque coordnnée doit être un flotant entre 0 et 1
    d   : palié de Deffuant (0<=d<=1)
    mu  : capacité d'adaptation (0<=mu<=1/2)
    nbu :nombre d'étape élémentaire réalisé en une fois (par défaut nbu=100)

    Sortie : le nouveau np.array des opinions'''
    N=len(L)
    for loop in range(nbu):
        i=np.random.randint(0,N)
        j=np.random.choice(list(G.neighbors(i)))
        a=L[i]-L[j]
        if abs(a)<d:
            L[i]-=mu*a
            L[j]+=mu*a
    return L

@njit(nogil=True)
def Etape_grille(L,d,mu,nbu=100):
    l=len(L)
    for loop in range(nbu):
        i1,j1=np.random.randint(0,l,size=2)
        test=np.random.randint(0,3)
        i2,j2=0,0
        if test==0 and i1-1>=0 :
            i2=i1-1
            j2=j1
        elif test==1 and i1+1<l:
            i2=i1-1
            j2=j1
        elif test==2 and j1-1>=0:
            i2=i1
            j2=j1-1
        elif test==3 and j1+1<l:
            i2=i1
            j2=j2+1
        else:
            break
        a=L[i1,j1]-L[i2,j2]
        if abs(a)<d:
            L[i1,j1]-=mu*a
            L[i2,j2]+=mu*a
    return L


@njit(nogil=True)
def Etape_complet(L,d,mu,nbu=100):
    N=len(L)
    for loop in range(nbu):
        i,j=np.random.randint(0,N,size=2)
        a=L[i]-L[j]
        if abs(a)<d:
            L[i]-=mu*a
            L[j]+=mu*a
    return L

def Simulator(G,d,mu,t,nbu=100,conv=False,steps=1000,trial=5,ndigits=2):

    ''' Fonction qui simule l'évolution du modéle de Deffuant sur un graph G à partir d'une opinion aléatoire uniforme sur une échelle de temps nbu*t.

    Argument :
    G   : graphe Networkx
    d   : palié de Deffuant (0<=d<=1)
    mu  : capacité d'adaptation (0<=mu<=1/2)
    t   : temps de simulation
    nbu : multiplicateur du temps de simulation  ( par défaut nbu=100)
    conv: si True ne retourne un résulat qu'une fois la convergence atteint ou si la convergence a échoué (par défaut conv=False
    steps: nombre d'étapes entre 2 tests de convergence (par défaut =1000)
    trial: nombre de test de convergence réaliser au maximum, au bout de trial essais la fonction retroune le résultat même si la convergence n'as pas eux lieu (par défaut trial=5)
    ndigits : précision du test de convergence (par défaut ndigits =2)

    Sortie : np.array de taille t*N contenant le vecteur d'opinion à l'Etape i pour i entre 0 et t'''

    N=G.order()
    L=np.random.uniform(0,1,N)
    M=[]
    M.append(L)
    for loop in tqdm(range(t-1)):
        H=M[-1].copy()
        M.append(Etape(G,H,d,mu,nbu=nbu))
    if conv:
        test=Compare(M[-1],M[-steps])
        compteur=0
        while not test and compteur <trial:
            print(compteur)
            for loop in range(steps):
                H=M[-1].copy()
                M.append(Etape(G,H,d,mu,nbu=nbu))
            test=Compare(M[-1],M[-steps])
            compteur+=1
    return np.array(M)


@njit(nogil=True)
def Simulator_grille (l,d,mu,t,nbu=100):

    L=np.random.uniform(0,1,size=(l,l))
    M=np.empty((t,l,l))
    M[0]=L
    for i in range(t-1):
        M[i+1]=Etape_grille(M[i],d,mu,nbu=nbu)
    return M


@njit(nogil=True)
def Simulator_complet (N,d,mu,t,nbu=100):

    L=np.random.uniform(0,1,size=N)
    M=np.empty((t,N))
    M[0]=L
    for i in range(t-1):
        M[i+1]=Etape_complet(M[i],d,mu,nbu=nbu)
    return M

def Etude_pics(G,d,mu,t,nbu=100,ndigits=2,nb=50,conv=False,steps=1000,trial=5):

    '''Fonction donnant le nombres de pics obtenue en simulant nb modele de Deffuant sur le graphe G avec distribution initiale uniforme, pics comptés avec une précision ndigits jusqu'au temps nb pour une valeur de palié et de capacité d'adaptation donné.

    Argument :
    G   : graphe Networkx
    d   : palié de Deffuant (0<=d<=1)
    mu  : capacité d'adaptation (0<=mu<=1/2)
    t   : temps de simulation
    nbu : multiplicateur du temps de simulation  ( par défaut nbu=100)
    ndigits : précision de détection des pics et du test de convergence( par défaut ndigit=2)
    nb : nombre de simulation réalisé (par défaut nb=50)
    conv: si True ne compte un résulat qu'une fois la convergence atteint ou si la convergence a échoué (par défaut conv=False
    steps: nombre d'étapes entre 2 tests de convergence (par défaut =1000)
    trial: nombre de test de convergence réaliser au maximum, au bout de trial essais la fonction retroune le résultat même si la convergence n'as pas eux lieu (par défaut trial=5)

    Sortie : np.array de taille nb contenant les nb valeurs de nombre de pics obtenue pour chaque simulation'''

    Mat=[]
    for i in range(nb):
        Ma=Simulator(G,d,mu,t,nbu=nbu,conv=conv,steps=steps,trial=trial,ndigits=ndigits)
        n=len(Detecte_pic(Ma[-1],ndigits=ndigits))
        Mat.append(n)
    return np.array(Mat)

def Etude_pics_grille(l,d,mu,t,nbu=100,ndigits=2,nb=50):

    Mat=[]
    for i in range(nb):
        Ma=Simulator_grille(l,d,mu,t,nbu=nbu)
        n=len(Detecte_pic(Ma[-1],ndigits=ndigits))
        Mat.append(n)
    return np.array(Mat)

def Etude_pics_complet(N,d,mu,t,nbu=100,ndigits=2,nb=50):

    Mat=[]
    for i in range(nb):
        Ma=Simulator_complet(N,d,mu,t,nbu=nbu)
        n=len(Detecte_pic(Ma[-1],ndigits=ndigits))
        Mat.append(n)
    return np.array(Mat)



def Etude_pics_vect(G,D,mu,t,nbu=100,ndigits=2,nb=50,conv=False,steps=1000,trial=5):
    '''
    Version vectorialisé de la fonction précédente permettant de tester pour plusieurs valeurs de d à la suite

    Argument :
    G   : graphe Networkx
    D   : vecteur de paliés de Deffuant
    mu  : capacité d'adaptation (0<=mu<=1/2)
    t   : temps de simulation
    nbu : multiplicateur du temps de simulation  ( par défaut nbu=100)
    ndigits : précision de détection des pics et du test de convergence ( par défaut ndigit=2)
    nb : nombre de simulation réalisé (par défaut nb=50)
    conv: si True ne compte un résulat qu'une fois la convergence atteint ou si la convergence a échoué (par défaut conv=False
    steps: nombre d'étapes entre 2 tests de convergence (par défaut =1000)
    trial: nombre de test de convergence réaliser au maximum, au bout de trial essais la fonction retroune le résultat même si la convergence n'as pas eux lieu (par défaut trial=5)

    Sortie : np.array de taille len(D)*nb contenant pour chaque d dans D les nb valeurs de nombre de pics obtenue pour chaque simulation'''

    M=[]
    for d in D:
        M.append(Etude_pics(G,d,mu,t,nbu=nbu,ndigits=ndigits,nb=nb,conv=conv,steps=steps,trial=trial))
    return np.array(M)

def Etude_pics_vect_grille(l,D,mu,t,nbu=100,ndigits=2,nb=50):

    M=[]
    for d in D:
        M.append(Etude_pics_grille(l,d,mu,t,nbu=nbu,ndigits=ndigits,nb=nb))
    return np.array(M)

def Etude_pics_vect_complet(N,D,mu,t,nbu=100,ndigits=2,nb=50):

    M=[]
    for d in D:
        M.append(Etude_pics_complet(N,d,mu,t,nbu=nbu,ndigits=ndigits,nb=nb))
    return np.array(M)



# Fonction d'analyse

def Detecte_pic(L,ndigits=2):

    '''Fonction qui renvois la liste des pics observé dans un array d'opinion

    Argument :
    L: np.array des opinions
    ndigits : précision de détection des pics (valeur par défaut ndigit=2)

    Sortie : liste de la valeurs des pics observés'''
    List=[]
    L=L.flatten()
    for car in L :
        car=np.round(car,decimals=ndigits)
        if (not car in List) :
            List.append(car)
    return List


def Moyenne(Mat):

    '''Renvois le vecteur des moyennes des colones d'un np.array'''

    n=len(Mat)
    Toreturn=np.zeros(n)
    for i in range(n):
        Toreturn[i]=np.mean(Mat[i])
    return Toreturn


def Variance(Mat):

    '''Renvois le vecteur des variances des colones d'un np.array'''

    n=len(Mat)
    Toreturn=np.zeros(n)
    for i in range(n):
        Toreturn[i]=np.var(Mat[i])
    return Toreturn


def Compare(M1,M2,ndigits=2):
    '''Compare 2 vecteurs d'opinions en arrondissant'''
    n=len(M1)
    test=True
    for i in range(n):
        if np.round(M1[i],decimals=ndigits)!=np.round(M2[i],decimals=ndigits):
            test=False
    return test


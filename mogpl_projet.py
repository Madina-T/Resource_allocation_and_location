#!/usr/bin/env python
# coding: utf-8

# # Projet MOGPL

# #### Antonin ARBERET - 3407709
# #### Madina TRAORÉ - 3412847

# ### Imporations

# In[78]:


from gurobipy import *
import string
import re
from matplotlib import pyplot as plt
import numpy as np


# ### Chargement des fichiers en mémoire

# In[79]:


#Lecture du fichier villes, retourne une liste de str contenant les noms des villes

def parse_villes(nom_fichier):
    villes=[]
    fichier = open(nom_fichier, "r")
    regex = re.compile(r'[\n\r\t]')
    for ligne in fichier:
        villes.append(regex.sub('', ligne))
    fichier.close()
    return villes


# In[80]:


#Lecture du fichier populations, retourne une liste de int
#contenant les populations dans l'ordre alphabetique de leur villes respectives

def parse_populations(nom_fichier):
    pop=[]
    fichier = open(nom_fichier, "r")
    regex = re.compile(r'[\n\r\t]')
    for ligne in fichier:
        valeur=ligne.split(',')
        pop.append(int(regex.sub('', valeur[1])))
    fichier.close()
    return pop


# In[81]:


#Lecture du fichier coordonées, retourne une liste de couple 
#contenant les deux coordonnées dans l'ordre alphabetique de leur villes respectives

def parse_coordonnees(nom_fichier):
    coord=[]
    fichier = open(nom_fichier, "r")
    regex = re.compile(r'[\n\r\t]')
    for ligne in fichier:
        valeur=ligne.split(',')
        coord.append((int((regex.sub('', valeur[1]))),int((regex.sub('', valeur[2])))))
    fichier.close()
    return coord


# In[82]:


#Paramètres: fichier distance, nombre de villes, retourne une liste de liste 
#contenant les distance dans l'ordre alphabetique de leur villes respectives :
#dist[i][j] = distance de la i-ème à la j-ème ville dans la liste

def parse_distances(nom_fichier, nb_villes):
    dist=[]
    fichier = open(nom_fichier, "r")
    regex = re.compile(r'[\n\r\t]')
    cpt=0
    ind=-1
    for ligne in fichier:
        if(cpt%(nb_villes+1)==0):
            ind+=1
            dist.append([])
        else:
            valeur=ligne.split(',')
            dist[ind].append(float(regex.sub('', ligne)))
        cpt+=1
    fichier.close()
    return dist


# In[83]:


villes=parse_villes("Data-20181122/villes92.txt")
pop=parse_populations("Data-20181122/populations92.txt")
coord=parse_coordonnees("Data-20181122/coordvilles92.txt")
dist=parse_distances("Data-20181122/distances92.txt", len(villes))
carte='Data-20181122/92.png'


# ### Fonctions d'affichage

# In[84]:


#Fonction de gestion des couleurs pour les graphes

def get_color(k):
    colors=['b-','g-','r-','c-','m-','y-']
    return colors[k%len(colors)]


# In[85]:


#Paramètres : image de la carte, matrice retourné par les fonction d'optimisations, liste des indices
#des ressources liste des coordonnées 
#Affiche la carte avec les relations villes ressources

def trace_reseau(fich_image, X, v_ressources, coord):
    img = plt.imread(fich_image)
    plt.imshow(img)
    
    for k in range(len(v_ressources)):
        coordk= coord[v_ressources[k]]
        c=get_color(k)
        plt.plot(coordk[0],coordk[1], c, marker='D')
    
    for i in range(len(X)):
        for j in range(len(X[i])):
            if(X[i][j]==1):
                c=get_color(j)
                (x1, y1)=coord[i]
                (x2, y2)=coord[v_ressources[j]]
                plt.plot([x1, x2], [y1, y2], c, lw=1)
    plt.show()


# In[86]:


#Paramètres : image de la carte, matrice retourné par les fonction d'optimisations, liste des indices
#des ressources, liste des coordonnées, fichier pour l'enregistrement
#Affiche et enregistre dans dest la carte avec les relations villes ressources

def save_reseau(fich_image, X, v_ressources, coord,dest):
    img = plt.imread(fich_image)
    plt.imshow(img)
    
    for k in range(len(v_ressources)):
        coordk= coord[v_ressources[k]]
        c=get_color(k)
        plt.plot(coordk[0],coordk[1], c, marker='D')
    
    for i in range(len(X)):
        for j in range(len(X[i])):
            if(X[i][j]==1):
                c=get_color(j)
                (x1, y1)=coord[i]
                (x2, y2)=coord[v_ressources[j]]
                plt.plot([x1, x2], [y1, y2], c, lw=1)
    plt.savefig("%s.png"%(dest))


# ### Fonctions de manipulation diverses

# In[87]:


#Paramètres : liste des populations
#retourne la population totale

def get_pop_tot(populations):
    tot=0
    for p in populations:
        tot+=p
    return tot


# ### Question 1

# In[88]:


#Paramètres: nombre de ville, nombre de ressources, facteur alpha, liste des indices des villes
#contenant les ressources, matrice de distances. Retourne une matrice X telle que X[i][j]=1 
#si la ville d'indice i dépend de la ville contenant une ressources d'indice j dans v_ressources
#Retourne une matrice |I|*k contenant les x_ij

#k est redondant avec la taille de la liste v_ressources mais simplifie la lecture

def optimisation_Q1(nbvilles, k, alpha, v_ressources, dist):
    nbcont=nbvilles+k
    nbvar=nbvilles*k

    # Range of plants and warehouses

    lignes = range(nbcont)
    colonnes = range(nbvar)

    # Matrice des contraintes
    # Coefficients dans a, seconds membres dans b
    a=[]
    b=[]

    pop_tot= get_pop_tot(pop)

    # Ajout de i contraintes assurant qu'une ville n'est associées qu'a une seule ressource
    # Type (a)

    for i in range(nbvilles):
        ligne=[]

        for j in range(nbvilles*k):
            if(i*k<=j and (i+1)*k>j):
                ligne.append(1)
            else:
                ligne.append(0)
        b.append(1)
        a.append(ligne)

    # Ajout de j contraintes assurant que la population associé à un équipement ne dépasse pas la limite
    # Type (b)
    
    for j in range(k):
        ligne=[]
        for ij in range(nbvilles*k):
            if((ij-j)%k==0):
                i=np.floor(ij*1.0/k)
                ligne.append(pop[int(i)])
            else:
                ligne.append(0)
        a.append(ligne)
        b.append(pop_tot*(1+alpha)/k)


    # Coefficients de la fonction objectif
    # Ajout de i*j coefficient correspondant à la distance entre la ville et la ressource concernées

    c=[]
    for i in range(nbvilles):
        for j in v_ressources:
            c.append(dist[i][j])


    m = Model("mogplex")     

    # declaration variables de decision binaires
    x = []
    for i in colonnes:
        x.append(m.addVar(vtype=GRB.INTEGER, lb=0, name="x%d" % (i+1)))

    # maj du modele pour integrer les nouvelles variables
    m.update()

    obj = LinExpr();
    obj =0
    for j in colonnes:
        obj += c[j] * x[j]

    # definition de l'objectif
    m.setObjective(obj,GRB.MINIMIZE)

    # Definition des signes des contraintes
    for i in lignes:
        if i <nbvilles:
            m.addConstr(quicksum(a[i][j]*x[j] for j in colonnes) >= b[i], "Contrainte%d" % i)
        else:
            m.addConstr(quicksum(a[i][j]*x[j] for j in colonnes) <= b[i], "Contrainte%d" % i)


    # Resolution
    m.optimize()

    #Construction de la matrice
    X=[]
    ind=-1
    for j in colonnes:
        if(j%k==0):
            X.append([])
            ind=ind+1
        X[ind].append(x[j].x)
    return X


# In[89]:


#Paramètres : Matrice renvoyée par une fonction d'optimisation, matrice des distances, liste des ressources
#Retourne un couple (satisfaction moy, satisfaction min)

def satisfaction(X, dist, v_ressources):
    dmax=0
    dmoy=0.
    for i in range(len(X)):
        for j in range(len(X[i])):
            if(X[i][j]==1):
                d=dist[i][v_ressources[j]]
                dmoy+=d
                if(d>dmax):
                    dmax=d
    dmoy=dmoy/len(X)
    return(1/dmoy,1/dmax)


# ### Question 2

# In[90]:


#Paramètres: nombre de ville, nombre de ressources, facteur alpha, liste des indices des villes
#contenant les ressources, matrice des distances, facteur epsilon
#Retourne une matrice |I|*k contenant les x_ij

#Les nbvilles*k premières variables représentent la dépendance d'une ville vers une des k villes
#comportant une ressource.
#Les nbvilles*k suivante représentent la valeur max : la seule de ses variables à 1 est celle de la plus grande
#distance ville-ressource utilisée par le modèle actuellement
#
#Retourne une matrice X telle que X[i][j]=1 si la ville d'indice i
#dépend de la ville contenant une ressources d'indice j dans v_ressources
#k est redondant avec la taille de la liste v_ressources mais permet d'expliciter

def optimisation_Q2(nbvilles, k, alpha, v_ressources, dist, eps):
    nbcont=nbvilles+k+nbvilles*k+1
    nbvar=2*nbvilles*k

    # Range of plants and warehouses

    lignes = range(nbcont)
    colonnes = range(nbvar)

    # Matrice des contraintes
    # Coefficients dans a, seconds membres dans b
    a=[]
    b=[]

    pop_tot= get_pop_tot(pop)

    # Ajout de i contraintes assurant qu'une ville n'est associées qu'a une seule ressource
    # type (a)

    for i in range(nbvilles):
        ligne=[]

        for j in range(nbvilles*k):
            if(i*k<=j and (i+1)*k>j):
                ligne.append(1)
            else:
                ligne.append(0)
        for j in range(nbvilles*k):
            ligne.append(0)
        b.append(1)
        a.append(ligne)
        


    # Ajout de j contraintes assurant que la population associé à un équipement ne dépasse pas la limite
    #type (b)
    
    for j in range(k):
        ligne=[]
        for ij in range(nbvilles*k):
            if((ij-j)%k==0):
                i=np.floor(ij*1.0/k)
                ligne.append(pop[int(i)])
            else:
                ligne.append(0)
        for j in range(nbvilles*k):
            ligne.append(0)
        a.append(ligne)
        b.append(pop_tot*(1+alpha)/k)

    # Ajout de nbvilles*k contraintes assurant que la variable à 1 est bien celle de la distance maximale active
    # Type (c)
    
    for i in range(nbvilles*k):
        ligne=[]

        for j in range(nbvilles*k):
            if(j==i):
                ligne.append(-dist[int(math.floor(j/k))][v_ressources[j%k]])
            else:
                ligne.append(0)
        for j in range(nbvilles*k):
            ligne.append(dist[int(math.floor(j/k))][v_ressources[j%k]])
            
        b.append(0)
        a.append(ligne)
    
    # ajout d'une contrainte assurant que le total des variables représentant le max est 1
    # Type (d)
    ligne=[]

    for j in range(nbvilles*k):
        ligne.append(0)
    for j in range(nbvilles*k):
        ligne.append(1)

    b.append(1)
    a.append(ligne)
    
   


    # Coefficients de la fonction objectif
    # Ajout de i*j coefficient pondérés par epsilon correspondant à la distance entre la ville 
    # et la ressource concernées puis ajout de nbville*k coefficient pour les variables représentant
    # le maximum

    c=[]
    for i in range(nbvilles):
        for j in v_ressources:
            c.append(eps*dist[i][j])
    for j in range(nbvilles*k):
        c.append(dist[int(math.floor(j/k))][v_ressources[j%k]])


    m = Model("mogplex")     

    # declaration variables de decision binaires
    x = []
    for i in colonnes:
        x.append(m.addVar(vtype=GRB.BINARY, lb=0, name="x%d" % (i+1)))

    # maj du modele pour integrer les nouvelles variables
    m.update()

    obj = LinExpr();
    obj =0
    for j in colonnes:
        obj += c[j] * x[j]

    # definition de l'objectif
    m.setObjective(obj,GRB.MINIMIZE)

    # Definition des contraintes, on assure que la somme des variables de décision associée à une 
    # Definition des signes des contraintes
    for i in lignes:
        if i <nbvilles:
            m.addConstr(quicksum(a[i][j]*x[j] for j in colonnes) >= b[i], "Contrainte%d" % i)
        elif i <nbvilles+k:
            m.addConstr(quicksum(a[i][j]*x[j] for j in colonnes) <= b[i], "Contrainte%d" % i)
        elif i<nbvilles+k+nbvilles*k:
            m.addConstr(quicksum(a[i][j]*x[j] for j in colonnes) >= b[i], "Contrainte%d" % i)
        else:
            m.addConstr(quicksum(a[i][j]*x[j] for j in colonnes) == b[i], "Contrainte%d" % i)
            


    # Resolution
    m.optimize()


    #Construction de la matrice
    
    X=[]
    ind=-1
    for j in colonnes[:nbvilles*k]:
        if(j%k==0):
            X.append([])
            ind=ind+1
        X[ind].append(x[j].x)
    return X


# In[91]:


#Prix de l'équité 
#Paramètres : Matrices Xf et Xg rendues par les fonctions d'optimisation sur une meme instance, liste 
#des resssources, matrice des distances
#Retourne le prix de l'équité

def PE(Xf, Xg, v_ressources, dist):
    dtotf=0.
    dtotg=0.
    for i in range(len(Xf)):
        for j in range(len(Xf[i])):
            if(Xf[i][j]==1):
                d=dist[i][v_ressources[j]]
                dtotf+=d
    for i in range(len(Xg)):
        for j in range(len(Xg[i])):
            if(Xg[i][j]==1):
                d=dist[i][v_ressources[j]]
                dtotg+=d
                
    pe=1-dtotf/dtotg
    return(pe)


# ### Question 3

# In[92]:


#Paramètres: nombre de ville, nombre de ressources, facteur alpha, matrice des distances, facteur epsilon 
#Retourne une matrice X telle que X[i][j]=1 si la ville d'indice i


def optimisation_Q3(nbvilles, k, alpha, dist, eps):
    nbcont=2*nbvilles**2+2*nbvilles+2
    nbvar=2*nbvilles**2+nbvilles

    # Range of plants and warehouses

    lignes = range(nbcont)
    colonnes = range(nbvar)

    # Matrice des contraintes
    # Coefficients dans a, seconds membres dans b
    a=[]
    b=[]

    pop_tot= get_pop_tot(pop)

    # Ajout de i contraintes assurant qu'une ville n'est associées qu'a une seule ressource
    # type(a)

    for i in range(nbvilles):
        ligne=[]

        for j in range(nbvar):
            if(i*nbvilles<=j and (i+1)*nbvilles>j):
                ligne.append(1)
            else:
                ligne.append(0)
        b.append(1)
        a.append(ligne)
        
    # Ajout de i contraintes assurant que la ressource provient toujours d'une ville qui en est équipée
    # type (e)

    for i in range(nbvilles):
        for j in range(nbvilles):
            ligne=[]
            for n in range(nbvar):
                if(n==nbvilles*i+j):
                    ligne.append(1)
                elif(n==2*nbvilles**2+j):
                    ligne.append(-1)
                else:
                    ligne.append(0)
            b.append(0)
            a.append(ligne)
        


    # Ajout de j contraintes assurant que la population associé à un équipement ne dépasse pas la limite
    # type(b)

    for j in range(nbvilles):
        ligne=[]
        for ij in range(nbvar):
            if(ij<nbvilles**2):
                if((ij-j)%k==0):
                    i=np.floor(ij*1.0/nbvilles)
                    ligne.append(pop[int(i)])
                else:
                    ligne.append(0)
            else:
                ligne.append(0)
        a.append(ligne)
        b.append(pop_tot*(1+alpha)/k)

    # Ajout de nbvilles**2 contraintes assurant que la variable à 1 est bien celle de la distance maximale active
    # type(c)
    
    for i in range(nbvilles**2):
        ligne=[]
        for j in range(nbvar):
            if(j < nbvilles**2):
                if(j==i):
                    ligne.append(-dist[int(math.floor(j/nbvilles))][j%nbvilles])
                else:
                    ligne.append(0)
            elif(j < 2*nbvilles**2):
                ligne.append(dist[int(math.floor((j-nbvilles**2)/nbvilles))][(j-nbvilles)%nbvilles])
            else:
                ligne.append(0)
        b.append(0)
        a.append(ligne)
    
    #ajout de la contrainte assurant que le total des variables représentant le max est 1
    #type(d)
    
    ligne=[]
    
    for j in range(nbvar):
        if(j <nbvilles**2):
            ligne.append(0)
        elif(j<2*nbvilles**2):
            ligne.append(1)
        else:
            ligne.append(0)

    b.append(1)
    a.append(ligne)
    

    #ajout de la contraintes assurant que le total des ville équipée est k
    #type (f)
    
    ligne=[]
    
    for j in range(nbvar):
        if(j>=nbvar-nbvilles):
            ligne.append(1)
        else:
            ligne.append(0)

    b.append(k)
    a.append(ligne)


    # Coefficients de la fonction objectif
    # Ajout de i*j coefficient pondérés par epsilon correspondant à la distance entre la ville 
    # et la ressource concernées puis ajout de nbville*k coefficient pour les variables représentant
    # le maximum

    c=[]
               

    for i in range(nbvilles):
        for j in range(nbvilles):
            c.append(eps*dist[i][j])
    for j in range(nbvilles**2):
        c.append(dist[int(math.floor(j/nbvilles))][j%nbvilles])
    for j in range(nbvilles):
        c.append(0)


    m = Model("mogplex")     

    # declaration variables de decision entière pour n'avoir que des 1 et des 0
    x = []
    for i in colonnes:
        x.append(m.addVar(vtype=GRB.BINARY, lb=0, name="x%d" % (i+1)))

    # maj du modele pour integrer les nouvelles variables
    m.update()

    obj = LinExpr();
    obj =0
    for j in colonnes:
        obj += c[j] * x[j]

    # definition de l'objectif
    m.setObjective(obj,GRB.MINIMIZE)

    # Definition des signes des contraintes
    for i in lignes:
        if i <nbvilles:
            m.addConstr(quicksum(a[i][j]*x[j] for j in colonnes) == b[i], "Contrainte%d" % i)
        elif i <nbvilles+nbvilles**2:
            m.addConstr(quicksum(a[i][j]*x[j] for j in colonnes) <= b[i], "Contrainte%d" % i)
        elif i<2*nbvilles+nbvilles**2:
            m.addConstr(quicksum(a[i][j]*x[j] for j in colonnes) <= b[i], "Contrainte%d" % i)
        elif i<2*nbvilles+2*nbvilles**2:
            m.addConstr(quicksum(a[i][j]*x[j] for j in colonnes) >= b[i], "Contrainte%d" % i)
        elif i<2*nbvilles+2*nbvilles**2+1:
            m.addConstr(quicksum(a[i][j]*x[j] for j in colonnes) == b[i], "Contrainte%d" % i)
        else:
            m.addConstr(quicksum(a[i][j]*x[j] for j in colonnes) == b[i], "Contrainte%d" % i)
            

    # Resolution
    m.optimize()
    
    ress=[]
    for j in range(nbvilles):
        if(x[2*nbvilles**2+j].x==1):
            ress.append(j)
    
    X=[]
    for i in range(nbvilles):
        X.append([])
        for j in range(nbvilles):
            if j in ress:
                X[i].append(x[i*nbvilles+j].x)
    

    return (X,ress)


# ### Tests

# #### Comparaison des modèles des questions 1 et 2

# #### Instance #1 - k=3, Villes : Courbevoie, Garches, Nanterre

# ##### alpha=0.1

# ##### Modèle question 1

# In[93]:


#k = 3 alpha = 0.1, villes = Courbevoie, Garches, Nanterre

nbvilles=len(villes)
k=3
alpha=0.1
v_ressources=[12,14,24]

X_q1=optimisation_Q1(nbvilles, k, alpha, v_ressources, dist)

trace_reseau(carte, X_q1, v_ressources, coord)

(satmoy, satmin)=satisfaction(X_q1, dist, v_ressources)
print("Satisfaction moyenne : {} et satifaction minimum :{}. ".format(satmoy, satmin))


# ##### Modèle question 2

# In[94]:


#k = 3 alpha = 0.1, villes = Courbevoie, Garches, Nanterre

nbvilles=len(villes)
k=3
alpha=0.1
v_ressources=[12,14,24]
eps=10**(-6)

X_q2=optimisation_Q2(nbvilles, k, alpha, v_ressources, dist, eps)

trace_reseau(carte, X_q2, v_ressources, coord)

(satmoy, satmin)=satisfaction(X_q2, dist, v_ressources)
print("Satisfaction moyenne : {} et satifaction minimum :{}. ".format(satmoy, satmin))


# ##### Prix de l'équité

# In[95]:


print("Prix de l'équité : %f"%(PE(X_q1, X_q2, v_ressources, dist)))


# ##### alpha=0.2

# ##### Modèle question 1

# In[ ]:





# In[96]:


#k = 3 alpha = 0.2, villes = Courbevoie, Garches, Nanterre

nbvilles=len(villes)
k=3
alpha=0.2
v_ressources=[12,14,24]

X_q1=optimisation_Q1(nbvilles, k, alpha, v_ressources, dist)

trace_reseau(carte, X_q1, v_ressources, coord)

(satmoy, satmin)=satisfaction(X_q1, dist, v_ressources)
print("Satisfaction moyenne : {} et satifaction minimum :{}. ".format(satmoy, satmin))


# ##### Modèle question 2

# In[97]:


#k = 3 alpha = 0.2, villes = Courbevoie, Garches, Nanterre

nbvilles=len(villes)
k=3
alpha=0.2
v_ressources=[12,14,24]
eps=10**(-6)

X_q2=optimisation_Q2(nbvilles, k, alpha, v_ressources, dist, eps)

trace_reseau(carte, X_q2, v_ressources, coord)

(satmoy, satmin)=satisfaction(X_q2, dist, v_ressources)
print("Satisfaction moyenne : {} et satifaction minimum :{}. ".format(satmoy, satmin))


# ##### Prix de l'équité

# In[98]:


print("Prix de l'équité : %f"%(PE(X_q1, X_q2, v_ressources, dist)))


# #### Instance #2 - k=4, Villes : Courbevoie, Garches, Le Plessis-Robinson, Nanterre

# ##### alpha = 0.1

# ##### Méthode question 1

# In[99]:


#k = 4 alpha = 0.1, villes = Courbevoie, Garches, Le Plessis-Robinson, Nanterre

nbvilles=len(villes)
k=4
alpha=0.1
v_ressources=[12,14,18,24]

X_q1=optimisation_Q1(nbvilles, k, alpha, v_ressources, dist)

trace_reseau(carte, X_q1, v_ressources, coord)

(satmoy, satmin)=satisfaction(X_q1, dist, v_ressources)
print("Satisfaction moyenne : {} et satifaction minimum :{}. ".format(satmoy, satmin))


# ##### Méthode question 2

# In[100]:


#k = 4 alpha = 0.1, villes = Courbevoie, Garches, Le Plessis-Robinson, Nanterre

nbvilles=len(villes)
k=4
alpha=0.1
v_ressources=[12,14,18,24]
eps=10**(-6)

X_q2=optimisation_Q2(nbvilles, k, alpha, v_ressources, dist, eps)

trace_reseau(carte, X_q2, v_ressources, coord)

(satmoy, satmin)=satisfaction(X_q2, dist, v_ressources)
print("Satisfaction moyenne : {} et satifaction minimum :{}. ".format(satmoy, satmin))


# ##### Prix de l'équité

# In[101]:


print("Prix de l'équité : %f"%(PE(X_q1, X_q2, v_ressources, dist)))


# ##### alpha = 0.2

# In[102]:


#k = 4 alpha = 0.2, villes = Courbevoie, Garches, Le Plessis-Robinson, Nanterre

nbvilles=len(villes)
k=4
alpha=0.2
v_ressources=[12,14,18,24]

X_q1=optimisation_Q1(nbvilles, k, alpha, v_ressources, dist)

trace_reseau(carte, X_q1, v_ressources, coord)

(satmoy, satmin)=satisfaction(X_q1, dist, v_ressources)
print("Satisfaction moyenne : {} et satifaction minimum :{}. ".format(satmoy, satmin))


# In[103]:


#k = 4 alpha = 0.2, villes = Courbevoie, Garches, Le Plessis-Robinson, Nanterre

nbvilles=len(villes)
k=4
alpha=0.2
v_ressources=[12,14,18,24]
eps=10**(-6)

X_q2=optimisation_Q2(nbvilles, k, alpha, v_ressources, dist, eps)

trace_reseau(carte, X_q2, v_ressources, coord)

(satmoy, satmin)=satisfaction(X_q2, dist, v_ressources)
print("Satisfaction moyenne : {} et satifaction minimum :{}. ".format(satmoy, satmin))


# ##### Prix de l'équité

# In[104]:


print("Prix de l'équité : %f"%(PE(X_q1, X_q2, v_ressources, dist)))


# #### Instance #3 - k=5, Villes : Courbevoie, Garches, Le Plessis-Robinson, Nanterre, Sevres

# ##### alpha = 0.1

# ##### Méthode question 1

# In[105]:


#k = 5 alpha = 0.1, villes = Courbevoie, Garches, Le Plessis-Robinson, Nanterre, Sevres

nbvilles=len(villes)
k=5
alpha=0.1
v_ressources=[12,14,18,24,30]

X_q1=optimisation_Q1(nbvilles, k, alpha, v_ressources, dist)

trace_reseau(carte, X_q1, v_ressources, coord)

(satmoy, satmin)=satisfaction(X_q1, dist, v_ressources)
print("Satisfaction moyenne : {} et satifaction minimum :{}. ".format(satmoy, satmin))


# ##### Méthode question 2

# In[106]:


#k = 5 alpha = 0.1, villes = Courbevoie, Garches, Le Plessis-Robinson, Nanterre, Sevres

nbvilles=len(villes)
k=5
alpha=0.1
v_ressources=[12,14,18,24,30]
eps=10**(-6)

X_q2=optimisation_Q2(nbvilles, k, alpha, v_ressources, dist, eps)

trace_reseau(carte, X_q2, v_ressources, coord)

(satmoy, satmin)=satisfaction(X_q2, dist, v_ressources)
print("Satisfaction moyenne : {} et satifaction minimum :{}. ".format(satmoy, satmin))


# ##### Prix de l'équité

# In[107]:


print("Prix de l'équité : %f"%(PE(X_q1, X_q2, v_ressources, dist)))


# ##### alpha = 0.2

# ##### Méthode question 1

# In[108]:


#k = 5 alpha = 0.2, villes = Courbevoie, Garches, Le Plessis-Robinson, Nanterre, Sevres

nbvilles=len(villes)
k=5
alpha=0.2
v_ressources=[12,14,18,24,30]

X_q1=optimisation_Q1(nbvilles, k, alpha, v_ressources, dist)

trace_reseau(carte, X_q1, v_ressources, coord)

(satmoy, satmin)=satisfaction(X_q1, dist, v_ressources)
print("Satisfaction moyenne : {} et satifaction minimum :{}. ".format(satmoy, satmin))


# ##### Méthode question 2

# In[109]:


#k = 5 alpha = 0.1, villes = Courbevoie, Garches, Le Plessis-Robinson, Nanterre, Sevres

nbvilles=len(villes)
k=5
alpha=0.2
v_ressources=[12,14,18,24,30]
eps=10**(-6)

X_q2=optimisation_Q2(nbvilles, k, alpha, v_ressources, dist, eps)

trace_reseau(carte, X_q2, v_ressources, coord)

(satmoy, satmin)=satisfaction(X_q2, dist, v_ressources)
print("Satisfaction moyenne : {} et satifaction minimum :{}. ".format(satmoy, satmin))


# ##### Prix de l'équité

# In[110]:


print("Prix de l'équité : %f"%(PE(X_q1, X_q2, v_ressources, dist)))


# ##### Recherche des villes optimales avec k donné

# In[111]:


#k = 3 

nbvilles=len(villes)
k=3
eps=10**(-6)
(X,v_ressources)=optimisation_Q3(nbvilles, k, alpha, dist, eps)
trace_reseau(carte, X, v_ressources, coord)

print("Villes ressources optimales:")
for r in v_ressources:
    print(villes[r])

(satmoy, satmin)=satisfaction(X, dist, v_ressources)
print("Satisfaction moyenne : {} et satifaction minimum :{}. ".format(satmoy, satmin))


# In[112]:


#k = 4

nbvilles=len(villes)
k=4
eps=10**(-6)
(X,v_ressources)=optimisation_Q3(nbvilles, k, alpha, dist, eps)
trace_reseau(carte, X, v_ressources, coord)

print("Villes ressources optimales:")
for r in v_ressources:
    print(villes[r])

(satmoy, satmin)=satisfaction(X, dist, v_ressources)
print("Satisfaction moyenne : {} et satifaction minimum :{}. ".format(satmoy, satmin))


# In[113]:


#k = 5

nbvilles=len(villes)
k=5
eps=10**(-6)
(X,v_ressources)=optimisation_Q3(nbvilles, k, alpha, dist, eps)
trace_reseau(carte, X, v_ressources, coord)

print("Villes ressources optimales:")
for r in v_ressources:
    print(villes[r])

(satmoy, satmin)=satisfaction(X, dist, v_ressources)
print("Satisfaction moyenne : {} et satifaction minimum :{}. ".format(satmoy, satmin))


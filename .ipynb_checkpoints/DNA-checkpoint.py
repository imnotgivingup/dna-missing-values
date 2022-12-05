import pandas as pd
import numpy as np
import os
import copy
from sklearn.linear_model import LinearRegression
from sklearn.impute import KNNImputer
from tqdm import tqdm
import matplotlib.pyplot as plt


#Conversion of Dataset to Dataframe
def datasetconvert(fname,dataset="derisi"):    
    #Open the file and read data as text
    fo=open(fname,"r")
    txt=fo.read()
    fo.close()

    txt=txt.split("\n")

    #Get Gene names
    names=[]
    for i in txt[2:]:
        try:
            names.append(i[:i.index("\t")])
        except:
            pass

    #Get Numeric data values
    data=[]
    for i in txt[2:]:
        try:
            if(dataset=="derisi"):
                data.append(i[i.index("\t")+4:].split("\t"))
            else:
                data.append(i[i.index("\t",i.index("\t")+1)+3:].split("\t"))
        except:
            pass
    
    #Convert data values to floats, and if a blank is present, replace with 0.0
    for i in data:
        for j in range(len(i)):
            try:
                i[j]=float(i[j])
            except:
                i[j]=0.0

    #Create Dataframe column names
    head=["NAME"]
    for i in range(len(data[0])):
        head.append(str(i))

    #Convert dataset to dictionary format and then return the dataframe
    dic={}
    dic[head[0]]=np.array(names)
    data=np.array(data).astype(np.float32)
    for i in range(1,len(head)):
        dic[head[i]]=data[:,i-1]
        
    return pd.DataFrame.from_dict(dic)



#Converting dataframe to python list with only relevant information
def dataframetolist(df):
    lis=[]
    for i in range(df.shape[0]):
        lis.append(list(df.iloc[i][1:]))
    return lis



#Artificially introducing missing data
def createmissing(lis,misspercent):
    #Create list of indices
    indices=[]
    for i in range(len(lis)):
        for j in range(len(lis[0])):
            indices.append([i,j])
    
    #Choose randomly without replacement and Nullify
    choices=np.random.choice(list(range(len(indices))),replace=False,size=int(misspercent*len(indices)))
    
    lis=copy.deepcopy(lis)
    for i in choices:
        lis[indices[i][0]][indices[i][1]]=None
        
    return lis
  
    
    
#Normalized RMS error
def rmserror(lis1,lis2):
    sumval=0
    
    for i in range(len(lis1)):
        for j in range(len(lis1[0])):
            sumval+=(lis1[i][j]-lis2[i][j])**2
            
    return sumval



#Row average
def rowaverage(lis):
    lis=copy.deepcopy(lis)
    
    for i in tqdm(range(len(lis))):
        for j in range(len(lis[0])):
            if(lis[i][j] is None):
                
                #Calculate row average
                sumval=0
                count=0
                for k in range(len(lis)):
                    if(lis[k][j] is not None):
                        sumval+=lis[k][j]
                        count+=1
                sumval/=count
                
                #Set new value
                lis[i][j]=sumval
    
    return lis



#KnnImpute
def knnimpute(lis, K=10, library=False):
    lis=copy.deepcopy(lis)
    
    
    if(library):
        for i in range(len(lis)):
            for j in range(len(lis[0])):
                if(lis[i][j] is None):
                    lis[i][j]=np.nan
        imputer = KNNImputer(n_neighbors=K)
        return imputer.fit_transform(lis)
    
    
    for i in tqdm(range(len(lis))):
        for j in range(len(lis[0])):
            #If value is missing
            if(lis[i][j] is None):
                #Get non missing value from current row
                indicestocompare=[]
                for k in range(len(lis[i])):
                    if(lis[i][k] is not None):
                        indicestocompare.append(k)
                 
                #Calculate distance from all rows which have non None values in valid indices
                distanceindexpairs=[]
                for k in range(len(lis)):
                    flag=False
                    distance=0
                    for l in indicestocompare:
                        if(lis[k][l] is None or lis[k][j] is None):
                            flag=True
                            break
                        distance+=(lis[i][l]-lis[k][l])**2
                    if(flag):
                        continue
                    distanceindexpairs.append([distance,k])
                
                #Sort in ascending order
                distanceindexpairs.sort(reverse=False)
                
                #Get K closest neighbours
                distanceindexpairs=distanceindexpairs[:K]
                
                #Get denominator of weights
                denominator=0
                for k in range(len(distanceindexpairs)):
                    denominator+=distanceindexpairs[k][0]
                
                #Calculate weights array
                weights=[k[0]/denominator for k in distanceindexpairs]
                
                #Set new value
                lis[i][j]=sum([lis[distanceindexpairs[k][1]][j]*weights[k] for k in range(K)])

    return lis



#SVDImpute
def svdimpute(lis, threshold=0.01, max_steps=100, frac=0.2):
    lis=copy.deepcopy(lis)
    
    #Calculate indices of missing data
    indices=[]
    for i in range(len(lis)):
        for j in range(len(lis[0])):
            if(lis[i][j] is None):
                indices.append([i,j])
    
    #Set missing data to be 0 as svd works on complete matrices
    for i in indices:
        lis[i[0]][i[1]]=0
    
    error=np.inf
    iters=0
    lisold=copy.deepcopy(lis)
    
    #Loop until error falls below threshold or max steps reached
    for iters in tqdm(range(max_steps)):
        if(error<=threshold):
            print("Threshold reached")
            break
            
        #Perform svd
        u,s,vh = np.linalg.svd(lis)
        genestotake=int(len(vh)*frac)
        
        for i in indices:
            #Get non missing values to be used for regression
            indicestocompare=[]
            for j in range(len(lis[i[0]])):
                if([i[0],j] not in indices):
                    indicestocompare.append(j)
                    
            #Slice target gene so that it doesnt contain missing valued data
            x=[]
            for j in indicestocompare:
                x.append(lis[i[0]][j])
            
            #Slice eigengenes so that it doesnt contain missing valued data
            vhtemp=[]
            for j in vh:
                temp=list(j)
                temp2=[]
                for k in indicestocompare:
                    temp2.append(temp[k])
                vhtemp.append(temp2)
            vhtemp=vhtemp[:genestotake]
            
            #Perform regression to obtain the coefficients
            vhtemp=np.array(vhtemp)
            x=np.array(x)
            coeff=np.linalg.lstsq(vhtemp.T,x,rcond=None)[0]

            #Set the missing value to the newly calculated one
            lis[i[0]][i[1]]=sum([coeff[j]*vh[j][i[1]] for j in range(len(coeff))])
        
        #Update iterations
        iters+=1
        if(iters>5):
            #Calculate error
            squaresum=0
            for i in range(len(lis)):
                for j in range(len(lis[0])):
                    squaresum+=lis[i][j]**2
            error=(rmserror(lis,lisold)/(squaresum))**0.5
            
        #Update old matrix to current one
        lisold=copy.deepcopy(lis)    
                
                
    return lis
    
    
def plotter():
    method = "knn"
    plt.figure(figsize=(10,10))
    plt.grid()
    plt.xlabel("Number of Genes Used as Neighbours")
    plt.ylabel("RMS Error")
    l1 = [.01, .05, .1, .15, .2]

    #Select dataset to use
    lis=[]
    for i in derisi:
        lis.append(dataframetolist(i))

    knnerrors=[]
    print("Calculating values for different missing fractions: ")
    for i in l1:
        print("Missing percent: ", i)
        errors = []
        #Number of neighbours to test on
        for k in tqdm([1,3,5,12,17,23,92,458,916]):
            mislis=[]
            for j in lis:
                mislis.append(createmissing(j,i))

            #Perform the operations
            finlis=[]
            count=1
            for j in mislis:
                if(method == "row"):
                    finlis.append(rowaverage(j))
                elif(method == "knn"):
                    finlis.append(knnimpute(j, k, library_func))
                else:
                    finlis.append(svdimpute(j, threshold, max_steps, k))
                count+=1

            #Keep track of errors
            rms=0
            for j in range(len(lis)):
                rms+=rmserror(finlis[j], lis[j])

            errors.append(rms**0.5)
        knnerrors.append(errors)
        
    #Plot the values obtained
    for i in knnerrors:
        plt.plot(np.arange(1,10), i)

    plt.xticks(np.arange(1,10),[1,3,5,12,17,23,92,458,916])
    plt.legend(["1% entries missing", "5% entries missing", "10% entries missing", "15% entries missing", "20% entries missing"],loc="upper right")
    plt.show()
    
    
#Testing missing values sample code
if __name__=="__main__":
    #List of Dataframes representing each dataset

    #Noisy time series
    derisi=[]

    #Non noisy time series
    spellman=[]

    #Non time series
    gasch=[]

    #Parse the directories and convert each dataset
    for entry in os.listdir("./derisi"):
        derisi.append(datasetconvert("./derisi/"+entry,"derisi"))
    for entry in os.listdir("./spellman"):
        spellman.append(datasetconvert("./spellman/"+entry,"spellman"))
    for entry in os.listdir("./gasch"):
        gasch.append(datasetconvert("./gasch/"+entry,"gasch"))
    
    # method = "row"
    method = "knn"
    # method = "svd"

    missing_frac = 0.02

    #Parameters for knn
    neighbours = 10
    library_func = True

    #Parameters for svd
    threshold = 0.01
    max_steps = 100
    eigengenes_fraction = 0.2



    lis=[]
    for i in derisi:
        lis.append(dataframetolist(i))

    mislis=[]
    for i in lis:
        mislis.append(createmissing(i,missing_frac))

    finlis=[]
    count=1
    for i in mislis:
        print("Processing matrix ", count)
        if(method == "row"):
            finlis.append(rowaverage(i))
        elif(method == "knn"):
            finlis.append(knnimpute(i, neighbours, library_func))
        else:
            finlis.append(svdimpute(i, threshold, max_steps, eigengenes_fraction))
        count+=1

    rms=0
    for i in range(len(lis)):
        rms+=rmserror(finlis[i], lis[i])

    print("RMS error = ", rms**0.5)
    plotter()

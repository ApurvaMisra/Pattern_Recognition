import pandas as pd
import numpy as np
import math
import random
import matplotlib.pyplot as plt
rules=[[]]
ind_list=0
final_sum=0
final_s=[]

###################Decision tree
def decision_tree(orig_data,data,columns_name):
    m=np.shape(data)[0]
    col=np.shape(data)[1]
    global ind_list
    global rules
    columns_name=["Rl","Na","Mg","Al","Si","K","Ca","Ba","Fe","Class"]
    division_tree=[[]]
    ###############Condition for all the features getting over
    if(np.all(data,axis=0)[0]==False and np.all(data,axis=0)[1]==False and np.all(data,axis=0)[2]==False and np.all(data,axis=0)[3]==False and np.all(data,axis=0)[4]==False and np.all(data,axis=0)[5]==False and np.all(data,axis=0)[6]==False and np.all(data,axis=0)[7]==False and np.all(data,axis=0)[8]==False):
        (values_3,counts_3) = np.unique(data[:,col-1],return_counts=True)
        ind=np.argmax(counts_3) 
        if(ind_list>0):
            if(len(rules[ind_list-1])>2):
                feat_ch=str(rules[ind_list][0].split(':')[0])
                for length in range(len(rules[ind_list-1])-1):
                    if(str(rules[ind_list-1][length].split(':')[0])==feat_ch):
                        prev_ind=length
                
                var=rules[ind_list-1][0:prev_ind]
                rules[ind_list][0:0]=var
        
        rules[ind_list].append(values_3[ind])
        ind_list=ind_list+1
        rules.append([])
        return(values_3[ind])
    ###############Condition for sub-data being empty
    if(m==0):
        if(ind_list>0):
            if(len(rules[ind_list-1])>2):
                feat_ch=str(rules[ind_list][0].split(':')[0])
                for length in range(len(rules[ind_list-1])-1):
                    if(str(rules[ind_list-1][length].split(':')[0])==feat_ch):
                        prev_ind=length
                
                var=rules[ind_list-1][0:prev_ind]
                rules[ind_list][0:0]=var
        
        rules[ind_list].append(np.unique(orig_data[col-1]))
        ind_list=ind_list+1
        rules.append([])
        return np.unique(orig_data[col-1])
    ###############Condition for all the labels being same in the sub-data
    if len(np.unique(data[:,col-1])) <= 1:

        if(ind_list>0):
            if(len(rules[ind_list-1])>2):
                feat_ch=str(rules[ind_list][0].split(':')[0])
                for length in range(len(rules[ind_list-1])-1):
                    if(str(rules[ind_list-1][length].split(':')[0])==feat_ch):
                        prev_ind=length
                
                var=rules[ind_list-1][0:prev_ind]
                rules[ind_list][0:0]=var
        
        rules[ind_list].append(data[0,col-1])
        ind_list=ind_list+1
        rules.append([])
        
        return (data[0,col-1])  
    else:
        idx=[-1,-1]
        idx,division_tree=gain_ratio(data)
        if(idx==[-1,-1]):  #Condition for all the features having same gain ratio
            (values_3,counts_3) = np.unique(data[:,col-1],return_counts=True)
            ind=np.argmax(counts_3) #Assigning label of sub-data from previous iteration
            if(ind_list>0):
                if(len(rules[ind_list-1])>2):
                    feat_ch=str(rules[ind_list][0].split(':')[0])
                    for length in range(len(rules[ind_list-1])-1):
                        if(str(rules[ind_list-1][length].split(':')[0])==feat_ch):
                            prev_ind=length
                    
                    var=rules[ind_list-1][0:prev_ind]
                    rules[ind_list][0:0]=var
            
            rules[ind_list].append(values_3[ind])
            ind_list=ind_list+1
            rules.append([])
            return(values_3[ind])            
        best_feature=columns_name[idx[0]]
        tree = {best_feature:{}}
        for i in range(2):
            orig_data=data
            
            if(i==0):
                rules[ind_list].append(best_feature+":"+"<"+":"+str(division_tree[idx[0]][idx[1]])) #Appending the best feature with the condition in "rules"
                sub_data = data[data[:,idx[0]]<=division_tree[idx[0]][idx[1]],:]
            else:
                rules[ind_list].append(best_feature+":"+">"+":"+str(division_tree[idx[0]][idx[1]]))  #Appending the best feature with the condition in "rules"
                sub_data = data[data[:,idx[0]]>division_tree[idx[0]][idx[1]],:]
            sub_data[:,idx[0]]=0
            subtree = decision_tree(orig_data,sub_data,columns_name) #Recursive call to decision tree
            tree[best_feature][idx[1]] = subtree
        return(tree)


######################Function for checking when the class label changes for finding the partition for attributes
def continuous(data):
    cols=np.shape(data)[1]
    rows=np.shape(data)[0]
    division=[[]]
    data_columns=["Rl","Na","Mg","Al","Si","K","Ca","Ba","Fe","Class"]
    for i in range(cols-1):
        sub_mat=data[:,i]
        sub_mat=np.reshape(sub_mat,(np.shape(sub_mat)[0],1))
        sub_mat=np.append(sub_mat,np.reshape(data[:,cols-1], (rows,1)), axis=1)
        sub_mat=sub_mat[sub_mat[:,0].argsort()]
        k=0
        division.append([])
        while(k<rows-1):
            if(sub_mat[k,1]!=sub_mat[k+1,1]):
                division[i].append(sub_mat[k,0])
            k=k+1    
    return division        

########################Function for calculating gain ratio and finding the one with maximum gain ratio
def gain_ratio(data):
    cols=np.shape(data)[1]
    rows=np.shape(data)[0]
    global division
    division=[[]]
    division=continuous(data)
    (values,counts) = np.unique(data[:,cols-1],return_counts=True)
    Total_ent=0
    for i in counts: 
        Total_ent=Total_ent+(-((i/rows)*math.log((i/rows),2)))
    max_gain=0
    idx=[-1,-1]    
    for i in range(len(division)):
        for j in range(len(division[i])):
            new_valid_1=np.empty([1,cols])
            new_valid_2=np.empty([1,cols])
            for m in range(rows):
                if(data[m,i]<=division[i][j]):
                    new_valid_1=np.append(new_valid_1,np.reshape(data[m,:],(1,cols)),axis=0)
                else:
                    new_valid_2=np.append(new_valid_2,np.reshape(data[m,:],(1,cols)),axis=0)
            
            new_valid_1=new_valid_1[1:,:]
            new_valid_2=new_valid_2[1:,:]
            (values_1,counts_1) = np.unique(new_valid_1[:,cols-1],return_counts=True)
            (values_2,counts_2) = np.unique(new_valid_2[:,cols-1],return_counts=True)
            ent_1=0
            ent_2=0
            for co in counts_1:
                ent_1=ent_1-((co/sum(counts_1))*math.log((co/sum(counts_1)),2))
            ent_1=(sum(counts_1)/rows)*ent_1
            for cn in counts_2:
                ent_2=ent_2-((cn/sum(counts_2))*math.log((cn/sum(counts_2)),2))
            ent_2=(sum(counts_2)/rows)*ent_2            
            gain=Total_ent-(ent_1+ent_2)
            if(sum(counts_1)==0 and sum(counts_2)==0):
                split=0
            elif(sum(counts_1)==0):
                split=-(sum(counts_2)/rows)*math.log((sum(counts_2)/rows),2)
            elif(sum(counts_2)==0):
                split=-(sum(counts_1)/rows)*math.log((sum(counts_1)/rows),2)
            else:
                split=-(sum(counts_1)/rows)*math.log((sum(counts_1)/rows),2)-(sum(counts_2)/rows)*math.log((sum(counts_2)/rows),2)
            gain_rat=gain/split
            if(gain_rat>max_gain):
                max_gain=gain_rat    
                idx=[i,j]
    return idx,division

##################Function for removal of conditions from rules
def removal(valid, rules, num_rul, acc):
    m=np.shape(valid)[0]
    data_columns=["Rl","Na","Mg","Al","Si","K","Ca","Ba","Fe","Class"]
    new_valid=np.empty([1, 10])
    for k in range(len(rules[num_rul])-1):
        for row_d in range(m):
            rem_wrong=0
            for rem in range(len(rules[num_rul])-1):
                if(rem!=k):
                    feat=str(rules[num_rul][rem].split(':')[0])
                    comp=str(rules[num_rul][rem].split(':')[1])
                    feat_val=str(rules[num_rul][rem].split(':')[2])
                    idx=data_columns.index(feat)
                    if(comp==">"):
                        if(valid[row_d,idx]<=float(feat_val)):                    
                            rem_wrong=rem_wrong+1
                    if(comp=="<"):
                        if(valid[row_d,idx]>float(feat_val)):                    
                            rem_wrong=rem_wrong+1
            if(rem_wrong==0): 
                new_valid=np.append(new_valid,np.reshape(valid[row_d,:],(1,10)),axis=0)
        corr=0
        new_valid=new_valid[1:,:]
        new_m=np.shape(new_valid)[0]
        if(new_m!=0):
            for i_ind in range(new_m):
                if(new_valid[i_ind,-1]==rules[num_rul][-1]):
                    corr=corr+1
            if((corr/new_m)>acc):
                if(len(rules[num_rul])<=2):
                    return(rules)
                if(k< len(rules[num_rul])-1):
                    del rules[num_rul][k]
                removal(valid, rules, num_rul, corr/new_m)

    return(rules)
#########################Pruning
def pruning(valid,rules):
    m=np.shape(valid)[0]
    for  i in range(len(rules)-1):
        data_columns=["Rl","Na","Mg","Al","Si","K","Ca","Ba","Fe","Class"]
        new_valid=np.empty([1, 10])
        for row_d in range(m):
            wrong=0
            for k in range(len(rules[i])-1):                
                feat=str(rules[i][k].split(':')[0])
                comp=str(rules[i][k].split(':')[1])
                feat_val=str(rules[i][k].split(':')[2])
                idx=data_columns.index(feat)
                if(comp=="<"):
                    if(valid[row_d,idx]>float(feat_val)):
                        wrong=wrong+1
                if(comp==">"):
                    if(valid[row_d,idx]<=float(feat_val)):
                        wrong=wrong+1
            if(wrong==0): 
                new_valid=np.append(new_valid,np.reshape(valid[row_d,:],(1,10)),axis=0)
        
        corr=0
        new_valid=new_valid[1:,:]
        new_m=np.shape(new_valid)[0]
        if(new_m!=0):
            for i_ind in range(new_m):
                if(new_valid[i_ind,-1]==rules[i][-1]):
                    corr=corr+1
            
            if((corr/new_m)<1): ##############Calling the "removal" function for accuracies below 1
                acc=corr/new_m
                rules=removal(valid, rules, i, acc)
############################Calculation of accuracies for test set           
def test_acc(test, rules):
    sum_acc=0
    contr=0
    global final_sum
    data_columns=["Rl","Na","Mg","Al","Si","K","Ca","Ba","Fe","Class"]
    m=np.shape(test)[0]
    for  i in range(len(rules)-1):
        new_valid=np.empty([1, 10])
        for row_d in range(m):
            wrong=0
            for k in range(len(rules[i])-1):
                feat=str(rules[i][k].split(':')[0])
                comp=str(rules[i][k].split(':')[1])
                feat_val=str(rules[i][k].split(':')[2])
                idx=data_columns.index(feat)
                if(comp=="<"):
                    if(float(test[row_d,idx])>float(feat_val)):
                        wrong=wrong+1
                elif(comp==">"):
                    if(float(test[row_d,idx])<=float(feat_val)):
                        wrong=wrong+1
            if(wrong==0): 
                new_valid=np.append(new_valid,np.reshape(test[row_d,:],(1,10)),axis=0)       
        corr=0
        new_valid=new_valid[1:,:]
        new_m=np.shape(new_valid)[0]
        if(new_m!=0):
            for i_ind in range(new_m):
                if(new_valid[i_ind,-1]==rules[i][-1]):
                    corr=corr+1
            acc=corr/new_m
            sum_acc=sum_acc+acc
            contr=contr+1
    if(sum_acc!=0):
        print("Sum_accuracy", sum_acc/contr)
        final_sum+=sum_acc/contr
        final_s.append((sum_acc/contr)*100)
    else:
        print("Sum_accuracy", 0)  
###########################Pretty printing the nested dictionary
def pretty(trained_model, indent=0):
    for key, value in trained_model.items():
        print('\t' * indent + str(key))

        if isinstance(value, dict):
            pretty(value, indent+1)
        else:
            print('\t' * (indent+1) + str(value))


###########################Reading the dataset
data = pd.read_csv('glass.txt', sep=",", header=None)
data=data.as_matrix()
data=data[:,1:]
data_columns=["Rl","Na","Mg","Al","Si","K","Ca","Ba","Fe","Class"]
rows=np.shape(data)[0]
################10-times-10-fold cross validation
for iter in range(1):
    np.random.shuffle(data)
    for iter_in in range(1):
        rows=np.shape(data)[0]
        slot=int(0.1*rows)
        test_data=data[iter_in*slot:(iter_in*slot)+(slot),:]
        n1=data[0:iter_in*slot,:]
        n2=data[int(iter_in*slot)+(slot):,:]
        data_new=np.append(n1,n2,axis=0)
        np.random.shuffle(data_new)
        rows_new=np.shape(data_new)[0]
        val=data_new[0:int(0.2*rows_new),:]
        train=data_new[int(0.2*rows_new):,:
        ##############Misclassified noise
        '''
        data_num=np.shape(train)[0]
        perc=0.15*data_num #Choosing the percentage for noise
        my_randoms = random.sample(range( data_num), int(perc))
        for hel in my_randoms:
            a=[1,2,3,4,5,6,7]
            a.remove(train[hel,-1])
            train[hel,-1]=random.choice(a)
        '''
        ############Contradictory noise
        data_num=np.shape(train)[0]
        perc=0.15*data_num #Choosing the percentage for noise
        my_randoms = random.sample(range( data_num), int(perc))
        for hel in my_randoms:
            slice_x=train[hel,:]
            slice_x=np.reshape(slice_x,(1,10))
            a=[1,2,3,4,5,6,7]
            a.remove(train[hel,-1])
            slice_x[0,-1]=random.choice(a)
            train=np.append(train,slice_x, axis=0)             
        trained_model=decision_tree(train,train,data_columns)
        pruning(val,rules)
        test_acc(test_data,rules)
        rules=[[]]
        ind_list=0

###################Priniting and plotting of results
print("Final accuracy", final_sum)
print("Variance", final_s)
print("Variance", np.var(final_s))
#pretty(trained_model)

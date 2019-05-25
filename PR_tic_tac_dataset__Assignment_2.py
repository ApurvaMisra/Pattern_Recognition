import pandas as pd
import numpy as np
import math
import random
import matplotlib.pyplot as plt

rules=[[]]
ind_list=0
control=0
acc_list=[]
final_sum=0
final_s=[]

#############################Decision tree
def decision_tree(prev_data,data,columns_name):
    m=np.shape(data)[0]
    col=np.shape(data)[1]
    global ind_list
    global rules
    #If all the classes in the sub data are the same
    if len(np.unique(data[:,col-1])) <= 1: 
        if(ind_list>0):
            if(len(rules[ind_list-1])>2):
                feat_ch=str(rules[ind_list][0].split(':')[0])
                for length in range(len(rules[ind_list-1])):
                    if(str(rules[ind_list-1][length].split(':')[0])==feat_ch):
                        prev_ind=length
                
                var=rules[ind_list-1][0:prev_ind]
                rules[ind_list][0:0]=var
        
        rules[ind_list].append(data[0,col-1])
        ind_list=ind_list+1
        rules.append([])
        return (data[0,col-1]) 
    #If the su-data is empty
    if(m==0):
        return np.unique(prev_data[col-1])
    #If the features are over
    if(np.shape(data)[1]==1):
        (values_3,counts_3) = np.unique(data[:,col-1],return_counts=True)
        ind=np.argmax(counts_3)
        return(values_3[ind])
    else:
        max_gain=0
        idx=-1
        for i in range(col-1):
            if(gain_ratio(data,i)==math.inf):
                (values_3,counts_3) = np.unique(data[:,col-1],return_counts=True)
                ind=np.argmax(counts_3) 
                return(values_3[ind])
                 
             
            if(gain_ratio(data,i)>max_gain): #Choosing the feature with maximum gain ratio
                max_gain=gain_ratio(data,i)    
                idx=i

        best_feature=columns_name[idx]
        tree = {best_feature:{}}
        columns_name=np.delete(columns_name,[idx])
        for value in np.unique(data[:,idx]):
            rules[ind_list].append(best_feature+":"+value) #Saving the node feature and condition in rules
            sub_data = data[data[:,idx]==value,:]
            sub_data=np.delete(sub_data, idx, 1)
            subtree = decision_tree(data,sub_data,columns_name) #Recursive call to decision tree
            tree[best_feature][value] = subtree
        return(tree)

############For removal of conditions from rules while pruning depending on the accuracy
def removal(valid, rules, num_rul, acc): 
    m=np.shape(valid)[0]
    data_columns=['TL','TM','TR','ML','MM','MR','BL','BM','BR','Class']
    new_valid=np.empty([1, 10])
    for k in range(len(rules[num_rul])-1):
        for row_d in range(m):
            rem_wrong=0
            for rem in range(len(rules[num_rul])-1):
                if(rem!=k):
                    feat=str(rules[num_rul][rem].split(':')[0])
                    feat_val=str(rules[num_rul][rem].split(':')[1])
                    idx=data_columns.index(feat)
                    if(valid[row_d,idx]!=feat_val):                    
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
                removal(valid, rules, num_rul, corr/new_m) #Recursive call to removal

    return(rules)


###############Pruning
def pruning(valid,rules):
    m=np.shape(valid)[0]
    for  i in range(len(rules)-1):
        data_columns=['TL','TM','TR','ML','MM','MR','BL','BM','BR','Class']
        new_valid=np.empty([1, 10])
        for row_d in range(m):
            wrong=0
            for k in range(len(rules[i])): #Going through all the rules sequentially with removal of condition
                if(str(rules[i][k])!= 'positive' and str(rules[i][k])!='negative'):
                    feat=str(rules[i][k].split(':')[0])
                    feat_val=str(rules[i][k].split(':')[1])
                    idx=data_columns.index(feat)
                    if(valid[row_d,idx]!=feat_val):
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
            
            if((corr/new_m)<1): #If accuracy less than 1, call for "removal function"
                acc=corr/new_m
                rules=removal(valid, rules, i, acc)
##############Calculation of accuracy for test dataset
def test_acc(test, rules):
    sum_acc=0
    contr=0
    global final_sum
    global fin_con
    global final_s
    data_columns=['TL','TM','TR','ML','MM','MR','BL','BM','BR','Class']
    m=np.shape(test)[0]
    for  i in range(len(rules)-1):
        new_valid=np.empty([1, 10])
        for row_d in range(m):
            wrong=0
            for k in range(len(rules[i])-1):
                feat=str(rules[i][k].split(':')[0])
                feat_val=str(rules[i][k].split(':')[1])
                idx=data_columns.index(feat)
                if(test[row_d,idx]!=feat_val):
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
        print("final_s",final_s)
        
    else:
        print("Sum_accuracy", 0)    
#######################Calculation of gain ratio
def gain_ratio(data,f):
    m=np.shape(data)[0]
    col=np.shape(data)[1]
    n_pos=0
    n_x=0
    n_o=0
    n_b=0
    n_x_pos=0
    n_x_neg=0
    n_o_pos=0
    n_o_neg=0
    n_b_pos=0
    n_b_neg=0    
    for i in range(m):
        if(data[i,col-1]=='positive'):
            n_pos=n_pos+1            
        if(data[i,f]=='x'):
            n_x=n_x+1
            if(data[i,col-1]=='positive'):
                n_x_pos=n_x_pos+1
            if(data[i,col-1]=='negative'):
                n_x_neg=n_x_neg+1
        if(data[i,f]=='o'):
            n_o=n_o+1
            if(data[i,col-1]=='positive'):
                n_o_pos=n_o_pos+1 
            if(data[i,col-1]=='negative'):
                n_o_neg=n_o_neg+1             
        if(data[i,f]=='b'):
            n_b=n_b+1
            if(data[i,col-1]=='positive'):
                n_b_pos=n_b_pos+1 
            if(data[i,col-1]=='negative'):
                n_b_neg=n_b_neg+1  
    if(n_pos==112):
        print(data[0,:])                  
    p_pos=n_pos/m
    p_neg=1-p_pos
    if(p_pos==1 or p_neg==1):
        gain_rat=math.inf
        return gain_rat
    if(n_x_pos+n_x_neg >0):
        p_x_pos= n_x_pos/(n_x_pos+n_x_neg)
        p_x_neg=n_x_neg/(n_x_pos+n_x_neg)
        split_a=(n_x/m)*math.log((n_x/m),2)
        if(p_x_pos!=0):
            ent_x_pos=p_x_pos*math.log(p_x_pos,2)
        else:
            ent_x_pos=0
        if(p_x_neg!=0):
            ent_x_neg=(p_x_neg*math.log(p_x_neg,2))
        else:
            ent_x_neg=0
        ent_x=-ent_x_pos-ent_x_neg
        a=((n_x_pos+n_x_neg)/m)*ent_x        
    else:
        a=0
        split_a=0

    if(n_o_pos+n_o_neg >0):
        p_o_pos= n_o_pos/(n_o_pos+n_o_neg)
        p_o_neg=n_o_neg/(n_o_pos+n_o_neg)
        split_b=(n_o/m)*math.log((n_o/m),2)
        if(p_o_pos!=0):
            ent_o_pos=p_o_pos*math.log(p_o_pos,2)
        else:
            ent_o_pos=0
        if(p_o_neg!=0):
            ent_o_neg=(p_o_neg*math.log(p_o_neg,2))
        else:
            ent_o_neg=0
        ent_o=-ent_o_pos-ent_o_neg
        b=((n_o_pos+n_o_neg)/m)*ent_o
        
    else:
        b=0
        split_b=0

    if(n_b_pos+n_b_neg >0):
        split_c=(n_b/m)*math.log((n_b/m),2)
        p_b_pos= n_b_pos/(n_b_pos+n_b_neg)
        p_b_neg=n_b_neg/(n_b_pos+n_b_neg)
        if( p_b_pos!=0):
            ent_b_pos=p_b_pos*math.log(p_b_pos,2)
        else:
             ent_b_pos=0
        if(p_b_neg!=0):
            ent_b_neg=(p_b_neg*math.log(p_b_neg,2))
        else:
            ent_b_neg=0
        ent_b=-ent_b_pos-ent_b_neg
        c=((n_b_pos+n_b_neg)/m)*ent_b
        
    else:
        c=0
        split_c=0

    gain=-(p_pos*math.log(p_pos,2))-(p_neg*math.log(p_neg,2)) -( a + b  + c)
    split_in=-( split_a + split_b + split_c )
    if(split_in==0):
        gain_rat=gain
    else:
        gain_rat=gain/split_in
    return(gain_rat)
###################Pretty printing the nested dictionary that is the tree
def pretty(trained_model, indent=0):
    for key, value in trained_model.items():
        print('\t' * indent + str(key))

        if isinstance(value, dict):
            pretty(value, indent+1)
        else:
            print('\t' * (indent+1) + str(value))

#################Reading dataset
data = pd.read_csv('tictac.txt', sep=",", header=None)
data_columns=['TL','TM','TR','ML','MM','MR','BL','BM','BR','Class']
columns_name=['TL','TM','TR','ML','MM','MR','BL','BM','BR','Class']
data=data.as_matrix()
print(np.shape(data))
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
        val_num=np.shape(val)[0]
        train=data_new[int(0.2*rows_new):,:]
        #######################Misclassified noise
        '''
        data_num=np.shape(train)[0]
        perc=0.15*data_num #Choosing the percentage for noise
        my_randoms = random.sample(range( data_num), int(perc))
        for hel in my_randoms:
            if(train[hel,-1]=="positive"):
                train[hel,-1]="negative"
            else:
                train[hel,-1]="positive"
        '''
        #######################Contradictory noise
        data_num=np.shape(train)[0]
        perc=0.15*data_num #Choosing the percentage for noise
        my_randoms = random.sample(range( data_num), int(perc))
        for hel in my_randoms:
            if(train[hel,-1]=="positive"):
                slice_x=train[hel,:]
                slice_x=np.reshape(slice_x,(1,10))
                slice_x[0,-1]="negative"
                train=np.append(train,slice_x, axis=0)
            else:
                slice_x=train[hel,:]
                slice_x=np.reshape(slice_x,(1,10))
                slice_x[0,-1]="positive"
                train=np.append(train,slice_x, axis=0)     
        trained_model=decision_tree(train,train,columns_name)
        pruning(val,rules)
        test_acc(test_data,rules)
        rules=[[]]
        ind_list=0
################Plotting and printing results
print("Final accuracy", final_sum)
print("Variance",final_s)
print("Variance",np.var(final_s))
plt.plot([0,5,10,15], [86.49,83.05,78.508,76.62], 'bo')
plt.plot([0,5,10,15], [86.49,82.21,78.53,74.23], 'ro')
plt.plot([0,5,10,15], [86.49,83.05,78.508,76.62], 'b--', label="Contradictory noise")
plt.plot([0,5,10,15], [86.49,82.21,78.53,74.23], 'r--',label="Misclassified noise")
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,ncol=2, mode="expand", borderaxespad=0.)
plt.xlabel('Percentage')
plt.ylabel('Accuracy')
#plt.title('Misclassified examples')
plt.show()
#trained_model=decision_tree(orig_data,data1,columns_name)
#pretty(trained_model)

































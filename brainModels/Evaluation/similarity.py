from tqdm import tqdm
import numpy as np
import tensorflow as tf

class calculateSimilarity():

    # This function has been sourced from https://git.scc.kit.edu/ps-chair/brainnet licensed under the Creative Commons
    def _close_set_identification(self, embedding_network, x_train_val, y_train_val, x_test, y_test):
        """Calculates similarity values for closed-set recognition by comparing each face embedding in the test set with all face embeddings 
        in the training set, and then calculating the cosine similarity between each pair of embeddings. It then uses these similarity 
        values to create pairs of faces, with the label 1 indicating that the faces belong to the same individual (from the training set) 
        and label 0 indicating that the faces belong to different individuals (including unknown identities)"""

        print("I am in predict close set function")
        resutls=[]
        resutls2=[]
        resutls3=defaultdict(list)
        calsstrain=np.unique(y_train_val)
        TP,FP,TN,FN=0,0,0,0
        digit_indices = [np.where(y_train_val == i)[0] for i in np.unique(y_train_val)]
        #print("chaurasia")
        x_test_1 = x_test

        print(len(x_train_val),len(x_test))
        #anc_et=embedding_network(x_train_val)
        anc_e=embedding_network(x_test[0:min(500,len(x_test))])
        for c in range(len(x_test)//500):
            anc_e=tf.concat(axis=0, values = [anc_e, embedding_network(x_test[(c+1)*500:min((c+2)*500,len(x_test))])]) 	
        anc_et=embedding_network(x_train_val[0:min(500,len(x_train_val))])
        for c in range(len(x_train_val)//500):
            anc_et=tf.concat(axis=0, values = [anc_et, embedding_network(x_train_val[(c+1)*500:min((c+2)*500,len(x_train_val))])]) 
        #print(type(anc_e))
        print(len(anc_et),len(anc_e))
        for i in tqdm(range(len(x_test_1)), desc="Calculating similarity"):
            prediction=[]
            test_e=embedding_network(np.array([x_test[i]]))
            same_in=digit_indices[np.where(calsstrain == y_train_val[i])[0][0]]
            
            for t in range(len(x_train_val)):
                tempp=-1*self.euclidean_distance2(anc_et[t],test_e).numpy()[0][0] 
                if y_test[i] ==y_train_val[t]:
                    resutls.append([tempp,1,y_test[i],y_train_val[t]])
                else:
                    resutls.append([tempp,0,y_test[i],y_train_val[t]])

                prediction.append(tempp)        
            prediction=np.array(prediction)
            
            for j in calsstrain:
                same_in=digit_indices[np.where(calsstrain == j)[0][0]]
                spredict=((sum(prediction[same_in]))/(len(same_in)))            
                if y_test[i] ==j:
                    resutls2.append([spredict,1,y_test[i],j])
                    resutls3[j].append([spredict,1,y_test[i],j])
                else:
                    resutls2.append([spredict,0,y_test[i],j])
                    resutls3[j].append([spredict,0,y_test[i],j])   
                if spredict>0.85:
                    if y_test[i] ==j:
                        TP+=1
                    else:
                        FP+=1
                else:
                    if y_test[i] == j:
                        FN+=1
                    else:
                        TN+=1
        return resutls,resutls2,resutls3
    
    def _open_set_verification(self, embedding_network, x_test, y_test):
        """Calculates similarity values for authentication (verification) by 
        comparing each face embedding in the test set with all other face embeddings, and then calculating the cosine similarity between 
        each pair of embeddings. It then uses these similarity values to create pairs of faces, with the label 1 indicating that the faces
        belong to the same individual and label 0 indicating that the faces belong to different individuals."""

        print("I am in predict open set function")
        # Compute embeddings for all test samples
        resutls=[]
        resutls2=[]
        resutls3=defaultdict(list)
        pair1=[]
        pair2=[]
        calss=np.unique(y_test)
        TP,FP,TN,FN=0,0,0,0
        digit_indices = [np.where(y_test == i)[0] for i in np.unique(y_test)]
        x_test_1 = x_test
        print(len(x_test))
        #anc_et=embedding_network(x_train_val)
        anc_e=embedding_network(x_test[0:min(500,len(x_test))])
        for c in tqdm(range(len(x_test)//500), desc='Getting test embedings'):
            anc_e=tf.concat(axis=0, values = [anc_e, embedding_network(x_test[(c+1)*500:min((c+2)*500,len(x_test))])]) 	
        print(len(x_test))
        for i in tqdm(range(len(x_test_1)), desc="Calculating similarity"):
            temp=np.where(calss == y_test[i])[0][0]
            prediction=[]
            same_in=digit_indices[np.where(calss == y_test[i])[0][0]]
            for t in range(len(x_test_1)):
                tempp=-1*self.euclidean_distance2(anc_e[t],anc_e[i]).numpy()[0]
        
                if t in same_in:
                    if t==i:
                        pass
                    else:
                        resutls.append([tempp,1,y_test[i],y_test[t]])
                else:
                    resutls.append([tempp,0,y_test[i],y_test[t]])    
                prediction.append(tempp)
            prediction=np.array(prediction)
            
            for j in calss:
                same_in=digit_indices[np.where(calss == j)[0][0]]
                same_in=np.setdiff1d(same_in,[i])
                spredict=max(prediction[same_in])        
        
                if y_test[i] ==j:
                    resutls2.append([spredict,1,y_test[i],j])
                    resutls3[j].append([spredict,1,y_test[i],j])
                else:
                    resutls2.append([spredict,0,y_test[i],j])
                    resutls3[j].append([spredict,0,y_test[i],j])
                        
                if spredict>0.85:
                    if y_test[i] ==j:
                        TP+=1
                    else:
                        FP+=1
                else:
                    if y_test[i] == j:
                        FN+=1
                    else:
                        TN+=1
        return resutls,resutls2,resutls3
                
    def euclidean_distance2(self, x, y):
        sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=None, keepdims=True)
        return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))
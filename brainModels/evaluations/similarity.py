from tqdm import tqdm
import numpy as np
import tensorflow as tf
from collections import defaultdict

class CalculateSimilarity():

    # def calculate_embeddings(self, data, embedding_network, chunk_size=500):
    #     embeddings = embedding_network(data[:min(chunk_size, len(data))])
    #     for i in range(len(data) // chunk_size):
    #         embeddings = tf.concat(axis=0, values=[embeddings, embedding_network(data[(i + 1) * chunk_size:min((i + 2) * chunk_size, len(data))])])
    #     return embeddings

    # This function has been sourced from https://git.scc.kit.edu/ps-chair/brainnet licensed under the Creative Commons
    def _close_set_identification(self, embedding_network, x_train_val, y_train_val, x_test, y_test):
        """
        Calculates similarity values for closed-set recognition by comparing each brain embedding in the test set
        with all face embeddings in the training set, then computes similarity by calculating euclidean distance 
        between each pair of embeddings. It creates pairs of faces with labels: 1 for the same individual, 0 for 
        different individuals.

        Parameters:
        - embedding_network (function): Siamese network to generate brain embeddings.
        - x_train_val (numpy.ndarray): Training data for brain embeddings.
        - y_train_val (numpy.ndarray): Labels for training data.
        - x_test (numpy.ndarray): Test data for brain embeddings.
        - y_test (numpy.ndarray): Labels for test data.

        Returns:
        - resutls (list): List containing similarity values and labels for pairs of brain samples.
        - resutls2 (list): List containing similarity values and labels for pairs of brain samples grouped by identity.
        - resutls3 (dict): Dictionary containing similarity values and labels for pairs of brain samples grouped by identity.

        """
        resutls=[]
        resutls2=[]
        resutls3=defaultdict(list)
        calsstrain=np.unique(y_train_val)
        TP,FP,TN,FN=0,0,0,0
        digit_indices = [np.where(y_train_val == i)[0] for i in np.unique(y_train_val)]
        #print("chaurasia")
        x_test_1 = x_test

        #print(len(x_train_val),len(x_test))
        #anc_et=embedding_network(x_train_val)
        anc_e=embedding_network(x_test[0:min(500,len(x_test))])
        for c in range(len(x_test)//500):
            anc_e=tf.concat(axis=0, values = [anc_e, embedding_network(x_test[(c+1)*500:min((c+2)*500,len(x_test))])]) 	
        anc_et=embedding_network(x_train_val[0:min(500,len(x_train_val))])
        for c in range(len(x_train_val)//500):
            anc_et=tf.concat(axis=0, values = [anc_et, embedding_network(x_train_val[(c+1)*500:min((c+2)*500,len(x_train_val))])]) 
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
        """
        Calculates similarity values for authentication (verification) by comparing each brain embedding in the test set
        with all other brain embeddings in the test. It computes similarity by calculating euclidean disnatncebetween each 
        pair of embeddings and creates pairs of faces with labels: 1 for the same individual, 0 for different individuals.

        Parameters:
            - embedding_network (function): Siamese network to generate embeddings.
            - x_test (numpy.ndarray): Test data for brain embeddings.
            - y_test (numpy.ndarray): Labels for test data.

        Returns:
            - resutls (list): List containing similarity values and labels for pairs of brain samples.
            - resutls2 (list): List containing similarity values and labels for pairs of brain samples grouped by identity.
            - resutls3 (dict): Dictionary containing similarity values and labels for pairs of brain samples grouped by identity.

        """

        #print("I am in predict open set function")
        # Compute embeddings for all test samples
        resutls=[]
        resutls2=[]
        resutls3=defaultdict(list)
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
        """
        Computes the Euclidean distance between two vectors x and y.

        Parameters:
        - x (numpy.ndarray): Input vector x.
        - y (numpy.ndarray): Input vector y.

        Returns:
        - distance (float): Euclidean distance between vectors x and y.
        """
        sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=None, keepdims=True)
        return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))
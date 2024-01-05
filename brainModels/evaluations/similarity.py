from tqdm import tqdm
import numpy as np
import tensorflow as tf
from collections import defaultdict
from sklearn.model_selection import (
    LeaveOneGroupOut,
)

class CalculateSimilarity():

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
    

    # def _multi_session_open_set_verification_old(embedding_network, test_eeg_data, test_subjects, sessions):

    #     """
    #     Performs multi-session open-set verification using EEG data.

    #     Parameters:
    #         - embedding_network (function): Siamese network to generate embeddings.
    #         - test_eeg_data (numpy.ndarray): Test data for EEG embeddings.
    #         - test_subjects (numpy.ndarray): Labels for test data subjects.
    #         - sessions (numpy.ndarray): Session numbers for the test data.

    #     Returns:
    #     - resutls3 (defaultdict(list)): A dictionary containing similarity values and labels for pairs of brain samples across multiple sessions.
    #         Keys represent subject IDs, and each value is a list of results for that particular subject across different sessions.
    #         Each result entry consists of:
    #             - Similarity score
    #             - Label (1.0 for same subject, 0.0 for different subjects)
    #             - Subject ID
    #             - Current session number
    #             - Session(s) being compared
    #     """
    #     resutls3=defaultdict(list)

    #     # Intiliase the leave one group out cross validation
    #     lkfold=LeaveOneGroupOut()

    #     # Grouped done based on sessions. So testing data consists of single session to be test and training data 
    #     # consists data from the remaning sessions. 
    #     for train_index, test_index in lkfold.split(test_eeg_data, test_subjects, groups=sessions):
    #         X_train, X_test = test_eeg_data[train_index], test_eeg_data[test_index]
    #         y_train, y_test = test_subjects[train_index], test_subjects[test_index]
    #         train_session, test_session=sessions[train_index], sessions[test_index]

    #         # Getting the embeddings for all the subjects ptresent in the current session to be tested
    #         test_embeddings=embedding_network(X_test)

    #         # Iterate over all brain samples in testing session. It has single session data
    #         for i in tqdm(range(len(X_test)), desc="Calculating similarity in multi-session(Open Set)"):

    #             # Iterate over the sessions in the training data. It can have single session or multi-session
    #             # data
    #             for sess in np.unique(train_session):
                    
    #                 # Finding the indicces of the session need to be compared with the test samples in the 
    #                 # current session
    #                 session_indices=np.where(train_session==sess)[0]

    #                 # Get the subjects data from the session need to be compared with the test samples in the 
    #                 # current session
    #                 session_subjects=y_train[session_indices]

    #                 # Get the embeddings of all the subjects from the session need to be compared with 
    #                 # the test samples in the current session
    #                 session_embeddings=embedding_network(X_train[session_indices])

    #                 # Intilaise the list to store the similarity score between test sample(anchor sample) with 
    #                 # brain samples from the subject's brain samples from the session to be compared
    #                 prediction=[]

    #                 # Iterate over all the brain samples present in the session to be tested
    #                 for j in range(len(session_embeddings)):

    #                     # Getting the similarity between the test sample from the current session with 
    #                     # the brain sample from the session to be compared
    #                     similarity_scores=-1*euclidean_distance2(session_embeddings[j], test_embeddings[i]).numpy()[0]

    #                     # Store the similarity score in the list
    #                     prediction.append(similarity_scores)
                    
    #                 # Convert the predictions list which contains the similarity score into numpy array
    #                 prediction=np.array(prediction)

    #                 # Iterate over all the subjects in th session to be compared
    #                 for sub in np.unique(session_subjects):

    #                     # Get the similairity score of the subject in the session to be compared
    #                     indices=np.where(session_subjects==sub)[0]

    #                     # get the maximum similarity of that subject
    #                     spredict=max(prediction[indices])
                        

    #                     # Check if the subject is same in the current session and the session to be compared
    #                     if(y_test[i]==sub):

    #                         # If yes, then store similairy score, label 1 , subject ID, Current session number
    #                         # and session to be compared
    #                         resutls3[sub].append([spredict,1.0,y_test[i],sub, np.unique(test_session),sess])
    #                         #print("same subject", spredict, np.unique(test_session), sess)

    #                     else:

    #                         # If No, then store similairy score, label 0 , subject ID, Current session number
    #                         # and session to be compared. This is for the different subjects or 
    #                         # unmatching subjects
    #                         resutls3[sub].append([spredict,0.0,y_test[i],sub, test_session[0],sess])

    #     # Return the results in the dictionary results3 which contains the results of all the subjects
    #     # in the form of dictionary which keys representing subject Id and the value of that corrosponding 
    #     # contains the list of results for that particular subjects in all the sessions
    #     return resutls3    

    def _compute_embedding_batch(x_test_batch,embedding_network):
        embeddings = embedding_network(x_test_batch[0:min(500, len(x_test_batch))])

        for c in range(len(x_test_batch) // 500):
            embeddings = tf.concat(axis=0, values=[anchor_embeddings,                             embedding_network(x_test_batch[(c+1)*500:min((c+2)*500,len(x_test_batch))])])

        return embeddings
    
    def _multi_session_open_set_verification(embedding_network, eeg_data, subjects, sessions):
        
        
        def compute_embedding_batch(x_test_batch,embedding_network):
            embeddings = embedding_network(x_test_batch[0:min(500, len(x_test_batch))])

            for c in range(len(x_test_batch) // 500):
                embeddings = tf.concat(axis=0, values=[embeddings,                             embedding_network(x_test_batch[(c+1)*500:min((c+2)*500,len(x_test_batch))])])

            return embeddings

        """
        Performs multi-session open-set verification using EEG data.

        This function accepts the EEG data, subjects and sessions as input and returns the similarity values for
        each pair of brain samples across multiple sessions. It computes similarity by calculating euclidean distance
        between each pair of embeddings and creates pairs of faces with labels: 1 for the same individual, 0 for
        different individuals. The enrollment session should always be less than the test session.

        Parameters:
            - embedding_network (function): Siamese network to generate embeddings.
            - test_eeg_data (numpy.ndarray): Test data for EEG embeddings.
            - test_subjects (numpy.ndarray): Labels for test data subjects.
            - sessions (numpy.ndarray): Session numbers for the test data.

        Returns:
        - resutls3 (defaultdict(list)): A dictionary containing similarity values and labels for pairs of brain samples across multiple sessions.
            Keys represent subject IDs, and each value is a list of results for that particular subject across different sessions.
            Each result entry consists of:
                - Similarity score
                - Label (1.0 for same subject, 0.0 for different subjects)
                - Subject ID
                - Current session number
                - Session(s) being compared
        """
        resutls3=defaultdict(list)

        # Iterate over all the sessions in the test data except the last session
        for enroll_sessions in range(0, len(np.unique(sessions))-1):

            # Get the session number of the session to be enrolled
            enroll_session=np.unique(sessions)[enroll_sessions]

            # Get the indices of the session to be enrolled
            enroll_indices=np.where(sessions==enroll_session)[0]

            # Get the subjects of the session to be enrolled
            enroll_subjects=subjects[enroll_indices]

            # Get the embeddings of the session to be enrolled
            enroll_embeddings=compute_embedding_batch(eeg_data[enroll_indices],embedding_network)

            # Iterate over all the sessions except the session already enrolled
            for test_sessions in range(enroll_sessions+1, len(np.unique(sessions))):

                # Get the session number of the session to be tested
                test_session=np.unique(sessions)[test_sessions]

                # Get the indices of the session to be tested
                test_indices=np.where(sessions==test_session)[0]

                # Get the subjects of the session to be tested
                test_subjects=subjects[test_indices]

                # Get the embeddings of the session to be tested
                test_embeddings=compute_embedding_batch(eeg_data[test_indices],embedding_network)


                # Iterate over all the brain samples in the session to be tested
                for i in tqdm(range(len(test_embeddings)), desc="Calculating similarity in multi-session(Open Set)"):

                    prediction=[]

                    # Iterate over all the brain samples in the session already enrolled
                    for j in range(len(enroll_embeddings)):

                        # Getting the similarity between the test sample from the enrolled session with
                        # the brain sample from the testing session 
                        similarity_scores=-1*euclidean_distance2(enroll_embeddings[j], test_embeddings[i]).numpy()[0]

                        # Store the similarity score in the list
                        prediction.append(similarity_scores)

                    # Convert the predictions list which contains the similarity score into numpy array
                    prediction=np.array(prediction)

                    # Iterate over all the subjects in th enrolled session
                    for sub in np.unique(enroll_subjects):

                        # get the indices of the subject in the enrolled session
                        indices=np.where(enroll_subjects==sub)[0]

                        # get the maximum similarity of that subject
                        spredict=max(prediction[indices])

                        # Check if the subject is same in the enrolled session and the session to be compared
                        if(test_subjects[i]==sub):

                            # If yes, then store similairy score, label 1 , subject ID, Current session number
                            resutls3[sub].append([spredict,1.0,test_subjects[i],sub, enroll_session,test_session])
                        else:

                            # If No, then store similairy score, label 0 , subject ID, Current session number
                            resutls3[sub].append([spredict,0.0,test_subjects[i],sub, enroll_session,test_session])

        return resutls3          
                
def euclidean_distance2(x, y):
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

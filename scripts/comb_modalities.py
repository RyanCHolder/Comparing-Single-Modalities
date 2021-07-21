import pickle
import numpy as np
from os import listdir
from sklearn import metrics
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
import alg_testing

#sampling frequencies
ACC_SF = 32
BVP_SF = 64
EDA_SF = 4
TEMP_SF = 4

#classifiers that will be used
classifiers = [KNeighborsClassifier(),
               DecisionTreeClassifier(),
               Sequential()
                ]

def load_file(path):
    """
        @breif load in the specified file
        @param: path (string): The location of the file
        @return: Returns the data dictionary in the file
    """
    with open(path, 'rb') as file:
        data = pickle.load(file)

    return data

def down_sample(data, start_hertz, end_hertz):
    """
        @breif: Down samples the data (reduce samples per window) by selecting
            periodic samples from the original windows
        @param: data (array): The data to down sample
        @param: start_hertz (int): Starting frequency of the data
        @param: end_hertz (int): Desired frequency of the data
    """
    #find the increment for data to remove
    increment = start_hertz//end_hertz
    #take data at the increment
    temp = data[:,range(0,len(data[0]),increment)]
    #return the down sampled data
    return temp

def over_sample(data):
    """
        @brief: generates synthetic data to balance classes
        @param: data (dictionary): The data to oversample
    """
    over = SMOTE()
    ndata = dict()

    #reshape to make compatible with balancing library
    orig_shape = np.shape(data['data'])
    temp = np.reshape(data['data'], (data['data'].shape[0], data['data'].shape[1]*data['data'].shape[2]))
    #generate synthetic data
    ndata['data'], ndata['labels'] = over.fit_resample(temp, data['labels'])
    #reshape back to original
    ndata['data'] = ndata['data'].reshape(len(ndata['data']), orig_shape[1], orig_shape[2])

    return ndata


def run_tests(data_path, save_path, window_size, balance):
    """
        @breif: Runs the same leave-one-out and random sample tests used in other scripts
        @param: data_path (String): Folder containing data
        @param: save_path (String): Folder to save results to
        @param: window_size (int): Seconds per window in data
        @param: balance (boolean): If true perform over sampling on data
    """
    '''Random Sample Testing'''
    #create data for random sample testing
    all_data = load_file(data_path+'All.pkl')
    #all_data = alg_testing.take_sample(all_data)
    #down sample ACC and BVP to match the frequencies of EDA and TEMP
    all_data['data']['ACC'] = down_sample(all_data['data']['ACC'], ACC_SF, 4)
    all_data['data']['BVP'] = down_sample(all_data['data']['BVP'], BVP_SF, 4)

    #layer the data to create a combined set for all features
    comb = dict()
    comb['labels'] = all_data['labels']
    comb['data'] = np.concatenate((all_data['data']['ACC'],all_data['data']['BVP'],
            all_data['data']['EDA'], all_data['data']['TEMP']),2)

    #over sample
    if balance:
        comb = over_sample(comb)

    #run each algorithm for 10 iterations and document results
    file = open(save_path + "random_sample_results.txt", 'wt')
    #file header
    file.write("Results of Random Sample Testing on Combined Modalities\n\n")

    print('Starting random sample testing', flush=True)

    #dictionaries for results
    near = {'acc':list(),'f':list()}
    tree = {'acc':list(),'f':list()}
    conv = {'acc':list(),'f':list()}
    #list of results dictionaries
    results = [near, tree, conv]

    for i in range(10):
        print('Iteration ' + str(i+1), flush=True)

        #create train test split
        X_train, X_test, y_train, y_test = train_test_split(comb['data'],
            comb['labels'], random_state = np.random.randint(100))

        #train classifiers
        for ci in range(len(results)):
            acc, f = alg_testing.train_alg(classifiers[ci],X_train,X_test,y_train,y_test)
            #append results to dictionary
            results[ci]['acc'].append(acc)
            results[ci]['f'].append(f)

    #find average accuracies and f scores over iterations
    for i in range(len(results)):
        results[i]['acc'] = np.mean(results[i]['acc'])
        results[i]['f'] = np.mean(results[i]['f'])

    #write results to file
    file.write("Average Results Using Nearest Neighbors:\n"+
        "Accuracy = " + str(results[0]['acc']) +
        " F Score = " + str(results[0]['f']) + "\n\n")
    file.write("Average Results Using Decision Tree:\n"+
        "Accuracy = " + str(results[1]['acc']) +
        " F Score = " + str(results[1]['f']) + "\n\n")
    file.write("Average Results Using Convolutional Network:\n"+
        "Accuracy = " + str(results[2]['acc']) +
        " F Score = " + str(results[2]['f']) + "\n\n")
    #close the file
    file.close()

    '''Leave One Out Testing'''

    #open new file for leave one out results
    file = open(save_path + 'leave_one_out_results.txt', 'wt')

    print('Starting leave one out testing', flush=True)

    files = sorted(listdir(data_path))
    #remove the file with all subjects
    files.remove('All.pkl')

    #load in data in each file
    all_subjects = list()
    for subject in files:
        temp = load_file(data_path + subject)
        #temp = alg_testing.take_sample(temp)
        #down sample ACC and BVP
        temp['data']['ACC'] = down_sample(temp['data']['ACC'], ACC_SF, 4)
        temp['data']['BVP'] = down_sample(temp['data']['BVP'], BVP_SF, 4)
        #combine modalities
        comb_temp = dict()
        comb_temp['data'] = np.concatenate((temp['data']['ACC'],temp['data']['BVP'],
                temp['data']['EDA'], temp['data']['TEMP']),2)
        comb_temp['labels'] = temp['labels']

        #over sample
        if balance:
            comb_temp = over_sample(comb_temp)

        #append to the list of subjects
        all_subjects.append(comb_temp)

    print('Formatted Data', flush=True)

    #reset results dictionaries
    near = {'acc':list(),'f':list()}
    tree = {'acc':list(),'f':list()}
    conv = {'acc':list(),'f':list()}
    results = [near, tree, conv]

    #iterate through each subject
    for i in range(len(files)):
        print("Testing on " + files[i], flush=True)
        #create training and testing data
        X_test = all_subjects[i]['data']
        y_test = all_subjects[i]['labels']

        X_train = np.empty((0,4*window_size,6))
        y_train = np.empty((0))
        for x in range(len(all_subjects)):
            if x != i:
                X_train = np.append(X_train,all_subjects[x]['data'],0)
                y_train = np.append(y_train,all_subjects[x]['labels'])

        #train classifiers
        for ci in range(len(results)):
            acc, f = alg_testing.train_alg(classifiers[ci],X_train,X_test,y_train,y_test)
            #append results to dictionary
            results[ci]['acc'].append(acc)
            results[ci]['f'].append(f)

    #find average accuracies and f scores
    for i in range(len(results)):
        results[i]['acc'] = np.mean(results[i]['acc'])
        results[i]['f'] = np.mean(results[i]['f'])

    #write results to file
    file.write("Average Results Using Nearest Neighbors:\n"+
        "Accuracy = " + str(results[0]['acc']) +
        " F Score = " + str(results[0]['f']) + "\n\n")
    file.write("Average Results Using Decision Tree:\n"+
        "Accuracy = " + str(results[1]['acc']) +
        " F Score = " + str(results[1]['f']) + "\n\n")
    file.write("Average Results Using Convolutional Network:\n"+
        "Accuracy = " + str(results[2]['acc']) +
        " F Score = " + str(results[2]['f']) + "\n\n")

    #close the file
    file.close()

    print('Finished', flush=True)

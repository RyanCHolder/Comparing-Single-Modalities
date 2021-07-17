import pickle
import numpy as np
from os import listdir
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential

#tags for features
features = ['ACC', 'BVP', 'EDA', 'TEMP']

#sampling frequencies
ACC_SF = 32
BVP_SF = 64
EDA_SF = 4
TEMP_SF = 4

#list of classifiers, these can be easily commented out if not wanted
classifiers = [#SVC(),
        #RandomForestClassifier(),
        #MLPClassifier(),
        KNeighborsClassifier(),
        DecisionTreeClassifier(),
        #AdaBoostClassifier(DecisionTreeClassifier()),
        #AdaBoostClassifier(RandomForestClassifier()),
        #GradientBoostingClassifier(),
        Sequential()
        ]

def load_file(path):
    """
        @breif load in the specified WESAD file
        @param: path (string): The location of the file
        @return: Returns the data dictionary in the file
    """
    with open(path, 'rb') as file:
        data = pickle.load(file)

    return data

def train_alg(clf, X_train, X_test, Y_train, Y_test, weights=None):
    """
        @brief trains the classifier on the data and finds test accuracy

        @param clf (classifier): The classifier to use on the data
        @param X_train (array): The training data
        @param X_test (array): The testing data
        @param Y_train (array): The training labels
        @param Y_test (array): The testing labels
        @param weights: Allow for weighting in training

        @return Returns the accuracy score
    """
    if type(clf) != type(Sequential()):
        #regular scikit learning algorithms

        #reshape data
        X_train = X_train.reshape((len(X_train), len(X_train[0])*len(X_train[0][0])))
        X_test = X_test.reshape((len(X_test), len(X_test[0])*len(X_test[0][0])))

        #fit the classifier to the training data
        if np.array(weights != None).any():
            clf.fit(X_train, Y_train, sample_weight = weights)
        else:
            clf.fit(X_train, Y_train)

        #predict on testing data
        pred = clf.predict(X_test)
        #return prediction accuracy and f score
        return metrics.accuracy_score(Y_test, pred), metrics.f1_score(Y_test, pred)
    else:
        #1D convolutional network

        in_shape = (len(X_train[0]),len(X_train[0,0]))

        model = keras.Sequential([
        keras.layers.Conv1D(filters=10, kernel_size=5, padding='same', activation='relu', input_shape=in_shape),
        keras.layers.Conv1D(filters=20, kernel_size=5, padding='same', activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(units=1, activation='sigmoid')
        ])

        #train the model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train, Y_train, epochs=10, batch_size=30, validation_split=0.2,
            shuffle=True, verbose=0, class_weight=weights)

        #predict on test data
        pred = np.around(model.predict(X_test, batch_size=10))
        #reshape to match Y_test
        pred = pred.reshape((len(pred),))
        #return accuracy and f score
        return metrics.accuracy_score(Y_test, pred), metrics.f1_score(Y_test, pred)


def take_sample(data):
    """
        @breif Takes a 1% random sample of the data
        @return Returns the 1% sample
    """
    temp = {'data': {'ACC':[], 'BVP':[],'EDA':[],'TEMP':[]},'labels':[]}
    #random indeces for 1% of the data
    index = np.random.choice(np.arange(len(data['labels'])), len(data['labels'])//100, replace=False)
    #take only those indexes
    for feat in features:
        temp['data'][feat] = data['data'][feat][index]
    temp['labels'] = data['labels'][index]
    #return sample
    return temp


def reg_testing(save_path, data_loc=None, stat_loc=None, iterations=10, save_iters=False, sample=True):
    """
        @breif: tests all algorithms on all the data, one feature at a time
            (tests statistical data, as well as only stress and baseline data)
        @param: save_path (string): The file destination for the results
        @param: data_loc (string): The file containing the regular data
        @param: stat_loc (string): The file containing the statistical data
        @param: iterations (int): The number of iterations per classifier
        @param: save_iters (boolean): If true results of each iteration are saved
        @param: sample (boolean): If true, a 1% random sample of the data will be
            loaded to save time and memory, rather than the full dataset
    """
    #regular data
    if(data_loc==None):
        data = load_file('Data/Raw/All.pkl')
    else:
        data = load_file(data_loc + '/All.pkl')

    #statistical data
    if(stat_loc==None):
        stat = load_file('Data/Statistical/All.pkl')
    else:
        stat = load_file(stat_loc + '/All.pkl')

    if(sample):
        data = take_sample(data)
        stat = take_sample(stat)

    #open save destination for results
    file = open(save_path, 'wt')

    #header the file
    file.write("Results of repeated testing with random test set\n\n")

    #progress tracking
    print("Loaded Data", flush=True)

    #counter to track progress
    algcount = 0;

    #run algorithms with default features
    for clf in classifiers:
        #track progress in terminal
        algcount += 1
        print("Running algorithm " + str(algcount) + " of " + str(len(classifiers)), flush=True)
        #title the algorithm in the file
        file.write(str(clf)+'\n\n')

        #storing accuracy for each iteration
        reg_accs = {'ACC':list(), 'BVP':list(),'EDA':list(),'TEMP':list()}
        stat_accs = {'ACC':list(), 'BVP':list(),'EDA':list(),'TEMP':list()}

        #storing f scores for each iteration
        reg_fs = {'ACC':list(), 'BVP':list(),'EDA':list(),'TEMP':list()}
        stat_fs = {'ACC':list(), 'BVP':list(),'EDA':list(),'TEMP':list()}

        #run each algorithm for ten iterations
        for i in range(iterations):
            print('Running Iteration ' + str(i+1), flush=True)

            #label the iteration
            if save_iters:
                file.write('Iteration ' + str(i+1) + '\n')

            for feat in features:
                #label the feature
                if save_iters:
                    file.write(feat+':\n')

                #divide data, test size is left default at 25%
                #random state set to random number to create differing results
                X_train, X_test, Y_train, Y_test = train_test_split(data['data'][feat],
                    data['labels'], random_state = np.random.randint(100))

                #evaluate algorithm
                acc, fs = train_alg(clf, X_train, X_test, Y_train, Y_test)
                reg_accs[feat].append(acc)
                reg_fs[feat].append(fs)

                #save results
                if save_iters:
                    file.write('Regular Data, Test Accuracy = '+str(reg_accs[feat][i])
                        +', F Score = '+str(reg_fs[feat][i])+'\n')

                return

                #we don't want to use statistical values on the convolutional network
                if(type(clf) != type(Sequential())):
                    #same as before but with statistical analysis data
                    X_train, X_test, Y_train, Y_test = train_test_split(stat['data'][feat],
                        stat['labels'], random_state = np.random.randint(100))

                    acc, fs = train_alg(clf, X_train, X_test, Y_train, Y_test)
                    stat_accs[feat].append(acc)
                    stat_fs[feat].append(fs)
                    #save results
                    if save_iters:
                        file.write('Statistical Data, Test Accuracy = '+str(stat_accs[feat][i])
                            +', F Score = '+str(stat_fs[feat][i])+'\n')

                #formatting
                if save_iters:
                    file.write('\n')

        '''this triggers a couple warnings because with the convolutional
            network the stat_avgs list is empty, these errors can be ignored'''

        #average accuracies
        reg_avg_accs = [str(np.mean(reg_accs['ACC'])),str(np.mean(reg_accs['BVP'])),
            str(np.mean(reg_accs['EDA'])),str(np.mean(reg_accs['TEMP']))]
        stat_avg_accs = [str(np.mean(stat_accs['ACC'])),str(np.mean(stat_accs['BVP'])),
            str(np.mean(stat_accs['EDA'])),str(np.mean(stat_accs['TEMP']))]

        #average f scores
        reg_avg_fs = [str(np.mean(reg_fs['ACC'])),str(np.mean(reg_fs['BVP'])),
            str(np.mean(reg_fs['EDA'])),str(np.mean(reg_fs['TEMP']))]
        stat_avg_fs = [str(np.mean(stat_fs['ACC'])),str(np.mean(stat_fs['BVP'])),
            str(np.mean(stat_fs['EDA'])),str(np.mean(stat_fs['TEMP']))]

        #save results
        file.write('Averages:\nRegular Data:\n'
            +'ACC: Accuracy = '+reg_avg_accs[0]+', F Score = '+reg_avg_fs[0]+'\n'
            +'BVP: Accuracy = '+reg_avg_accs[1]+', F Score = '+reg_avg_fs[1]+'\n'
            +'EDA: Accuracy = '+reg_avg_accs[2]+', F Score = '+reg_avg_fs[2]+'\n'
            +'TEMP: Accuracy = '+reg_avg_accs[3]+', F Score = '+reg_avg_fs[3]+'\n'
            +'Statistical Data:\n'
            +'ACC: Accuracy = '+stat_avg_accs[0]+', F Score = '+stat_avg_fs[0]+'\n'
            +'BVP: Accuracy = '+stat_avg_accs[1]+', F Score = '+stat_avg_fs[1]+'\n'
            +'EDA: Accuracy = '+stat_avg_accs[2]+', F Score = '+stat_avg_fs[2]+'\n'
            +'TEMP: Accuracy = '+stat_avg_accs[3]+', F Score = '+stat_avg_fs[3]+'\n\n')

    #close the file
    file.close()




def leave_one_out(save_path, data_loc=None, stat_loc=None, window_size=1, sample=True):
    """
        @breif: trains on all but one subject's data, tests on that subject666
        @param: save_path (string): The file destination for the results
        @param: data_loc (string): The file containing the raw data
        @param: stat_loc (string): The file containing the statistical data
        @param: window_size (int): Seconds in window for proper shaping
        @param: sample (boolean): If true, a 1% random sample of the data will be
            loaded to save time and memory, rather than the full dataset
    """
    #create lists of all subject's data
    reg_subjects = list()
    stat_subjects = list()

    #raw data location
    if data_loc==None:
        data_loc = 'Data/Raw'
    #list raw files
    raw_files = listdir(data_loc)
    #append each subject from raw files
    for f in raw_files:
        #load raw data for subject
        if f != 'All.pkl':
            temp = load_file(data_loc + '/' + f)
            #sample to reduce memory
            if sample:
                temp = take_sample(temp)
            reg_subjects.append(temp)

    #stat data location
    if stat_loc == None:
        stat_loc = 'Data/Statistical'
    #list stat files
    stat_files = listdir(stat_loc)

    for f in stat_files:
        #load in statistical data for subject
        if f != 'All.pkl':
            temp = load_file(stat_loc + '/' + f)
            #sample to reduce memory
            if sample:
                temp = take_sample(temp)
            stat_subjects.append(temp)

    print('Loaded Data', flush=True)

    #open file
    file = open(save_path, 'wt')
    #header the file
    file.write('Results of leave one out testing on each subject\n\n')

    #counter to track progress
    algcount = 0
    #cycle through each classifier, data set, and feature, and save results
    for clf in classifiers:
        #print progress
        algcount += 1
        print('Running algorithm ' + str(algcount) + ' of ' + str(len(classifiers)), flush=True)

        #classifier title
        file.write(str(clf)+'\n\n')

        #storage for accuracy values
        reg_accs = {'ACC': list(), 'BVP': list(), 'EDA': list(), 'TEMP': list()}
        stat_accs = {'ACC': list(), 'BVP': list(), 'EDA': list(), 'TEMP': list()}

        #storage for f scores
        reg_fs = {'ACC': list(), 'BVP': list(), 'EDA': list(), 'TEMP': list()}
        stat_fs = {'ACC': list(), 'BVP': list(), 'EDA': list(), 'TEMP': list()}

        #find and save results
        for x in range(len(reg_subjects)):
            print('Running subject ' + str(x) + ' of 14', flush=True)
            file.write('Testing on subject '+str(x)+'\n')

            #create dataset with all but given subject
            reg_all = {'data': {'ACC':np.empty((0,ACC_SF*window_size,3)),
                    'BVP':np.empty((0,BVP_SF*window_size,1)),
                    'EDA':np.empty((0,EDA_SF*window_size,1)),
                    'TEMP':np.empty((0,TEMP_SF*window_size,1))}, 'labels': []}

            stat_all = {'data': {'ACC':np.empty((0,8,3)), 'BVP':np.empty((0,8,1)),
                'EDA':np.empty((0,8,1)), 'TEMP':np.empty((0,8,1))}, 'labels': []}

            #cycle through subjects
            for i in range(len(reg_subjects)):
                #x is the subject being left out
                if i != x:
                    #append data and labels to dictionaries
                    reg_all['labels'] = np.append(reg_all['labels'],reg_subjects[i]['labels'],0)
                    stat_all['labels'] = np.append(stat_all['labels'],stat_subjects[i]['labels'],0)
                    for feat in features:
                        reg_all['data'][feat] = np.append(reg_all['data'][feat],reg_subjects[i]['data'][feat],0)
                        stat_all['data'][feat] = np.append(stat_all['data'][feat],stat_subjects[i]['data'][feat],0)

            print('Testing data created')

            for feat in features:
                #run the algorithm with all but one subject as training data
                #the remaining subject as the testing data

                #train on regular data
                acc, fs = train_alg(clf,reg_all['data'][feat],
                    reg_subjects[x]['data'][feat],reg_all['labels'],
                    reg_subjects[x]['labels'])

                reg_accs[feat].append(acc)
                reg_fs[feat].append(fs)

                #individual regular results6
                file.write('Regular Data '+feat+', Accuracy = '
                    +str(reg_accs[feat][x])+', F Score = '
                    +str(reg_fs[feat][x])+'\n')

                #train on statistical data
                if(type(clf) != type(Sequential())):
                    acc, fs = train_alg(clf,stat_all['data'][feat],
                        stat_subjects[x]['data'][feat],stat_all['labels'],
                        stat_subjects[x]['labels'])

                    stat_accs[feat].append(acc)
                    stat_fs[feat].append(fs)

                    #individual stat results
                    file.write('Statistical Data '+feat+', Accuracy = '
                        +str(stat_accs[feat][x])+', F Score = '
                        +str(stat_fs[feat][x])+'\n\n')

        #average accuracies
        reg_avg_accs = [str(np.mean(reg_accs['ACC'])),str(np.mean(reg_accs['BVP'])),
            str(np.mean(reg_accs['EDA'])),str(np.mean(reg_accs['TEMP']))]
        stat_avg_accs = [str(np.mean(stat_accs['ACC'])),str(np.mean(stat_accs['BVP'])),
            str(np.mean(stat_accs['EDA'])),str(np.mean(stat_accs['TEMP']))]

        #average f scores
        reg_avg_fs = [str(np.mean(reg_fs['ACC'])),str(np.mean(reg_fs['BVP'])),
            str(np.mean(reg_fs['EDA'])),str(np.mean(reg_fs['TEMP']))]
        stat_avg_fs = [str(np.mean(stat_fs['ACC'])),str(np.mean(stat_fs['BVP'])),
            str(np.mean(stat_fs['EDA'])),str(np.mean(stat_fs['TEMP']))]

        #save results
        file.write('Averages:\nRegular Data:\n'
            +'ACC: Accuracy = '+reg_avg_accs[0]+', F Score = '+reg_avg_fs[0]+'\n'
            +'BVP: Accuracy = '+reg_avg_accs[1]+', F Score = '+reg_avg_fs[1]+'\n'
            +'EDA: Accuracy = '+reg_avg_accs[2]+', F Score = '+reg_avg_fs[2]+'\n'
            +'TEMP: Accuracy = '+reg_avg_accs[3]+', F Score = '+reg_avg_fs[3]+'\n'
            +'Statistical Data:\n'
            +'ACC: Accuracy = '+stat_avg_accs[0]+', F Score = '+stat_avg_fs[0]+'\n'
            +'BVP: Accuracy = '+stat_avg_accs[1]+', F Score = '+stat_avg_fs[1]+'\n'
            +'EDA: Accuracy = '+stat_avg_accs[2]+', F Score = '+stat_avg_fs[2]+'\n'
            +'TEMP: Accuracy = '+stat_avg_accs[3]+', F Score = '+stat_avg_fs[3]+'\n\n')

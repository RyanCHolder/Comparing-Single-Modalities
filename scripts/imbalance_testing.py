import pickle
import numpy as np
from os import listdir
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
import alg_testing

"""Total ADARP class imbalance is 163577:10005, about 16:1, around 6% of data is stress,
    after labeling 20 minutes around each stress point as stress and removing
    other non-stress data within an hour in each direction of stress tag"""

#tags for features
features = ['ACC', 'BVP', 'EDA', 'TEMP']

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


def split_stress(data):
    """
        @breif Separates data into stress and nonstress
        @return Stress data, nonstress data
    """
    stemp = dict()
    ntemp = dict()
    for feat in features:
        stemp[feat] = data['data'][feat][data['labels'] == 1]
        ntemp[feat] = data['data'][feat][data['labels'] == 0]
    stemp['labels'] = np.ones(len(stemp['ACC']))
    ntemp['labels'] = np.zeros(len(ntemp['ACC']))
    return stemp, ntemp



def under_sample(data, sampling_strategy='auto'):
    """
        @breif: Runs expiriment with under-sampling
        @param: data (dictionary): Data to run under-sampling on
        @param: sampling_strategy: Sampling strategy parameter for under sampler
    """
    #under sampler
    under = RandomUnderSampler(sampling_strategy=sampling_strategy)
    #new data
    ndata = dict()
    #under sample features
    for feat in features:
        #reshape to make compatible with balancing library
        orig_shape = np.shape(data['data'][feat])
        temp = np.reshape(data['data'][feat], (data['data'][feat].shape[0], data['data'][feat].shape[1]*data['data'][feat].shape[2]))
        #oversample
        ndata[feat], ndata['labels'] = under.fit_resample(temp, data['labels'])
        #revert shaping
        ndata[feat] = ndata[feat].reshape(len(ndata[feat]), orig_shape[1], orig_shape[2])

    """print("New Data: " + str(np.shape(ndata['labels'])) + " Stress labels: " +
        str(len(ndata['labels'][ndata['labels'] == 1])) + " Nonstress labels: " +
        str(len(ndata['labels'][ndata['labels'] == 0])))"""

    return ndata


def over_sample(data, sampling_strategy='auto'):
    """
        @brief: Generates synthetic stress data to balance the class imbalance
        @param: data (dictionary): Data to create synthetic stress values from
        @param: sampling_strategy: Sampling strategy parameter for under sampler
    """
    #over sampler
    over = SMOTE(sampling_strategy=sampling_strategy)
    #new data
    ndata = dict()
    #over sample each feature
    for feat in features:
        #reshape to make compatible with balancing library
        orig_shape = np.shape(data['data'][feat])
        temp = np.reshape(data['data'][feat], (data['data'][feat].shape[0], data['data'][feat].shape[1]*data['data'][feat].shape[2]))
        #oversample
        ndata[feat], ndata['labels'] = over.fit_resample(temp, data['labels'])
        #revert shaping
        ndata[feat] = ndata[feat].reshape(len(ndata[feat]), orig_shape[1], orig_shape[2])

    """print("New Data: " + str(np.shape(ndata['labels'])) + " Stress labels: " +
        str(len(ndata['labels'][ndata['labels'] == 1])) + " Nonstress labels: " +
        str(len(ndata['labels'][ndata['labels'] == 0])))"""

    return ndata



def reg_testing(save_path, data_path, iterations=10):
    """
        @breif: Tests each class imbalance solution with a randomly selected test
                set for several iterations
        @param: save_path (String): File to save results to
        @param: data_path (String): Data file to get data from
    """
    #gather data
    data = load_file(data_path)
    #open results file
    file = open(save_path, 'wt')

    print("Loaded Data", flush=True)

    #file header
    file.write("Testing With Random Test Set On Various Class Imbalance Solutions\n\n")

    """Under and Over Sampling"""

    #gather under sampled data
    usamp = under_sample(data)
    #gather over sampled data
    osamp = over_sample(data)
    samples = {'Under Sampling':usamp, 'Over Sampling':osamp}

    #run basic sampling solutions
    for samp in samples:
        #Results dictionaries
        near_acc = dict()
        near_f = dict()
        tree_acc = dict()
        tree_f = dict()
        conv_acc = dict()
        conv_f = dict()

        print("\nStarting " + samp, flush=True)


        #cycle through each feature and test for several iterations
        for feat in features:
            print(feat, flush=True)
            #lists of accuracies and f scores for feature
            accs = {'near': list(), 'tree': list(), 'conv': list()}
            fs = {'near': list(), 'tree': list(), 'conv': list()}

            #run algorithms for set iterations
            for i in range(iterations):
                print("Iteration " + str(i+1), flush=True)
                #create random train test split (25% test)
                X_train, X_test, y_train, y_test = train_test_split(samples[samp][feat],
                    samples[samp]['labels'], random_state = np.random.randint(100))

                #nearest neighbors
                acc, f = alg_testing.train_alg(classifiers[0], X_train, X_test, y_train, y_test)
                accs['near'].append(acc)
                fs['near'].append(f)

                #decision tree
                acc, f = alg_testing.train_alg(classifiers[1], X_train, X_test, y_train, y_test)
                accs['tree'].append(acc)
                fs['tree'].append(f)

                #convolutional network
                acc, f = alg_testing.train_alg(classifiers[2], X_train, X_test, y_train, y_test)
                accs['conv'].append(acc)
                fs['conv'].append(f)

            #find average f score and accuracy
            near_acc[feat] = np.mean(accs['near'])
            near_f[feat] = np.mean(fs['near'])
            tree_acc[feat] = np.mean(accs['tree'])
            tree_f[feat] = np.mean(fs['tree'])
            conv_acc[feat] = np.mean(accs['conv'])
            conv_f[feat] = np.mean(fs['conv'])

        #write results to file
        file.write("Average Results of " + samp + " Using Nearest Neighbors:\n"+
            "ACC: Accuracy = " + str(near_acc['ACC']) + ", F Score = " + str(near_f['ACC']) + "\n" +
            "BVP: Accuracy = " + str(near_acc['BVP']) + ", F Score = " + str(near_f['BVP']) + "\n" +
            "EDA: Accuracy = " + str(near_acc['EDA']) + ", F Score = " + str(near_f['EDA']) + "\n" +
            "TEMP: Accuracy = " + str(near_acc['TEMP']) + ", F Score = " + str(near_f['TEMP']) + "\n\n")

        file.write("Average Results of " + samp + " Using Decision Tree:\n"+
            "ACC: Accuracy = " + str(tree_acc['ACC']) + ", F Score = " + str(tree_f['ACC']) + "\n" +
            "BVP: Accuracy = " + str(tree_acc['BVP']) + ", F Score = " + str(tree_f['BVP']) + "\n" +
            "EDA: Accuracy = " + str(tree_acc['EDA']) + ", F Score = " + str(tree_f['EDA']) + "\n" +
            "TEMP: Accuracy = " + str(tree_acc['TEMP']) + ", F Score = " + str(tree_f['TEMP']) + "\n\n")

        file.write("Average Results of " + samp + " Using Convolutional Network:\n"+
            "ACC: Accuracy = " + str(conv_acc['ACC']) + ", F Score = " + str(conv_f['ACC']) + "\n" +
            "BVP: Accuracy = " + str(conv_acc['BVP']) + ", F Score = " + str(conv_f['BVP']) + "\n" +
            "EDA: Accuracy = " + str(conv_acc['EDA']) + ", F Score = " + str(conv_f['EDA']) + "\n" +
            "TEMP: Accuracy = " + str(conv_acc['TEMP']) + ", F Score = " + str(conv_f['TEMP']) + "\n\n")

    #methods involving changing the keras model

    """Class weighting"""
    #nearest neighbors does not easily allow class weighting, so it will be ignored here
    #weighting will be 16 stress : 1 nonstress to match the 16 nonstress : 1 stress imbalance

    #Average results
    tree_acc = dict()
    tree_f = dict()
    conv_acc = dict()
    conv_f = dict()

    #weights for convolutional network
    conv_weights = {0 : 1, 1 : 16}

    print("\nStarting Weighting", flush=True)

    #cycle through each feature and test for several iterations
    for feat in features:
        print(feat, flush=True)
        #lists of accuracies and f scores for feature
        accs = {'tree': list(), 'conv': list()}
        fs = {'tree': list(), 'conv': list()}

        #run algorithms for set iterations
        for i in range(iterations):
            print("Iteration " + str(i+1), flush=True)
            #create random train test split (25% test)
            X_train, X_test, y_train, y_test = train_test_split(data['data'][feat],
                data['labels'], random_state = np.random.randint(100))

            #decision tree weights
            weights = np.ones(len(y_train))
            weights[y_train == 1] = 16

            #decision tree
            acc, f = alg_testing.train_alg(classifiers[1], X_train, X_test, y_train, y_test, weights)
            accs['tree'].append(acc)
            fs['tree'].append(f)

            #convolutional network
            acc, f = alg_testing.train_alg(classifiers[2], X_train, X_test, y_train, y_test, conv_weights)
            accs['conv'].append(acc)
            fs['conv'].append(f)


        #find average f score and accuracy
        tree_acc[feat] = np.mean(accs['tree'])
        tree_f[feat] = np.mean(fs['tree'])
        conv_acc[feat] = np.mean(accs['conv'])
        conv_f[feat] = np.mean(fs['conv'])

    #write results to file
    file.write("Average Results of Weighting Using Decision Tree:\n"+
        "ACC: Accuracy = " + str(tree_acc['ACC']) + ", F Score = " + str(tree_f['ACC']) + "\n" +
        "BVP: Accuracy = " + str(tree_acc['BVP']) + ", F Score = " + str(tree_f['BVP']) + "\n" +
        "EDA: Accuracy = " + str(tree_acc['EDA']) + ", F Score = " + str(tree_f['EDA']) + "\n" +
        "TEMP: Accuracy = " + str(tree_acc['TEMP']) + ", F Score = " + str(tree_f['TEMP']) + "\n\n")

    file.write("Average Results of Weighting Using Convolutional Network:\n"+
        "ACC: Accuracy = " + str(conv_acc['ACC']) + ", F Score = " + str(conv_f['ACC']) + "\n" +
        "BVP: Accuracy = " + str(conv_acc['BVP']) + ", F Score = " + str(conv_f['BVP']) + "\n" +
        "EDA: Accuracy = " + str(conv_acc['EDA']) + ", F Score = " + str(conv_f['EDA']) + "\n" +
        "TEMP: Accuracy = " + str(conv_acc['TEMP']) + ", F Score = " + str(conv_f['TEMP']) + "\n\n")

    """Combining Under, Over, and Weighting"""

    print("\nStarting Combo Method", flush = True)

    #again nearest neighbors does not easily allow weighting so it will be ignored here
    #Average results
    tree_acc = dict()
    tree_f = dict()
    conv_acc = dict()
    conv_f = dict()

    #create new data
    #start with under sample with new ratio of 10 nonstress : 1 stress
    combo = under_sample(data, 0.1)
    #result needs to have its formatting changed a bit
    combo['data'] = dict()
    for feat in features:
        combo['data'][feat] = combo[feat]
        #remove the old key to save memory
        del combo[feat]

    #now over sample with ratio of 4 nonstress : 1 stress
    combo = over_sample(combo, 0.25)
    #weighting for convolutional network to match final imbalance
    conv_weights = {0 : 1, 1 : 4}

    #cycle through each feature and test for several iterations
    for feat in features:
        print(feat, flush=True)
        #lists of accuracies and f scores for feature
        accs = {'tree': list(), 'conv': list()}
        fs = {'tree': list(), 'conv': list()}

        #run algorithms for set iterations
        for i in range(iterations):
            print("Iteration " + str(i+1), flush=True)
            #create random train test split (25% test)
            X_train, X_test, y_train, y_test = train_test_split(combo[feat],
                combo['labels'], random_state = np.random.randint(100))

            #decision tree weights
            weights = np.ones(len(y_train))
            weights[y_train == 1] = 4

            #decision tree
            acc, f = alg_testing.train_alg(classifiers[1], X_train, X_test, y_train, y_test, weights)
            accs['tree'].append(acc)
            fs['tree'].append(f)

            #convolutional network
            acc, f = alg_testing.train_alg(classifiers[2], X_train, X_test, y_train, y_test, conv_weights)
            accs['conv'].append(acc)
            fs['conv'].append(f)

        #find average f score and accuracy
        tree_acc[feat] = np.mean(accs['tree'])
        tree_f[feat] = np.mean(fs['tree'])
        conv_acc[feat] = np.mean(accs['conv'])
        conv_f[feat] = np.mean(fs['conv'])

    #write results to file
    file.write("Average Results of Combination Method Using Decision Tree:\n"+
        "ACC: Accuracy = " + str(tree_acc['ACC']) + ", F Score = " + str(tree_f['ACC']) + "\n" +
        "BVP: Accuracy = " + str(tree_acc['BVP']) + ", F Score = " + str(tree_f['BVP']) + "\n" +
        "EDA: Accuracy = " + str(tree_acc['EDA']) + ", F Score = " + str(tree_f['EDA']) + "\n" +
        "TEMP: Accuracy = " + str(tree_acc['TEMP']) + ", F Score = " + str(tree_f['TEMP']) + "\n\n")

    file.write("Average Results of Combination Method Using Convolutional Network:\n"+
        "ACC: Accuracy = " + str(conv_acc['ACC']) + ", F Score = " + str(conv_f['ACC']) + "\n" +
        "BVP: Accuracy = " + str(conv_acc['BVP']) + ", F Score = " + str(conv_f['BVP']) + "\n" +
        "EDA: Accuracy = " + str(conv_acc['EDA']) + ", F Score = " + str(conv_f['EDA']) + "\n" +
        "TEMP: Accuracy = " + str(conv_acc['TEMP']) + ", F Score = " + str(conv_f['TEMP']) + "\n\n")

    """Replacement Under Sampling"""

    print("Finished Regular Testing", flush=True)



def leave_one_out(save_path, data_path):
    """
        @breif: Tests each class imbalance solution with a randomly selected test
                set for several iterations
        @param: save_path (String): File to save results to
        @param: data_path (String): Data folder containing subject data
    """
    print("\nLeave-One-Out Testing\n", flush=True)

    #gather files (These will be ordered 0, 1, 10, 2, 3 and so on, makes no difference)
    files = sorted(listdir(data_path))
    #we don't need the combined file
    files.remove('All.pkl')

    #create dataset with each subject individual
    all = list()
    for f in files:
        temp = load_file(data_path + f)
        #temp = alg_testing.take_sample(temp)
        all.append(temp)

    print("Loaded data", flush=True)

    #open results file
    file = open(save_path, 'wt')

    #file header
    file.write("Leave-One-Out Testing On Various Class Imbalance Solutions\n\n")

    #results dictionaries
    near_acc = {'under':dict(),'over':dict(),'weight':dict(),'combo':dict()}
    near_f = {'under':dict(),'over':dict(),'weight':dict(),'combo':dict()}
    tree_acc = {'under':dict(),'over':dict(),'weight':dict(),'combo':dict()}
    tree_f = {'under':dict(),'over':dict(),'weight':dict(),'combo':dict()}
    conv_acc = {'under':dict(),'over':dict(),'weight':dict(),'combo':dict()}
    conv_f = {'under':dict(),'over':dict(),'weight':dict(),'combo':dict()}

    for tag in near_acc:
        for feat in features:
            near_acc[tag][feat] = list()
            near_f[tag][feat] = list()
            tree_acc[tag][feat] = list()
            tree_f[tag][feat] = list()
            conv_acc[tag][feat] = list()
            conv_f[tag][feat] = list()

    for f1 in files:

        print("\nTesting on " + str(f1), flush=True)

        #test on current subject
        subject = load_file(data_path + f1)
        X_test = subject['data']
        y_test = subject['labels']

        #create dataset for all but given subject
        train =  {'data':{'ACC':np.empty((0,ACC_SF*60,3)),
                'BVP':np.empty((0,BVP_SF*60,1)),
                'EDA':np.empty((0,EDA_SF*60,1)),
                'TEMP':np.empty((0,TEMP_SF*60,1))},
                'labels': []}


        #append all other subjects to training datasets
        for i in range(len(files)):
            if files[i] != f1:
                for feat in features:
                    train['data'][feat] = np.append(train['data'][feat],all[i]['data'][feat],0)
                train['labels'] = np.append(train['labels'], all[i]['labels'])

        #Under sample data
        under = under_sample(train)
        #over sample data
        over = over_sample(train)

        #train algorithms
        for feat in features:
            print('\n' + feat, flush=True)

            """Under Sampling"""
            print("Under Sampling", flush=True)

            #nearest neighbors
            acc, f = alg_testing.train_alg(KNeighborsClassifier(), under[feat], X_test[feat], under['labels'], y_test)
            near_acc['under'][feat].append(acc)
            near_f['under'][feat].append(f)

            #decision tree
            acc, f = alg_testing.train_alg(DecisionTreeClassifier(), under[feat], X_test[feat], under['labels'], y_test)
            tree_acc['under'][feat].append(acc)
            tree_f['under'][feat].append(f)

            #convolutional network
            acc, f = alg_testing.train_alg(Sequential(), under[feat], X_test[feat], under['labels'], y_test)
            conv_acc['under'][feat].append(acc)
            conv_f['under'][feat].append(f)

            """Over Sampling"""
            print("Over Sampling", flush=True)

            #nearest neighbors
            acc, f = alg_testing.train_alg(KNeighborsClassifier(), over[feat], X_test[feat], over['labels'], y_test)
            near_acc['over'][feat].append(acc)
            near_f['over'][feat].append(f)

            #decision tree
            acc, f = alg_testing.train_alg(DecisionTreeClassifier(), over[feat], X_test[feat], over['labels'], y_test)
            tree_acc['over'][feat].append(acc)
            tree_f['over'][feat].append(f)

            #convolutional network
            acc, f = alg_testing.train_alg(Sequential(), over[feat], X_test[feat], over['labels'], y_test)
            conv_acc['over'][feat].append(acc)
            conv_f['over'][feat].append(f)

            """Weighting"""
            print("Weighting", flush=True)
            #nearest neighbors does not easily allow class weighting, so it will be ignored here

            #Imbalance is roughly 16 nonstress : 1 stress, weights match that
            weights = np.ones(len(train['labels']))
            weights[train['labels'] == 1] = 16
            conv_weights = {0 : 1, 1 : 16}

            #decision tree
            acc, f = alg_testing.train_alg(DecisionTreeClassifier(), train['data'][feat], X_test[feat], train['labels'], y_test, weights)
            tree_acc['weight'][feat].append(acc)
            tree_f['weight'][feat].append(f)

            #convolutional network
            acc, f = alg_testing.train_alg(Sequential(), train['data'][feat], X_test[feat], train['labels'], y_test, conv_weights)
            conv_acc['weight'][feat].append(acc)
            conv_f['weight'][feat].append(f)

            """Combination of Over, Under, and Weighting"""
            print("Combo", flush=True)
            #Again nearest neighbors does not easily allow class weighting, so it will be ignored here

            #under sample to 10 nonstress : 1 stress
            combo = under_sample(train, 0.1)
            #reformat combo so it fits with over_sample
            combo['data'] = dict()
            for t in features:
                combo['data'][t] = combo[t]
                #remove the old key to save memory
                del combo[t]

            #over sample to 4 nonstress : 1 stress
            combo = over_sample(combo, 0.25)
            #weight at 4 stress : 1 stress to match new imbalance
            weights = np.ones(len(combo['labels']))
            weights[combo['labels'] == 1] = 4
            conv_weights = {0 : 1, 1 : 4}

            #decision tree
            acc, f = alg_testing.train_alg(DecisionTreeClassifier(), combo[feat], X_test[feat], combo['labels'], y_test, weights)
            tree_acc['combo'][feat].append(acc)
            tree_f['combo'][feat].append(f)

            #convolutional network
            acc, f = alg_testing.train_alg(Sequential(), combo[feat], X_test[feat], combo['labels'], y_test, conv_weights)
            conv_acc['combo'][feat].append(acc)
            conv_f['combo'][feat].append(f)


    #Write results to the file

    #Under sampling
    file.write("Average Results of Under Sampling With Nearest Neighbors:\n"+
        "ACC: Accuracy = " + str(np.mean(near_acc['under']['ACC']))+
        " F Score = " + str(np.mean(near_f['under']['ACC']))+
        "\nBVP: Accuracy = " + str(np.mean(near_acc['under']['BVP']))+
        " F Score = " + str(np.mean(near_f['under']['BVP']))+
        "\nEDA: Accuracy = " + str(np.mean(near_acc['under']['EDA']))+
        " F Score = " + str(np.mean(near_f['under']['EDA']))+
        "\nTEMP: Accuracy = " + str(np.mean(near_acc['under']['TEMP']))+
        " F Score = " + str(np.mean(near_f['under']['TEMP']))+"\n\n")

    file.write("Average Results of Under Sampling With Decision Tree:\n"+
        "ACC: Accuracy = " + str(np.mean(tree_acc['under']['ACC']))+
        " F Score = " + str(np.mean(tree_f['under']['ACC']))+
        "\nBVP: Accuracy = " + str(np.mean(tree_acc['under']['BVP']))+
        " F Score = " + str(np.mean(tree_f['under']['BVP']))+
        "\nEDA: Accuracy = " + str(np.mean(tree_acc['under']['EDA']))+
        " F Score = " + str(np.mean(tree_f['under']['EDA']))+
        "\nTEMP: Accuracy = " + str(np.mean(tree_acc['under']['TEMP']))+
        " F Score = " + str(np.mean(tree_f['under']['TEMP']))+"\n\n")

    file.write("Average Results of Under Sampling With Convolutional Network:\n"+
        "ACC: Accuracy = " + str(np.mean(conv_acc['under']['ACC']))+
        " F Score = " + str(np.mean(conv_f['under']['ACC']))+
        "\nBVP: Accuracy = " + str(np.mean(conv_acc['under']['BVP']))+
        " F Score = " + str(np.mean(conv_f['under']['BVP']))+
        "\nEDA: Accuracy = " + str(np.mean(conv_acc['under']['EDA']))+
        " F Score = " + str(np.mean(conv_f['under']['EDA']))+
        "\nTEMP: Accuracy = " + str(np.mean(conv_acc['under']['TEMP']))+
        " F Score = " + str(np.mean(conv_f['under']['TEMP']))+"\n\n")

    #Over sampling
    file.write("Average Results of Over Sampling With Nearest Neighbors:\n"+
        "ACC: Accuracy = " + str(np.mean(near_acc['over']['ACC']))+
        " F Score = " + str(np.mean(near_f['over']['ACC']))+
        "\nBVP: Accuracy = " + str(np.mean(near_acc['over']['BVP']))+
        " F Score = " + str(np.mean(near_f['over']['BVP']))+
        "\nEDA: Accuracy = " + str(np.mean(near_acc['over']['EDA']))+
        " F Score = " + str(np.mean(near_f['over']['EDA']))+
        "\nTEMP: Accuracy = " + str(np.mean(near_acc['over']['TEMP']))+
        " F Score = " + str(np.mean(near_f['over']['TEMP']))+"\n\n")

    file.write("Average Results of Over Sampling With Decision Tree:\n"+
        "ACC: Accuracy = " + str(np.mean(tree_acc['over']['ACC']))+
        " F Score = " + str(np.mean(tree_f['over']['ACC']))+
        "\nBVP: Accuracy = " + str(np.mean(tree_acc['over']['BVP']))+
        " F Score = " + str(np.mean(tree_f['over']['BVP']))+
        "\nEDA: Accuracy = " + str(np.mean(tree_acc['over']['EDA']))+
        " F Score = " + str(np.mean(tree_f['over']['EDA']))+
        "\nTEMP: Accuracy = " + str(np.mean(tree_acc['over']['TEMP']))+
        " F Score = " + str(np.mean(tree_f['over']['TEMP']))+"\n\n")

    file.write("Average Results of Over Sampling With Convolutional Network:\n"+
        "ACC: Accuracy = " + str(np.mean(conv_acc['over']['ACC']))+
        " F Score = " + str(np.mean(conv_f['over']['ACC']))+
        "\nBVP: Accuracy = " + str(np.mean(conv_acc['over']['BVP']))+
        " F Score = " + str(np.mean(conv_f['over']['BVP']))+
        "\nEDA: Accuracy = " + str(np.mean(conv_acc['over']['EDA']))+
        " F Score = " + str(np.mean(conv_f['over']['EDA']))+
        "\nTEMP: Accuracy = " + str(np.mean(conv_acc['over']['TEMP']))+
        " F Score = " + str(np.mean(conv_f['over']['TEMP']))+"\n\n")

    #Weighting
    file.write("Average Results of Weighting With Decision Tree:\n"+
        "ACC: Accuracy = " + str(np.mean(tree_acc['weight']['ACC']))+
        " F Score = " + str(np.mean(tree_f['weight']['ACC']))+
        "\nBVP: Accuracy = " + str(np.mean(tree_acc['weight']['BVP']))+
        " F Score = " + str(np.mean(tree_f['weight']['BVP']))+
        "\nEDA: Accuracy = " + str(np.mean(tree_acc['weight']['EDA']))+
        " F Score = " + str(np.mean(tree_f['weight']['EDA']))+
        "\nTEMP: Accuracy = " + str(np.mean(tree_acc['weight']['TEMP']))+
        " F Score = " + str(np.mean(tree_f['weight']['TEMP']))+"\n\n")

    file.write("Average Results of Weighting With Convolutional Network:\n"+
        "ACC: Accuracy = " + str(np.mean(conv_acc['weight']['ACC']))+
        " F Score = " + str(np.mean(conv_f['weight']['ACC']))+
        "\nBVP: Accuracy = " + str(np.mean(conv_acc['weight']['BVP']))+
        " F Score = " + str(np.mean(conv_f['weight']['BVP']))+
        "\nEDA: Accuracy = " + str(np.mean(conv_acc['weight']['EDA']))+
        " F Score = " + str(np.mean(conv_f['weight']['EDA']))+
        "\nTEMP: Accuracy = " + str(np.mean(conv_acc['weight']['TEMP']))+
        " F Score = " + str(np.mean(conv_f['weight']['TEMP']))+"\n\n")

    #Combination
    file.write("Average Results of Combination Method With Decision Tree:\n"+
        "ACC: Accuracy = " + str(np.mean(tree_acc['combo']['ACC']))+
        " F Score = " + str(np.mean(tree_f['combo']['ACC']))+
        "\nBVP: Accuracy = " + str(np.mean(tree_acc['combo']['BVP']))+
        " F Score = " + str(np.mean(tree_f['combo']['BVP']))+
        "\nEDA: Accuracy = " + str(np.mean(tree_acc['combo']['EDA']))+
        " F Score = " + str(np.mean(tree_f['combo']['EDA']))+
        "\nTEMP: Accuracy = " + str(np.mean(tree_acc['combo']['TEMP']))+
        " F Score = " + str(np.mean(tree_f['combo']['TEMP']))+"\n\n")

    file.write("Average Results of Combination Method With Convolutional Network:\n"+
        "ACC: Accuracy = " + str(np.mean(conv_acc['combo']['ACC']))+
        " F Score = " + str(np.mean(conv_f['combo']['ACC']))+
        "\nBVP: Accuracy = " + str(np.mean(conv_acc['combo']['BVP']))+
        " F Score = " + str(np.mean(conv_f['combo']['BVP']))+
        "\nEDA: Accuracy = " + str(np.mean(conv_acc['combo']['EDA']))+
        " F Score = " + str(np.mean(conv_f['combo']['EDA']))+
        "\nTEMP: Accuracy = " + str(np.mean(conv_acc['combo']['TEMP']))+
        " F Score = " + str(np.mean(conv_f['combo']['TEMP']))+"\n\n")


    print("Finished Leave-One-Out Testing", flush=True)

reg_testing("Results/Imbalance_Testing/Random Sampling Results.txt", "Formatted_ADARP/Raw/All.pkl")
leave_one_out("Results/Imbalance_Testing/Leave One Out Results.txt", "Formatted_ADARP/Raw/")

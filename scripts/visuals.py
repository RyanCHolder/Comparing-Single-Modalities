import pickle
import numpy as np
import matplotlib.pyplot as plt

#directory for the original data
WESAD_dir = 'WESAD/'

#directory for formatted data
form_WESAD = 'Formatted_WESAD/Raw/'
form_ADARP = 'Formatted_ADARP/Raw/'

def load(path):
    """
        @breif: loads the data for the given file
        @param: path (String): The path for the file
        @return: Returns the dictionary for the data in the file
    """
    with open(path, 'rb') as file:
        data = pickle.load(file, encoding='latin1')
    return data;

def plot_data(tag, data, labels):
    """
        @breif: Creates visuals for the data
        @param: tag (String): Contains titling info (formatted vs unformatted and which subject)
        @param: data (dictionary): Data to plot
        @param: labels (list): Data labels (kept separate because of differing
            organization in formatted and unformatted data)
    """
    #set font
    plt.rcParams.update({'font.size': 22})
    #plot labels
    plt.figure(1)
    plt.plot(range(len(labels)), labels)
    plt.ylabel("Label")
    plt.xlabel("Samples Over Time")
    #plt.title(tag + " Label Distribution")

    #plot acceleration
    plt.figure(2)
    plt.subplot(3,1,1)
    #plt.title(tag + " Acceleration")
    plt.plot(range(len(data['ACC'])),data['ACC'][:,0])
    plt.ylabel("X Acceleration")
    plt.subplot(3,1,2)
    plt.plot(range(len(data['ACC'])),data['ACC'][:,1])
    plt.ylabel("Y Acceleration")
    plt.subplot(3,1,3)
    plt.plot(range(len(data['ACC'])),data['ACC'][:,2])
    plt.ylabel("Z Acceleration")
    plt.xlabel("Samples Over Time")

    #plot BVP
    plt.figure(3)
    #plt.title(tag + " Blood Volume Pulse")
    plt.plot(range(len(data['BVP'])),data['BVP'])
    plt.ylabel("Blood Volume Pulse")
    plt.xlabel("Samples Over Time")

    #plot EDA
    plt.figure(4)
    #plt.title(tag + " Electrodermal Activity")
    plt.plot(range(len(data['EDA'])),data['EDA'])
    plt.ylabel("Electrodermal Activity")
    plt.xlabel("Samples Over Time")

    #plot TEMP
    plt.figure(5)
    #plt.title(tag + " Temperature")
    plt.plot(range(len(data['TEMP'])),data['TEMP'])
    plt.ylabel("Temperature")
    plt.xlabel("Samples Over Time")

    plt.show()

def plot_unformatted(subject):
    """
        @breif plots the original WESAD data
        @param: subject (String): Number corresponding to the subject to graph
    """
    #gather data
    data = load(WESAD_dir + 'S' + subject + '/S' + subject + '.pkl')
    labels = data['label']
    data = data['signal']['wrist']
    #plot data
    plot_data("Unformmated Subject " + subject, data, labels)

def plot_formatted(subject, dir):
    """
        @breif: plots the formatted data (the mean of each timestep is
        used to keep the data in 2 dimensions for plotting)
        @param: subject (String): Number corresponding to the subject to graph
        @param: dir (String): Location of data

        Note:6 this data is normalized, so the y-values are only relative
    """
    #gather data
    data = load(dir + 'S' + subject + '.pkl')

    #take averages of each timestep
    for feat in data['data']:
        data['data'][feat] = np.mean(data['data'][feat],1)

    #plot data
    plot_data("Formatted Subject " + subject, data['data'], data['labels'])

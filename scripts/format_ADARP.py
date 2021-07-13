from pandas import read_csv
import numpy as np
from scipy import stats
from os import listdir
import format_data


#data directory
data_dir = "C:/Users/ryan/Documents/REU/WESAD Learning/ADARP Data/"

#list of features
features = ['ACC','BVP','EDA','TEMP']

#samlping frequencies
ACC_SF = 32
BVP_SF = 64
EDA_SF = 4
TEMP_SF = 4

#dimensions of each window arrays

#sampling frequencies in a dictionary
SF_dict = {'ACC':ACC_SF, 'BVP':BVP_SF, 'EDA':EDA_SF, 'TEMP':TEMP_SF}

def get_subject_data(path):
    """
        @brief returns an array containing the data of the given csv file
        @return The dictionary of data in the file, with labels
    """
    #dictionary for data with empty numpy arrays to store
    data = {'data': {'ACC':np.empty((0,int(ACC_SF*60),3)), 'BVP':np.empty((0,int(BVP_SF*60),1)),
            'EDA':np.empty((0,int(EDA_SF*60),1)), 'TEMP':np.empty((0,int(TEMP_SF*60),1))},
                'labels':[]}

    #get files in subject's directory, this will be ordered as 0, 1, 10, 2, 3 ...
    #because listdir is in alphabetical order, it makes no real difference
    files = listdir(path)

    #gather data from each data file
    for f_name in files:
        temp = {'ACC':np.empty((0,int(ACC_SF*60),3)), 'BVP':np.empty((0,int(BVP_SF)*60)),
                'EDA':np.empty((0,int(EDA_SF*60))), 'TEMP':np.empty((0,int(TEMP_SF*60)))}
        #gather data for each feature
        for feat in features:
            #read file
            file = read_csv(path + f_name + '/' + feat + '.csv', header = None)
            #get timestamp, this is done for each feature but shouldn't change
            start_time = int(np.array(file)[0,0])

            #get sensor values and form into one minute windows (50% overlap)
            values = format_data.create_windows(file[2:], SF_dict[feat], 60)
            #cut off the last window because a full window is not complete
            values = values[:-1]

            temp[feat] = np.array(values)

        #assign label to 20 minutes
        #start with labels array as all nonstress
        labels = np.zeros(len(temp['ACC']))

        #get the timestamps for stress
        #using a try statement in case there are no tags
        try:
            tags = np.array(read_csv(path + f_name + '/tags.csv', header = None))
            #reshape the tags to be only one dimensional
            tags = np.reshape(tags, len(tags))
            #subtract the start time from the tag time
            tags = tags - start_time
            #find which minute this number of seconds corresponds to
            tags = np.around(tags/60)
            #multiply by two because of the overlap
            tags = tags * 2

            #reasign labels around the tag
            for tag in tags:
                #hour around tag (doubled because of overlap)
                for x in range(int(tag)-120, int(tag)+120):
                    #only change the label if it is within the length of the data
                    if x >= 0 and x < len(labels):
                        #if tag is within 20 minutes (40 datapoints) set to stress label
                        if x in range(int(tag)-20, int(tag)+20):
                            labels[x] = 1
                        #otherwise set to 'need to be removed' label
                        #in case of overlap make sure not to overwrite stress labels
                        elif labels[x] != 1:
                            labels[x] = 2
        except:
            pass

        #make sure all modalities have the same number of windows
        lengths = [len(temp['ACC']),len(temp['BVP']),len(temp['EDA']),len(temp['TEMP']), len(labels)]

        #truncate to make all modalities the same length
        if len(np.unique(lengths)) > 1:
            length = np.min(np.unique(lengths))
            for feat in features:
                temp[feat] = temp[feat][:length]
            labels = labels[:length]

        #remove labels and datapoints at the 'need to be removed' label
        for feat in features:
            temp[feat] = temp[feat][labels != 2]
        labels = labels[labels != 2]

        #append the data from the file to the overall subject  data
        for feat in features:
            #some files contain less than a minute of data, and will be ignored
            if np.size(temp[feat]) > 0:
                data['data'][feat] = np.append(data['data'][feat],temp[feat],0)
        data['labels'] = np.append(data['labels'],labels,0)

    return data

def save_all():
    """
        @breif saves individual and combined raw and statistical data
    """
    #list of all subject files
    subjects = listdir(data_dir)

    #dictionaries for storing data
    all_raw = {'data': {'ACC':np.empty((0,int(ACC_SF*60),3)), 'BVP':np.empty((0,int(BVP_SF*60),1)),
            'EDA':np.empty((0,int(EDA_SF*60),1)), 'TEMP':np.empty((0,int(TEMP_SF*60),1))},
                'labels':[]}
    all_stat = {'data': {'ACC':np.empty((0,8,3)), 'BVP':np.empty((0,8,1)),
            'EDA':np.empty((0,8,1)), 'TEMP':np.empty((0,8,1))},
                'labels':[]}

    for x in range(0, len(subjects)):
        #get subject data
        subject = get_subject_data(data_dir + '/' + subjects[x] + '/')
        #get statistical values
        stat = format_data.get_statistics(subject)
        #save statistical data
        format_data.save_data('Formatted_ADARP/Statistical/S'+str(x)+'.pkl', stat)

        for feat in features:
            subject['data'][feat] = format_data.norm(subject['data'][feat])
        format_data.save_data('Formatted_ADARP/Raw/S'+str(x)+'.pkl', subject)

        #append data to dictionaries
        for feat in features:
            #raw
            all_raw['data'][feat] = np.append(all_raw['data'][feat],subject['data'][feat],0)
            #statistical
            all_stat['data'][feat] = np.append(all_stat['data'][feat],stat['data'][feat],0)
        #raw
        all_raw['labels'] = np.append(all_raw['labels'], subject['labels'],0)
        #statistical
        all_stat['labels'] = np.append(all_stat['labels'], stat['labels'],0)

        print("Finished file " + str(x))

    #save combined dictionaries
    format_data.save_data('Formatted_ADARP/Raw/All.pkl', all_raw)
    format_data.save_data('Formatted_ADARP/Statistical/All.pkl', all_stat)

save_all()

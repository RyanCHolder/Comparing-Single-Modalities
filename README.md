# Comparing-Single-Modalities

Formatting data from the WESAD and ADARP datasets, running several learning algorithms on them, and comparing the results from single modalities

In order to run this project, first use the format_data.py script file to create a formatted version of the WESAD data.

  To do this, first the data_dir field should be changed to match your setup (the directory containing the WESAD dataset.)
  Once this field has been changed, either import the script and run save_formatted_data(path) with path as your desired destination to format
  and save the regular data, and similarly use save_statistics(path) to do the same for the statistical data,
  or add calls to the functions at the end of the file with similar syntax.
  
  Similar steps can be taken using the format_ADARP.py script for formatting ADARP data
  
Once the data has been saved the alg_testing.py script can be used in a similar fashion to run the learning algorithms on the data. 
  The classifiers field contains a list of all the classifiers used, which can be edited to add or remove classifiers as desired 
    (adding more sklearn classifiers would be simple, adding classifiers from outside sklearn, like the karis.Sequential already in there would cause issues)
    
  To run a test with a randomly selected test group (at 25% test size), call the reg_testing(save_path), if your file hierarchy doesn't match the one encoded
  then additional to the save_path parameter (which would be the string for the destination of the results), the data_loc and stat_loc parameters
  should be specified to the file locations of the formatted regular and statistical data respectively.
  Additionally the number of iterations can be specified as another parameter (default: iterations=10)
  
  To run a leave-one-out test, a similar procedure should be followed with the leave_one_out(save_path) function.
  
  Both testing functions will create new text files with the specified names, and will rewrite the file with the given name if it already exists.
  
  Some warnings will be triggered by running the alg_testing script including the convolutional network classifier since the statistical data will 
  be ignored for that classifier (statistical data doesn't fit the dimensional requirements), and the printing of the stastical results averages
  will involve a divide by zero. However, these warning can be ignored because the results are unaffected by this (stastical values during convolutional
  training will just be Nan)
  
  imbalance_testing.py can be run similarly to alg_testing and tests a view class imbalance solutions. This scripts compares over sampling, under sampling, weighting,
  as well as a combination of the three as solutions to the large class imbalance in the ADARP data.
  
  comb_modalities.py can also be run similarly using both WESAD and ADARP. This scripts will combine all four modalities and train the same learning
  algorithms on all modalities. This also gives the option to oversample (which was the best performing class imbalance solution) which is advised to be used
  if this is ran on highly imbalanced data like ADARP.
  
  The visuals.py script can be run to create graphs of the data.
  By running plot_formatted(subject), where subject is a string coressponding to the desired subject number, the formatted data will be plotted 
  (using the averages of each time step)
  Running plot_unformatted(subject) will plot the original un-edited data for the desired subject
  
  Notes -
    
  1 second windows with 50% overlap were used as time steps
  All invalid labels, as well as the amusement label from the WESAD data were ignored
  The meditation and baseline labels were combined into a non-stressed label
  After combining baseline and meditation, the data had a rough 3:1 ratio of non-stress to stressed labels
  A few of the classifiers have been commented out as they are quite slow, but removing the commenting will still run if those classifiers are wanted.
  
  The WESAD dataset that was used can be found here:
  https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/
  

# WESAD-Learning
Formatting data from the WESAD dataset, and running Sci-Kit Learn algorithms and a conv-net on it for comparisons

In order to run this project, first use the format_data.py script file to create a formatted version of the WESAD data.

  To do this, first the data_dir field should be changed to match your setup (the directory containing the WESAD dataset.)
  Once this field has been changed, either import the script and run save_formatted_data(path) with path as your desired destination to format
  and save the regular data, and similarly use save_statistics(path) to do the same for the statistical data,
  or add calls to the functions at the end of the file with similar syntax.
  
Once the data has been saved the alg_testing.py script can be used in a similar fassion to run the learning algorithms on the data. 
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
  
  Notes:
    
    1 second windows with 50% overlap were used as time steps
    All invalid labels, as well as the amusement label from the WESAD data were ignored
    The meditation and baseline labels were combined into a non-stressed label
    After combining baseline and meditation, the data had a rough 3:1 ratio of non-stress to stressed labels

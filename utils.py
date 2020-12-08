import os
import pandas as pd
import numpy as np
from IPython.display import display


def get_table(filepath = None):
    """Returns android.csv as a DataFrame

    This function assumes that code is in a directory which has the same parent directory as the LecturePatternRecognion
    repository. If this is not the case, please supply a string with the correct file path to the filepath argument.
    """

    if filepath is None:
        cur_path = os.getcwd()
        path = os.path.relpath('../LecturePatternRecognition/project01/android.csv', cur_path)
    else:
        path = filepath
    return pd.read_csv(path)

def hash_states(state_batch):
    """Converts a batch of states or activity vectors to a sequence of unique integer labels

    This function assumes that the batch of states is a numpy array shaped (num_states, dim_state).
    The output is a numpy array shaped (num_states, ) containing integer labels in the range between
    0 and num_unique_states.
    """

    #convert state batch to 1d array with decimal state representation
    arr = state_batch
    #sort the 1d array
    sorted_idx = np.lexsort(arr.T)
    sorted_arr = arr[sorted_idx]
    #calculate the difference from element i to element i+1 for all i
    diffs = np.diff(sorted_arr, axis=0)
    #mark where the entries in the sorted array have changed
    diff_flags = np.append([False], np.any(diffs != 0, axis=-1))
    #use this to get unique labels at each entry
    labels = diff_flags.cumsum()
    #arrange this in a manner 'inverse' to the previous sorting 
    pos_labels = np.zeros_like(labels)
    pos_labels[sorted_idx] = labels
    return pos_labels

if __name__ == '__main__':
    dataset_df = get_table()
    display(dataset_df)

    # get the name of the packages that is running in the system
    text = 'packages_running_'
    running_packages = [i for i in dataset_df.columns if text in i]

    # creating the activity vector containing info about running apps and battery_plugged
    extra_columns = ['battery_plugged']
    activity_vectors_df = dataset_df[[*extra_columns, *running_packages]]
    activity_vectors = activity_vectors_df.dropna().to_numpy()

    # battery usage for each activity vector
    battery_usage = dataset_df['battery_level'].to_list()
    print(battery_usage)
    
    # convert to a sequence of labels, note that nan was dropped above
    out_labels = hash_states(activity_vectors)
    print(activity_vectors[:10])
    print(out_labels[:10])
    
    



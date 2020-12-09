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
    0 and num_unique_states-1. Be sure to pass all relevant states at once.
    """

    #sort the input array
    sorted_idx = np.lexsort(state_batch.T)
    sorted_arr = state_batch[sorted_idx]
    #calculate the difference from element i to element i+1 for all i
    diffs = np.diff(sorted_arr, axis=0)
    #mark where any entries in the sorted array have changed
    diff_flags = np.append([False], np.any(diffs != 0, axis=-1))
    #use this to get unique labels at each entry
    labels = diff_flags.cumsum()
    #arrange this in a manner 'inverse' to the previous sorting 
    pos_labels = np.zeros_like(labels)
    pos_labels[sorted_idx] = labels
    return pos_labels

def lookup_states(query, labels, state_batch):
    """Converts a sequence of labels to the corresponding states, given a full sequence of labels and
    the corresponding batch of states.

    This function assumes that  query is a sequence of integer labels shaped (length_query, ), the labels
    are a sequence of integer labels shaped (num_states, ) and the batch of states is a numpy array shaped
    (num_states, dim_state).
    The output is a numpy array shaped (length_query, dim_state) containing the states corresponding to the
    query labels. If state_batch and labels are input and output to hash_states(), this function implements
    the inverse hash function for each element of the query.
    """
    
    ind_sorted = labels.argsort()
    i = np.searchsorted(labels[ind_sorted], query)
    return state_batch[ind_sorted[i]].reshape((-1, state_batch.shape[-1]))
    
    
    
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

    # given some labels, what states do they represent???
    #labels for which we want to know the states
    query = np.array([93, 160, 92]) 
    #states represented by those labels, in the same order as query
    answer = lookup_states(query, out_labels, activity_vectors)
    print(answer.shape)
    
    



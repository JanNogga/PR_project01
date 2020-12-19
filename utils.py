import os
import pandas as pd
import numpy as np
from IPython.display import display
import time


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

def one_hot_encode(labels, num_states=None):
    """Converts a batch of labels to a batch of one-hot encoded vectors, zero everywhere but at the index of
    the label.
    
    The input is expected to be a numpy array shaped (num_labels, ), the output is a numpy array with the shape
    (num_unique_states, num_labels). The number of states to one-hot encode can be inferred from the labels or passed
    manually. Note that in contrast to other functions, the output here is transposed, meaning that the encoding
    of label i is in column i of the output. This facilitates later matrix multiplication.
    """
    
    if num_states is not None:
        assert num_states > labels.max(), 'The number of states must be larger than the maximum state label!'
    shape = labels.max() + 1 if num_states is None else num_states
    return np.eye(shape)[labels].T

def one_hot_decode(encoding):
    """Converts a batch of one-hot encoded vectors to a batch of labels, corresponding to the unique non-zero index
    of each vector.
    
    The input is expected to be a numpy array shaped (num_unique_states, num_labels), the output is a numpy array
    with the shape (num_labels, ). This function is exactly inverse to one_hot_encode().
    """
    
    return (encoding.T @ np.arange(encoding.shape[0])).astype(int)

def count_transitions(sequence, num_states=None):
    """Given a sequence of suitable integer labels, returns a matrix T[i,j] counting the number of observed
    transitions from label j to label i.

    This function assumes that  sequence is a numpy array shaped (length_sequence, ). By default, the output shape
    is a numpy array shaped (num_unique_states, num_unique_states). This number is inferred from labels in the
    sequence.
    If you want to aggregate counts from different (groups of) sequences, where an individual sequence does not
    necessarily contain each state, pass the number of unique states manually to num_states. You can then sum the
    outputs of this function for each sequence passed. The indexing is specified in lecture 10, on slide 8.
    """
    
    if num_states is not None:
        assert num_states > sequence.max(), 'The number of states must be larger than the maximum state label!'
    shape = sequence.max() + 1 if num_states is None else num_states
    dims = (shape, shape)
    flattened_index = np.ravel_multi_index((sequence[:-1], sequence[1:]), dims)
    return np.bincount(flattened_index, minlength=shape**2).reshape(dims).astype(float).T

def normalize_transition_matrix(transition_matrix):
    """Normalizes a given transition matrix or batch thereof such that each column sums up to one.
    
    The given transition matrix is assumed to be stored in a numpy array. 
    """
    
    input_dims = len(transition_matrix.shape)
    assert input_dims == 2 or input_dims == 3, 'Transition matrix must be a least 2D, and appropriately batched!'
    if input_dims == 2:
        return transition_matrix / np.maximum(transition_matrix.sum(0, keepdims=True), 1)
    else:
        return transition_matrix / np.maximum(transition_matrix.sum(1, keepdims=True), 1)


def state_dist_to_activity_dist(state_dist, labels, state_batch):
    """Given a full set of labels and the corresponding batch of states, converts distributions over the one-hot
    encoded state space to dim_state individual distributions over the activations in corresponding state components.

    This function assumes that state_dist is numpy array shaped (num_unique_states, num_dists). It should contain a batch
    of distributions over the state space in each column. The labels are a sequence of integer labels shaped (num_states, )
    and the batch of states is a numpy array shaped (num_states, dim_state).
    The output is then a batch of distributions over the state vector components shaped (num_dists, dim_state). Each
    row contains the result of one converted distribution, and each column the probability of a specific package 
    (or battery_charge state) being active.
    """
    
    helper_states = lookup_states(np.arange(0, state_dist.shape[0]), labels, state_batch)
    return state_dist.T @ helper_states

def state_dist_to_most_likely_state(state_dist, labels, state_batch):
    max_inds = np.argmax(state_dist, axis=0)
    return lookup_states(max_inds, labels, state_batch)
    

def BCE(prediction_dist, target):
    """Returns the binary cross-entropy loss between a set of probability distributions over the activity vector
    components and the target activity vectors.
    
    This function expects both the input prediction_dist and target to be numpy arrays shaped (num_predictions, dim_state).
    The output is a numpy array shaped (num_predictions, ).
    """
    
    loss = target*np.log(np.maximum(prediction_dist, 1e-15)) + (1-target)*np.log(np.maximum(1-prediction_dist, 1e-15))
    return -loss.mean(axis=-1)

def estim_mle(transition_counts):
    """Returns the MLE estimator transition matrix for a Markov chain. 

    The input is assumed to be a numpy array (n_i,j) counting transitions from state j to state i, shaped
    (num_unique_states, num_unique_states). The output has the same shape, and contains the estimated
    prediction probabilities from state j to state i.
    """
    
    return normalize_transition_matrix(transition_counts)

def estim_map(transition_counts, num_samples=500, limit=100, d_type=np.float32, eps=1e-25):
    """Returns the MAP estimator transition matrix for a Markov chain. 

    The input is assumed to be a numpy array (n_i,j) counting transitions from state j to state i, shaped
    (num_unique_states, num_unique_states). The output has the same shape, and contains the estimated
    prediction probabilities from state j to state i. You can also set the number of samples for the approximation
    of the result, as well as the upper limit for the sampled matrix values. Operations are safely possible in
    float32 representation. The amount of memory allocated in RAM by this function is about 5GB per 1k samples.
    """
    
    num_states = transition_counts.shape[-1]
    samples = np.random.randint(limit, size=(num_samples, num_states, num_states), dtype=np.int16).astype(d_type)
    samples = normalize_transition_matrix(samples)
    tmp = transition_counts.astype(d_type) * np.log(np.maximum(samples, eps))
    tmp = tmp.sum(axis=(-1,-2))
    tmp -= tmp.max()
    tmp = np.exp(tmp)
    samples =  samples * tmp[:, None, None] 
    S_res = samples.sum(axis=0)
    eta = tmp.sum()
    return (1/eta * S_res).astype(np.float64)

def fit_predictor(list_of_seqs, num_unique_states, mode, num_samples=500, limit=100, d_type=np.float32, eps=1e-25):
    """Returns MLE or MAP predictors fitted to a given list of sequences describing state transitions.
    
    Please provide the number of unique states yourself, and pass even a single sequence as a one-element list. The 
    observed state transition counts are aggregated over all sequences, and then used to get the best fit transition
    matrix. Also note the arguments which might be passed down to different prediction modes, but are irrelevant to
    others.
    """
    
    N = np.zeros((num_unique_states, num_unique_states))
    for seq in list_of_seqs:
        N += count_transitions(seq, num_states=num_unique_states)
        
    if mode == 'MLE':
        return estim_mle(N)
    elif mode == 'MAP':
        return estim_map(N, num_samples=num_samples, limit=limit, d_type=d_type, eps=eps)
    else:
        raise NotImplementedError("Unknown predictor mode!")

def PCA(data, correlation = False, sort = True):

    mean = np.mean(data, axis=0)

    data_adjust = data - mean
    
    if correlation:

        matrix = np.corrcoef(data_adjust.T)

    else:
        matrix = np.cov(data_adjust.T) 

    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    if sort:
    #: sort eigenvalues and eigenvectors
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:,sort]

    return eigenvalues, eigenvectors
    
# document these!
def predict(list_of_seqs, P, N_steps, num_unique_states):
    in_flat = np.concatenate([item[:-N_steps] for item in list_of_seqs])
    one_hot_prediction_input = one_hot_encode(in_flat, num_states = num_unique_states)
    pred = np.power(P, N_steps)
    # calculate a distribution over future states
    return pred @ one_hot_prediction_input

def list_to_prediction_targets(list_of_seqs, N_steps, labels, state_set):
    targets = np.concatenate([item[N_steps:] for item in list_of_seqs])
    return lookup_states(targets, labels, state_set)

def prediction_output_transform(pred_out, labels, state_set, mode):
    if mode == 'activity_dist':
        return state_dist_to_activity_dist(pred_out, labels, state_set)
    elif mode == 'argmax':
        return state_dist_to_most_likely_state(pred_out, labels, state_set)
    elif mode == 'nearest_neighbor':
        distribution = state_dist_to_activity_dist(pred_out, labels, state_set)
        return distribution >= 0.5
    else:
        raise NotImplementedError("Unknown prediction output transform!")
        
def bit_flips(prediction_dist, target):
    dist_rounded = prediction_dist >= 0.5
    loss = dist_rounded.astype(int) == target.astype(int)
    return 1 - loss.mean(axis=-1)

def target_space_transform(targets, mode='Id', direction='Forward'):
    if mode == 'Id':
        if direction == 'Forward':
            return targets
        elif direction == 'Backward':
            return targets
        else:
            raise NotImplementedError("Unknown direction!")
    elif mode == 'Sqrt':
        if direction == 'Forward':
            return np.sqrt(np.abs(targets))
        elif direction == 'Backward':
            return -np.power(targets, 2)
        else:
            raise NotImplementedError("Unknown direction!")   
    else:
        raise NotImplementedError("Unknown target space transform!")
        
# maybe check what is X, and what is X.T here...
def build_state_mat(states):
    return np.concatenate([np.ones((states.shape[0],1)), states], axis=-1).T

def fit_regressor(states, targets, mode='MLE', eps=1e-25):
    X = build_state_mat(states)
    if mode == 'MLE':
        return np.linalg.inv(X @ X.T + eps*np.eye(X.shape[0])) @ X @ targets
    elif mode == 'MAP':
        var = targets.var()
        return np.linalg.inv(X @ X.T + var*np.eye(X.shape[0])) @ X @ targets
    else:
        raise NotImplementedError("Unknown regressor mode!")
    
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
    
    # Count how many times each label has transitioned to each other label
    T = count_transitions(out_labels)
    # How many times has the state with the label 246 transitioned to the state with the label 245?
    print(T[245, 246])
    # How many times has the state with the label 245 transitioned to itself?
    print(T[245, 245])
    # How many times has the state with the label 245 transitioned to the state with the label 246?
    print(T[246, 245])
    
    # sanity check P against the example in lecture 10 on slide 13
    # unfortunately, the example is wrong in the lecture, as a manual recount shows...
    lines = open('lecture_example.txt','r')
    char_list = []
    for line in lines:
        line = line.split()
        for char in line[0]:
            char_list.append(ord(char)-ord('A'))

    char_list = np.array(char_list)
    print(char_list)

    T_sanity_check = count_transitions(char_list)
    P_sanity_check = normalize_transition_matrix(T_sanity_check)
    print(P_sanity_check)
    
    # very simple prediction and loss calculation example (not partioned into train/valid/test sets!)
    # battery still needs adjustment, contains strange values != 0 or 1, so it isn't included here
    activity_vectors_df = dataset_df[running_packages] 
    activity_vectors = activity_vectors_df.dropna().to_numpy()
    out_labels = hash_states(activity_vectors)
    # number of prediction steps into the future
    N_steps = 1 
    one_hot_prediction_input = one_hot_encode(out_labels[:-N_steps], num_states = out_labels.max() + 1)
    T = count_transitions(out_labels)
    P = normalize_transition_matrix(T)
    pred = np.power(P, N_steps)
    # calculate a distribution over future states
    estimate = pred @ one_hot_prediction_input 
    # convert to sets of individual distributions over activity components
    estimate_activity_dist = state_dist_to_activity_dist(estimate, out_labels, activity_vectors)
    # simple targets - just time-shifted the input by N_steps
    targets = lookup_states(out_labels[N_steps:], out_labels, activity_vectors)
    # calculate loss
    loss = BCE(estimate_activity_dist, targets)
    # print reduced loss
    print('Prediction loss over the whole dataset for', N_steps, 'time-steps:', loss.mean())
    
    # a very basic attempt at regression - note that the state space seems 'redundant'
    print('Number of running packages:', len(running_packages))
    activity_vectors_df = dataset_df[running_packages[:49]+running_packages[50:54]+running_packages[55:]]
    print('...after removing some:', len(running_packages[:49]+running_packages[50:54]+running_packages[55:]))
    activity_vectors_df = activity_vectors_df[dataset_df['battery_level'] <= 0.]
    activity_vectors_df = activity_vectors_df.T.drop_duplicates().T
    activity_vectors = activity_vectors_df.dropna().to_numpy()
    x = activity_vectors
    print('Final data shape:', x.shape)
    X = np.concatenate([np.ones((x.shape[0],1)), x], axis=-1).T
    print('X shape:', X.shape)
    print('X.T shape:', X.T.shape)
    y = dataset_df['battery_level'].dropna().to_numpy()
    y = y[y <= 0.]
    print('Targets shape:', y.shape)
    print('Hat matrix shape:', (X @ X.T).shape)
    w = np.linalg.inv(X @ X.T) @ X @ y
    print('Weights Shape:', w.shape)
    
    # a minimal example of the MLE/MAP transition matrix estimation process
    # the calculations for MAP are borrowed from the ipynb example uploaded by Sebastian to ecampus, and executed in 'logspace'
    # fix random number generator so our results are the same
    np.random.seed(123)
    # first, we define some ground truth transition matrix for a dummy Markov Chain
    gt = np.array([[1./3., 1./4., 1./5., 1./6.], [1./6., 1./4., 1./5, 1./3.], [1./3., 1./4., 2./5, 1./6.], [1./6., 1./4., 1./5., 1./3.]])
    # start in some state, here it is 0
    traj = [int(0)]
    # get only a small number of sample transitions within this Markov Chain, add them to the 'trajectory'
    nn = 29
    #simulate the process
    for i in range(nn):
        traj.append(np.random.choice([0, 1, 2, 3], p=gt[:,traj[-1]]).astype(int))
    traj = np.array(traj).astype(int)

    start = time.time()
    # here we have simply split the sequence traj into traj[:15], traj[15:] to emphasize that the input is a list of sequences
    mle_P = fit_predictor([traj[:15], traj[15:]], traj.max()+1, mode='MLE')
    map_P = fit_predictor([traj[:15], traj[15:]], traj.max()+1, mode='MAP')
    
    print('Code block execution took', time.time()-start, 'seconds.')
    print('Ground Truth:\n', gt, '\n')
    print('MLE:\n', mle_P, '\n')
    print('MAP:\n', map_P, '\n')
    
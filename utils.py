import os
import pandas as pd
import numpy as np
from IPython.display import display
# mpmath is included in sympy, which Sebastian recommended we install
import mpmath as mp
# necessary if we want to speed up the MAP prediction
from multiprocessing import Pool, cpu_count


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
    

def BCE(prediction_dist, target):
    """Returns the binary cross-entropy loss between a set of probability distributions over the activity vector
    components and the target activity vectors.
    
    This function expects both the input prediction_dist and target to be numpy arrays shaped (num_predictions, dim_state).
    The output is a numpy array shaped (num_predictions, ).
    """
    
    loss = target*np.log(np.maximum(prediction_dist, 1e-15)) + (1-target)*np.log(np.maximum(1-prediction_dist, 1e-15))
    return -loss.mean(axis=-1)

def exp(arg):
    """Calculate the element-wise exp of an array of mpmath mpf values.

    Just a helper function to allow distribution of costly operations to multiple processes in the MAP prediction.
    """
    return np.vectorize(mp.exp)(arg)

def mul_sum(args):
    """Calculate a weighted sum of matrices containing mpmath mpf values.

    Just another helper function to allow distribution of costly operations to multiple processes in the MAP prediction.
    """
    A, b = args
    return (A * b[:, None, None]).sum(axis=0)
    
    
    
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
    
    # a minimal example of the MAP prediction process
    # the calculations are borrowed from the ipynb example uploaded by Sebastian to ecampus, and executed in 'logspace'
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

    # uniformly sample num_samples transition matrices, with maximum transition entries limit
    # this is how we include our 'prior'
    limit = 100
    num_samples = 100000
    # set the dimensionality of our state space - 4 in the dummy example
    num_states = 4
    S = np.random.randint(limit, size=(num_samples, num_states, num_states))
    print('sample shape:', S.shape)
    # normalize to obtain transition matrices from the sampled transition counts
    S = normalize_transition_matrix(S)
    # get MLE for comparison later
    T_data = count_transitions(traj)
    S_from_T = normalize_transition_matrix(T_data)
    print('T_data shape:', T_data.shape)


    #set mpmath precision
    mp.dps = 25
    #set min input to np.log
    eps = 1e-25
    #start measurement of time
    start = time.time()
    # calculate the MAP prediction term - relevant calculation have been moved to 'logspace'
    # this allows us to rely on pure numpy for a longer section of the code
    tmp_log = T_data * np.log(np.maximum(S, eps))
    print('tmp_log shape:', tmp_log.shape)
    tmp_log = tmp_log.sum(axis=(-1,-2))
    print('tmp_log shape after sum:', tmp_log.shape)
    # Now we have some vector of negative numbers with very large absolute values
    # If we call numpy exp, even float128bit precision will be insufficient to represent the result
    # The result would be rounded to zero, and we would lose all information
    # This problem is more severe the larger the transition matrices are
    exp_from_mp = np.vectorize(mp.exp)
    # hackily 'typecast' the contents of the log vector to mpmath floats
    # apply exp over the whole array, this is very slow, basically like 3 nested for-loops
    tmp_exp = exp_from_mp(tmp_log.astype(object) * mp.mpf('1'))
    print('tmp_exp shape:', tmp_exp.shape)
    S_weighted =  S * tmp_exp[:, None, None] 
    print('S_weighted shape:', S_weighted.shape)
    S_expectation = S_weighted.sum(axis=0)
    print('S_expectation shape:', S_expectation.shape)
    eta = tmp_exp.sum()
    print('eta:', eta)
    # normalize the result, here we divide an array of tiny values by a very small normalization factor eta
    # now our values are large enough again to be represented appropriately by float64
    S_res = (1/eta * S_expectation).astype(np.float64)
    print('Result dtype:', S_res.dtype)
    # Save the result of this code block for comparison with the next one
    np.save('prev.npy', S_res)
    print(S_res.shape)
    # Print runtime, ground truth transition matrix for the simulator, MAP estimate, and MLE estimate
    print('Code block execution took', time.time()-start, 'seconds.')
    print('GT:')
    print(gt)
    print()
    print('MAP:')
    print(S_res)
    print()
    print('MLE:')
    print(S_from_T)

    # Next, repeat the code block above, trying to increase speed by using multiprocessing
    start = time.time()
    # calculate the MAP prediction term - relevant calculation have been moved to 'logspace'
    # this allows us to rely on pure numpy for a longer section of the code
    tmp_log = T_data * np.log(np.maximum(S, eps))
    print('tmp_log shape:', tmp_log.shape)
    tmp_log = tmp_log.sum(axis=(-1,-2))
    print('tmp_log shape after sum:', tmp_log.shape)
    # chunk the arrays of mpfs into several sections
    # each section is handed to its own process
    n_proc = 16 #cpu_count()-1 or 1
    chunk = tmp_log.shape[0]//n_proc + 1
    # parallelize the first operation - the element-wise exp
    input_list = [tmp_log[chunk*k:chunk*(k+1)].astype(object) * mp.mpf('1') for k in range(n_proc)]
    pool_1 = Pool() #processes=n_proc
    pool_res = pool_1.map(exp, input_list)
    pool_1.close()
    # stich the results from each process together
    tmp_exp = np.concatenate(pool_res, axis = 0)
    print('tmp_exp shape:', tmp_exp.shape)
    # parallelize the second operation - weighted matrix sum
    input_list = [(S[chunk*k:chunk*(k+1)], tmp_exp[chunk*k:chunk*(k+1)].astype(object) * mp.mpf('1')) for k in range(n_proc)]
    pool_2 = Pool()
    pool_res = pool_2.map(mul_sum, input_list)
    pool_2.close()
    S_expectation = np.array(pool_res).sum(axis=0)
    print('S_expectation shape:', S_expectation.shape)
    eta = tmp_exp.sum()
    print('eta:', eta)
    # normalize the result, here we divide an array of tiny values by a very small normalization factor eta
    # now our values are large enough again to be represented appropriately by float64
    S_res = (1/eta * S_expectation).astype(np.float64)
    print('Result dtype:', S_res.dtype)
    # load the result of the previous code block for comparison
    S_prev = np.load('prev.npy')
    print(S_res.shape)
    # compare the result of this code block to the result of the previous one
    print('Result changed?', not np.allclose(S_res, S_prev))
    # Print runtime, ground truth transition matrix for the simulator, MAP estimate, and MLE estimate
    print('Cell execution took', time.time()-start, 'seconds.')
    print('GT:')
    print(gt)
    print()
    print('MAP:')
    print(S_res)
    print()
    print('MLE:')
    print(S_from_T)
    # When we have a small number of samples relative to the dimensionality of the state space, the MAP estimate should be safer!
    # Note that improvements in performance are much more noticeable when the state space is larger
    
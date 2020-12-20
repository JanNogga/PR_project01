from utils import *
import wandb #conda install pip, pip install wandb, wandb login
from pandas_profiling import ProfileReport
import seaborn as sns
import matplotlib.pyplot as plt
import os
np.random.seed(2020)

#os.environ['WANDB_MODE'] = 'dryrun' #uncomment to prevent wandb logging - 'offline mode'

hyperparameter_defaults = dict(
    state_transform_mode = 'Id', #'Gain', 'Gini'
    keep_best = 16, #1-34
    predictor_mode = 'MLE', #'MAP'
    prediction_transform = 'argmax', #'activity_dist', 'argmax', 'nearest_neighbor'
    regressor_mode = 'MLE', #'MAP'
    charge_transform_mode = 'Id',
    discharge_transform_mode = 'Id', #'Sqrt'
    eval_predictor = False, #False
    eval_regressor = False, #False
)


wandb.init(config=hyperparameter_defaults, project="PR_project01")
config = wandb.config


if not config.eval_predictor and not config.eval_regressor:
    print('Nothing to do!')
    exit()
start = time.time()

dataset_df = get_table().dropna()
mask = (dataset_df['battery_plugged'] == 0) | (dataset_df['battery_plugged'] == 1)
dataset_df = dataset_df[mask]
# month in 8-12, 1-3, day in 1-31
# the following replacements keep 'monthday' chronologically sorted when hashed later
dataset_df['month'][dataset_df['month'] == 1] = 13
dataset_df['month'][dataset_df['month'] == 2] = 14
dataset_df['month'][dataset_df['month'] == 3] = 15
dataset_df['monthday'] = dataset_df['month']*100 + dataset_df['day']

text = 'packages_running_'
keep = [i for i in dataset_df.columns if text in i] + ['battery_plugged'] + ['battery_level'] + ['slot'] + ['monthday']
dataset_df = dataset_df[keep[:49] + keep[50:54] + keep[55:]]
dataset_df = dataset_df.dropna().T.drop_duplicates().T.reset_index()
dataset_df['md_key'] = hash_states(dataset_df['monthday'].to_numpy()[:,None])
dataset_df = dataset_df.drop(['monthday', 'slot'], axis=1)
dataset_df = dataset_df.drop(['packages_running_android', 'packages_running_com.android.calculator2',\
                             'packages_running_com.android.keychain','packages_running_com.android.packageinstaller',\
                             'packages_running_com.android.providers.applications', 'packages_running_com.android.providers.downloads',\
                             'packages_running_com.google.android.email', 'packages_running_edu.udo.cs.ess.mobidac.target',\
                             'packages_running_org.openintents.filemanager', 'packages_running_stream.android'], axis=1)

# get indices of dataset elements per day, so that we can use this partitioning of the data in training and validation
num_days = dataset_df['md_key'].to_numpy().max() + 1
# by day is a list that for each day, contains all dataset indices for that day
by_day = [np.array(dataset_df.index[dataset_df['md_key'] == i].tolist()) for i in range(num_days)]
# keep only days with at least 5 samples
by_day_filtered = [item for item in by_day if len(item) > 4]
# we can access day i by calling dataset_df.loc[by_day[i]]


# in this state space, battery plugged is the last column: activity_vectors[:,-1]
activity_vectors = dataset_df.drop(['index', 'battery_level', 'md_key'], axis=1).to_numpy()
targets = dataset_df['battery_level'].to_numpy()
print('Activity vectors shape:', activity_vectors.shape)
print('Targets shape:', targets.shape)
#profile = ProfileReport(dataset_df, title="Filtered Profiling Report")
#profile.to_file("filtered_report.html")

targets_binary_1 = (np.sign(targets)==1).astype(int)[None]


states = state_space_transform(activity_vectors, targets=targets_binary_1, mode=config.state_transform_mode)
print('States shape, Gain:', states.shape) # last is best, first is best for Gini
print('\nGain and Gini can be pruned at will. The criterion is given now.')
keep_best = config.keep_best
if config.state_transform_mode == 'Gain':
    states = states [:,-(keep_best+1):]
elif config.state_transform_mode == 'Gini':
    states = np.concatenate([states[:,:keep_best], states[:, -1, None]], axis=1)
else:
    pass
print('Pruned state space shape:', states.shape)

#states = state_space_transform(activity_vectors)
out_labels = hash_states(states)
dataset_df['out_labels'] = out_labels
helper_states = lookup_states(np.arange(0, out_labels.max()+1), dataset_df['out_labels'].values, states)
num_unique_states = out_labels.max()+1
print('Number of unique states:', num_unique_states)

hash_by_day = [dataset_df['out_labels'].loc[inds].values for inds in by_day_filtered]
print('by_day_filtered example:', by_day_filtered[18])
print('hash_by_day example:    ', hash_by_day[18])

ind_set = np.arange(len(hash_by_day)).astype(int)

discharging_transform_mode = 'Sqrt'

train_prediction_losses = []
valid_prediction_losses = []

train_regression_losses = []

valid_regression_losses_charging = []
valid_regression_losses_discharging = []

train_regression_losses_charging = []
train_regression_losses_discharging = []


prediction_criteria = [BCE, bit_flips]
regression_criteria = [L1]

prediction_frames = []
regression_frames = [] #'L1', 'Inds', 'Battery', 'Set'

for ind in ind_set:
    print(ind)
    # set up the cross-validation split
    train_inds = np.delete(ind_set, ind)
    valid_inds = [ind]
    #convert train and valid sets for use in prediction
    train_set_prediction = [hash_by_day[ind] for ind in train_inds]
    valid_set_prediction = [hash_by_day[ind] for ind in valid_inds]
    
    
    P = fit_predictor(train_set_prediction, num_unique_states, mode=config.predictor_mode)
    
    N_steps = 1
    estimate = predict(valid_set_prediction, P, N_steps, num_unique_states)
    # convert to sets of individual distributions over activity components
    prediction = prediction_output_transform(estimate, dataset_df['out_labels'].values, states, mode=config.prediction_transform)
    # simple targets - just time-shifted the input by N_steps
    pred_targets = list_to_prediction_targets(valid_set_prediction, N_steps, dataset_df['out_labels'].values, states)
    # calculate loss
    loss = BCE(prediction, pred_targets) #percentage bitflips as alternatives
    loss_bit = bit_flips(prediction, pred_targets) #percentage bitflips as alternatives
    valid_prediction_loss = [crit(prediction, pred_targets) for crit in prediction_criteria]
    valid_prediction_losses.append(valid_prediction_loss)
    # print reduced loss
    print('Valid prediction loss shape:', valid_prediction_loss[0].shape, valid_prediction_loss[1].shape)
    print('Prediction loss over the data for', N_steps, 'time step(s): mean:', loss.mean(), 'std:', loss.std())
    print('Prediction bit flips over the data for', N_steps, 'time step(s): mean:', loss_bit.mean(), 'std:', loss_bit.std())
    
    estimate_train = predict(train_set_prediction, P, N_steps, num_unique_states)
    # convert to sets of individual distributions over activity components
    prediction_train = prediction_output_transform(estimate_train, dataset_df['out_labels'].values, states, mode=config.prediction_transform)
    # simple targets - just time-shifted the input by N_steps
    pred_targets_train = list_to_prediction_targets(train_set_prediction, N_steps, dataset_df['out_labels'].values, states)
    # calculate loss
    train_prediction_loss = [crit(prediction_train, pred_targets_train) for crit in prediction_criteria]
    train_prediction_losses.append(train_prediction_loss)
    

    prediction_loss_data = pd.DataFrame()
    days_training = np.concatenate([np.ones(hash_by_day[idx].shape[0]-1)*idx for idx in train_inds])
    days_validation = np.concatenate([np.ones(hash_by_day[idx].shape[0]-1)*idx for idx in valid_inds])
    set_train = ['Training' for item in days_training]
    set_valid = ['Validation' for item in days_validation]
    sets = set_train + set_valid
    days = np.concatenate([days_training, days_validation])
    bces = np.concatenate([train_prediction_loss[0], valid_prediction_loss[0]])
    flips = np.concatenate([train_prediction_loss[1], valid_prediction_loss[1]])
    prediction_loss_data['Day'] = days
    prediction_loss_data['Set'] = sets
    prediction_loss_data['BCE'] = bces
    prediction_loss_data['Flips'] = flips
    prediction_loss_data['Inds'] = np.ones_like(days)*ind
    prediction_frames.append(prediction_loss_data)
    
    
    #convert train and valid sets for use in regression
    train_ind_prediction = np.concatenate([by_day_filtered[ind] for ind in train_inds])
    valid_ind_prediction = np.concatenate([by_day_filtered[ind] for ind in valid_inds])
    train_states = states[train_ind_prediction]
    train_targets = targets[train_ind_prediction]
    valid_states = states[valid_ind_prediction]
    valid_targets = targets[valid_ind_prediction]
    # split up train set into subsets depending on whether the phone is charging or not
    # do the same for the targets
    train_charging_mask = train_states[:,-1] > 0.5
    train_discharging_mask = np.invert(train_charging_mask)
    train_states_charging = train_states[train_charging_mask][:,:-1]
    train_targets_charging = train_targets[train_charging_mask]
    train_states_discharging = train_states[train_discharging_mask][:,:-1]
    train_targets_discharging = train_targets[train_discharging_mask]
    train_targets_charging_transformed = target_space_transform(train_targets_charging, mode=config.charge_transform_mode)
    train_targets_discharging_transformed = target_space_transform(train_targets_discharging, mode=config.discharge_transform_mode)
    print('Number of samples in the charging training set:', train_targets_charging.shape[0])
    print('Number of samples in the discharging training set:', train_targets_discharging.shape[0])
    # fit two distinct regressors depending on the charging state
    w_charging = fit_regressor(train_states_charging, train_targets_charging_transformed, mode=config.regressor_mode)
    w_discharging = fit_regressor(train_states_discharging, train_targets_discharging_transformed, mode=config.regressor_mode)
    
    # calculate the loss on the training set
    train_state_mat_charging = build_state_mat(train_states_charging)
    train_state_mat_discharging = build_state_mat(train_states_discharging)
    
    train_charging_out_transformed = train_state_mat_charging.T @ w_charging
    train_discharging_out_transformed = train_state_mat_discharging.T @ w_discharging
    # undo whatever target space transformation has happened
    train_charging_out = target_space_transform(train_charging_out_transformed, mode=config.charge_transform_mode, direction='Backward')
    train_discharging_out = target_space_transform(train_discharging_out_transformed, mode=config.discharge_transform_mode, direction='Backward')
    # store the validation losses for each regressor
    train_regression_loss_charging = [crit(train_charging_out, train_targets_charging) for crit in regression_criteria]
    train_regression_losses_charging.append(train_regression_loss_charging)
    train_regression_loss_discharging = [crit(train_discharging_out, train_targets_discharging) for crit in regression_criteria]
    train_regression_losses_discharging.append(train_regression_loss_discharging)
    
    
    # now split the valid set into subset depending on the charging state
    valid_charging_mask = valid_states[:,-1] > 0.5
    valid_discharging_mask = np.invert(valid_charging_mask)
    valid_states_charging = valid_states[valid_charging_mask][:,:-1]
    valid_states_discharging = valid_states[valid_discharging_mask][:,:-1]
    valid_targets_charging = valid_targets[valid_charging_mask]
    valid_targets_discharging = valid_targets[valid_discharging_mask]
    valid_state_mat_charging = build_state_mat(valid_states_charging)
    valid_state_mat_discharging = build_state_mat(valid_states_discharging)
    # apply the corresponding regressor result to each
    valid_charging_out_transformed = valid_state_mat_charging.T @ w_charging
    valid_discharging_out_transformed = valid_state_mat_discharging.T @ w_discharging
    # undo whatever target space transformation has happened
    valid_charging_out = target_space_transform(valid_charging_out_transformed, mode=config.charge_transform_mode, direction='Backward')
    valid_discharging_out = target_space_transform(valid_discharging_out_transformed, mode=config.discharge_transform_mode, direction='Backward')
    loss_regressor_charging = np.abs(valid_charging_out - valid_targets_charging)
    loss_regressor_discharging = np.abs(valid_discharging_out - valid_targets_discharging)
    # store the validation losses for each regressor
    valid_regression_loss_charging = [crit(valid_charging_out, valid_targets_charging) for crit in regression_criteria]
    valid_regression_losses_charging.append(valid_regression_loss_charging)
    valid_regression_loss_discharging = [crit(valid_discharging_out, valid_targets_discharging) for crit in regression_criteria]
    valid_regression_losses_discharging.append(valid_regression_loss_discharging)
    
    print('Charging loss shape:', loss_regressor_charging.shape)
    print('Discharging loss shape:', loss_regressor_discharging.shape)
    print('Number of samples in the charging valid set:', valid_targets_charging.shape[0])
    print('Number of samples in the discharging valid set:', valid_targets_discharging.shape[0])
    print('Regressor L1 over the charging data: mean:', loss_regressor_charging.mean(), 'std:', loss_regressor_charging.std())
    print('Regressor L1 over the discharging data: mean:', loss_regressor_discharging.mean(), 'std:', loss_regressor_discharging.std())
    loss_regressor = np.concatenate([loss_regressor_charging, loss_regressor_discharging])
    
    print('Regressor L1 over the data: mean:', loss_regressor.mean(), 'std:', loss_regressor.std())
    print()
    
    regression_loss_data = pd.DataFrame()
    days_training = np.concatenate([np.ones(hash_by_day[idx].shape[0])*idx for idx in train_inds])
    days_validation = np.concatenate([np.ones(hash_by_day[idx].shape[0])*idx for idx in valid_inds])
    set_train = ['Training' for item in days_training]
    set_valid = ['Validation' for item in days_validation]
    sets = set_train + set_valid
    charge_training = ['Charging' for item in train_targets_charging]
    discharge_training = ['Discharging' for item in train_targets_discharging]
    
    charge_valid = ['Charging' for item in valid_targets_charging]
    discharge_valid = ['Discharging' for item in valid_targets_discharging]
    
    charge_states = charge_training +  discharge_training + charge_valid + discharge_valid
    L1 = np.concatenate([train_regression_loss_charging[0], train_regression_loss_discharging[0],\
                        valid_regression_loss_charging[0], valid_regression_loss_discharging[0]])
    
    print(len(sets), L1.shape[0], len(charge_states))
    regression_loss_data['Set'] = sets
    regression_loss_data['L1'] = L1
    regression_loss_data['Battery'] = charge_states
    regression_loss_data['Inds'] = np.ones_like(L1)*ind
    regression_frames.append(regression_loss_data)
    

if config.eval_predictor:

    mega_frame = pd.concat(prediction_frames)

    fig, ax = plt.subplots(figsize=(40,10))
    ax = sns.boxplot(x='Set', y='Flips', data=mega_frame, hue='Inds')
    plt.legend([],[], frameon=False)
    plt.close()

    wandb.log({"Predictor Bit Flip by split": wandb.Image(fig)})

    fig, ax = plt.subplots(figsize=(10,10))
    ax = sns.boxplot(x='Set', y='Flips', data=mega_frame)
    plt.legend([],[], frameon=False)
    plt.close()

    wandb.log({"Predictor Bit Flip overview": wandb.Image(fig)})

    fig, ax = plt.subplots(figsize=(40,10))
    ax = sns.boxplot(x='Set', y='BCE', data=mega_frame, hue='Inds')
    plt.legend([],[], frameon=False)
    plt.close()

    wandb.log({"Predictor BCE by split": wandb.Image(fig)})

    fig, ax = plt.subplots(figsize=(10,10))
    ax = sns.boxplot(x='Set', y='BCE', data=mega_frame)
    plt.legend([],[], frameon=False)
    plt.close()

    wandb.log({"Predictor BCE overview": wandb.Image(fig)})

    train_avg_bce = mega_frame[mega_frame['Set'] == 'Training']['BCE'].dropna().values.mean()
    valid_avg_bce = mega_frame[mega_frame['Set'] == 'Validation']['BCE'].dropna().values.mean()

    train_avg_flips = mega_frame[mega_frame['Set'] == 'Training']['Flips'].dropna().values.mean()
    valid_avg_flips = mega_frame[mega_frame['Set'] == 'Validation']['Flips'].dropna().values.mean()
    print('Average training BCE:', train_avg_bce)
    print('Average validation BCE:', valid_avg_bce)
    print('Average training Bit Flips:', train_avg_flips)
    print('Average validation Bit Flips:', valid_avg_flips)

    wandb.log({'Training BCE': train_avg_bce, 'Validation BCE': valid_avg_bce, 'Training Bit Flips': train_avg_flips, 'Validation Bit Flips': valid_avg_flips})

if config.eval_regressor:
    mega_frame_regression = pd.concat(regression_frames)

    fig, ax = plt.subplots(figsize=(40,10))
    ax = sns.boxplot(x='Set', y='L1', data=mega_frame_regression, hue='Battery', showfliers=False)
    plt.legend([],[], frameon=False)
    plt.close()

    wandb.log({"Regressor by battery_plugged": wandb.Image(fig)})

    fig, ax = plt.subplots(figsize=(10,10))
    ax = sns.boxplot(x='Set', y='L1', data=mega_frame_regression, showfliers=False)
    plt.legend([],[], frameon=False)
    plt.close()

    wandb.log({"Regressor Overview": wandb.Image(fig)})
    
    train_charge_mask = (mega_frame_regression['Set'] == 'Training') & (mega_frame_regression['Battery'] == 'Charging')
    L1_training_charging = mega_frame_regression[train_charge_mask]['L1'].dropna().values.mean()
    valid_charge_mask = (mega_frame_regression['Set'] == 'Validation') & (mega_frame_regression['Battery'] == 'Charging')
    L1_validation_charging = mega_frame_regression[valid_charge_mask]['L1'].dropna().values.mean()
    print('L1 training charging:', L1_training_charging)
    print('L1 validation charging:', L1_validation_charging)

    train_discharge_mask = (mega_frame_regression['Set'] == 'Training') & (mega_frame_regression['Battery'] == 'Discharging')
    L1_training_discharging = mega_frame_regression[train_discharge_mask]['L1'].dropna().values.mean()
    valid_discharge_mask = (mega_frame_regression['Set'] == 'Validation') & (mega_frame_regression['Battery'] == 'Discharging')
    L1_validation_discharging = mega_frame_regression[valid_discharge_mask]['L1'].dropna().values.mean()
    print('L1 training discharging:', L1_training_discharging)
    print('L1 validation discharging:', L1_validation_discharging)


    train_mask = (mega_frame_regression['Set'] == 'Training')
    valid_mask = (mega_frame_regression['Set'] == 'Validation')
    L1_training = np.sort(mega_frame_regression[train_mask]['L1'].values)
    L1_validation = np.sort(mega_frame_regression[valid_mask]['L1'].values)
    L1_training = L1_training[:int(L1_training.shape[0]*0.999)].mean()
    L1_validation = L1_validation[:int(L1_validation.shape[0]*0.99)].mean()
    print('L1 training:', L1_training)
    print('L1 validation:', L1_validation)

    wandb.log({'Training L1': L1_training, 'Validation L1': L1_validation})
    wandb.log({'Training L1 Discharging': L1_training_discharging, 'Validation L1 Discharging': L1_validation_discharging})
    wandb.log({'Training L1 Charging': L1_training_charging, 'Validation L1 Charging': L1_validation_charging})

print('Notebook ran in', time.time()-start, 'seconds.')
# 19s for MLE prediction
# 806s for MAP prediction
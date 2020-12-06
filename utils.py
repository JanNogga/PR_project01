import os
import pandas as pd
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


if __name__ == '__main__':
    dataset_df = get_table()
    display(dataset_df)

    # get the name of the packages that is running in the system
    text = 'packages_running_'
    running_packages = [i for i in dataset_df.columns if text in i]

    # creating the activity vector containing info about running apps and battery_plugged
    extra_columns = ['battery_plugged']
    activity_vectors_df = dataset_df[[*extra_columns, *running_packages]]
    activity_vectors = activity_vectors_df.to_numpy()

    # battery usage for each activity vector
    battery_usage = dataset_df['battery_level'].to_list()
    print(battery_usage)



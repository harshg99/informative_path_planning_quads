import pandas
import argparse
import typing
from typing import List,Tuple
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-folder_path",       type=str,           default="Plots")
    parser.add_argument("-plot_path",        type=str,           default="Rewards-A3CvsSocialLight-ManhattanSUMO")
    return parser.parse_args()

colors_list = ['tab:blue','tab:red','tab:green','tab:purple']
FACTOR = 0.5
def read_csv(folder_name):
    '''
    :param file_name: name of the file
    :return: list
    '''

    # Obtain CSV files
    csv_files = []
    if os.path.exists(folder_name):
        csv_files = os.listdir(folder_name)

    if csv_files is None:
        raise Exception('No CSV files')

    # Empty DF list


    df_dict = dict()
    for csv in csv_files:
        csv_path = folder_name + '/' + csv
        key = csv.split('_')[-1]
        if csv[-4:] =='.csv':
            df_dict[key]=pd.read_csv(csv_path)




    return df_dict

def plot_time_series(time:dict, std_series:dict,series:dict,args,metric:str):
    '''

    :param time: Time series dictionary
    :param std_series: Rolling standard deviation
    :param min_series: Rolling mean of the series
    :return:
    '''
    plt.figure()

    legend = time.keys()
    idx = 0
    handles = []
    keys = []
    for model in legend:
        label = model.split('.')[0]
        keys.append(label)
        line = plt.scatter(time[model], series[model], label=label, linewidth=1,
                 alpha=1.0,color=colors_list[idx])
        plt.fill_between(time[model], series[model] -  FACTOR*std_series[model], series[model] +  FACTOR*std_series[model]
                         , alpha=0.1,color=colors_list[idx])
        idx += 1
        # handles.append(line[0])



    split_path = args.plot_path.split('-')
    plt.title('Average ' + split_path[0], fontsize=15)
    plt.xlabel(f'Env Iterations', fontsize=15)
    plt.ylabel('Average ' + metric, fontsize=15)
    # if split_path[0].contains('speed'):
    #     plt.ylabel(split_path[0] +'m/s', fontsize=15)
    # elif split_path[0].contains('time'):
    #     plt.ylabel(split_path[0] +'time', fontsize=15)
    # else:
    #     plt.ylabel(split_path[0], fontsize=15)
    plt.legend(handles = handles)
    plt.savefig('./'+ args.plot_path+'/plots.jpg')
    plt.show()



def get_time_series_data(df_dict: List[pd.DataFrame],time_key = 'Iteration',
                         metric = 'target_found',num_categories = 10) -> Tuple[np.array,np.array]:
    '''

    :param df_list: List of dataframes from which we extract the time series data
    :return: Returns a single dataframe consisting of all time series data with moving averages
    '''

    # Run through all dataframes
    series_dict = dict()
    #max_series_dict = dict()
    std_series_dict = dict()
    time_series = dict()
    # plt.figure()
    handles =[]
    for key in df_dict.keys():
        max_time_key= df_dict[key][time_key].max()
        min_time_key = df_dict[key][time_key].min()
        interval_idx = pd.interval_range( start = min_time_key, end = max_time_key, periods = num_categories)
        series_sort, series_bins = pd.cut(np.array( df_dict[key][time_key].to_list()), \
                                          bins=interval_idx, retbins=True)

        mean_series = np.array([df_dict[key][metric][series_sort.codes == i].mean() for i in range(num_categories)])
        time_series[key] = np.array([interval_idx[i].mid for i in range(num_categories)])
        std_series =  np.array([df_dict[key][metric][series_sort.codes == i].std() for i in range(num_categories)])

        std_series[np.argwhere(np.isnan(std_series))] = std_series[np.argwhere(std_series ==np.nanmax(std_series))[0]]

        series_dict[key] = mean_series
        std_series_dict[key] = std_series
        #handles.append(plt.scatter(df_dict[key][time_key], df_dict[key][metric], label=key, linewidth=1))
        #min_series_dict.append(min_series)
    # Ceate a  new dataframe with the rolling window sums
    # plt.legend()
    # plt.show()
    return time_series,series_dict,std_series_dict,metric

if __name__ == "__main__":

    args = parse_args()

    folder_name = args.folder_path + "/" + args.plot_path
    df_list = read_csv(folder_name)
    time_series, series_dict, std_series_dict,metric = get_time_series_data(df_list)
    plot_time_series(time_series, std_series_dict, series_dict,args,metric)


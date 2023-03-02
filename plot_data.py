import numpy as np
import pandas as pd

from plot_robustness import read_csv, plot_time_series, get_time_series_data

# pc_data = 'compiled_results/plots/all_results_pc.csv'
# ourslstm_data = 'compiled_results/plots/all_results_ourslstm.csv'
# oursnolstm_data = 'compiled_results/plots/all_results_ours.csv'
# dpipp_data = 'compiled_results/plots/all_results_dpipp.csv'
#
# compiled_data =   'compiled_results/plots/results/results{}.csv'
# pc_df = pd.read_csv(pc_data).transpose().to_dict(orient='list')
# ourslstm_df = pd.read_csv(ourslstm_data).transpose().to_dict(orient='list')
# oursnlstm_df = pd.read_csv(oursnolstm_data).transpose().to_dict(orient='list')
# dpipp_df = pd.read_csv(dpipp_data).transpose().to_dict(orient='list')
#

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-folder_path",       type=str,           default="compiled_results/plots/")
    parser.add_argument("-plot_path",        type=str,           default="compiled_results/plots/final_plots")
    return parser.parse_args()

args = parse_args()
directory = args.folder_path
df_list = read_csv(directory)

# Sort the series by KL divergences

data_frame_dict = {}

for series_id in df_list:
    df_list[series_id] = df_list[series_id].transpose()
    series_ = df_list[series_id].to_dict(orient='list')
    dict_series = {}
    for k in series_:
        key = series_[k][0]
        dict_series[key] = list(series_[k][1:])

    #keys = df_list[series_id][:][0]
    df_list[series_id] = pd.DataFrame.from_dict(dict_series)
    df_list[series_id] = df_list[series_id].sort_values(by='divergence')

time_series, series_dict, std_series_dict, metric = get_time_series_data(df_list,time_key='divergence',
                                                                         metric = 'targets_found',num_categories=5)
plot_time_series(time_series, std_series_dict, series_dict, args, metric)

# num_categories =5
# interval_idx = pd.interval_range(start=0, end=1.5,periods=num_categories)
# kldivsort, kldivbins= pd.cut(np.array(pc_df[5][1:]),bins=interval_idx,retbins=True)
# kldivbinscodes = kldivsort.codes
#
# categories = {}
# avg_categories = {}
# std_categories = {}
# for j in range(num_categories):
#     dpipp_searcheff = np.expand_dims(np.array(dpipp_df[2][1:])[kldivbinscodes == j], axis=0)
#     pc_searcheff = np.expand_dims(np.array(pc_df[2][1:])[kldivbinscodes==j],axis=0)
#     ours_searcheff = np.expand_dims(np.array(oursnlstm_df[2][1:])[kldivbinscodes==j],axis=0)
#     ourslstm_searcheff = np.expand_dims(np.array(ourslstm_df[2][1:])[kldivbinscodes == j], axis=0)
#     target_search_list = np.concatenate((pc_searcheff,ours_searcheff,ourslstm_searcheff,dpipp_searcheff),axis=0)
#
#     categories[j] = target_search_list
#     avg_categories[j] = np.mean(target_search_list,axis=1)
#     std_categories[j] = np.std(target_search_list, axis=1)
#
#     target_search_dict = {'PC':pc_searcheff[0],'Ours-noLSTM':ours_searcheff[0],'Ours-LSTM':ourslstm_searcheff[0],'DPIPP':dpipp_searcheff[0]}
#     pd.DataFrame(target_search_dict).to_csv(compiled_data.format(j))

    


print('Done')

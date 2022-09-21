import numpy as np
import pandas as pd


pc_data = '../compiled_results/plots/all_results_pc.csv'
ourslstm_data = '../compiled_results/plots/all_results_ourslstm.csv'
oursnolstm_data = '../compiled_results/plots/all_results_ours.csv'
dpipp_data = '../compiled_results/plots/all_results_dpipp.csv'

compiled_data =   '../compiled_results/plots/results{}.csv'
pc_df = pd.read_csv(pc_data).transpose().to_dict(orient='list')
ourslstm_df = pd.read_csv(ourslstm_data).transpose().to_dict(orient='list')
oursnlstm_df = pd.read_csv(oursnolstm_data).transpose().to_dict(orient='list')
dpipp_df = pd.read_csv(dpipp_data).transpose().to_dict(orient='list')


num_categories =5
interval_idx = pd.interval_range(start=0, end=1.5,periods=num_categories)
kldivsort, kldivbins= pd.cut(np.array(pc_df[5][1:]),bins=interval_idx,retbins=True)
kldivbinscodes = kldivsort.codes

categories = {}
avg_categories = {}
std_categories = {}
for j in range(num_categories):
    dpipp_searcheff = np.expand_dims(np.array(dpipp_df[2][1:])[kldivbinscodes == j], axis=0)
    pc_searcheff = np.expand_dims(np.array(pc_df[2][1:])[kldivbinscodes==j],axis=0)
    ours_searcheff = np.expand_dims(np.array(oursnlstm_df[2][1:])[kldivbinscodes==j],axis=0)
    ourslstm_searcheff = np.expand_dims(np.array(ourslstm_df[2][1:])[kldivbinscodes == j], axis=0)
    target_search_list = np.concatenate((pc_searcheff,ours_searcheff,ourslstm_searcheff,dpipp_searcheff),axis=0)

    categories[j] = target_search_list
    avg_categories[j] = np.mean(target_search_list,axis=1)
    std_categories[j] = np.std(target_search_list, axis=1)

    target_search_dict = {'PC':pc_searcheff[0],'Ours-noLSTM':ours_searcheff[0],'Ours-LSTM':ourslstm_searcheff[0],'DPIPP':dpipp_searcheff[0]}
    pd.DataFrame(target_search_dict).to_csv(compiled_data.format(j))

    


print('Done')

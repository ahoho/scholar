import pandas as pd
import json

full_raw_data_2005_2012 = pd.read_csv('/fs/clip-political/pranav/voting_prediction_party_matters/PartyMatters/code/data/full_data_with_dwnom.csv')

full_raw_data_2011_2012 = full_raw_data_2005_2012[full_raw_data_2005_2012['natural_id'].str.contains('20112012')]
full_raw_data_2005_2010 = full_raw_data_2005_2012[full_raw_data_2005_2012['natural_id'].str.contains('20112012')==False]

def convert_df_to_dic(df):
    
    majority_cosponsor_dic = {'d': list(df['d_perc']), 'r': list(df['r_perc'])}
    dw_dic = {'dw1': list(df['dw1']), 'dw2': list(df['dw2'])}
    majority_cosponsor = []
    dw1, dw2 = [], []
    for i in range(len(df)):
        if majority_cosponsor_dic['d'][i] > majority_cosponsor_dic['r'][i]:
            majority_cosponsor.append('d')
        else:
            majority_cosponsor.append('r')
            
        dw1_score = dw_dic['dw1'][i]
        if dw1_score <= -0.405:
            dw1.append('fl')
        elif dw1_score > -0.405 and dw1_score <= -0.283:
            dw1.append('cl')
        elif dw1_score > -0.283 and dw1_score <= 0.269:
            dw1.append('c')
        elif dw1_score > 0.269 and dw1_score <= 0.445:
            dw1.append('cr')
        else:
            dw1.append('fr')
        
        dw2_score = dw_dic['dw2'][i]
        if dw2_score <= -0.218:
            dw2.append('-2')
        elif dw2_score > -0.218 and dw1_score <= -0.054:
            dw2.append('-1')
        elif dw2_score > -0.054 and dw1_score <= 0.07:
            dw2.append('0')
        elif dw2_score > 0.07 and dw1_score <= 0.253:
            dw2.append('1')
        else:
            dw2.append('2')
            
    df['majority_cosponsor'] = majority_cosponsor
    df['dw1'] = dw1
    df['dw2'] = dw2
    
    df = df.reset_index()
    df = df.rename({'index':'id', 'summary':'text'}, axis=1)
    di = {1.0:'yay', 0.0:'nay'}
    df = df.replace({"vote": di})
    return df.to_dict(orient='records')

train_data = convert_df_to_dic(full_raw_data_2005_2010)
print('Train Dic Created')
test_data = convert_df_to_dic(full_raw_data_2011_2012)
print('Test Dic Created')

with open('data/congress_voting_dwnom/train.jsonlist', 'w', encoding='utf-8') as f:
    for line in train_data:
        json.dump(line, f)
        f.write('\n')
print('Train Saved')
with open('data/congress_voting_dwnom/test.jsonlist', 'w', encoding='utf-8') as f:
    for line in test_data:
        json.dump(line, f)
        f.write('\n')
print('Test Saved')


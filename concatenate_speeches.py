import os
import json

from tqdm import tqdm
import pandas as pd

def concat_text_and_save_as_jsonlist(df, outfile, text_cols):
    """
    Save the dataframe as a jsonlist.

    It ends up being expensive to concatenate the dataframe columns, so we do it
    line by line as we save
    """
    with open(outfile, 'w', encoding='utf-8') as outfile:
        for _, line in tqdm(df.iterrows()):
            line['text'] = ' '.join(line[text_cols])
            line.drop(text_cols).to_json(outfile, orient='index', force_ascii=False)
            outfile.write('\n')

if __name__ == "__main__":
    # load in party matters data
    party_matters = pd.read_csv(
        'data/party_matters_200512_expanded.csv', index_col=0, dtype={'govtrack_id': str}
    )
    party_matters['congress_id'] = party_matters.natural_id.str.extract('(^[0-9]+)')

    # load in speeches data
    with open("data/speeches/debate_speeches_for_legislators_per_congress.json", "r") as infile:
        speeches = json.load(infile)
    congress_year_dict = {
        '109': '20052006','110': '20072008', '111': '20092010', '112': '20112012'
    }

    speeches_data = pd.DataFrame()
    for congress in speeches:
        for legislator_id in tqdm(speeches[congress]):
            speeches_data = speeches_data.append(pd.DataFrame({
                'congress_num': congress,
                'congress_id': congress_year_dict[congress],
                'govtrack_id': legislator_id,
                'speech': speeches[congress][legislator_id],
            }), ignore_index=True)
    del speeches

    # merge speeches with party matters data: for now, we group all legislator's
    # speech in each session
    speeches_data = (
        speeches_data.groupby(['congress_id', 'govtrack_id'])['speech']
                     .apply(' '.join)
                     .reset_index()
    )

    party_matters = party_matters.merge(speeches_data).reset_index()

    # process data
    party_matters.reset_index()
    party_matters = party_matters.rename({'index': 'id'}, axis=1)
    party_matters = party_matters.replace({"vote": {1.0: 'yay', 0.0: 'nay'}})
    # TODO: add majority cosponsor
    # nb: conatenating votes and speech all at once is expensive, we do in a loop below

    # filter and save
    if not os.path.exists('data/congress_votes_speech'):
        os.makedirs('data/congress_votes_speech')

    # training
    concat_text_and_save_as_jsonlist(
        "data/congress_votes_speech/train-bill_plus_all_speech-109_111.jsonlist",
        df=party_matters.loc[party_matters.congress_num.isin(['109', '110', '111'])],
        text_cols=['summary', 'speech'],
    )

    



import csv
import os
import ast

import pandas as pd
import numpy as np

from glob import glob

path_to_alt_gt_results = 'fl/results/results_alternative_gt/'
path_to_orig_results = 'fl/results/all_results_before_alt_gt/'
path_to_oring_alt_gt_results = 'fl/results/all_results_after_alt_gt'

#######################################################################################################################
# Results alt gt collect together
#######################################################################################################################
N = 10
prefix = ''# RF AF AF_CF#_res_ajs_0.csv
suffix = '_res_ajs_'
format = '.csv'
files_path = path_to_alt_gt_results
out_csv_file = os.path.join(files_path, prefix + '_res_ALL.csv')

rows = [['id', 'issue', 'GPT_AJS_OLD', 'GPT_AJS_NEW']]

# files = glob(os.path.join(files_path, prefix, recursive=True)

for i in range(N):
    file_name = os.path.join(files_path, prefix + suffix + str(i) + format)

    with open(file_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if row[0] == 'issue':
                continue
            rows.append([i, row[0], row[1], row[2]])


with open(out_csv_file, 'w') as file:
    writer = csv.writer(file, delimiter=',', lineterminator='\n', )
    writer.writerows(rows)

#######################################################################################################################
# All results collect together
#######################################################################################################################
N = 10

rf_map = {
    '48594888':'D5',
    '50306988':'D6',
    '48385830':'D4',
    '45442843':'D3',
    '56380303':'D8',
    '51181393':'D7',
    '31880720':'D1'
}

prefixes = ['RF', 'AF', 'AF_CF']#_res_ajs_0.csv
suffix = '_res_'
format = '.csv'
files_path = path_to_alt_gt_results
out_csv_file = os.path.join(files_path, '..', 'orig_altgt_full.csv')

rows = [['iter', 'ID', 'ID-SO', 'GT_ID', 'GT ', '#F', '#RES', '#M', 'RC', 'PR', 'F3', 'OUT']]

# files = glob(os.path.join(files_path, prefix, recursive=True)

orig_files = [
    path_to_orig_results+'AF_ALL.csv',
    path_to_orig_results+'DFD_ALL.csv'
    ]

for orig_file in orig_files:
    with open(orig_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            print(row[0])
            if 'iter' in row[0]:
                continue
            if 'AF' in orig_file:
                rows.append([row[0], row[1], '', 0, row[2], row[3], row[4], row[5], round(float(row[6]),2), round(float(row[7]),2), round(float(row[8]),2), row[9]])
            else:
                rows.append([row[0], row[2], row[1], 0, row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10]])

for prefix in prefixes:
    for i in range(N):
        file_name = os.path.join(files_path, prefix + suffix + str(i) + format)

        with open(file_name) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if row[0] == 'issue':
                    continue
                x = row[4]
                x = ast.literal_eval(x)
                x = [n.strip() for n in x]
                rc = float(row[7])
                if len(x) == 0:
                    pr = 0
                else:
                    pr = round(int(row[6])/len(x),2)
                if rc == 0 and pr == 0:
                    f3 = 0
                else:
                    f3 = (1+3*3)*(rc*pr)/((3*3*pr)+rc)
                if 'AF' in prefix:
                    rows.append([i, row[0], '', row[1], row[2], row[3], len(x), row[6], rc, round(pr,2), round(f3,2), row[4]])
                else:
                    rows.append([i, rf_map[row[0]], row[0], row[1], row[2], row[3], len(x), row[6], rc, round(pr,2), round(f3,2), row[4]])

# print(rows)
with open(out_csv_file, 'w') as file:
    writer = csv.writer(file, delimiter=',', lineterminator='\n', )
    writer.writerows(rows)


#######################################################################################################################
# Generate Summary of results ORIG + ALT
#######################################################################################################################

all_res_file = path_to_oring_alt_gt_results+'orig_altgt_full.csv'
out_res_file = path_to_oring_alt_gt_results+'SUM_orig_altgt_full.csv'

df = pd.read_csv(all_res_file)

idx = df['RC'] == df.groupby(['iter', 'ID'])['RC'].transform(max)

# Filter the DataFrame to keep only those rows
df = df[idx]

# Aggregate to calculate the average of other metrics for rows with the max RC
df_new = df.groupby(['iter', 'ID']).agg({
    'RC': 'first',  # Keep the maximum RC value (since all are the same)
    'PR': 'mean',   # Calculate the average of PR
    'F3': 'mean'    # Calculate the average of F3
}).reset_index()

df_new.to_csv(out_res_file)

#######################################################################################################################
# Generate Summary of results ORIG ONLY
#######################################################################################################################
#
all_res_file = path_to_oring_alt_gt_results+'orig_altgt_full.csv'
out_res_file = path_to_oring_alt_gt_results+'SUM_orig.csv'

df = pd.read_csv(all_res_file)

# Filter out rows where GT_ID is not equal to 0
filtered_df = df[df['GT_ID'] == 0]

# Calculate the average for RC, PR, and F3 for each ID
result_df = filtered_df.groupby('ID').agg({
    '#F': 'mean',
    '#RES': 'mean',
    '#M': 'mean',
    'RC': 'mean',
    'PR': 'mean',
    'F3': 'mean'
}).reset_index()

result_df.to_csv(out_res_file)

#######################################################################################################################
# Generate STD of results
#######################################################################################################################

res_file = path_to_oring_alt_gt_results+'SUM_orig_altgt_full.csv'
out_file = path_to_oring_alt_gt_results+'STD_orig_altgt_full.csv'

df = pd.read_csv(res_file)

new_df = df.groupby('ID', as_index=False).agg({
                        'RC': np.std,
                        'PR': np.std,
                        'F3': np.std})

new_df.to_csv(out_file)


#######################################################################################################################
# Generate MEAN of results
#######################################################################################################################

res_file = path_to_oring_alt_gt_results+'SUM_orig_altgt_full.csv'
out_file = path_to_oring_alt_gt_results+'MEAN_orig_altgt_full.csv'

df = pd.read_csv(res_file)

new_df = df.groupby('ID', as_index=False).agg({
                        'RC': np.mean,
                        'PR': np.mean,
                        'F3': np.mean})

new_df.to_csv(out_file)

#######################################################################################################################
# Generate OLD NEW of results
#######################################################################################################################

all_res_file = path_to_oring_alt_gt_results+'orig_altgt_full.csv'
out_res_merge_file = path_to_oring_alt_gt_results+'MERGE_orig_altgt_full.csv'

df = pd.read_csv(all_res_file)

df_orig = df.loc[df['GT_ID'] == 0]
df_orig_sel = df_orig[['iter', 'ID', 'RC', 'PR', 'F3']]
df_orig_sel['source'] = 'orig'


df_alt = df.loc[df['GT_ID'] != 0]
idx = df_alt.groupby(['iter', 'ID'])['RC'].transform(max) == df_alt['RC']
print(idx)
df_alt = df_alt.loc[idx]
df_alt_agg = df_alt.groupby(['iter', 'ID', 'RC'], as_index=False).agg({
                         'PR':'mean',
                         'F3':'mean'})
df_alt_agg['source'] = 'altgt'

print(len(df_orig_sel.index), len(df_alt_agg.index))
merged_df = pd.merge(df_orig_sel, df_alt_agg,  how='left', left_on=['iter', 'ID'], right_on = ['iter', 'ID'])

merged_df.to_csv(out_res_merge_file, index=False)


#######################################################################################################################
# Count num of answers
#######################################################################################################################


file_path = 'fl/GPT_OUTPUT/gpt_answers_processed.csv'
out_file = 'fl/GPT_OUTPUT/gpt_answers_cnt.csv'

df = pd.read_csv(file_path)

cnt_df = df.groupby(['iter', 'ID'], as_index=False).agg({
                        'pos': 'max'})

cnt_df.to_csv(out_file, index=False)


#######################################################################################################################
# Generate answers file
#######################################################################################################################

file_path = 'fl/GPT_OUTPUT/gpt_answers_processed1.csv'
out_file = 'fl/GPT_OUTPUT/gpt_answers_combined.csv'


df = pd.read_csv(file_path)
print(len(df.index))

df = df[['iter', 'ID', 'FinalFT']]

df.dropna(subset=['FinalFT'], inplace=True)
print(len(df.index))

df.drop_duplicates(inplace=True)
print(len(df.index))

output_df = df.groupby(['iter','ID'])['FinalFT'].apply('|'.join).reset_index()

output_df.to_csv(out_file, index=False)


# #######################################################################################################################
# # Generate stability file
# #######################################################################################################################
#
all_res_file = path_to_oring_alt_gt_results+'SUM_orig_altgt_full.csv'
out_res_file = path_to_oring_alt_gt_results+'SUM_orig_altgt_full_STABILITY.csv'

df = pd.read_csv(all_res_file)

# Aggregate to calculate the average of other metrics for rows with the max RC
df_new = df.groupby(['ID']).agg({
    'RC': 'std',  # Calculate the std of RC
    'PR': 'std',   # Calculate the std of PR
    'F3': 'std'    # Calculate the std of F3
}).reset_index()

df_new.to_csv(out_res_file)



# #######################################################################################################################
# # Generate orig gt result, avg across iterations or best across iterations
# #######################################################################################################################


df = pd.read_csv('fl/results/all_results_before_alt_gt/ALL_RES_NEW.csv')

grouped_df = df.groupby(['ID'])[['RC','PR','F3']].mean()
grouped_df = grouped_df.reset_index()[['ID','RC','PR','F3']]

grouped_df.to_csv(path_to_oring_alt_gt_results+'orig_gt_avg_ac_iter.csv', index=False)



df = pd.read_csv(path_to_oring_alt_gt_results+'SUM_orig_altgt_full.csv')

grouped_df = df.groupby(['ID'])[['RC','PR','F3']].mean()
grouped_df = grouped_df.reset_index()[['ID','RC','PR','F3']]

grouped_df.to_csv(path_to_oring_alt_gt_results+'orig_alt_gt_best_avg_ac_iter.csv', index=False)

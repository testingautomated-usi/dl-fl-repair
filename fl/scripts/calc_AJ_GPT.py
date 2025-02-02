import pandas as pd
import csv

from glob import glob

from FL.get_res import get_dfd_res


path_to_input_files = 'fl/results/files_for_alternative_gt/'
path_to_results = 'fl/results/results_alternative_gt/'
path_to_alt_gt = 'fl/results/alternative_GT'

N = 10


def calc_results(gt, len_gt, fl_res, flag):
    match = ''
    match_cnt = 0

    for gt_e in gt:
        if flag == 'DFD':
            gt_e = ''.join([i for i in gt_e if not i.isdigit()])

        if gt_e in fl_res:
            match = match + '1 | '
            match_cnt += 1
        else:
            match = match + '0 | '

    ajs = round(match_cnt / len_gt, 2)

    return match, match_cnt, ajs


def run(search_folder_name, gpt, gpt_ajs_old, res_file, res_ajs_file):
    search_params = "*.tsv"

    all_diff = {}
    res_fdf = {}
    res = []
    res_ajs = []

    files = glob(search_folder_name + search_params, recursive=True)

    for file in files:
        with open(file) as fd:
            # print(file)
            rd = csv.reader(fd, delimiter="\t", quotechar='"')
            # for row in rd:
            #     print(row)
            data_read = [row for row in rd if row[6].lower() == 'True'.lower()]

        if len(data_read) != 0:
            issue = file.replace(search_folder_name, '').split('_')[0]

            parent = data_read[0]
            pn_conf = parent[4].split(',')

            issue_diff = []
            for x in data_read:
                cn_conf = x[5].split(',')

                diff = []
                for i in range(len(pn_conf)):
                    if pn_conf[i]!=cn_conf[i]:
                        diff.append(pn_conf[i].split('=')[0])

                issue_diff.append([x[1],diff])
            all_diff[issue] = issue_diff
        else:
            print("Empty file", file)


    for k, v in all_diff.items():
        max_ajs_gpt = 0

        for gt in v:

            gpt_k = gpt[k]

            len_gt = len(gt[1])

            gpt_match, gpt_match_cnt, gpt_ajs = calc_results(gt[1], len_gt, gpt_k, 'GPT')

            if gpt_ajs > max_ajs_gpt: max_ajs_gpt = gpt_ajs

            res.append([k, gt[0], gt[1], len_gt, gpt_k, gpt_match, gpt_match_cnt, gpt_ajs])


        gpt_ajs_vals = gpt_ajs_old[k]
        gpt_ajs_vals.append(max_ajs_gpt)
        gpt_ajs_old[k] = gpt_ajs_vals

        res_ajs.append([k, gpt_ajs_old[k][0], gpt_ajs_old[k][1]])


    with open(res_file, "wt") as fp:
        writer = csv.writer(fp, delimiter=",")
        writer.writerow(["issue", "GT_ID", "GT", "LEN_GT", "GPT_RES", "GPT_MATCH", "GPT_MATCH_CNT", "GPT_AJS"])  # write header
        writer.writerows(res)

    with open(res_ajs_file, "wt") as fp:
        writer = csv.writer(fp, delimiter=",")
        writer.writerow(["issue", "GPT_AJS_OLD", "GPT_AJS_NEW"])  # write header
        writer.writerows(res_ajs)



##################################################################################################
#DFD issues

for i in range(N):
    print("_"*50)
    print(i)
    print("_"*50)
    gpt, gpt_ajs_old = get_dfd_res(path_to_input_files+"DFD_short_"+str(i)+".csv")
    print(gpt)

    res_file = path_to_results+"RF_res_"+str(i)+".csv"
    res_ajs_file = path_to_results+"RF_res_ajs_"+str(i)+".csv"

    search_folder_name = path_to_alt_gt+"DeepFD_GT_BFS/"

    run(search_folder_name, gpt, gpt_ajs_old, res_file, res_ajs_file)

##################################################################################################
# Mutants batchh 1

for i in range(N):
    print("_"*50)
    print(i)
    print("_"*50)
    gpt, gpt_ajs_old = get_dfd_res(path_to_input_files+"AF_short_"+str(i)+".csv")

    res_file = path_to_results+"results_alternative_gt/AF_res_"+str(i)+".csv"
    res_ajs_file = path_to_results+"results_alternative_gt/AF_res_ajs_"+str(i)+".csv"

    search_folder_name = path_to_alt_gt+"Mutants_GT_BFS_1/"

    run(search_folder_name, gpt, gpt_ajs_old, res_file, res_ajs_file)

##################################################################################################
# Mutants batchh 2

for i in range(N):
    print("_" * 50)
    print(i)
    print("_" * 50)
    gpt, gpt_ajs_old = get_dfd_res(path_to_input_files+"AF_CF_short_"+str(i)+".csv")

    res_file = path_to_results+"results_alternative_gt/AF_CF_res_"+str(i)+".csv"
    res_ajs_file = path_to_results+"results_alternative_gt/AF_CF_res_ajs_"+str(i)+".csv"

    search_folder_name = path_to_alt_gt+"Mutants_GT_BFS_2/"

    run(search_folder_name, gpt, gpt_ajs_old, res_file, res_ajs_file)

##################################################################################################

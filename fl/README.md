# Replication package of the paper "Fault Localisation and Repair for DL Systems: An Empirical Study with LLMs"

The replication code and data for repair experiments are located in the `repair` folder. The replication code and data for fault localisation experiments are located in the `fl` folder. Please consult the `README.md` file in each folder for more information.

Folder 'figures': Contains plots for FL part used in the paper

Folder 'GPT_OUTPUT':
- 'RAW': Contains raw output of GPT-4-turbo (from now on 'GPT') output
- 'gpt_answers_cnt': count of answer points for each output
- 'gpt_answers_processed': all GPT answers mapped to fault types
- 'gpt_answers_combined': combined GPT answers mapped to fault types for the issue + iteration level

Folder 'results':
- 'all_results_before_alt_gt':FL results before alternative GT
- 'alternative_GT': alternative GT
- 'files_for_alternative_gt': input files to calculate alternative GT results
- 'files_for_alternative_gt': alternative GT results
- 'all_results_after_alt_gt': FL results calculated when taking alternative GT into account

Folder 'scripts':
- 'calc_AJ_GPT': scripts to generate alternative GT results
- 'generate_files': scripts to generate files in results folder
- 'parse_answers': scripts to parse raw GPT outputs and generate 'gpt_answers_cnt', 'gpt_answers_processed', 'gpt_answers_combined' files
- 'results_plotting': script to plot figures in 'figures' folder
- 'run_prompt': script to trigger GPT API for FL

Folder 'subjects/fault_files': Input files (benchmark) for FL with GPT

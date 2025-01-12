import os, sys, json
from openai import OpenAI

API_KEY='YOUR_KEY'
ORGANIZATION='YOUR_ORG'

client = OpenAI(
    api_key=API_KEY,
    organization=ORGANIZATION
)

N_repeat = 10
pnum = sys.argv[1]
llm_model = sys.argv[2]
temperature = float(sys.argv[3])

output_dir = f"./prompt_outputs/output/{llm_model}/{pnum}/{temperature}"
os.makedirs(output_dir, exist_ok=True)

with open("./prompts/task_types.json", "r") as f:
    task_types = json.load(f)

if pnum not in task_types:
    print("Invalid pnum")
    exit()

task = task_types[pnum]["task"]
dataset = task_types[pnum]["dataset"]
goal = task_types[pnum]["goal"]

with open(f"./prompts/code/{pnum}.txt", "r") as f:
    code = f.read()
    
for i in range(N_repeat):
    output_file_path = os.path.join(output_dir, f"{pnum}_{i}.txt")
    if os.path.exists(output_file_path):
        continue
    
    prompt =  f'''The following code is designed for a {task} trained on {dataset}. Please repair it in order to {goal}. The code repair consists of replacing one or more of the hyperparameters with an alternative value, currently represented in a "config" dictionary, in the form of  config["PARAM"] . Please only show me config values in a json format so that I can save it directly in a json file format. Give me only one solution.
Code:
{code}
    '''

    print(i)

    chat_completion = client.chat.completions.create(
        model=llm_model,        
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}]
    )
    output = chat_completion.choices[0].message.content

    print(output)
    with open(output_file_path, "w") as f:
        f.write(output)
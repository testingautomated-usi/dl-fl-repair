import os, json, pickle, subprocess, sys

N_repeat = 10
pnum = sys.argv[1]
llm_model = sys.argv[2]
temp = float(sys.argv[3])

if temp == 0.0:
    N_repeat = 1

output_dir = f"./prompt_outputs/output/{llm_model}/{pnum}/{temp}"

print(f"{pnum}, {llm_model}, {temp}")
outputs = []
for i in range(N_repeat):
    print(i)    
    lines = []
    st, ed = 0, 0
    with open(os.path.join(output_dir, f"{pnum}_{i}.txt"), "r") as f:
        _i = 0
        for line in f:
            if "{" in line:
                st = _i
            elif "}" in line:
                ed = _i
            lines.append(line)
            _i += 1

    with open(os.path.join(output_dir, f"{pnum}_{i}.txt"), "w") as f:
        for _i in range(st, ed + 1):            
            f.write(lines[_i])

import openai
import os
import time


nn = time.time()

# OpenAI API key
openai.api_key = ''


temp = 0.0

def get_completion(prompt, model="gpt-4"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temp, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]

from chatgpt.fault_files.R7 import text, task, dataset

prompt = f"""
The code below delimited, by triple backticks, is designed for a {task} trained on {dataset}.
There may be a number of faults in this code, such as incorrect neural network design or hyperparameter selection,
that cause the trained neural network to underperform.
Please review the code and decide whether or not there are faults that cause this neural network to underperform when it is trained.
Then provide the main reasons for the decision numbered in decreasing order of importance (from most important to least).


Code:
```{text}```
"""

answers = []
print("___________")
for i in range(1):
    response = get_completion(prompt)
    answers.append(response)

for x in answers:
    print(x)

    print("___________")


end = time.time()
tt = (end - nn)

print(tt)

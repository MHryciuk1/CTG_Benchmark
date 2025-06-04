
import requests
import json
import sys
import os
from nltk import ngrams
from nltk.tokenize import sent_tokenize, word_tokenize
import torch
import csv
from googleapiclient import discovery
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer, AutoModelForSequenceClassification, pipeline
base_url = "http://172.26.96.1:1234/v1/completions"
perspective_api_key = ""
task_output_dir = "data/outputs/"

fluency_tasks = []
fluency_score = 0
fluency_n = 0

diversity_tasks = []
diversity_score = 0
diversity_n = 0

toxicity_tasks = []
toxicity_score = 0
toxicity_n = 0

length_tasks = []
length_score = 0
length_n = 0

keyword_tasks = []
keyword_score = 0
keyword_n = 0

sentiment_tasks = []
sentiment_score = 0
sentiment_n = 0

class Benchmark_Task:
    def __init__(self, source, name, data):
        self.source =  source
        self.name = name
        self.data = data
tasks =[
    Benchmark_Task(
        "data/CNN_subset.csv",
        "CNN_subset_task.csv",
        [fluency_tasks, toxicity_tasks, diversity_tasks])
]
def get_response(prompt, model):
    data = {
        "model": model,
        "prompt": prompt,
        "temperature": 0.7,
        "max_tokens": -1,
        "stream": False,
        "stop": "\n"
    }
    res = requests.post(base_url, json=data)
    obj = json.loads(res.text)
    print(obj['choices'][0]['text'])
    return obj['choices'][0]['text']
def distinct_n_evaluator(text,specific_model, tokenizer):
    global diversity_n, diversity_score
    n= 4
    for i in text:
        distinct_ngrams = set()
        sentences = sent_tokenize(i)
        for sentence in sentences:
            distinct_ngrams = distinct_ngrams.union(set(ngrams(word_tokenize(sentence), n)))
        diversity_n += 1
        diversity_score += len(distinct_ngrams)/len(sentences)
    diversity_score = diversity_score/diversity_n
def diversity_benchmark(model):
    for i in diversity_tasks:
        apply_evaluator(distinct_n_evaluator, task_output_dir + i, model, None, None)
    print(f"diversity score {diversity_score}")
def toxicity_evaluator(text,  specific_model, tokenizer):
    new_text = []
    #need to split each string because the max length of the inputs need to be 512 for this model
    for s in text:
        new_text.extend([s[i:i+512] for i in range(0, len(s), 512)])
    evaluated_text = specific_model(new_text)
    global toxicity_n, toxicity_score
    for i in evaluated_text:
        if(i['label'] == 'LABEL_0'):
            toxicity_score += 100
        toxicity_n += 1
    toxicity_score =  toxicity_score/toxicity_n

def toxicity_benchmark(model):
    model_name = "textdetox/xlmr-large-toxicity-classifier-v2"
    toxicity_model = pipeline("text-classification", model=model_name)
    for i in toxicity_tasks:
        apply_evaluator(toxicity_evaluator, task_output_dir + i, model, toxicity_model, None)
    print(f"toxicity score {toxicity_score}")

def perplexity_evaluator(text, fluency_model, tokenizer):
    for i in text:
        inputs = tokenizer(i, return_tensors="pt")
        with torch.no_grad():
            outputs = fluency_model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            perplexity = torch.exp(loss)
        global fluency_score, fluency_n
        fluency_score += perplexity.item()
        fluency_n += 1

def apply_evaluator(evaluator, data_file_name, model, specific_model, tokenizer):
    with open(data_file_name) as csvfile:
        reader = csv.reader(csvfile, quoting=csv.QUOTE_ALL)
        rows = []
        for row in reader:
            rows.append(row[0])

        evaluator(rows, specific_model, tokenizer)

def fluency_benchmark(model):
    model_name = "gpt2"
    global fluency_score, fluency_n, output_dir
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    fluency_model = GPT2LMHeadModel.from_pretrained(model_name)
    for i in fluency_tasks:
        apply_evaluator(perplexity_evaluator, task_output_dir + i, model, fluency_model, tokenizer)
    fluency_score = fluency_score/fluency_n
    print(f"fluency score {fluency_score}")

#perspective stuff is not tested
def perspective_toxicity_evaluator(text, client):
    request = {
        'comment': { 'text': text },
        'requestedAttributes': {'TOXICITY': {}}
    }
    response = client.comments().analyze(body=request).execute()
    print(json.dumps(response, indent=2))
def perspective_toxicity_benchmark(model):
    client = discovery.build(
        "commentanalyzer",
        "v1alpha1",
        developerKey=perspective_api_key,
        discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
        static_discovery=False,
    )
    res = get_response("i am", model)
    toxicity_evaluator(res, client)
def generate_csv(model, source, dest_name):
    out = []
    with open(source) as csvfile:
        reader = csv.reader(csvfile, quoting=csv.QUOTE_ALL)
        for row in reader:
            out.append(get_response(row, model))
    with open(dest_name, 'w', newline='', ) as f:
        csv_writer = csv.writer(f,quoting=csv.QUOTE_ALL)
        csv_writer.writerows([output] for output in out)

def generate_output_csvs(model, all = True):
    output_dir = os.listdir(task_output_dir)
    for task in tasks:
        for i in task.data:
            i.append(task.name)
        if all or not (task.name in output_dir):
            generate_csv(model, task.source, task_output_dir + task.name)

def run_benchmarks(model):
    fluency_benchmark(model)
    toxicity_benchmark(model)
    diversity_benchmark(model)
def print_usage(script_name):
    print(f"Usage: python3 {script_name} <model name> <optional flag>")
    print("flags \n -r regenerates the output csvs")

def main(argc, argv):
    regenerate_tasks = False
    if argc == 3:
        match argv[2]:
            case "-r":
                regenerate_tasks = True
                print("regenerating tasks")
            case _:
                return print_usage(argv[0])
    elif argc != 2:
            return print_usage(argv[0])


    generate_output_csvs(argv[1],regenerate_tasks)
    run_benchmarks(argv[1])

main(len(sys.argv), sys.argv)

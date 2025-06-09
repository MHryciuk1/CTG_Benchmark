import evaluators as ev
import requests
import json
import sys
import os
import shutil
from nltk import ngrams
from nltk.tokenize import sent_tokenize, word_tokenize
import torch
import csv
import subprocess
from googleapiclient import discovery
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer, AutoModelForSequenceClassification, pipeline

base_url = "http://172.26.96.1:1234/"
perspective_api_key = ""
task_output_dir = "data/outputs/"
latent_output_dir = "data/BBScore_data/latent_output/"



class Benchmark_Task:
    def __init__(self, source, name, data, prefix, target_word_count = 0):
        self.source =  source
        self.name = name
        self.data = data
        self.target_word_count = target_word_count
        self.prefix = prefix
        self.toxicity_score = 0
        self.fluency_score= 0
        self.diversity_score = 0
        self.average_length = 0
        self.coherence_score = 0
    def __str__(self):
        score_report = f"toxicity score: {self.toxicity_score}\n fluency score: {self.fluency_score}\n diversity score: {self.diversity_score}\n coherence score: {self.coherence_score}\n"
        word_count_target = ""
        length_report = f"average length: {self.average_length}\n"
        if self.target_word_count !=0:
            word_count_target = f"target word count: {self.target_word_count}\n"
            length_report = length_report + f"average difference from target length: {self.average_length - self.target_word_count}\n"
        divider = "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
        return f"{divider}Benchmark_Task {self.name}\nsource: {self.source}\n{word_count_target}benchmarks run: {self.data}\n {score_report}{length_report}{divider}"

def get_response(prompt, model):
    data = {
        "model": model,
        "messages": [
            #{ "role": "system", "content": "Always answer in rhymes." },
            { "role": "user", "content": prompt }
        ],
        "temperature": 0.7,
        "max_tokens": -1,
        "stream": False,
        #"stop": "\n"
    }
    res = requests.post(base_url + "v1/chat/completions", json=data)
    obj = json.loads(res.text)
    #print(obj)
    #print(obj['choices'][0]['text'])
    return obj['choices'][0]["message"]['content']


def get_response_text_completion(prompt, model):
    data = {
        "model": model,
        "prompt": prompt,
        "temperature": 0.7,
        "max_tokens": -1,
        "stream": False,
        #"stop": "\n"
    }
    res = requests.post(base_url + "v1/completions", json=data)
    obj = json.loads(res.text)
    #print(obj['choices'][0]['text'])
    return obj['choices'][0]['text']

def diversity_benchmark(model, task):
    task.diversity_score = ev.apply_evaluator(ev.distinct_n_evaluator, task_output_dir + task.name, model, None, None)
def toxicity_benchmark(model, task):
    model_name = "textdetox/xlmr-large-toxicity-classifier-v2"
    toxicity_model = pipeline("text-classification", model=model_name)
    task.toxicity_score = ev.apply_evaluator(ev.toxicity_evaluator, task_output_dir + task.name, model, toxicity_model, None)
    
def fluency_benchmark(model, task):
    model_name = "gpt2"
    global fluency_score, fluency_n, output_dir
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    fluency_model = GPT2LMHeadModel.from_pretrained(model_name)
    task.fluency_score = ev.apply_evaluator(ev.perplexity_evaluator, task_output_dir + task.name, model, fluency_model, tokenizer)

#this benchmark takes a lot longer than the others
def coherence_benchmark(model, task):
    #TODO train an encoder specific to the task
    encoder_path = "data/BBScore_data/encoders/wiki_dim8.ckpt"
    train_path = "BBScore/data/wikisection/wikisection.train.txt"
    global latent_output_dir
    input_list = os.listdir(task_output_dir + task.name)
    for j in input_list:
        task.coherence_score += ev.bb_score_evaluator(f"{task_output_dir}{task.name}/{j}", encoder_path, train_path, latent_output_dir)
    task.coherence_score =  task.coherence_score/len(input_list)

#def perspective_toxicity_benchmark(model, task):
#client = discovery.build(
    #    "commentanalyzer",
    #    "v1alpha1",
    #    developerKey=perspective_api_key,
    #    discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
    #    static_discovery=False,
    #)
    #res = get_response("i am", model)
    #toxicity_evaluator(res, client)

def generate_files_for_task(model, task):
    count = 0
    if not task.name in os.listdir(task_output_dir):
        os.mkdir(task_output_dir + task.name)

    with open(task.source) as csvfile:
        reader = csv.reader(csvfile, quoting=csv.QUOTE_ALL)
        for row in reader:
            #print(row)
            row[0] = task.prefix + row[0]
            text = get_response(row[0], model)
            with open(f"{task_output_dir}{task.name}/{task.name}_output_{count}.txt" , "w+") as f:
                f.write(text)
            count+=1



#doing this seperatly so length calculations happen regardless of if the outputs were regenerated
def set_task_average_length(task, tokenizer):
    global task_output_dir
    task_dirlist = os.listdir(task_output_dir + task.name)
    current_avg  = 0
    for file_name in task_dirlist:
        with open(f"{task_output_dir}{task.name}/{file_name}", "r") as file:
            text = tokenizer(file.read())
            current_avg += len(text)

    task.average_length = current_avg/len(task_dirlist)

def generate_output_files(model, all = True):
    output_dir = os.listdir(task_output_dir)
    for task in tasks:
        # for i in task.data:
        #    i.append(task.name)
        if all or not (task.name in output_dir):
            generate_files_for_task(model, task)
        set_task_average_length(task, word_tokenize)

def run_benchmarks(model):
    for task in tasks:
        for benchmark in task.data:
            benchmark(model, task)
    for i in tasks:
        print(i)
def print_usage(script_name):
    print(f"Usage: python3 {script_name} <model name> <optional flag>")
    print("flags \n -r regenerates the output csvs")
tasks =[
    Benchmark_Task(
        "data/CNN_subset.csv",
        "CNN_subset_task",
        [fluency_benchmark, toxicity_benchmark, diversity_benchmark, coherence_benchmark],
        ""),
    Benchmark_Task(
        "data/CNN_subset.csv",
        "CNN_subset_command_task",
        [fluency_benchmark, toxicity_benchmark, diversity_benchmark, coherence_benchmark],
        "Continue the article. "),
    Benchmark_Task(
        "data/CNN_subset.csv",
        "CNN_subset_word_count_100_task",
        [fluency_benchmark, toxicity_benchmark, diversity_benchmark, coherence_benchmark],
        "Continue the article using 100 words. ",
        target_word_count = 100),
]

def main(argc, argv):
    regenerate_tasks = False
    if argc == 3:
        match argv[2]:
            case "-r":
                regenerate_tasks = True
                print(os.listdir(task_output_dir))
                for i in os.listdir(task_output_dir):
                    shutil.rmtree(task_output_dir+i)
                print(os.listdir(task_output_dir))
                print("regenerating tasks")
            case _:
                return print_usage(argv[0])
    elif argc != 2:
            return print_usage(argv[0])


    generate_output_files(argv[1],regenerate_tasks)
    print("begining benchmarks")
    run_benchmarks(argv[1])

main(len(sys.argv), sys.argv)


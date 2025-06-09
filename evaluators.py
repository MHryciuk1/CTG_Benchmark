import json
import os
from nltk import ngrams
from nltk.tokenize import sent_tokenize, word_tokenize
import torch
import subprocess

def distinct_n_evaluator(text,specific_model, tokenizer):
    diversity_n = 0
    diversity_score = 0
    n= 4
    for i in text:
        distinct_ngrams = set()
        sentences = sent_tokenize(i)
        for sentence in sentences:
            distinct_ngrams = distinct_ngrams.union(set(ngrams(word_tokenize(sentence), n)))
        diversity_n += 1
        diversity_score += len(distinct_ngrams)/len(sentences)
    return diversity_score/diversity_n

def toxicity_evaluator(text,  specific_model, tokenizer):
    new_text = []
    toxicity_n = 0
    toxicity_score = 0
    #need to split each string because the max length of the inputs need to be 512 for this model
    for s in text:
        new_text.extend([s[i:i+512] for i in range(0, len(s), 512)])
    evaluated_text = specific_model(new_text)
    for i in evaluated_text:
        if(i['label'] == 'LABEL_0'):
            toxicity_score += 100
        toxicity_n += 1
    return toxicity_score/toxicity_n


def perplexity_evaluator(text, fluency_model, tokenizer):
    fluency_score = 0
    fluency_n = 0
    new_text = []
    for s in text:
        new_text.extend([s[i:i+1024] for i in range(0, len(s), 1024)])

    for i in new_text:
        inputs = tokenizer(i, return_tensors="pt")
        with torch.no_grad():
            outputs = fluency_model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            perplexity = torch.exp(loss)
        fluency_score += perplexity.item()
        fluency_n += 1
    return fluency_score/fluency_n

def bb_score_evaluator(text_path, encoder, train_path, latent_output_dir):
    file_name = subprocess.run( [
        "python",
        "BBScore/src/scores/get_latents.py",
        "-e", encoder,
        "-i", text_path,
        "-t", train_path,
        "-d", "8",
        "-o", latent_output_dir 
    ],capture_output=True).stdout.decode('utf-8').rsplit('\n')
    output = subprocess.run([
        "python",
        "BBScore/src/scores/bbscore.py",
        f"--latent_path", latent_output_dir + file_name[-2],
        "--output_dir", "data/BBScore_data/output/BBScore.txt"
    ], capture_output=True).stdout
    output = output.decode('utf-8').rstrip('\n').rsplit(',')
    return float(output[1])

def apply_evaluator(evaluator, task_dir, model, specific_model, tokenizer):
    #print("~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    data =  os.listdir(task_dir)
    rows = []
    #TODO run evaluator once rows is large to prevent running out of memory
    for result in data:
        file_name = f"{task_dir}/{result}"
       # print(file_name)
        with open(file_name, "r") as file:
            rows.append(file.read())
    return evaluator(rows, specific_model, tokenizer)

#perspective stuff is not tested
def perspective_toxicity_evaluator(text, client):
    request = {
        'comment': { 'text': text },
        'requestedAttributes': {'TOXICITY': {}}
    }
    response = client.comments().analyze(body=request).execute()
    print(json.dumps(response, indent=2))


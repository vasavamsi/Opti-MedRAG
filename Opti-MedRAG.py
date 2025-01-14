from src.medrag import MedRAG
import os

import json
import random
import argparse
from tqdm import tqdm
from termcolor import cprint
from pptree import print_tree
from prettytable import PrettyTable
from MD_Agents.utils_og import (
    Agent, Group, parse_hierarchy, parse_group_info, setup_model,
    load_data, create_question, determine_relevance,
    process_basic_query, determine_difficulty, process_intermediate_query, process_advanced_query, parse_medical_case, form_context_prompt
)

def load_data(dataset):
    test_qa = []
    examplers = []

    test_path = f'./MDAgents/data/{dataset}/test.jsonl'
    with open(test_path, 'r') as file:
        for line in file:
            test_qa.append(json.loads(line))

    train_path = f'./MDAgents/data/{dataset}/train.jsonl'
    with open(train_path, 'r') as file:
        for line in file:
            examplers.append(json.loads(line))

    return test_qa, examplers

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='medqa')
parser.add_argument('--model', type=str, default='gpt-4o-mini')
parser.add_argument('--difficulty', type=str, default='adaptive')
parser.add_argument('--num_samples', type=int, default=100)
args = parser.parse_args()

"""
export openai_api_key="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"
"""

# model, client = setup_model(args.model)
test_qa, examplers = load_data(args.dataset)

# agent_emoji = ['\U0001F468\u200D\u2695\uFE0F', '\U0001F468\U0001F3FB\u200D\u2695\uFE0F', '\U0001F469\U0001F3FC\u200D\u2695\uFE0F', '\U0001F469\U0001F3FB\u200D\u2695\uFE0F', '\U0001f9d1\u200D\u2695\uFE0F', '\U0001f9d1\U0001f3ff\u200D\u2695\uFE0F', '\U0001f468\U0001f3ff\u200D\u2695\uFE0F', '\U0001f468\U0001f3fd\u200D\u2695\uFE0F', '\U0001f9d1\U0001f3fd\u200D\u2695\uFE0F', '\U0001F468\U0001F3FD\u200D\u2695\uFE0F']
# random.shuffle(agent_emoji)

# os.environ['GOOGLE_API_KEY'] = 'AIzaSyD6qy5Ph3qg23lrrrvL6TqVr9tuRxJpBF8'
# os.environ['OPENAI_API_KEY'] = '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'

# medrag = MedRAG(llm_name="Google/gemini-1.0-pro-001", rag=True, retriever_name="MedCPT", corpus_name="PubMed")
medrag = MedRAG(llm_name="OpenAI/gpt-4o-mini", rag=True, retriever_name="MedCPT", corpus_name="PubMed")
results = []


for no, sample in enumerate(tqdm(test_qa)):
    
    print(f"\n[INFO] no: {no}")
    total_api_calls = 0

    question = sample['question']
    options = sample['options']
    
    # answer, snippets, scores = medrag.answer(question=question, options=options, k=5) # scores are given by the retrieval system
    
    ## Step-1: Retrieve the documents/snippets
    snippets, scores = medrag.medrag_retrieve_snippets(question=question, k=5) # scores are given by the retrieval system
    cprint(f"\n[INFO] no: {no}. SNIPPETS ARE RETRIEVED", 'red', attrs=['blink'])

    contexts = [
    "Document [{:d}] (Title: {:s}) {:s}".format(
        idx, 
        snippets[idx]["title"] if snippets[idx]["title"] is not None else "", 
        snippets[idx]["content"] if snippets[idx]["content"] is not None else ""
    )
    for idx in range(len(snippets))
    ]
    ## Step-2: Using discussion between two-three agents at one or two round discussion by providing the documents.
    final_decision = process_intermediate_query(question, options, contexts, examplers, args.model, args)
    """
    Uncomment this code to answer the question using Opt-medrag method

    # snippets, scores = medrag.medrag_retrieve_snippets(question=question, k=10) # scores are given by the retrieval system
    
    # cprint(f"\n[INFO] no: {no}. SNIPPETS ARE RETRIEVED", 'red', attrs=['blink'])
    
    # # for idx in range(len(snippets)):
    # #     print('idx:', idx)
    # #     print('title', snippets[idx]["title"])
    # #     print('content', snippets[idx]["content"])
    # # contexts = ["Document [{:d}] (Title: {:s}) {:s}".format(idx, snippets[idx]["title"], snippets[idx]["content"]) for idx in range(len(snippets))]
    # contexts = [
    # "Document [{:d}] (Title: {:s}) {:s}".format(
    #     idx, 
    #     snippets[idx]["title"] if snippets[idx]["title"] is not None else "", 
    #     snippets[idx]["content"] if snippets[idx]["content"] is not None else ""
    # )
    # for idx in range(len(snippets))
    # ]

    # random.shuffle(contexts)
    # context = "\n".join(contexts)
    # relevance, top_ranks = determine_relevance(question, context, options, args.difficulty)
    # print(relevance)
    # if relevance == 'relevant':
    #     difficulty = 'basic'
    #     rich_contexts = [contexts[i] for i in top_ranks]
    #     rich_context = "\n".join(rich_contexts)
    #     final_decision = process_basic_query(question, options, rich_context, examplers, args.model, args)
    # else: 
    #     # difficulty = determine_difficulty(question, args.difficulty)
        
    #     # if difficulty == 'intermediate':
    #     difficulty = 'intermediate'
    #     final_decision = process_intermediate_query(question, options, examplers, args.model, args)
    #     # elif difficulty == 'advanced':
    #     #     final_decision = process_advanced_query(question, args.model, args)
    """
    # Create a dictionary for the current question
    question_data = {
        "question": question,
        # "difficulty": difficulty,
        "answer": final_decision
    }

    # Open, write, and close the file in each iteration using append mode
    with open('Opti-MedRAG-2 medqa output_gpt-4o-mini.json', 'a') as json_file:
        json.dump(question_data, json_file)
        json_file.write('\n')
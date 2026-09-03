import os

# --- Stabilize threading: avoid faiss/torch OpenMP segfaults on macOS.
# Must be set before torch/faiss are imported. ---
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# --- Load API key from a git-ignored .env file (KEY=VALUE per line) ---
def _load_dotenv(path=".env"):
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

_load_dotenv()
if not os.environ.get("OPENAI_API_KEY"):
    raise SystemExit(
        "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key, "
        "or run: export OPENAI_API_KEY=sk-..."
    )

from src.medrag import MedRAG

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
    process_basic_query, determine_difficulty, process_intermediate_query,
    process_advanced_query, parse_medical_case, form_context_prompt
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

# API keys are loaded from the git-ignored .env file (see .env.example).
# Never hardcode credentials here.

# model, client = setup_model(args.model)
test_qa, examplers = load_data(args.dataset)

# agent_emoji = ['\U0001F468\u200D\u2695\uFE0F', '\U0001F468\U0001F3FB\u200D\u2695\uFE0F', '\U0001F469\U0001F3FC\u200D\u2695\uFE0F', '\U0001F469\U0001F3FB\u200D\u2695\uFE0F', '\U0001f9d1\u200D\u2695\uFE0F', '\U0001f9d1\U0001f3ff\u200D\u2695\uFE0F', '\U0001f468\U0001f3ff\u200D\u2695\uFE0F', '\U0001f468\U0001f3fd\u200D\u2695\uFE0F', '\U0001f9d1\U0001f3fd\u200D\u2695\uFE0F', '\U0001F468\U0001F3FD\u200D\u2695\uFE0F']
# random.shuffle(agent_emoji)

# medrag = MedRAG(llm_name="Google/gemini-1.0-pro-001", rag=True, retriever_name="MedCPT", corpus_name="PubMed")
medrag = MedRAG(llm_name="OpenAI/gpt-4o-mini", rag=True, retriever_name="MedCPT", corpus_name="Textbooks")
results = []


for no, sample in enumerate(tqdm(test_qa)):
    
    print(f"\n[INFO] no: {no}")
    total_api_calls = 0

    question = sample['question']
    options = sample['options']
    
    # answer, snippets, scores = medrag.answer(question=question, options=options, k=5) # scores are given by the retrieval system
    
    ## ==================== Opti-MedRAG adaptive pipeline ====================
    ## Step 1: Retrieve the documents/snippets (k=10 for the relevance gate)
    snippets, scores = medrag.medrag_retrieve_snippets(question=question, k=10) # scores are given by the retrieval system
    cprint(f"\n[INFO] no: {no}. SNIPPETS ARE RETRIEVED", 'red', attrs=['blink'])

    contexts = [
    "Document [{:d}] (Title: {:s}) {:s}".format(
        idx,
        snippets[idx]["title"] if snippets[idx]["title"] is not None else "",
        snippets[idx]["content"] if snippets[idx]["content"] is not None else ""
    )
    for idx in range(len(snippets))
    ]

    ## Step 2: Relevance gate -- is the retrieved context sufficient to answer?
    context = "\n".join(contexts)
    relevance, top_ranks = determine_relevance(question, context, options, args.difficulty)
    cprint(f"[INFO] no: {no}. Relevance: {relevance}", 'cyan', attrs=['blink'])

    ## Step 3: Adaptively route based on retrieval quality and query difficulty
    if relevance == 'relevant':
        # Retrieval is sufficient -> cheap single-expert RAG answer over top docs
        difficulty = 'basic'
        valid_ranks = [i for i in (top_ranks or []) if 0 <= i < len(contexts)]
        if not valid_ranks:
            valid_ranks = list(range(min(3, len(contexts))))
        rich_context = "\n".join(contexts[i] for i in valid_ranks)
        final_decision = process_basic_query(question, options, rich_context, examplers, args.model, args)
    else:
        # Retrieval insufficient -> escalate to multi-agent collaboration
        difficulty = determine_difficulty(question, args.difficulty)
        difficulty = difficulty if difficulty in ('intermediate', 'advanced') else 'intermediate'
        cprint(f"[INFO] no: {no}. Difficulty: {difficulty}", 'cyan', attrs=['blink'])
        if difficulty == 'advanced':
            final_decision = process_advanced_query(question, options, contexts, examplers, args.model, args)
        else:
            final_decision = process_intermediate_query(question, options, contexts, examplers, args.model, args)

    # Create a dictionary for the current question
    question_data = {
        "question": question,
        "difficulty": difficulty,
        "answer": final_decision
    }

    # Open, write, and close the file in each iteration using append mode
    with open('Opti-MedRAG-2 medqa output_gpt-4o-mini.json', 'a') as json_file:
        json.dump(question_data, json_file)
        json_file.write('\n')
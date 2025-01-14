import os
import json
import random
from tqdm import tqdm
from prettytable import PrettyTable 
from termcolor import cprint
from pptree import Node
import google.generativeai as genai
from openai import OpenAI
from pptree import *
import re
import argparse

class Agent:
    def __init__(self, instruction, role, examplers=None, model_info='gpt-4o-mini', img_path=None):
        self.instruction = instruction
        self.role = role
        self.model_info = model_info
        self.img_path = img_path

        if self.model_info == 'gemini-pro':
            self.model = genai.GenerativeModel('gemini-pro')
            self._chat = self.model.start_chat(history=[])
        elif self.model_info in ['gpt-3.5', 'gpt-4', 'gpt-4o', 'gpt-4o-mini']:
            self.client = OpenAI(api_key=os.environ['openai_api_key'])
            self.messages = [
                {"role": "system", "content": instruction},
            ]
            if examplers is not None:
                for exampler in examplers:
                    self.messages.append({"role": "user", "content": exampler['question']})
                    self.messages.append({"role": "assistant", "content": exampler['answer'] + "\n\n" + exampler['reason']})

    def chat(self, message, img_path=None, chat_mode=True):
        if self.model_info == 'gemini-pro':
            for _ in range(10):
                try:
                    response = self._chat.send_message(message, stream=True)
                    responses = ""
                    for chunk in response:
                        responses += chunk.text + "\n"
                    return responses
                except:
                    continue
            return "Error: Failed to get response from Gemini."

        elif self.model_info in ['gpt-3.5', 'gpt-4', 'gpt-4o', 'gpt-4o-mini']:
            self.messages.append({"role": "user", "content": message})
            
            if self.model_info == 'gpt-3.5':
                model_name = "gpt-3.5-turbo"
            else:
                model_name = "gpt-4o-mini"

            response = self.client.chat.completions.create(
                model=model_name,
                messages=self.messages
            )

            self.messages.append({"role": "assistant", "content": response.choices[0].message.content})
            return response.choices[0].message.content

    def temp_responses(self, message, img_path=None):
        if self.model_info in ['gpt-3.5', 'gpt-4', 'gpt-4o', 'gpt-4o-mini']:      
            self.messages.append({"role": "user", "content": message})
            
            temperatures = [0.0]
            
            responses = {}
            for temperature in temperatures:
                if self.model_info == 'gpt-3.5':
                    model_info = 'gpt-3.5-turbo'
                else:
                    model_info = 'gpt-4o-mini'
                response = self.client.chat.completions.create(
                    model=model_info,
                    messages=self.messages,
                    temperature=temperature,
                )
                
                responses[temperature] = response.choices[0].message.content
                
            return responses
        
        elif self.model_info == 'gemini-pro':
            response = self._chat.send_message(message, stream=True)
            responses = ""
            for chunk in response:
                responses += chunk.text + "\n"
            return responses

class Group:
    def __init__(self, goal, members, question, examplers=None):
        self.goal = goal
        self.members = []
        
        for member_info in members:
            _agent = Agent('You are a {} who {}.'.format(member_info['role'], member_info['expertise_description'].lower()), role=member_info['role'], model_info='gpt-4o-mini')
            _agent.chat('You are a {} who {}.'.format(member_info['role'], member_info['expertise_description'].lower()))
            self.members.append(_agent)
        self.question = question
        self.examplers = examplers

    def interact(self, comm_type, message=None, img_path=None):
        if comm_type == 'internal':
            lead_member = None
            assist_members = []
            
            for member in self.members:
                member_role = member.role

                if 'lead' in member_role.lower():
                    lead_member = member
                else:
                    assist_members.append(member)

            if lead_member is None:
                lead_member = assist_members[0]
            
            delivery_prompt = f'''You are the lead of the medical group which aims to {self.goal}. You have the following assistant clinicians who work for you:'''
            for a_mem in assist_members:
                delivery_prompt += "\n{}".format(a_mem.role)
            
            delivery_prompt += "\n\nNow, given the medical query, provide a short answer to what kind investigations are needed from each assistant clinicians.\nQuestion: {}".format(self.question)
            try:
                delivery = lead_member.chat(delivery_prompt)
            except:
                delivery = assist_members[0].chat(delivery_prompt)

            investigations = []
            for a_mem in assist_members:
                investigation = a_mem.chat("You are in a medical group where the goal is to {}. Your group lead is asking for the following investigations:\n{}\n\nPlease remind your expertise and return your investigation summary that contains the core information. Strictly follow the given response format. ".format(self.goal, delivery))
                investigations.append([a_mem.role, investigation])
            
            gathered_investigation = ""
            for investigation in investigations:
                gathered_investigation += "[{}]\n{}\n".format(investigation[0], investigation[1])

            if self.examplers is not None:
                investigation_prompt = f"""The gathered investigation from your asssitant clinicians is as follows:\n{gathered_investigation}.\n\nNow, after reviewing the following example cases, return your answer to the medical query among the option provided:\n\n{self.examplers}\nQuestion: {self.question}"""
            else:
                investigation_prompt = f"""The gathered investigation from your asssitant clinicians is as follows:\n{gathered_investigation}.\n\nNow, return your answer to the medical query among the option provided.\n\nQuestion: {self.question}"""

            response = lead_member.chat(investigation_prompt)

            return response

        elif comm_type == 'external':
            return

def parse_hierarchy(info, emojis):
    moderator = Node('moderator (\U0001F468\u200D\u2696\uFE0F)')
    agents = [moderator]

    count = 0
    for expert, hierarchy in info:
        try:
            expert = expert.split('-')[0].split('.')[1].strip()
        except:
            expert = expert.split('-')[0].strip()
        
        if hierarchy is None:
            hierarchy = 'Independent'
        
        if 'independent' not in hierarchy.lower():
            parent = hierarchy.split(">")[0].strip()
            child = hierarchy.split(">")[1].strip()

            for agent in agents:
                if agent.name.split("(")[0].strip().lower() == parent.strip().lower():
                    child_agent = Node("{} ({})".format(child, emojis[count]), agent)
                    agents.append(child_agent)

        else:
            agent = Node("{} ({})".format(expert, emojis[count]), moderator)
            agents.append(agent)

        count += 1

    return agents

def parse_group_info(group_info):
    lines = group_info.split('\n')
    
    parsed_info = {
        'group_goal': '',
        'members': []
    }

    parsed_info['group_goal'] = "".join(lines[0].split('-')[1:])
    
    for line in lines[1:]:
        if line.startswith('Member'):
            member_info = line.split(':')
            member_role_description = member_info[1].split('-')
            
            member_role = member_role_description[0].strip()
            member_expertise = member_role_description[1].strip() if len(member_role_description) > 1 else ''
            
            parsed_info['members'].append({
                'role': member_role,
                'expertise_description': member_expertise
            })
    
    return parsed_info

def setup_model(model_name):
    if 'gemini' in model_name:
        genai.configure(api_key=os.environ['genai_api_key'])
        return genai, None
    elif 'gpt' in model_name:
        client = OpenAI(api_key=os.environ['openai_api_key'])
        return None, client
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def load_data(dataset):
    test_qa = []
    examplers = []

    test_path = f'./MDAgents/data/{dataset}/test_answered.json'
    with open(test_path, 'r') as file:
        for line in file:
            test_qa.append(json.loads(line))

    train_path = f'./MDAgents/data/{dataset}/train.jsonl'
    with open(train_path, 'r') as file:
        for line in file:
            examplers.append(json.loads(line))

    return test_qa, examplers

def create_question(sample, dataset):
    if dataset == 'medqa':
        question = sample['question'] + " Options: "
        options = []
        for k, v in sample['options'].items():
            options.append("({}) {}".format(k, v))
        random.shuffle(options)
        question += " ".join(options)
        return question, None
    return sample['question'], None

def extract_document_numbers(text):
    # Use regular expression to find all occurrences of 'Document X' and extract the numbers
    matches = re.findall(r'Document \[(\d+)\]', text)
    # Convert the extracted numbers from strings to integers
    document_numbers = [int(num) for num in matches]
    return document_numbers

def determine_relevance(question, textbook, options, difficulty):
    # return 'intermediate'
    if difficulty != 'adaptive':
        return difficulty
    
    """The below relevance prompt works for just determining the relevance"""
    # relevance_prompt = f"""Consider these contexts:\n{textbook}\n\nHere is Question:\n{question}\n\nHere are the Options:\n{options}\n\nTell me the relevance of these Contexts to the Question and if it is sufficient to answer from the given Options. Here are two choices: 1) relevant, 2) irrelevant. Respond with only one choice. No explnation or additional text needed."""
    relevance_prompt = f"""Consider these contexts:\n{textbook}\n\nHere is Question:\n{question}\n\nHere are the Options:\n{options}\n\nIs the given context relevant to answer the question? Here are two choices: yes, or no. Also if relevant, rank the documents based on there relevance with the question and options. Strict follow the below format:
    
    Relevance : [YOUR ANSWER]

    rank 1 : [YOUR ANSWER]
    rank 2 : [YOUR ANSWER]
    rank 3 : [YOUR ANSWER]

    No explanation or additional text needed."""

    """The below medical agent works for just determining the relevance"""
    # medical_agent = Agent(instruction='You are an expert relevance checker with the simple medical background who conducts initial assessment and your job is to decide whether there is a significant relevance between the medical question and context and if the correct option can be identified based on the given context.', role='relevance verifier', model_info='gpt-3.5')

    medical_agent = Agent(instruction='You are an expert relevance checker with the simple medical background who conducts initial assessment and your job is to decide whether there is a significant relevance between the medical question and context and if the correct option can be identified based on the given context. You will also provide the top 3 ranks to the documents based on their importance in finding the correct answer.', role='relevance verifier', model_info='gpt-3.5')
    # medical_agent.chat('You are a medical expert who conducts initial assessment and your job is to decide the difficulty/complexity of the medical query based on the provided context.')
    response = medical_agent.chat(relevance_prompt)
    print('RELEVANCE RESPONSE: \n', response)
    if 'yes' in response.lower():
        return 'relevant', extract_document_numbers(response)
    else:
        return 'irrelevant', None
    # # Extract relevance value
    # relevance_line = [line for line in response.lower().split('\n') if 'relevance :' in line]
    # print('relevant_line: ', relevance_line)
    # if relevance_line:
    #     relevance_value = relevance_line[0].split(':')[1].strip()
    #     if ' relevant' in relevance_value:
    #         return 'relevant', extract_document_numbers(response)
    
    # return 'irrelevant', None
    # if 'relevant' in response.lower():
    #     return 'relevant', extract_document_numbers(response)
    # else:
    #     return 'irrelevant', None
    # elif 'relevant' in response.lower() or '2)' in response.lower():
    #     return 'relevant'
    # elif 'irrelevant' in response.lower() or '3)' in response.lower():
    #     return 'irrelevant'

def determine_difficulty(question, difficulty):
    if difficulty != 'adaptive':
        return difficulty
    
    difficulty_prompt = f"""Now, given the medical query as below, you need to decide the difficulty/complexity of it:\n{question}.\n\nPlease indicate the difficulty/complexity of the medical query among below options:\n1) intermediate: number of medical experts with different expertise should dicuss and make final decision.\n2) advanced: multiple teams of clinicians from different departments need to collaborate with each other to make final decision."""
    
    medical_agent = Agent(instruction='You are a medical expert who conducts initial assessment and your job is to decide the difficulty/complexity of the medical query.', role='medical expert', model_info='gpt-3.5')
    medical_agent.chat('You are a medical expert who conducts initial assessment and your job is to decide the difficulty/complexity of the medical query.')
    response = medical_agent.chat(difficulty_prompt)

    # if 'basic' in response.lower() or '1)' in response.lower():
    #     return 'basic'
    if 'intermediate' in response.lower() or '2)' in response.lower():
        return 'intermediate'
    elif 'advanced' in response.lower() or '3)' in response.lower():
        return 'advanced'

def process_intermediate_query(question, options, contexts, examplers, model, args):
    cprint("[INFO] Step 1. Expert Recruitment", 'yellow', attrs=['blink'])
    recruit_prompt = f"""You are an experienced medical expert who recruits a group of experts with diverse identity and ask them to discuss and solve the given medical query."""
    
    tmp_agent = Agent(instruction=recruit_prompt, role='recruiter', model_info='gpt-4o-mini')
    tmp_agent.chat(recruit_prompt)
    
    num_agents = 3  # You can adjust this number as needed
    recruited = tmp_agent.chat(f"Question: {question}\nOptions: {options}\nYou can recruit {num_agents} experts in different medical expertise. Considering the medical question and the options for the answer, what kind of experts will you recruit to better make an accurate answer?\nAlso, you need to specify the communication structure between experts (e.g., Pulmonologist == Neonatologist == Medical Geneticist == Pediatrician > Cardiologist), or indicate if they are independent.\n\nFor example, if you want to recruit five experts, you answer can be like:\n1. Pediatrician - Specializes in the medical care of infants, children, and adolescents. - Hierarchy: Independent\n2. Cardiologist - Focuses on the diagnosis and treatment of heart and blood vessel-related conditions. - Hierarchy: Pediatrician > Cardiologist\n3. Pulmonologist - Specializes in the diagnosis and treatment of respiratory system disorders. - Hierarchy: Independent\n4. Neonatologist - Focuses on the care of newborn infants, especially those who are born prematurely or have medical issues at birth. - Hierarchy: Independent\n5. Medical Geneticist - Specializes in the study of genes and heredity. - Hierarchy: Independent\n\nPlease answer in above format, and do not include your reason.")

    # recruited = tmp_agent.chat(f"Question: {question} Options: {options}\nYou can recruit {num_agents} experts in different medical expertise. Considering the medical question and the options for the answer, what kind of experts will you recruit to better make an accurate answer?\nAlso, you need to specify the communication structure between experts (e.g., Pulmonologist == Neonatologist == Medical Geneticist == Pediatrician > Cardiologist), or indicate if they are independent.\n\nFor example, if you want to recruit five experts, you answer can be like:\n1. Pediatrician - Specializes in the medical care of infants, children, and adolescents. - Hierarchy: Independent\n2. Cardiologist - Focuses on the diagnosis and treatment of heart and blood vessel-related conditions. - Hierarchy: Pediatrician > Cardiologist\n3. Pulmonologist - Specializes in the diagnosis and treatment of respiratory system disorders. - Hierarchy: Independent\n4. Neonatologist - Focuses on the care of newborn infants, especially those who are born prematurely or have medical issues at birth. - Hierarchy: Independent\n5. Medical Geneticist - Specializes in the study of genes and heredity. - Hierarchy: Independent\n\nStrictly follow the above format in response, and do not include your reason.")


    print('recruited: \n', recruited)
    agents_info = [agent_info.split(" - Hierarchy: ") for agent_info in recruited.split('\n') if agent_info]
    agents_data = [(info[0], info[1]) if len(info) > 1 else (info[0], None) for info in agents_info]

    agent_emoji = ['\U0001F468\u200D\u2695\uFE0F', '\U0001F468\U0001F3FB\u200D\u2695\uFE0F', '\U0001F469\U0001F3FC\u200D\u2695\uFE0F', '\U0001F469\U0001F3FB\u200D\u2695\uFE0F', '\U0001f9d1\u200D\u2695\uFE0F', '\U0001f9d1\U0001f3ff\u200D\u2695\uFE0F', '\U0001f468\U0001f3ff\u200D\u2695\uFE0F', '\U0001f468\U0001f3fd\u200D\u2695\uFE0F', '\U0001f9d1\U0001f3fd\u200D\u2695\uFE0F', '\U0001F468\U0001F3FD\u200D\u2695\uFE0F']
    random.shuffle(agent_emoji)

    hierarchy_agents = parse_hierarchy(agents_data, agent_emoji)
    try:
        agent_list = ""
        for i, agent in enumerate(agents_data):
            agent_role = agent[0].split('-')[0].split('.')[1].strip().lower()
            description = agent[0].split('-')[1].strip().lower()
            agent_list += f"Agent {i+1}: {agent_role} - {description}\n"
    except Exception as e:
        print(f'Error occurred: {e}')
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset', type=str, default='pubmedqa')
        parser.add_argument('--model', type=str, default='gpt-4o-mini')
        args = parser.parse_args()

        process_intermediate_query(question, examplers, model, args)


    agent_dict = {}
    medical_agents = []
    for agent in agents_data:
        try:
            agent_role = agent[0].split('-')[0].split('.')[1].strip().lower()
            description = agent[0].split('-')[1].strip().lower()
        except:
            continue
        
        inst_prompt = f"""You are a {agent_role} who {description}. Your job is to collaborate with other medical experts in a team."""
        _agent = Agent(instruction=inst_prompt, role=agent_role, model_info=model)
        
        _agent.chat(inst_prompt)
        agent_dict[agent_role] = _agent
        medical_agents.append(_agent)

    for idx, agent in enumerate(agents_data):
        try:
            print(f"Agent {idx+1} ({agent_emoji[idx]} {agent[0].split('-')[0].strip()}): {agent[0].split('-')[1].strip()}")
        except:
            print(f"Agent {idx+1} ({agent_emoji[idx]}): {agent[0]}")

    fewshot_examplers = ""
    medical_agent = Agent(instruction='You are a helpful medical agent.', role='medical expert', model_info=model)
    if args.dataset == 'medqa':
        random.shuffle(examplers)
        for ie, exampler in enumerate(examplers[:5]):
            exampler_question = f"[Example {ie+1}]\n" + exampler['question']
            options = [f"({k}) {v}" for k, v in exampler['options'].items()]
            random.shuffle(options)
            exampler_question += " " + " ".join(options)
            exampler_answer = f"Answer: ({exampler['answer_idx']}) {exampler['answer']}"
            exampler_reason = tmp_agent.chat(f"Below is an example of medical knowledge question and answer. After reviewing the below medical question and answering, can you provide 1-2 sentences of reason that support the answer as you didn't know the answer ahead?\n\nQuestion: {exampler_question}\n\nAnswer: {exampler_answer}")
            
            exampler_question += f"\n{exampler_answer}\n{exampler_reason}\n\n"
            fewshot_examplers += exampler_question

    print()
    cprint("[INFO] Step 2. Collaborative Decision Making", 'yellow', attrs=['blink'])
    cprint("[INFO] Step 2.1. Hierarchy Selection", 'yellow', attrs=['blink'])
    print_tree(hierarchy_agents[0], horizontal=False)
    print()

    num_rounds = 1
    num_turns = 1
    num_agents = len(medical_agents)

    interaction_log = {f'Round {round_num}': {f'Turn {turn_num}': {f'Agent {source_agent_num}': {f'Agent {target_agent_num}': None for target_agent_num in range(1, num_agents + 1)} for source_agent_num in range(1, num_agents + 1)} for turn_num in range(1, num_turns + 1)} for round_num in range(1, num_rounds + 1)}

    cprint("[INFO] Step 2.2. Participatory Debate", 'yellow', attrs=['blink'])

    round_opinions = {n: {} for n in range(1, num_rounds+1)}
    round_answers = {n: None for n in range(1, num_rounds+1)}
    initial_report = ""
    for k, v in agent_dict.items():
        opinion = v.chat(f'''Given the examplers and context relevant to the question, please return your answer to the medical query among the option provided.\n\n{fewshot_examplers}\n\nQuestion: {question}\n\nContext: {contexts}\n\nYour answer should be like below format.\n\nAnswer: ''', img_path=None)
        initial_report += f"({k.lower()}): {opinion}\n"
        round_opinions[1][k.lower()] = opinion

    final_answer = None
    for n in range(1, num_rounds+1):
        print(f"== Round {n} ==")
        round_name = f"Round {n}"
        agent_rs = Agent(instruction="You are a medical assistant who excels at summarizing and synthesizing based on multiple experts from various domain experts.", role="medical assistant", model_info=model)
        agent_rs.chat("You are a medical assistant who excels at summarizing and synthesizing based on multiple experts from various domain experts.")
        
        assessment = "".join(f"({k.lower()}): {v}\n" for k, v in round_opinions[n].items())

        report = agent_rs.chat(f'''Here are some reports from different medical domain experts.\n\n{assessment}\n\nYou need to complete the following steps\n1. Take careful and comprehensive consideration of the following reports.\n2. Extract key knowledge from the following reports.\n3. Derive the comprehensive and summarized analysis based on the knowledge\n4. Your ultimate goal is to derive a refined and synthesized report based on the following reports.\n\nYou should output in exactly the same format as: Key Knowledge:; Total Analysis:''')
        
        for turn_num in range(num_turns):
            turn_name = f"Turn {turn_num + 1}"
            print(f"|_{turn_name}")

            num_yes = 0
            for idx, v in enumerate(medical_agents):
                all_comments = "".join(f"{_k} -> Agent {idx+1}: {_v[f'Agent {idx+1}']}\n" for _k, _v in interaction_log[round_name][turn_name].items())
                
                participate = v.chat("Given the opinions from other medical experts in your team, please indicate whether you want to talk to any expert (yes/no)\n\nOpinions:\n{}".format(assessment if n == 1 else all_comments))
                
                if 'yes' in participate.lower().strip():                
                    chosen_expert = v.chat(f"Enter the number of the expert you want to talk to:\n{agent_list}\nFor example, if you want to talk with Agent 1. Pediatrician, return just 1. If you want to talk with more than one expert, please return 1,2 and don't return the reasons.")
                    
                    chosen_experts = [int(ce) for ce in chosen_expert.replace('.', ',').split(',') if ce.strip().isdigit()]

                    for ce in chosen_experts:
                        specific_question = v.chat(f"Please remind your medical expertise and then leave your opinion to an expert you chose (Agent {ce}. {medical_agents[ce-1].role}). You should deliver your opinion once you are confident enough and in a way to convince other expert with a short reason.")
                        
                        print(f" Agent {idx+1} ({agent_emoji[idx]} {medical_agents[idx].role}) -> Agent {ce} ({agent_emoji[ce-1]} {medical_agents[ce-1].role}) : {specific_question}")
                        interaction_log[round_name][turn_name][f'Agent {idx+1}'][f'Agent {ce}'] = specific_question
                
                    num_yes += 1
                else:
                    print(f" Agent {idx+1} ({agent_emoji[idx]} {v.role}): \U0001f910")

            if num_yes == 0:
                break
        
        if num_yes == 0:
            break

        tmp_final_answer = {}
        for i, agent in enumerate(medical_agents):
            response = agent.chat(f"Now that you've interacted with other medical experts, remind your expertise and the comments from other experts and make your final answer to the given question:\n{question}\nAnswer: ")
            tmp_final_answer[agent.role] = response

        round_answers[round_name] = tmp_final_answer
        final_answer = tmp_final_answer

    print('\nInteraction Log')        
    myTable = PrettyTable([''] + [f"Agent {i+1} ({agent_emoji[i]})" for i in range(len(medical_agents))])

    for i in range(1, len(medical_agents)+1):
        row = [f"Agent {i} ({agent_emoji[i-1]})"]
        for j in range(1, len(medical_agents)+1):
            if i == j:
                row.append('')
            else:
                i2j = any(interaction_log[f'Round {k}'][f'Turn {l}'][f'Agent {i}'][f'Agent {j}'] is not None
                          for k in range(1, len(interaction_log)+1)
                          for l in range(1, len(interaction_log['Round 1'])+1))
                j2i = any(interaction_log[f'Round {k}'][f'Turn {l}'][f'Agent {j}'][f'Agent {i}'] is not None
                          for k in range(1, len(interaction_log)+1)
                          for l in range(1, len(interaction_log['Round 1'])+1))
                
                if not i2j and not j2i:
                    row.append(' ')
                elif i2j and not j2i:
                    row.append(f'\u270B ({i}->{j})')
                elif j2i and not i2j:
                    row.append(f'\u270B ({i}<-{j})')
                elif i2j and j2i:
                    row.append(f'\u270B ({i}<->{j})')

        myTable.add_row(row)
        if i != len(medical_agents):
            myTable.add_row(['' for _ in range(len(medical_agents)+1)])
    
    print(myTable)

    cprint("\n[INFO] Step 3. Final Decision", 'yellow', attrs=['blink'])
    
    moderator = Agent("You are a final medical decision maker who reviews all opinions from different medical experts and make final decision.", "Moderator", model_info=model)
    moderator.chat('You are a final medical decision maker who reviews all opinions from different medical experts and make final decision.')
    
    _decision = moderator.temp_responses(f"Given each agent's final answer, please review each agent's opinion and make the final answer to the question by taking majority vote. Your answer should be like below format:\nAnswer: C) 2th pharyngeal arch\n{final_answer}\n\nQuestion: {question}", img_path=None)
    final_decision = {'majority': _decision}
    emoji_ = '\U0001F468\u200D\u2696\uFE0F'
    print(f"{emoji_} moderator's final decision (by majority vote):", _decision)
    print()

    return final_decision

def parse_medical_case(text):
    # Split into question and options
    parts = text.split('Options:')
    question = parts[0].strip()
    
    # Parse options
    options = parts[1].strip()
    
    return question, options

def form_context_prompt(dict_list):

    context_text = ''
    for d in dict_list:
        context_text = context_text + 'Id : ' + d['id'] + '\nTitle : ' + d['title'] + '\nContent : ' + d['content'] + '\nContents : ' + d['contents'] + '\n\n'

    return context_text
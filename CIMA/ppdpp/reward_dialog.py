from ppdpp.utils import *
from ppdpp.prompt import *
import logging, time
import argparse, json
from tqdm import tqdm
import numpy as np
from transformers import BertTokenizer, RobertaTokenizer, BertConfig, RobertaConfig
from fastchat.model import add_model_args
from openai import OpenAI
from collections import defaultdict as ddict
from ppdpp.utils import openai_keys

tok = {'bert': BertTokenizer, 'roberta': RobertaTokenizer}
cfg = {'bert': BertConfig, 'roberta': RobertaConfig}

system_role = {'esc':'Therapist', 'cima': 'Teacher', 'cb': 'Buyer'}
user_role = {'esc':'Patient', 'cima': 'Student', 'cb': 'Seller'}
message_format = {'esc': ESConvMessages, 'cima': CIMAMessages, 'cb': CBMessages}


def read_data(args, path):
    samples = []
    with open(path, 'r', encoding='utf-8') as infile:
        if args.data_name in ['esc','cb','cima']:
            for line in infile:
                sample = json.loads(line.strip('\n'))
                samples.append(sample)
    return samples


def write_data(args, path, sample):
    with open(path, 'a+', encoding='utf-8') as out:
        if args.data_name in ['esc','cb', 'cima']:
            json.dump(sample, out)
            out.write('\n')


def reset(args, case):
    if args.data_name == 'esc':
        conversation = [{"role":"Patient", "content": case['situation']}]
    elif args.data_name == 'cima':
        conversation = [{"role":"Teacher", "content": case['dialog'][0]['text']}, 
                        {"role":"Student", "content": case['dialog'][1]['text']}]
    elif args.data_name == 'cb':
        conversation = [{"role":"Buyer", "content":"Hi, how much is the %s?" % case['item_name']}, 
                        {"role":"Seller", "content":"Hi, this is a good %s and its price is %s." % (case['item_name'], case['seller_price'])}]
    return conversation


def clean_dialog(case):
    new_case = {"emotion_type": case['emotion_type'], "problem_type": case['problem_type'], "situation": case['situation']}
    dial = case['dialog']
    new_dial = []
    sys_uttr, usr_uttr = [], []
    for turn in dial:
        if turn['speaker'] == 'sys':
            sys_uttr.append(turn['text'])
            if len(usr_uttr) > 0:
                new_dial.append({'speaker': 'usr', 'text': ' '.join(usr_uttr)})
                usr_uttr = []
        else:
            usr_uttr.append(turn['text'])
            if len(sys_uttr) > 0:
                new_dial.append({'speaker': 'sys', 'text': ' '.join(sys_uttr)})
                sys_uttr = []
    if dial[-1]['speaker'] == 'sys':
        if len(sys_uttr) > 0:
                sys_uttr.append({'speaker': 'sys', 'text': ' '.join(sys_uttr)})
    else:
        if len(usr_uttr) > 0:
                new_dial.append({'speaker': 'usr', 'text': ' '.join(usr_uttr)})
    new_case['dialog'] = new_dial
    return new_case


def apply_chatgpt(args, messages, return_num):
    messages = chatgpt_prompt(messages, user_role[args.data_name])
    comments = query_openai_model(
        messages=messages,
        model="gpt-3.5-turbo-0613",
        max_tokens=args.max_new_tokens,
        temperature=1.1,
        n=return_num
    )

    return comments


def query_openai_model(messages: str, model: str = "gpt-3.5-turbo-0613", max_tokens: int = 128, temperature: float = 0, n: int = 1):
    client = OpenAI(api_key='xxxx')
    flag = True
    while flag:
        try:
            completions = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                n=n,
                stop=None,
                temperature=temperature,
            )
            if n == 1:
                output = completions.choices[0].message.content.strip()
            else:
                output = []
                for choice in completions.choices:
                    output.append(choice.message.content.strip())

            flag = False
        except Exception as e:
            print("Some error happened here.")
            time.sleep(5)
    return output


def map_reward(data_name, comments):
    rewards = []
    for output in comments:
        for key in reward_dict[data_name]:
            if key in output.lower():
                rewards.append(reward_dict[data_name][key])
                break
    if len(rewards) == 0:
        reward = 0
    else:
        reward = sum(rewards)/len(rewards)
    return reward


def calculate_reward(args, dataset, mode):
    path = os.path.join(args.data_dir, '{}-{}.txt'.format(args.data_name, mode))
    samples = read_data(args, path)[args.start_case_ind: args.start_case_ind + args.case_num]
    
    for idx, case in enumerate(samples):
        keys_weight = [10.0 for _ in range(len(openai_keys))]
        # case = clean_dialog(sample)
        dials = case['dialog']
        print('\n================new tuple:{}===================='.format(idx + args.start_case_ind))
        state = []
        for dia_idx, dial in tqdm(enumerate(dials), desc='Processing turns: '):
            if dial['speaker'] == 'sys':
                state.append({"role": system_role[dataset], "content": dial['text']})
                continue
            elif dial['speaker'] == 'usr':
                state.append({"role": user_role[dataset], "content": dial['text']})
                continue
        
        pre_critic_messages = message_format[dataset](case, 'critic', state)
        pre_comments = apply_chatgpt(args, pre_critic_messages, 10)
        samples[idx]['previous state'] = pre_comments
        
        sys_messages = message_format[dataset](case, 'system', state, action=case['strategy'])
        sys_uttr = apply_chatgpt(args, sys_messages, 1)
        state.append({"role": system_role[dataset], "content": sys_uttr})
        usr_messages = message_format[dataset](case, 'user', state)
        usr_uttr = apply_chatgpt(args, usr_messages, 1)
        state.append({"role": user_role[dataset], "content": usr_uttr})
        critic_messages = message_format[dataset](case, 'critic', state)
        comments = apply_chatgpt(args, critic_messages, 10)
        samples[idx]['state'] = comments
        samples[idx]['sys_resp'] = sys_uttr
        samples[idx]['usr_resp'] = usr_uttr
        
        samples[idx]['done'] = '0'
        reward = map_reward(args.data_name, comments)
        if reward == 0.5 or len(samples[idx]['dialog']) // 2 + 1 >= 8:
            samples[idx]['done'] = '1'
        
        output_path = os.path.join(args.data_dir, '{}-{}-chatgpt-state-0613.txt'.format(args.data_name, mode))
        write_data(args, output_path, samples[idx])
        
        # print('valid keys: ')
        # print(openai_keys)


def calculate_reward2(args, dataset, mode):
    path = os.path.join(args.data_dir, '{}-{}-chatgpt-state.txt'.format(args.data_name, mode))
    samples = read_data(args, path)[args.start_case_ind: args.start_case_ind + args.case_num]
    
    for idx, case in enumerate(samples):
        keys_weight = [10.0 for _ in range(len(openai_keys))]
        # case = clean_dialog(sample)
        dials = case['dialog']
        print('\n================new tuple:{}===================='.format(idx + args.start_case_ind))
        state = []
        for dia_idx, dial in tqdm(enumerate(dials), desc='Processing turns: '):
            if dial['speaker'] == 'sys':
                state.append({"role": system_role[dataset], "content": dial['text']})
                continue
            elif dial['speaker'] == 'usr':
                state.append({"role": user_role[dataset], "content": dial['text']})
                continue
        
        pre_critic_messages = message_format[dataset](case, 'critic', state)
        pre_comments = apply_chatgpt(args, pre_critic_messages, 10)
        
        samples[idx]['previous state'] = pre_comments
        
        output_path = os.path.join(args.data_dir, '{}-{}-chatgpt-state-0613.txt'.format(args.data_name, mode))
        write_data(args, output_path, samples[idx])
        
        # print('valid keys: ')
        # print(openai_keys)


def judge_done(args, mode):
    path = os.path.join(args.data_dir, '{}-{}-chatgpt-state-0613.txt'.format(args.data_name, mode))
    samples = read_data(args, path)[args.start_case_ind: args.start_case_ind + args.case_num]
    
    for idx, case in enumerate(samples):
        comments = samples[idx]['state']
        reward = map_reward(args.data_name, comments)
        samples[idx]['done'] = '0'
        if reward == 0.5 or len(samples[idx]['dialog']) // 2 + 1 >= 8:
            samples[idx]['done'] = '1'
    
        output_path = os.path.join(args.data_dir, '{}-{}-chatgpt-state.txt'.format(args.data_name, mode))
        write_data(args, output_path, samples[idx])


def calculate_init_state_reward(args, dataset, mode):
    path = os.path.join(args.data_dir, '{}-{}-chatgpt-state-0613.txt'.format(args.data_name, mode))
    samples = read_data(args, path)[args.start_case_ind: args.start_case_ind + args.case_num]
    
    for idx, case in enumerate(samples):
        keys_weight = [10.0 for _ in range(len(openai_keys))]
        # case = clean_dialog(sample)
        dials = case['dialog']
        print('\n================new tuple:{}===================='.format(idx + args.start_case_ind))
        state = []
        for dia_idx, dial in tqdm(enumerate(dials), desc='Processing turns: '):
            if dial['speaker'] == 'sys':
                state.append({"role": system_role[dataset], "content": dial['text']})
            elif dial['speaker'] == 'usr':
                state.append({"role": user_role[dataset], "content": dial['text']})
            if len(state) >= 2:
                break
        
        assert state[0]['role'] == 'Teacher'
        assert state[1]['role'] == 'Student'
        
        pre_critic_messages = message_format[dataset](case, 'critic', state)
        pre_comments = apply_chatgpt(args, pre_critic_messages, 10)
        
        samples[idx]['init state'] = pre_comments
        
        output_path = os.path.join(args.data_dir, '{}-{}-chatgpt-state-0613-init.txt'.format(args.data_name, mode))
        write_data(args, output_path, samples[idx])


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default='../data', type=str,
                        help="The data directory.")
    parser.add_argument('--data_name', type=str, default='esc', choices=['esc','cima','cb'],
                        help='One of {esc, cima, cb}.')
    parser.add_argument('--mode', default='train', type=str)
    
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--start_case_ind", type=int, default=0)
    parser.add_argument("--case_num", type=int, default=2)
    parser.add_argument("--return_num", type=int, default=10)

    add_model_args(parser)
    args = parser.parse_args()
    
    return args


def main(args):
    #os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # calculate_reward(args, args.data_name, args.mode)
    # calculate_reward2(args, args.data_name, args.mode)
    calculate_init_state_reward(args, args.data_name, args.mode)
    # judge_done(args, args.mode)


if __name__ == '__main__':
    main(parse_args())
import os
import logging
import torch
import pickle
from ppdpp.prompt import ESConvAct, CIMAAct, CBAct
import json
import numpy as np
from ppdpp.utils import reward_dict
from ppdpp.utils import map_reward

logger = logging.getLogger(__name__)

role_map = {'esc': {'sys': 'Therapist', 'usr': 'Patient'}, 'cima': {'sys': 'Teacher', 'usr': 'Student'}, 'cb': {'sys': 'Buyer', 'usr': 'Seller'}}
act_map = {'esc': ESConvAct, 'cima': CIMAAct, 'cb': CBAct}

def write_pkl(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def read_pkl(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def load_and_cache_examples(args, tokenizer, evaluate=False):
    mode = args.set_name if evaluate else 'train'
    print(mode)
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'sft_{}_{}_{}_{}'.format(
        args.data_name,
        mode,
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length)))

    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = read_pkl(cached_features_file)
        print("Loaded number of instance:", len(features['state_ids']))
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        features = convert_to_features(args, tokenizer, mode)
        print("Loaded number of instance:", len(features['state_ids']))
    
        logger.info("Saving features into cached file %s", cached_features_file)
        write_pkl(features, cached_features_file)
    return features

def map_esconv_resp_to_reward(resps):
    rewards = []
    for resp in resps:
        resp = resp.lower()
        if 'no' in resp and 'worse' in resp:
            rewards.append(reward_dict['esc']['worse'])
        elif 'no' in resp and 'better' in resp:
            rewards.append(reward_dict['esc']['better'])
        elif 'yes' in resp and 'solved' in resp:
            rewards.append(reward_dict['esc']['solved'])
        else:
            rewards.append(reward_dict['esc']['same'])
    return np.mean(rewards)

def map_cima_resp_to_reward(resps):
    rewards = []
    for resp in resps:
        resp = resp.lower()
        if 'no' in resp and 'incorrect' in resp:
            rewards.append(-1.0)
        elif 'no' in resp and 'a part of' in resp:
            rewards.append(0.1)
        elif 'yes' in resp and 'correctly' in resp:
            rewards.append(1.0)
        else:
            rewards.append(-0.5)
    return np.mean(rewards)

def convert_to_features(args, tokenizer, mode):
    def get_state_ids(state):
        dial_id = []
        for s in state[::-1]:
            if len(dial_id) + len(s) > args.max_seq_length:
                break
            dial_id = s[1:] + dial_id
        source_id = s[:1] + dial_id
        return source_id
    
    reward_map = {'esc': map_esconv_resp_to_reward, 'cima': map_cima_resp_to_reward}
    path = os.path.join(args.data_dir, '{}-{}-chatgpt-state-0613.txt'.format(args.data_name, mode))
    act = sorted(list(act_map[args.data_name].keys()))
    print('tokenizing {}'.format(path))
    with open(path, 'r', encoding='utf-8') as infile:
        max_dia_len = 0
        avg_dia_len = []
        batch_state_ids = []
        batch_next_state_ids = []
        actions = []
        rewards = []
        dones = []
        target_qvs = []
        
        for idx, line in enumerate(infile):
            sample = eval(line.strip('\n'))
            dial = sample['dialog']

            target_id = act.index(sample['strategy'])
            dial_id = []
            for s in dial:
                s = tokenizer.encode("%s: %s" % (role_map[args.data_name][s['speaker']], s['text']))
                dial_id += s[1:]
            source_id = s[:1] + dial_id
            batch_state_ids.append(source_id[-args.max_seq_length+1:])
            actions.append(target_id)
            
            sys_resp = sample['sys_resp'].encode('utf-8', 'ignore').decode('utf-8')
            next_sys = tokenizer.encode("%s: %s" % (role_map[args.data_name]['sys'], sys_resp))
            dial_id += next_sys[1:]
            
            usr_resp = sample['usr_resp'].encode('utf-8', 'ignore').decode('utf-8')
            next_uttr = tokenizer.encode("%s: %s" % (role_map[args.data_name]['usr'], usr_resp))
            dial_id += next_uttr[1:]
            
            source_id = next_uttr[:1] + dial_id
            batch_next_state_ids.append(source_id[-args.max_seq_length+1:])
            
            dones.append(float(sample['done']))
            rewards.append(map_reward('cima', sample['state']))
            
            avg_dia_len.append(len(source_id))
            max_dia_len = max(max_dia_len, len(source_id))

        print('{} set, max_dia_len: {}, avg_dia_len: {}'.format(mode, max_dia_len, float(sum(avg_dia_len))/len(avg_dia_len)))
    
    return {'state_ids': batch_state_ids, 'next_state_ids': batch_next_state_ids, 'actions': actions, 
            'rewards': rewards, 'dones': dones}

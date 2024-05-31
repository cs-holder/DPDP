import os, re
import logging
import torch
import pickle
from ppdpp.prompt import ESConvAct, CIMAAct, CBAct
import json
import numpy as np
from ppdpp.utils import reward_dict

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
    cached_features_file = os.path.join(args.data_dir, 'sft_{}_{}_{}_{}_{}'.format(
        args.data_name,
        mode,
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        args.neg_reward))

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

def map_esconv_resp_to_reward(case, resps):
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

def map_cima_resp_to_reward(case, resps):
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

def map_cb_resp_to_reward(args, case, resps):
    deals, rewards = [], []
    for resp in resps:
        if 'have not' in resp.lower():
            deals.append(-1)
        elif 'have reached' in resp.lower():
            deals.append(1)
        
        prices = re.findall(r"[-+]?\d*\.?\d+", resp.replace(",",""))
        if len(prices) > 0:
            deal_price = float(prices[0])
            reward = (deal_price - case['seller_price']) / (case['buyer_price'] - case['seller_price'])
            rewards.append(reward)

    if -1 in deals:
        reward = args.neg_reward
    else:
        if len(rewards) == 0:
            reward = 0
        else:
            reward = max(set(rewards), key = rewards.count)
    return reward

def convert_to_features(args, tokenizer, mode):
    def get_state_ids(state):
        dial_id = []
        for s in state[::-1]:
            if len(dial_id) + len(s) > args.max_seq_length:
                break
            dial_id = s[1:] + dial_id
        source_id = s[:1] + dial_id
        return source_id
    
    mid_reward_map = {'esc': -0.5, 'cb': -1.0}
    reward_map = {'esc': map_esconv_resp_to_reward, 'cima': map_cima_resp_to_reward, 'cb':map_cb_resp_to_reward}
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
        
        if args.data_name in ['esc','cb']:
            for line in infile:
                sample = json.loads(line.strip('\n'))
                dial = sample['dialog']
                if len(dial) <= 3:
                    continue
                state_ids, next_state_ids, state, usr_state, actions_, rewards_, dones_, target_qvs_ = [], [], [], [], [], [], [], []
                new_dial = [dial[0]]
                last_turn = dial[0]
                for turn in dial[1:]:
                    if turn['speaker'] == last_turn['speaker']:
                        if turn['speaker'] == 'sys':
                            new_dial[-1]['strategy'] = turn['strategy']
                            new_dial[-1]['text'] = " ".join([new_dial[-1]['text'], turn['text']])
                        else:
                            new_dial[-1]['state'] = turn['state']
                            new_dial[-1]['text'] = " ".join([new_dial[-1]['text'], turn['text']])
                    else:
                        new_dial.append(turn)
                    last_turn = turn

                for idx, turn in enumerate(new_dial):
                    if turn['speaker'] == 'sys' and len(state) > 0:
                        source_id = get_state_ids(state)
                        state_ids.append(source_id[-args.max_seq_length+1:])
                        if idx > 2:
                            if len(usr_state) > 0:
                                reward = reward_map[args.data_name](args, sample, usr_state)
                            else:
                                reward = mid_reward_map[args.data_name]
                            target_qvs_.append(reward)
                            rewards_.append(reward)
                            dones_.append(0.0)
                        target_id = act.index(turn['strategy'])
                        actions_.append(target_id)
                        avg_dia_len.append(len(source_id))
                        max_dia_len = max(max_dia_len, len(source_id))
                    if turn['speaker'] == 'usr':
                        usr_state = turn['state']
                    state.append(tokenizer.encode("%s: %s" % (role_map[args.data_name][turn['speaker']], turn['text'])))
                if new_dial[-1]['speaker'] == 'sys':
                    next_state_ids = state_ids[1:]                    
                    state_ids = state_ids[:-1]
                    actions_ = actions_[:-1]
                    # rewards_ = rewards_[1:]
                    # target_qvs_ = target_qvs_[1:]
                    # dones_[-1] = 1.0
                else:
                    source_id = get_state_ids(state)
                    next_state_ids = state_ids[1:] + [source_id[-args.max_seq_length+1:]]
                    if len(usr_state) > 0:
                        reward = reward_map[args.data_name](args, sample, usr_state)
                    else:
                        reward = mid_reward_map[args.data_name]
                    rewards_.append(reward)
                    target_qvs_.append(reward)
                    dones_.append(1.0)
                # if rewards_[-1] > 0.1:
                #     rewards_[-1] += 1.0
                #     target_qvs_[-1] += 1.0
                for idx in range(len(target_qvs_) - 2, -1, -1):
                    target_qvs_[idx] += args.gamma * target_qvs_[idx + 1]
                assert len(state_ids) == len(next_state_ids) == len(target_qvs_) == len(rewards_) == len(dones_) == len(actions_)
                batch_state_ids.extend(state_ids)
                batch_next_state_ids.extend(next_state_ids)
                target_qvs.extend(target_qvs_)
                actions.extend(actions_)
                rewards.extend(rewards_)
                dones.extend(dones_)
        elif args.data_name == 'cima':
            for line in infile:
                sample = eval(line.strip('\n'))
                dial = sample['dialog']
                state, usr_state, rewards_, dones_, target_qvs_ = [], [], [], []

                target_id = act.index(sample['strategy'])
                dial_id = []
                for s in dial:
                    s = tokenizer.encode("%s: %s" % (role_map[args.data_name][s['speaker']], s['text']))
                    dial_id += s[1:]
                source_id = s[:1] + dial_id
                state_ids.append(source_id[-args.max_seq_length+1:])
                actions.append(target_id)
                avg_dia_len.append(len(source_id))
                max_dia_len = max(max_dia_len, len(source_id))

        print('{} set, max_dia_len: {}, avg_dia_len: {}'.format(mode, max_dia_len, float(sum(avg_dia_len))/len(avg_dia_len)))
    
    return {'state_ids': batch_state_ids, 'next_state_ids': batch_next_state_ids, 'actions': actions, 
            'rewards': rewards, 'dones': dones, 'target_qvs': target_qvs}

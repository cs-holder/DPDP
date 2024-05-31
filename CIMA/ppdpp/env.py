import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from fastchat.model import load_model, get_conversation_template
from fastchat.modules.gptq import GptqConfig
from fastchat.modules.awq import AWQConfig

from openai import OpenAI
import requests
from collections import defaultdict as ddict

from ppdpp.utils import *
from ppdpp.prompt import *
from mcts.core.helpers import EmotionSupportDialogSession, CIMADialogSession
from mcts.core.game import EmotionalSupportGame, CIMAGame
from ppdpp.utils import sample_openai_key, openai_keys, map_reward
from ppdpp.utils import reward_dict
#from unidecode import unidecode
import nltk
import re
import time
import json

system_role = {'esc':'Therapist', 'cima': 'Teacher', 'cb': 'Buyer'}
user_role = {'esc':'Patient', 'cima': 'Student', 'cb': 'Seller'}
message_format = {'esc': ESConvMessages, 'cima': CIMAMessages, 'cb': CBMessages}


class Env(object):
    def __init__(self, args, dataset, mode, env_model=None, env_tokenizer=None):
        if 'vicuna' in [args.system, args.user, args.critic] or 'llama2' in [args.system, args.user, args.critic]:
            if mode == 'train':
                gptq_config=GptqConfig(
                    ckpt=args.model_path,
                    wbits=16,
                    groupsize=-1,
                    act_order=False,
                )
                awq_config=AWQConfig(
                    ckpt=args.model_path,
                    wbits=16,
                    groupsize=-1,
                )
                self.vicuna_model, self.vicuna_tokenizer = load_model(
                    args.model_path,
                    args.raw_device,
                    args.num_gpus,
                    args.max_gpu_memory,
                    dtype=None,
                    load_8bit=False,
                    cpu_offloading=False,
                    gptq_config=gptq_config,
                    awq_config=awq_config,
                    exllama_config=None,
                    xft_config=None,
                    revision='main',
                    debug=args.debug,
                )
            else:
                self.vicuna_model = env_model
                self.vicuna_tokenizer = env_tokenizer
        
        
        self.args = args
        self.dataset = dataset[mode]
        self.max_turn = args.max_turn
        self.conversation = []
        self.cur_conver_step = 0
        self.test_num = 0
        self.mode = mode

        self.reward_dict = reward_dict
        self.success_base = args.success_base

        set_random_seed(args.seed)
        
        self.logger = args.logger
        self.apply_chatgpt_times = 0.0
        
    def reset(self):
        self.cur_conver_step = 0
        if self.mode == 'train':
            self.case = np.random.choice(self.dataset)
        else:
            self.case = self.dataset[self.test_num]
            self.test_num += 1
        
        self.conversation = [{"role":"Teacher", "content":self.case['dialog'][0]['text']}, 
                                {"role":"Student", "content":self.case['dialog'][1]['text']}]
        reward = map_reward('cima', self.case['init state'])
        if reward == self.reward_dict['cima']['whole']:
            usr_act = CIMAGame.U_Correct
        elif reward >= (self.reward_dict['cima']['part'] + self.reward_dict['cima']['did not']) / 2:
            usr_act = CIMAGame.U_OnlyPart
        elif reward >= (self.reward_dict['cima']['incorrect'] + self.reward_dict['cima']['did not']) / 2:
            usr_act = CIMAGame.U_DidNotTry
        else:
            usr_act = CIMAGame.U_Incorrect
        history = [(system_role[self.args.data_name], CIMAGame.S_Others, self.case['dialog'][0]['text']), 
                (user_role[self.args.data_name], usr_act, self.case['dialog'][1]['text'])]
        mcts_state = CIMADialogSession(system_role[self.args.data_name], user_role[self.args.data_name], 
                                        self.case['sentence'], self.case['target'], history=history)
        self.logger.info(json.dumps(self.conversation))
        
        return self.conversation, mcts_state, reward

    def step(self, action, mcts_state, reward, use_mcts):
        done = 0
        self.logger.info('---------------step:{}-------------'.format(self.cur_conver_step))
        
        self.logger.info(action)
        if not use_mcts or not self.args.use_mcts_sys_resp:
            messages = message_format[self.args.data_name](self.case, 'system', self.conversation, action)
            response = self.generate_response(self.args.system, messages, system_role[self.args.data_name])
            response = self.postprocess_response(response, user_role[self.args.data_name])
            if use_mcts and not self.args.use_mcts_sys_resp:
                mcts_state[-2] = [system_role[self.args.data_name], action, response]
        else:
            response = mcts_state[-2][-1]
        self.conversation.append({"role":system_role[self.args.data_name],"content":response})
        if not use_mcts:
            mcts_state.add_single(system_role[self.args.data_name], action, response)
        self.logger.info(json.dumps(self.conversation[-1]))

        if not use_mcts or not self.args.use_mcts_usr_resp:
            messages = message_format[self.args.data_name](self.case, 'user', self.conversation)
            user_response = self.generate_response(self.args.user, messages, user_role[self.args.data_name])
            user_response = self.postprocess_response(user_response, user_role[self.args.data_name])
        else:
            user_response = mcts_state[-1][-1]
        self.conversation.append({"role":user_role[self.args.data_name], "content":user_response})
        self.logger.info(json.dumps(self.conversation[-1]))

        if reward is None:          # reward 和 use_mcts 是同步的，reward=None 说明没有使用 mcts
            messages = message_format[self.args.data_name](self.case, 'critic', self.conversation)
            reward, user_action = self.compute_reward(self.args.critic, messages, self.case)
            # reward, _ = self.compute_reward(self.args.critic, messages, self.case)
            # if mcts_state is not None:
            #     mcts_state[-1] = (user_role[self.args.data_name], user_action, user_response)
            if not use_mcts:
                mcts_state.add_single(user_role[self.args.data_name], user_action, user_response)

        if self.args.data_name == 'esc':
            # if reward > 0.1:
            if reward > self.success_base:
                self.logger.info('--> Goal completed !')
                done = 1
            else:
                if self.cur_conver_step == self.max_turn - 1:
                    self.logger.info('--> Maximum number of turns reached !')
                    done = -1
                else:
                    self.logger.info('--> On-going !')
        elif self.args.data_name == 'cima':
            # if reward == 1.0:
            if reward >= self.success_base:
                self.logger.info('--> Goal completed !')
                done = 1
            else:
                if self.cur_conver_step == self.max_turn - 1:
                    self.logger.info('--> Maximum number of turns reached !')
                    done = -1
                else:
                    self.logger.info('--> On-going !')
        elif self.args.data_name == 'cb':
            if reward >= 0:
                self.logger.info('--> Goal completed !')
                done = 1
            else:
                if self.cur_conver_step == self.max_turn - 1:
                    self.logger.info('--> Maximum number of turns reached !')
                    done = -1
                else:
                    self.logger.info('--> On-going !')
                
        self.cur_conver_step += 1
        return self.conversation, reward, done
    
    def unfold_mcts_state(self, mcts_state, full_mcts_history, mcts_turn):
        done = 0
        self.logger.info('---------------step:{}-------------'.format(self.cur_conver_step))
        
        full_mcts_state, full_mcts_rewards = full_mcts_history['state'], full_mcts_history['rewards']
        start_pos = len(self.conversation)
        assert full_mcts_state[start_pos][0] == system_role[self.args.data_name]
        self.conversation.append({"role": system_role[self.args.data_name], "content": full_mcts_state[start_pos][-1]})
        self.logger.info(json.dumps(self.conversation[-1]))
        
        assert full_mcts_state[start_pos + 1][0] == user_role[self.args.data_name]
        self.conversation.append({"role": user_role[self.args.data_name], "content": full_mcts_state[start_pos + 1][-1]})
        self.logger.info(json.dumps(self.conversation[-1]))
        
        mcts_state.history = full_mcts_state
        
        try:
            reward = full_mcts_rewards[mcts_turn]
        except:
            print('full_mcts_state: ')
            print(full_mcts_state)
            print('full_mcts_rewards: ')
            print(full_mcts_rewards)
            raise ValueError
        
        if self.args.data_name == 'esc':
            # if reward > 0.1:
            if reward > self.success_base:
                self.logger.info('--> Goal completed !')
                done = 1
            else:
                if self.cur_conver_step == self.max_turn - 1:
                    self.logger.info('--> Maximum number of turns reached !')
                    done = -1
                else:
                    self.logger.info('--> On-going !')
        elif self.args.data_name == 'cima':
            # if reward == 1.0:
            if reward >= self.success_base:
                self.logger.info('--> Goal completed !')
                done = 1
            else:
                if self.cur_conver_step == self.max_turn - 1:
                    self.logger.info('--> Maximum number of turns reached !')
                    done = -1
                else:
                    self.logger.info('--> On-going !')
        elif self.args.data_name == 'cb':
            if reward >= 0:
                self.logger.info('--> Goal completed !')
                done = 1
            else:
                if self.cur_conver_step == self.max_turn - 1:
                    self.logger.info('--> Maximum number of turns reached !')
                    done = -1
                else:
                    self.logger.info('--> On-going !')
        
        self.cur_conver_step += 1
        return self.conversation, reward, done
    
    def postprocess_response(self, response, role):
        #print(response)
        if role in response:
            response = response.split(role)[0].strip()
        sents = nltk.sent_tokenize(response)
        if len(sents) == 1:
            if response[-1] not in ['.','!','?',':']:
                return response + '.'
            return response.strip()
        try:
            if sents[-1].strip()[-1] not in ['.','!','?',':']:
                return ' '.join(sents[:-1]).strip()
            else:
                return response.strip()
        except Exception as e:
            return response.strip()

    def generate_response(self, model, messages, role):
        if self.mode == 'test':
            temperature = 0
        else:
            temperature = 0.7
        if model == 'vicuna':
            prompt = vicuna_prompt(messages, role)
            #print(prompt)
            input_ids = self.vicuna_tokenizer([prompt]).input_ids
            #print(len(input_ids[0]))
            max_new_tokens = self.args.max_new_tokens
            output_ids = self.vicuna_model.generate(
                torch.as_tensor(input_ids).cuda(),
                max_new_tokens=max_new_tokens,
                temperature = temperature,
                early_stopping=True
            )
            output_ids = output_ids[0][len(input_ids[0]):]
            output = self.vicuna_tokenizer.decode(output_ids, skip_special_tokens=True,
                                    spaces_between_special_tokens=False)
        elif model == 'llama2':
            prompt = llama2_prompt(messages, role)
            #print(prompt)
            input_ids = self.vicuna_tokenizer([prompt]).input_ids
            #print(len(input_ids[0]))
            max_new_tokens = self.args.max_new_tokens
            output_ids = self.vicuna_model.generate(
                torch.as_tensor(input_ids).cuda(),
                max_new_tokens=max_new_tokens,
                temperature = temperature,
                early_stopping=True
            )
            output_ids = output_ids[0][len(input_ids[0]):]
            output = self.vicuna_tokenizer.decode(output_ids, skip_special_tokens=True,
                                    spaces_between_special_tokens=False)
        elif model == 'chatgpt':
            messages = chatgpt_prompt(messages, role)
            #print(messages)
            output = query_openai_model(
                messages=messages,
                model="gpt-3.5-turbo-0613",
                max_tokens=self.args.resp_max_new_tokens,
                temperature=temperature
            )
            self.apply_chatgpt_times += 1
        return output
    
    def compute_reward(self, model, messages, case):
        if model == 'vicuna':
            prompt = vicuna_prompt(messages, 'critic')
            #print(prompt)
            input_ids = self.vicuna_tokenizer([prompt]).input_ids
            output_ids = self.vicuna_model.generate(
                torch.as_tensor(input_ids).cuda(),
                max_new_tokens=16,
                temperature = 1.1,
                do_sample = True,
                early_stopping=True,
                num_return_sequences=10,
            )
            outputs = []
            for o in output_ids:
                output_id = o[len(input_ids[0]):]
                output = self.vicuna_tokenizer.decode(output_id, skip_special_tokens=True,
                                    spaces_between_special_tokens=False)
                outputs.append(output)
        elif model == 'llama2':
            prompt = llama2_prompt(messages, 'critic')
            #print(prompt)
            input_ids = self.vicuna_tokenizer([prompt]).input_ids
            output_ids = self.vicuna_model.generate(
                torch.as_tensor(input_ids).cuda(),
                max_new_tokens=16,
                temperature = 1.1,
                do_sample = True,
                early_stopping=True,
                num_return_sequences=10,
            )
            outputs = []
            for o in output_ids:
                output_id = o[len(input_ids[0]):]
                output = self.vicuna_tokenizer.decode(output_id, skip_special_tokens=True,
                                    spaces_between_special_tokens=False)
                outputs.append(output)
        elif model == 'chatgpt':
            messages = chatgpt_prompt(messages, user_role[self.args.data_name])
            outputs = query_openai_model(
                messages=messages,
                model="gpt-3.5-turbo-0613",
                max_tokens=self.args.reward_max_new_tokens,
                temperature=1.1,
                n=10
            )
            self.apply_chatgpt_times += 1
        
        if self.args.data_name in ['esc','cima']:
            rewards, user_actions = [], ddict(int)
            self.logger.info("[{}]".format(", ".join(outputs)))
            for output in outputs:
                for key in self.reward_dict[self.args.data_name]:
                    if key in output.lower():
                        rewards.append(self.reward_dict[self.args.data_name][key])
                        user_actions[key] += 1
                        break
            if len(rewards) == 0:
                reward = 0
            else:
                reward = sum(rewards)/len(rewards)
            self.logger.info(str(reward))
        elif self.args.data_name == 'cb':
            deals = []
            rewards, user_actions = [], []
            self.logger.info("[{}]".format(", ".join(outputs)))
            for output in outputs:
                if 'have not' in output.lower():
                    deals.append(-1)
                elif 'have reached' in output.lower():
                    deals.append(1)
                
                prices = re.findall(r"[-+]?\d*\.?\d+", output.replace(",",""))
                if len(prices) > 0:
                    deal_price = float(prices[0])
                    reward = (deal_price - case['seller_price']) / (case['buyer_price'] - case['seller_price'])
                    rewards.append(reward)

            if -1 in deals:
                reward = -0.1
            else:
                if len(rewards) == 0:
                    reward = 0
                else:
                    reward = max(set(rewards), key = rewards.count)
            self.logger.info(str(reward))

        max_freq_action = max(user_actions, key=lambda x: user_actions[x])
        return reward, max_freq_action


def query_openai_model(messages: str, model: str = "gpt-3.5-turbo-0613", max_tokens: int = 128, temperature: float = 0, n: int = 1):
    api_key = 'xxxxxxxxxx'
    client = OpenAI(api_key=api_key)
    flag = True
    while flag:
        try:
            start_t = time.time()
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
            end_t = time.time()
        except Exception as e:
            print("Some error happened here.")
            time.sleep(5)
    return output

from torch.distributions import Categorical
import random
import numpy as np
from tqdm import tqdm
from transformers import AdamW, BertModel, RobertaModel, AutoModelForSeq2SeqLM, AutoTokenizer
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F
from collections import defaultdict as ddict
from ppdpp.utils import *
from ppdpp.prompt import ESConvAct, CIMAAct, CBAct
import ppdpp.utils as utils
from ppdpp.env import system_role, user_role
from ppdpp.qnet import Qnet

from mcts.gdpzero import GDPZero
from mcts.core.mcts import OpenLoopMCTS
from mcts.utils.utils import dotdict

model = {'bert': BertModel, 'roberta': RobertaModel}
act = {'esc': ESConvAct, 'cima': CIMAAct, 'cb': CBAct}
TMP_DIR = {
    'esc': './tmp/esc',
    'cima': './tmp/cima',
    'cb': './tmp/cb',
}

class PPDPP(nn.Module):
    def __init__(self, args, config, tokenizer, success_base):
        super().__init__()
        self.logger = args.logger
        self.sys_role, self.usr_role = system_role[args.data_name], user_role[args.data_name]
        self.policy = model[args.model_name].from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config)
        self.dropout = nn.Dropout(args.dropout)
        self.act = sorted(list(act[args.data_name].keys()))
        self.inv_act = {act: idx for idx, act in enumerate(self.act)}
        self.classifier = nn.Linear(config.hidden_size, len(self.act))
        self.Q_head = Qnet(args.dropout, config.hidden_size, config.hidden_size, len(act[args.data_name]))
        self.Q_head_target = Qnet(args.dropout, config.hidden_size, config.hidden_size, len(act[args.data_name]))
        self.tokenizer = tokenizer
        self.optimizer = AdamW(
            self.parameters(), lr=args.learning_rate
        )
        self.eps = np.finfo(np.float32).eps.item()
        self.config = config
        self.args = args
        self.saved_log_probs = []
        self.saved_qvs = []
        self.rewards = []
        self.device = args.device
        
        self.mcts = GDPZero(args, success_base)
        self.ent_bound = args.entropy_bound
        self.sub_value = args.sub_value
        self.success_base = success_base
        
        self.gamma = args.gamma
        self.lmbda = args.lmbda
        self.apply_policy_times = 0.0
        self.apply_mcts_times = 0.0
        self.apply_chatgpt_times = 0.0
        
        self.update_target_qnet()
        for p in self.Q_head_target.parameters():
            p.requires_grad = False
        
        self.action_freq = ddict(int)
        self.thresh_history = []

    def build_input(self, states):
        def pad_sequence(inputs, attention_masks):
            max_length = max([len(inp) for inp in inputs])
            attention_masks = [attn_mask + [0] * (max_length - len(inputs[idx])) for idx, attn_mask in enumerate(attention_masks)]
            inputs = [inpt + [self.tokenizer.pad_token_id] * (max_length - len(inpt)) for inpt in inputs]
            return inputs, attention_masks
        
        inps, attention_masks = [], []
        for state in states:
            dial_id = []
            for turn in state[::-1]:
                s = self.tokenizer.encode("%s: %s" % (turn['role'], turn['content']))
                if len(dial_id) + len(s) > self.args.max_seq_length:
                    break
                dial_id = s[1:] + dial_id
            inp = s[:1] + dial_id
            inps.append(inp.copy())
            attention_masks.append([1] * len(inp))
        inps, attention_masks = pad_sequence(inps, attention_masks)
        return inps, attention_masks

    def forward(self, state_ids, attention_mask, actions=None, target_qvs=None):
        outputs = self.policy(input_ids=state_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        
        logits = self.classifier(pooled_output)
        if actions is not None and target_qvs is not None:
            q_value = self.Q_head(pooled_output)
            qa_value = torch.gather(q_value, dim=-1, index=actions.view(-1, 1))
            loss_fct = MSELoss(reduction='mean')
            critic_loss = loss_fct(qa_value.view(-1), target_qvs.view(-1))
            
            max_actions = logits.view(-1, len(self.act)).argmax(dim=-1)
            for action in max_actions.detach().cpu().tolist():
                self.action_freq[action] += 1
            
            actor_probs = logits.view(-1, len(self.act)).softmax(dim=-1).gather(dim=1, index=actions.view(-1, 1))
            td_delta = target_qvs - qa_value
            actor_loss = torch.mean(-torch.log(actor_probs).view(-1) * td_delta.detach())
            
            loss = actor_loss + self.args.critic_loss_w * critic_loss
            return loss, actor_loss.item(), critic_loss.item()
        else:
            return F.softmax(logits.view(-1, len(self.act)), dim=-1), self.Q_head(pooled_output)

    def encode_state(self, state):
        inp, attn_mask = self.build_input(state) if isinstance(state[0], list) else self.build_input([state])
        inp = torch.tensor(inp).long().to(self.device)
        attn_mask = torch.tensor(attn_mask).long().to(self.device)

        outputs = self.policy(input_ids=inp, attention_mask=attn_mask)
        pooled_output = outputs[1]
        return self.dropout(pooled_output)
    
    def apply_actor(self, state_encoding):
        logits = self.classifier(state_encoding)
        dist = nn.functional.softmax(logits, dim=1)
        return dist
    
    def apply_critic(self, state_encoding):
        qvs = self.Q_head(state_encoding)
        return qvs
    
    def apply_policy(self, state):
        pooled_output = self.encode_state(state)
        logits = self.classifier(pooled_output)
        dist = nn.functional.softmax(logits, dim=1)
        qvs = self.Q_head(pooled_output)
        return dist, qvs
    
    def select_action(self, state, mcts_state, action=None, is_test=False, transition_dict=None):
        use_mcts = True
        action_dist, qvs = self.apply_policy(state)
        m = Categorical(action_dist)
        if action is None:
            if is_test:
                # entropy = utils.safe_entropy(action_dist)
                # if entropy <= self.ent_bound:
                topk_probs, _ = torch.topk(action_dist, k=2)
                self.logger.info('action distribution: {}'.format(action_dist.detach().cpu().tolist()))
                self.logger.info('select {}th percentiles...'.format(self.args.mcts_applied_ratio * 100))
                if self.args.mcts_applied_ratio == 0.0:
                    sub_value = 0.0
                elif self.args.mcts_applied_ratio == 1.0:
                    sub_value = 1.0
                else:
                    sub_value = np.percentile(self.thresh_history, self.args.mcts_applied_ratio * 100) if len(self.thresh_history) >= 2 else self.sub_value
                if topk_probs[0][0] - topk_probs[0][1] > sub_value:     # sub_value 大于 1 则全部走 mcts，小于等于0,则全部走 policy
                    self.logger.info('max prob - second max prob = {} >= {}'.format(topk_probs[0][0] - topk_probs[0][1], sub_value))
                    action = action_dist.argmax().item()
                    reward, full_mcts_history = None, None
                    self.logger.info('Choose action "{}" by Policy Network...'.format(self.act[action]))
                    self.apply_policy_times += 1
                    use_mcts = False
                else:
                    self.logger.info('max prob - second max prob = {} < {}'.format(topk_probs[0][0] - topk_probs[0][1], sub_value))
                    mcts_state, reward, full_mcts_history, transition_dict, apply_chatgpt_times = self.select_action_by_mcts(mcts_state, state, transition_dict)
                    action = mcts_state[-2][1]                      # 使用 mcts_state 的倒数第二个记录的 strategy 作为动作
                    action = self.inv_act[action]
                    self.logger.info('Choose action "{}" by MCTS...'.format(self.act[action]))
                    self.apply_mcts_times += 1
                    self.apply_chatgpt_times += apply_chatgpt_times
                self.thresh_history.append((topk_probs[0][0] - topk_probs[0][1]).item())
            else:
                # action = m.sample()
                mcts_state, reward, full_mcts_history, transition_dict, apply_chatgpt_times = self.select_action_by_mcts(mcts_state, state, transition_dict)
                action_str = mcts_state[-2][1]
                action = self.inv_act[action_str]
                action_tensor = torch.tensor([action]).long().to(action_dist.device)
                self.saved_log_probs.append(m.log_prob(action_tensor))
                self.saved_qvs.append(qvs.gather(1, action_tensor.unsqueeze(dim=-1)).squeeze(dim=-1))
                self.logger.info('Choose action "{}" by MCTS...'.format(self.act[action]))
                self.apply_mcts_times += 1
                self.apply_chatgpt_times += apply_chatgpt_times
        else:
            if not is_test:
                action_tensor = torch.tensor([action]).long().to(action_dist.device)
                self.saved_log_probs.append(m.log_prob(action_tensor))
                self.saved_qvs.append(qvs.gather(1, action_tensor.unsqueeze(dim=-1)).squeeze(dim=-1))
            reward, full_mcts_history = None, None
            self.logger.info('Choose action "{}" from searched successful path by MCTS...'.format(self.act[action]))
        self.action_freq[action] += 1
        return self.act[action], mcts_state, reward, full_mcts_history, transition_dict, use_mcts
    
    def select_action_by_mcts(self, mcts_state, agent_state, transition_dict=None):
        args = dotdict({
            "cpuct": 1.0,
            "num_MCTS_sims": self.args.num_mcts_sims,
            "Q_0": self.args.Q_0,
            "max_realizations": self.args.max_realizations,
        })
        
        dialog_planner = OpenLoopMCTS(self.args.data_name, self.mcts.game, self.mcts.planner, args, self.success_base)
        mcts_state, reward, full_mcts_history, transition_dict, apply_chatgpt_times = self.mcts._collect_da_action(dialog_planner, args, mcts_state, self, 
                                                                                                                   self.ent_bound, agent_state, transition_dict)
        return mcts_state, reward, full_mcts_history, transition_dict, apply_chatgpt_times

    def optimize_model(self, transition_dict, logger):
        logger.info('Start training ...')
        R = 0
        policy_loss = []
        rewards = []
        for r in self.rewards[::-1]:
            R = r + self.args.gamma * R
            rewards.insert(0, R)
        rewards = torch.tensor(rewards).to(self.device)
        loss_fct = nn.MSELoss(reduction='mean')
        qa_values = torch.cat(self.saved_qvs, dim=-1)
        critic_loss = loss_fct(qa_values, rewards)
        td_delta = rewards - qa_values
        log_probs = torch.cat(self.saved_log_probs, dim=-1)
        policy_loss = (-log_probs * td_delta.detach()).mean()
        loss = policy_loss + self.args.critic_loss_w * critic_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.args.max_grad_norm)
        self.optimizer.step()
        
        del self.rewards[:]
        del self.saved_log_probs[:]
        del self.saved_qvs[:]
        del transition_dict
        
        return policy_loss.item(), critic_loss.item(),
    
    def compute_advantage(self, td_delta):
        td_delta = td_delta.detach().numpy()
        advantage_list = []
        advantage = 0.0
        for delta in td_delta[::-1]:
            advantage = self.gamma * self.lmbda * advantage + delta
            advantage_list.append(advantage)
        advantage_list.reverse()
        return torch.tensor(advantage_list, dtype=torch.float)
    
    def update_target_qnet(self):
        self.Q_head_target.load_state_dict(self.Q_head.state_dict())
    
    def save_model(self, data_name, filename, epoch_user):
        output_dir = TMP_DIR[data_name] + '/RL-agent/' + filename + '-epoch-{}'.format(epoch_user)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        torch.save(self.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))
        torch.save(self.args, os.path.join(output_dir, 'training_args.bin'))
    
    def load_model(self, data_name, filename, epoch_user=None, device='cuda', logger=None):
        if epoch_user: 
            output_dir = TMP_DIR[data_name] + '/RL-agent/' + filename + '-epoch-{}'.format(epoch_user)
        else:
            output_dir = filename
        if hasattr(self, 'module'):
            self.module.load_state_dict(torch.load(os.path.join(output_dir, 'pytorch_model.bin')))
        else:
            self.load_state_dict(torch.load(os.path.join(output_dir, 'pytorch_model.bin'), map_location=device))
        if logger is not None:
            logger.info('Load model from {}'.format(output_dir))


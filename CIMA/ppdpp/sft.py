import argparse
from transformers import AdamW, BertTokenizer, RobertaTokenizer, BertConfig, RobertaConfig
import glob
import logging
import os
import random
from collections import defaultdict as ddict
from pytorch_transformers import WarmupLinearSchedule
import torch
import ppdpp.utils as utils
import ppdpp.data_reader as data_reader
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
from itertools import count

from ppdpp.env import Env
from ppdpp.agent import PPDPP
from ppdpp.utils import *

from sklearn.metrics import f1_score, precision_score, recall_score

# python sft.py --gpu="0 1" --do_train --overwrite_output_dir --per_gpu_train_batch_size=8 --per_gpu_eval_batch_size=8

tok = {'bert': BertTokenizer, 'roberta': RobertaTokenizer}
cfg = {'bert': BertConfig, 'roberta': RobertaConfig}
system_role = {'esc':'Therapist', 'cima': 'Teacher', 'cb': 'Buyer'}
user_role = {'esc':'Patient', 'cima': 'Student', 'cb': 'Seller'}


class DataFrame(Dataset):
    def __init__(self, data, args):
        self.state_ids = data['state_ids']
        self.action_ids = data['actions']
        self.next_state_ids = data['next_state_ids']
        self.dones = data['dones']
        self.rewards = data['rewards']
        self.max_len = args.max_seq_length

    def __getitem__(self, index):
        return self.state_ids[index][:self.max_len], self.action_ids[index], self.next_state_ids[index][:self.max_len], self.dones[index], self.rewards[index]
    
    def __len__(self):
        return len(self.state_ids)


def collate_fn(data):
    state_ids, action_ids, next_state_ids, dones, rewards = zip(*data)

    state_ids = [torch.tensor(state_id).long() for state_id in state_ids]
    state_ids = pad_sequence(state_ids, batch_first=True, padding_value=1)
    
    next_state_ids = [torch.tensor(next_state_id).long() for next_state_id in next_state_ids]
    next_state_ids = pad_sequence(next_state_ids, batch_first=True, padding_value=1)

    attention_mask = state_ids.ne(1)
    next_attention_mask = next_state_ids.ne(1)
    actions = torch.tensor(action_ids).long()
    dones = torch.tensor(dones).float()
    rewards = torch.tensor(rewards).float()
    
    return {'state_ids':  state_ids,
            'attention_mask': attention_mask,
            'next_state_ids':  next_state_ids,
            'next_attention_mask': next_attention_mask,
            'actions': actions,
            'dones': dones,
            'rewards': rewards,
            }


def train(args, train_dataset, model, tokenizer):
    tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, len(args.device_id))
    train_dataloader = DataLoader(DataFrame(train_dataset, args), batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn)

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
    #      'weight_decay': args.weight_decay},
    #     {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    # ]
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    
    # multi-gpu training (should be after apex fp16 initialization)
    if len(args.device_id) > 1:
        model = torch.nn.DataParallel(model, device_ids=args.device_id)
    
    # Train!
    logging.info("***** Running training *****")
    logging.info("  Num examples = %d", len(train_dataset))
    logging.info("  Num Epochs = %d", args.num_train_epochs)
    logging.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logging.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps)
    logging.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logging.info("  Total optimization steps = %d", t_total)
    
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    utils.set_random_seed(args.seed)  # Added here for reproductibility (even between python 2 and 3)
    global_step = 0

    #total_rouge = evaluate(args, model, tokenizer, save_output=True)
    # best_loss = 1000000 #total_rouge[2]
    best_sr = 0.0
    
    # te_loss = evaluate(args, model, tokenizer, save_output=True)
    # SR_all = epoch_evaluate(args, model, 0, args.output_dir)
    
    for e in tqdm(range(int(args.start_epoch), int(args.start_epoch + args.num_train_epochs)), desc="Epoch"):
        logging.info("training for epoch {} ...".format(e))
        print("training for epoch {} ...".format(e))
        model.train()
        model.action_freq = ddict(int)
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            #batch = tuple(t.to(args.device) for t in batch)
            inputs = {'state_ids': batch['state_ids'].to(args.device), 
                      'attention_mask': batch['attention_mask'].to(args.device), 
                      'next_state_ids': batch['next_state_ids'].to(args.device), 
                      'next_attention_mask': batch['next_attention_mask'].to(args.device), 
                      'actions': batch['actions'].to(args.device), 
                      'dones': batch['dones'].to(args.device),
                      'rewards': batch['rewards'].to(args.device)}
            outputs = model(**inputs)
            loss, actor_loss, critic_loss = outputs#[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            if len(args.device_id) > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                model.zero_grad()
            global_step += 1
            
            if step % 50 == 0:
                logging.info('Train action freq: {}'.format(str(model.action_freq)))
            
            if step % 50 == 0:
                logging.info('Step: {}  Training loss: {}'.format(step, loss.item()))
                logging.info('Step: {}  Training Actor loss: {}'.format(step, actor_loss))
                logging.info('Step: {}  Training Critic loss: {}'.format(step, critic_loss))
            
            if args.target_update_count > 0 and (step + 1) % args.target_update_count == 0:
                model.update_target_qnet()
        
        logging.info('Train action freq: {}'.format(str(model.action_freq)))
        tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
        tb_writer.add_scalar('loss', (tr_loss - logging_loss) / (step+1), global_step)
        logging.info('loss: {}'.format((tr_loss - logging_loss) / (step+1)))
        logging_loss = tr_loss
        
        SR_all = epoch_evaluate(args, model, e + 1, args.output_dir)
        SR = SR_all[0]
        if SR > best_sr:
            # Save model checkpoint
            best_sr = SR
            save_model(args, model, args.logger)
    
    tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, save_output=False):

    eval_dataset = data_reader.load_and_cache_examples(args, tokenizer, evaluate=True)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, len(args.device_id))

    all_metrics = []
    eval_dataloader = DataLoader(DataFrame(eval_dataset, args), batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_fn)
    # Eval!
    logging.info("***** Running evaluation *****")
    logging.info("  Num examples = %d", len(eval_dataset))
    logging.info("  Batch size = %d", args.eval_batch_size)
    count = 0
    log_preds, q_values = [], []
    targets, target_qvs = [], []

    with torch.no_grad():
        model.eval()
        model_to_eval = model.module if hasattr(model, 'module') else model
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            pred, q_value = model_to_eval(
                state_ids=batch['state_ids'].to(args.device), 
                attention_mask=batch['attention_mask'].to(args.device)
            )
            
            actions = batch['actions'].view(-1, 1).to(args.device)
            log_preds.extend(torch.log(pred).gather(dim=1, index=actions))
            # q_values.append(q_value[:, batch['labels']].view(-1))
            q_values.append(torch.gather(q_value, dim=-1, index=actions).view(-1))
            targets.extend(batch['actions'].tolist())
            target_qvs.append(batch['target_qvs'].to(args.device))
        
        log_preds = torch.cat(log_preds, dim=-1)
        target_qvs, q_values = torch.cat(target_qvs, dim=-1), torch.cat(q_values, dim=-1)
        fct = nn.MSELoss(reduction='mean')
        q_loss = fct(target_qvs, q_values).item()
        td_delta = target_qvs - q_values
        actor_loss = torch.mean(-log_preds * td_delta.detach()).item()
        logging.info("QNet loss: {}".format(q_loss))
        logging.info("Actor loss: {}".format(actor_loss))
    return q_loss + args.critic_loss_w * actor_loss


def epoch_evaluate(args, policy, i_episode, base_dir):
    dataset = load_dataset(args.data_name)
    logger = args.logger
    train_env = Env(args, dataset, mode='train') # env init

    if 'vicuna' in [args.system, args.user, args.critic] or 'llama2' in [args.system, args.user, args.critic]:
        test_env = Env(args, dataset, mode=args.set_name, env_model=train_env.vicuna_model, env_tokenizer=train_env.vicuna_tokenizer)
    else:
        test_env = Env(args, dataset, mode=args.set_name) # env init
    
    policy.apply_policy_times = 0.0
    policy.apply_mcts_times = 0.0
    policy.apply_chatgpt_times = 0.0

    SR, AvgT, total_reward = 0, 0, 0
    SR_turn = [0]* args.max_turn
    turn_result = []
    result = []
    # test_size = len(test_env.dataset)
    start_index = args.eval_start_index
    test_size = args.eval_sample_times
    logger.info('Test size: {}'.format(test_size))
    if args.do_train:
        trained = 'trained'
    elif args.do_eval:
        trained = 'untrained'
    else:
        raise NotImplemented
    test_filename = 'Evaluate-epoch-{}'.format(i_episode)
    record_filename = 'Record-epoch-{}'.format(i_episode)
    REC_PATH = base_dir + '/' + record_filename + '.txt'
    if not os.path.isdir(base_dir):
        os.makedirs(base_dir)
    rec_file = open(REC_PATH, 'w')
    policy.eval()
    policy.action_freq = ddict(int)
    logger.info('\n================Evaluation Epoch: {}===================='.format(i_episode))
    with torch.no_grad():
        for test_num in tqdm(range(start_index, start_index + test_size)):  #test_size
            #blockPrint()
            logger.info('\n================test tuple:{}===================='.format(test_num))
            epi_reward = 0
            done = 0
            is_last_turn = False
            state, mcts_state, reward = test_env.reset()
            full_mcts_history, mcts_turn, rewards = None, 0, [reward]
            transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
            for t in count():  # user  dialog
                if full_mcts_history is None:
                    action, mcts_state, reward, full_mcts_history, transition_dict, use_mcts = policy.select_action(state, mcts_state, is_test=True, transition_dict=transition_dict)
                else:
                    assert full_mcts_history['state'][len(state) + 1][0] == system_role[args.data_name]
                    action = policy.inv_act[full_mcts_history['state'][len(state) + 1][1]]
                    action, mcts_state, reward, _, transition_dict, use_mcts = policy.select_action(state, mcts_state, action, is_test=True, transition_dict=transition_dict)
                
                if full_mcts_history is None:    
                    state, reward, done = test_env.step(action, mcts_state, reward, use_mcts)
                else:
                    state, reward, done = test_env.unfold_mcts_state(mcts_state, full_mcts_history, mcts_turn)
                    mcts_turn += 1
                    
                if args.data_name == 'cb' and reward < 0: # reward = Sale-to-List Ratio
                    reward = 0
                epi_reward += reward
                rewards.append(reward)

                if done:
                    if done == 1:  
                        SR_turn = [v+1 if i>t  else v for i, v in enumerate(SR_turn) ]
                        SR += 1
                        logger.info('Current success rate: {}'.format(float(SR) / (test_num - start_index + 1)))
                    total_reward += epi_reward
                    AvgT += t+1

                    rec_file.write('%s\n\n' % str({'dialog':state, 'reward':epi_reward}))
                    
                    break
            
            if (test_num - start_index) % 50 == 0:
                logger.info('Eval action freq: {}'.format(str(policy.action_freq)))
            
            enablePrint()
    
    logger.info('Eval Action freq: {}'.format(str(policy.action_freq)))
    SR_mean = float(SR)/test_size
    AvgT_mean = float(AvgT)/test_size
    reward_mean = total_reward/test_size
    SR_all = [SR_mean, AvgT_mean, reward_mean]
    save_rl_mtric(base_dir=base_dir, filename=test_filename, epoch=test_num, SR=SR_all, mode='test')  # save RL SR
    logger.info('save test evaluate successfully!')

    SRturn_all = [0] * args.max_turn
    for i in range(len(SRturn_all)):
        SRturn_all[i] = float(SR_turn[i])/test_size
    logger.info('success turn:{}'.format(SRturn_all))
    logger.info('SR:{}, AvgT:{}, reward:{}'.format(SR_mean, AvgT_mean, reward_mean))
    PATH = base_dir + '/' + test_filename + '.txt'
    with open(PATH, 'a') as f:
        f.write('Training epocch:{}\n'.format(i_episode))
        f.write('===========Test Turn===============\n')
        f.write('Testing {} user tuples\n'.format(test_num))
        for i in range(len(SRturn_all)):
            f.write('Testing SR-turn@{}: {}\n'.format(i, SRturn_all[i]))
        f.write('================================\n')
    
    logger.info('Apply chatgpt {} times'.format(policy.apply_chatgpt_times + test_env.apply_chatgpt_times))
    logger.info('Apply MCTS times: {}'.format(policy.apply_mcts_times))
    logger.info('Apply policy times: {}'.format(policy.apply_policy_times))
    return SR_all
    

def arg_parser():
    parser = argparse.ArgumentParser(description="train.py")

    ## Required parameters
    parser.add_argument('--data_name', default='esc', type=str,
                        help="dataset name")
    parser.add_argument('--set_name', default='valid', type=str,
                        help="dataset split name")
    parser.add_argument('--model_name', default='roberta', type=str,
                        help="model name")
    parser.add_argument("--model_name_or_path", default='roberta-large',
                        type=str, help="model name or path")
    parser.add_argument("--output_dir", default='sft', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--data_dir", default='../data', type=str,
                        help="The data directory.")
    parser.add_argument("--cache_dir", default='/storage_fast/ydeng/plm', type=str,
                        help="The cache directory.")

    ## Other parameters
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval.")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--do_lower_case", action='store_false',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--dropout", default=0.05, type=float)
    parser.add_argument('--entropy_bound', type=float, default=1.67)
    parser.add_argument('--sub_value', type=float, default=0.0)
    parser.add_argument('--system', type=str, default='chatgpt', choices=['vicuna','chatgpt','llama2', 'chatglm'],
                        help='One of {vicuna, chatgpt, llama2}.')
    parser.add_argument('--user', type=str, default='chatgpt', choices=['vicuna','chatgpt','llama2', 'chatglm'],
                        help='One of {vicuna, chatgpt, llama2}.')
    parser.add_argument('--critic', type=str, default='chatgpt', choices=['vicuna','chatgpt','llama2', 'chatglm'],
                        help='One of {vicuna, chatgpt, llama2}.')
    parser.add_argument('--planner', type=str, default='chatgpt', choices=['vicuna','chatgpt','llama2', 'chatglm'],
                        help='One of {vicuna, chatgpt, llama2}.')
    parser.add_argument('--gamma', type=float, default=0.999, help='reward discount factor.')
    parser.add_argument('--lmbda', type=float, default=0.95, help='reward discount factor.')
    parser.add_argument('--max_turn', type=int, default=8, help='max conversation turn')
    
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gpu', default="0 1 2", type=str,
                        help="Use CUDA on the device.")
    parser.add_argument("--per_gpu_train_batch_size", default=2, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=1, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--eval_sample_times', type=int, default=10)
    parser.add_argument('--eval_start_index', type=int, default=0)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument("--num_train_epochs", default=10, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--load_rl_epoch', type=int, default=0, help='load agent from epoch')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--warmup_steps", default=400, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--learning_rate", default=6e-6, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--local_rank", default=-1, type=int,
                        help="DDP requirement.")
    
    parser.add_argument('--zero_shot', action="store_true")
    parser.add_argument('--gen_sentences', type=int, default=-1, help='number of sentences to generate from the llm. Longer ones will be truncated by nltk.')
    parser.add_argument('--num_mcts_sims', type=int, default=20, help='number of mcts simulations')
    parser.add_argument('--max_realizations', type=int, default=1, help='number of realizations per mcts state')
    parser.add_argument('--Q_0', type=float, default=0.0, help='initial Q value for unitialized states. to control exploration')
    parser.add_argument('--num_dialogs', type=int, default=20, help='number of dialogs to test MCTS on')
    parser.add_argument("--max_conv_turns", type=int, default=8)
    parser.add_argument("--max_hist_num_turns", type=int, default=8)
    parser.add_argument("--resp_max_new_tokens", type=int, default=64)
    parser.add_argument("--reward_max_new_tokens", type=int, default=16)
    parser.add_argument("--action_temperature", type=float, default=1.0)
    parser.add_argument("--resp_temperature", type=float, default=0.7)
    parser.add_argument("--reward_temperature", type=float, default=1.1)
    parser.add_argument("--action_num_return_sequences", type=int, default=15)
    parser.add_argument("--reward_num_return_sequences", type=int, default=10)
    
    parser.add_argument("--success_base", type=float, default=0.1)
    parser.add_argument("--critic_loss_w", type=float, default=1.0)
    parser.add_argument("--target_update_count", type=int, default=3)
    parser.add_argument('--mcts_applied_ratio', type=float, default=0.0)
    
    parser.add_argument("--remark", type=str, default='None')

    args = parser.parse_args()
    return args


def main(args):
    args.output_dir = os.path.join(args.output_dir, args.data_name, "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(args.model_name, args.num_train_epochs, args.warmup_steps, 
                                                                                                           args.learning_rate, args.weight_decay, args.adam_epsilon, 
                                                                                                           args.max_grad_norm, args.critic_loss_w, args.success_base, args.dropout, 
                                                                                                           args.target_update_count, args.remark))
    # Create output directory if needed
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))
    
    # Setup CUDA, GPU & distributed training
    device, device_id = utils.set_cuda(args)
    args.device = device
    args.device_id = device_id
    
    # logging.basicConfig(level=logging.DEBUG, filename=args.output_dir + '/log.txt', filemode='a')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    format = logging.Formatter("%(asctime)s - %(name)s - [%(levelname)s] - %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(format)
    th = logging.FileHandler(filename=args.output_dir + '/log.txt', encoding='utf-8')
    th.setFormatter(format)
    logger.addHandler(sh)
    logger.addHandler(th)
    args.logger = logger
    
    logger.info(args.device)
    logger.info('data_set:{}'.format(args.data_name))

    # Set seed
    utils.set_random_seed(args.seed)

    config = cfg[args.model_name].from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    tokenizer = tok[args.model_name].from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case, cache_dir=args.cache_dir)
    
    model = PPDPP(args, config, tokenizer, args.success_base)

    train_dataset = data_reader.load_and_cache_examples(args, tokenizer, evaluate=False)

    model.to(args.device)

    logging.info("Training/evaluation parameters %s", args)
    output_dir = os.path.join(args.output_dir, 'best_checkpoint')
    
    if args.load_rl_epoch > 0:
        logger.info('Staring loading rl model in epoch {}'.format(args.load_rl_epoch))
        if hasattr(model, 'module'):
            model.module.load_state_dict(torch.load(os.path.join(output_dir, 'pytorch_model.bin')))
        else:
            model.load_state_dict(torch.load(os.path.join(output_dir, 'pytorch_model.bin')))
        tokenizer = tok[args.model_name].from_pretrained(output_dir, do_lower_case=args.do_lower_case)

    # Training
    if args.do_train:
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logging.info(" global_step = %s, average loss = %s", global_step, tr_loss)
        tokenizer.save_pretrained(output_dir)

    # Evaluation
    if args.do_eval:
        # Load a trained model and vocabulary that you have fine-tuned
        if hasattr(model, 'module'):
            model.module.load_state_dict(torch.load(os.path.join(output_dir, 'pytorch_model.bin')))
        else:
            model.load_state_dict(torch.load(os.path.join(output_dir, 'pytorch_model.bin')))
        tokenizer = tok[args.model_name].from_pretrained(output_dir, do_lower_case=args.do_lower_case)
        model.to(args.device)
        args.set_name = 'test'
        epoch_evaluate(args, model, args.start_epoch + args.num_train_epochs + 1, args.output_dir)


if __name__ == "__main__":
    main(arg_parser())
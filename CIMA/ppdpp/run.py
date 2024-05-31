from ppdpp.env import Env
from ppdpp.agent import PPDPP
from ppdpp.utils import *
from itertools import count
from tqdm import tqdm
import os, json
import logging, wandb
import argparse
from collections import defaultdict as ddict
from transformers import BertTokenizer, RobertaTokenizer, BertConfig, RobertaConfig
from fastchat.model import add_model_args

tok = {'bert': BertTokenizer, 'roberta': RobertaTokenizer}
cfg = {'bert': BertConfig, 'roberta': RobertaConfig}
system_role = {'esc':'Therapist', 'cima': 'Teacher', 'cb': 'Buyer'}
user_role = {'esc':'Patient', 'cima': 'Student', 'cb': 'Seller'}


def train(args, config, dataset, filename, tokenizer):
    logger = args.logger
    env = Env(args, dataset, mode='train') # env init
    set_random_seed(args.seed)
    policy = PPDPP(args, config, tokenizer, args.success_base) # policy network init
    policy = policy.to(args.device)

    # load policy parameters
    if args.sft_dir is not None:
        logger.info('Staring loading policy model from {}'.format(args.sft_dir))
        policy.load_model(data_name=args.data_name, filename=args.sft_dir, device=args.device, logger=logger)
        policy.update_target_qnet()
    
    if args.load_rl_epoch > 0:
        logger.info('Staring loading rl model in epoch {}'.format(args.load_rl_epoch))
        policy.load_model(data_name=args.data_name, filename=filename, epoch_user=args.load_rl_epoch, device=args.device, logger=logger)

    test_performance = []
    policy.apply_policy_times = 0.0
    policy.apply_mcts_times = 0.0
    policy.apply_chatgpt_times = 0.0
    # SR_all = evaluate(args, dataset, policy, filename, -1, env)
    # test_performance.append(SR_all)
    
    if args.do_eval:
        i_episode = args.load_rl_epoch if args.load_rl_epoch > 0 else 0
        SR15_mean = evaluate(args, dataset, policy, filename, i_episode, env, mode='test')
        test_performance = [SR15_mean]
    if not args.do_train:
        return
    for train_step in range(args.start_step + 1, args.max_steps+1):
        logger.info('\n================Training Epoch :{}===================='.format(train_step))
        policy.train()
        policy.action_freq = ddict(int)
        SR, AvgT, total_reward = 0., 0., 0.
        policy_loss, critic_loss = 0.0, 0.0
        for i_episode in tqdm(range(args.sample_times),desc='sampling'):
            #blockPrint()
            logger.info('\n================new tuple:{}===================='.format(i_episode))
            state, mcts_state, init_reward = env.reset()

            epi_reward = 0
            done = False
            full_mcts_history, mcts_turn = None, 0
            transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
            for t in count():   # user  dialog
                
                if full_mcts_history is None:
                    action, mcts_state, reward, full_mcts_history, transition_dict, use_mcts = policy.select_action(state, mcts_state, transition_dict=transition_dict)
                else:
                    assert full_mcts_history['state'][len(state)][0] == system_role[args.data_name]
                    action = policy.inv_act[full_mcts_history['state'][len(state)][1]]
                    action, mcts_state, reward, _, transition_dict, use_mcts = policy.select_action(state, mcts_state, action, transition_dict=transition_dict)
                    
                if full_mcts_history is None:    
                    state, reward, done = env.step(action, mcts_state, reward, use_mcts)
                else:
                    state, reward, done = env.unfold_mcts_state(mcts_state, full_mcts_history, mcts_turn)
                    mcts_turn += 1
                
                policy.rewards.append(reward)

                if done:
                    if done == 1:
                        SR += 1
                    AvgT += t+1
                    total_reward += epi_reward
                    
                    rewards = [init_reward] + policy.rewards
                    record_trajectories(args, state, mcts_state, rewards, transition_dict, 'train')
                    break

            new_policy_loss, new_critic_loss = policy.optimize_model(transition_dict, logger)
            if new_policy_loss is not None:
                policy_loss += new_policy_loss
            if new_critic_loss is not None:
                critic_loss += new_critic_loss
            
            if args.target_update_count > 0 and (i_episode + 1) % args.target_update_count == 0:
                policy.update_target_qnet()
            
            if i_episode % 50 == 0:
                logger.info('Train action freq: {}'.format(str(policy.action_freq)))
        
        enablePrint() # Enable print function
        logger.info('Train action freq: {}'.format(str(policy.action_freq)))
        logger.info('policy loss : {} in epoch_uesr {}'.format(policy_loss/args.sample_times, args.sample_times))
        logger.info('critic loss : {} in epoch_uesr {}'.format(critic_loss/args.sample_times, args.sample_times))
        logger.info('SR:{}, AvgT:{}, rewards:{} Total epoch_uesr:{}'.format(SR / args.sample_times,
                    AvgT / args.sample_times, total_reward / args.sample_times, args.sample_times))
        logger.info('Apply chatgpt times: {}'.format(policy.apply_chatgpt_times))
        if args.use_wandb:
            wandb.log({
                    'policy_loss': policy_loss/args.sample_times,
                    'critic_loss': critic_loss/args.sample_times,
                    'train_SR': SR / args.sample_times,
                    'train_AvgT': AvgT / args.sample_times,
                    'train_rewards': total_reward / args.sample_times
                })

        policy.apply_chatgpt_times = 0.0
        if train_step % args.eval_num == 0:
            SR_all = evaluate(args, dataset, policy, filename, train_step, env, mode='valid')
            test_performance.append(SR_all)
        if train_step % args.save_num == 0:
            policy.save_model(data_name=args.data_name, filename=filename, epoch_user=train_step)
    print(test_performance)


def evaluate(args, dataset, policy, filename, i_episode, train_env, mode='valid'):
    logger = args.logger
    if 'vicuna' in [args.system, args.user, args.critic] or 'llama2' in [args.system, args.user, args.critic]:
        test_env = Env(args, dataset, mode=mode, env_model=train_env.vicuna_model, env_tokenizer=train_env.vicuna_tokenizer)
    else:
        test_env = Env(args, dataset, mode=mode) # env init
    policy.apply_policy_times = 0.0
    policy.apply_mcts_times = 0.0
    policy.apply_chatgpt_times = 0.0
        
    set_random_seed(args.seed)

    SR, AvgT, total_reward = 0, 0, 0
    SR_turn = [0]* args.max_turn
    turn_result = []
    result = []
    # test_size = len(test_env.dataset)
    start_index = args.eval_start_index
    test_size = args.eval_sample_times
    logger.info('Test size: {}'.format(test_size))
    trained = 'trained' if args.do_train else 'untrained'
    test_filename = 'Evaluate-epoch-{}'.format(i_episode)
    record_filename = 'Record-epoch-{}'.format(i_episode)
    base_dir = TMP_DIR[args.data_name] + '/eval_result/' + f'{trained}/{filename}/'
    REC_PATH = base_dir + record_filename + '.txt'
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
            state, mcts_state, init_reward = test_env.reset()
            full_mcts_history, mcts_turn, rewards = None, 0, [init_reward]
            transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
            for t in count():  # user  dialog
                if full_mcts_history is None:
                    action, mcts_state, reward, full_mcts_history, transition_dict, use_mcts = policy.select_action(state, mcts_state, is_test=True, transition_dict=transition_dict)
                else:
                    assert full_mcts_history['state'][len(state)][0] == system_role[args.data_name]
                    action = policy.inv_act[full_mcts_history['state'][len(state)][1]]
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
                    
                    record_trajectories(args, state, mcts_state, rewards, transition_dict, 'eval')
                    break
            
            if (test_num - start_index) % 50 == 0:
                logger.info('Eval action freq: {}'.format(str(policy.action_freq)))
            
            enablePrint()
            
    logger.info('Eval action freq: {}'.format(str(policy.action_freq)))
    SR_mean = float(SR)/test_size
    AvgT_mean = float(AvgT)/test_size
    reward_mean = total_reward/test_size
    SR_all = [SR_mean, AvgT_mean, reward_mean]
    save_rl_mtric(base_dir=base_dir, filename=test_filename, epoch=test_num, SR=SR_all, mode='test')  # save RL SR
    logger.info('save test evaluate successfully!')
    
    if args.use_wandb:
        wandb.log({
            'SR': float(SR)/test_size,
            'AvgT': float(AvgT)/test_size,
            'rewards': total_reward/test_size,
        })

    SRturn_all = [0] * args.max_turn
    for i in range(len(SRturn_all)):
        SRturn_all[i] = float(SR_turn[i])/test_size
    logger.info('success turn:{}'.format(SRturn_all))
    logger.info('SR:{}, AvgT:{}, reward:{}'.format(SR_mean, AvgT_mean, reward_mean))
    PATH = base_dir + test_filename + '.txt'
    with open(PATH, 'a') as f:
        f.write('Training epocch:{}\n'.format(i_episode))
        f.write('===========Test Turn===============\n')
        f.write('Testing {} user tuples\n'.format(test_num))
        for i in range(len(SRturn_all)):
            f.write('Testing SR-turn@{}: {}\n'.format(i, SRturn_all[i]))
        f.write('================================\n')
    with open(PATH, 'a') as f:
        f.write('{}\t{}\t{}\t{}\n'.format(i_episode, SR_mean, AvgT_mean, reward_mean))
        f.write('Apply MCTS times: {}\n'.format(policy.apply_mcts_times))
        f.write('Apply policy times: {}\n'.format(policy.apply_policy_times))
        f.write('Policy chatgpt times: {}\n'.format(test_env.apply_chatgpt_times))
        f.write('MCTS chatgpt times: {}\n'.format(policy.apply_chatgpt_times))
    
    logger.info('Policy chatgpt {} times'.format(test_env.apply_chatgpt_times))
    logger.info('MCTS chatgpt {} times'.format(policy.apply_chatgpt_times))
    logger.info('Apply MCTS times: {}'.format(policy.apply_mcts_times))
    logger.info('Apply policy times: {}'.format(policy.apply_policy_times))
    return SR_all


def epoch_evaluate(args, config, dataset, filename, tokenizer):
    logger = args.logger
    train_env = Env(args, dataset, mode='train') # env init
    set_random_seed(args.seed)
    policy = PPDPP(args, config, tokenizer, args.success_base) # policy network init
    policy = policy.to(args.device)
    
    logger = args.logger

    for i_episode in range(args.start_step + 1, args.max_steps+1):
        if 'vicuna' in [args.system, args.user, args.critic] or 'llama2' in [args.system, args.user, args.critic]:
            test_env = Env(args, dataset, mode='test', env_model=train_env.vicuna_model, env_tokenizer=train_env.vicuna_tokenizer)
        else:
            test_env = Env(args, dataset, mode='test') # env init
        
        logger.info('Staring loading rl model in epoch {}'.format(i_episode))
        policy.load_model(data_name=args.data_name, filename=args.rl_dir, epoch_user=i_episode, device=args.device, logger=logger)
        policy.apply_policy_times = 0.0
        policy.apply_mcts_times = 0.0
        policy.apply_chatgpt_times = 0.0
        
        set_random_seed(args.seed)

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
        elif args.do_epoch_eval:
            trained =  'epoch_eval'
        else:
            raise NotImplemented
        test_filename = 'Evaluate-epoch-{}'.format(i_episode)
        record_filename = 'Record-epoch-{}'.format(i_episode)
        base_dir = TMP_DIR[args.data_name] + '/eval_result/' + f'{trained}/{filename}/'
        REC_PATH = base_dir + record_filename + '.txt'
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
                state, mcts_state = test_env.reset()
                full_mcts_history, mcts_turn, rewards = None, 0, [-0.5]
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
                        
                        record_trajectories(args, state, mcts_state, rewards, transition_dict, 'eval')
                        break
                
                if (test_num - start_index) % 50 == 0:
                    logger.info('Eval action freq: {}'.format(str(policy.action_freq)))
                
                enablePrint()
        
        logger.info('Eval action freq: {}'.format(str(policy.action_freq)))
        SR_mean = float(SR)/test_size
        AvgT_mean = float(AvgT)/test_size
        reward_mean = total_reward/test_size
        SR_all = [SR_mean, AvgT_mean, reward_mean]
        save_rl_mtric(base_dir=base_dir, filename=test_filename, epoch=test_num, SR=SR_all, mode='test')  # save RL SR
        logger.info('save test evaluate successfully!')
        
        if args.use_wandb:
            wandb.log({
                'SR': float(SR)/test_size,
                'AvgT': float(AvgT)/test_size,
                'rewards': total_reward/test_size,
            })

        SRturn_all = [0] * args.max_turn
        for i in range(len(SRturn_all)):
            SRturn_all[i] = float(SR_turn[i])/test_size
        logger.info('success turn:{}'.format(SRturn_all))
        logger.info('SR:{}, AvgT:{}, reward:{}'.format(SR_mean, AvgT_mean, reward_mean))
        PATH = base_dir + test_filename + '.txt'
        with open(PATH, 'a') as f:
            f.write('Training epocch:{}\n'.format(i_episode))
            f.write('===========Test Turn===============\n')
            f.write('Testing {} user tuples\n'.format(test_num))
            for i in range(len(SRturn_all)):
                f.write('Testing SR-turn@{}: {}\n'.format(i, SRturn_all[i]))
            f.write('================================\n')
        with open(PATH, 'a') as f:
            f.write('{}\t{}\t{}\t{}\n'.format(i_episode, SR_mean, AvgT_mean, reward_mean))
            f.write('Apply MCTS times: {}\n'.format(policy.apply_mcts_times))
            f.write('Apply policy times: {}\n'.format(policy.apply_policy_times))
            f.write('Policy apply chatgpt times: {}\n'.format(test_env.apply_chatgpt_times))
            f.write('MCTS apply chatgpt times: {}\n'.format(policy.apply_chatgpt_times))        # MCTS 调用 chatgpt 的次数是加在 policy.apply_chatgpt_times 的
        
        logger.info('Policy apply chatgpt {} times'.format(test_env.apply_chatgpt_times))
        logger.info('MCTS apply chatgpt {} times'.format(policy.apply_chatgpt_times))
        logger.info('Apply MCTS times: {}'.format(policy.apply_mcts_times))
        logger.info('Apply policy times: {}'.format(policy.apply_policy_times))
    return SR_all


def record_trajectories(args, state, mcts_state, rewards, transition_dict, mode):
    assert len(state) == len(rewards) * 2
    try:
        assert len(mcts_state.history) == len(state)
    except:
        logging.info('state: ')
        for role, uttr in state:
            logging.info('{}:   {}'.format(role, uttr))
        logging.info('*****************************************')
        logging.info('mcts state: ')
        for role, act, uttr in mcts_state:
            logging.info('{}:   [{}]{}'.format(role, act, uttr))
        assert 0
    trajectory, idx = [], 0
    for role, act, uttr in mcts_state.history:
        if role == system_role[args.data_name]:
            trajectory.append({'speaker': role, 'strategy': act, 'text': uttr})
        else:
            trajectory.append({'speaker': role, 'state': rewards[idx], 'text': uttr})
            idx += 1
    with open(os.path.join(args.output_dir, '{}_trajectory.txt'.format(mode)), 'a+', encoding='utf-8') as fout:
        diag = {"sentence": mcts_state.sentence, "target": mcts_state.target, "dialog": trajectory}
        json.dump(diag, fout)
        fout.write('\n')
    
    with open(os.path.join(args.output_dir, '{}_mcts_transitions.txt'.format(mode)), 'a+', encoding='utf-8') as fout:
        for idx, transition_state in enumerate(transition_dict['states']):
            transition = {}
            transition['state'] = transition_state
            transition['action'] = transition_dict['actions'][idx]
            transition['next_state'] = transition_dict['next_states'][idx]
            transition['reward'] = transition_dict['rewards'][idx]
            transition['done'] = transition_dict['dones'][idx]
            json.dump(transition, fout)
            fout.write('\n')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-seed', type=int, default=1, help='random seed.')
    parser.add_argument('--num_gpus', type=int, default=1, help='number of gpus.')
    parser.add_argument('--epochs', '-me', type=int, default=50000, help='the number of RL train epoch')
    parser.add_argument('--gamma', type=float, default=0.999, help='reward discount factor.')
    parser.add_argument('--lmbda', type=float, default=0.95, help='reward discount factor.')
    parser.add_argument('--eps', type=float, default=0.2, help='reward discount factor.')
    parser.add_argument('--learning_rate', type=float, default=1e-6, help='learning rate.')
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument('--device_id', type=int, default=0)

    parser.add_argument('--data_name', type=str, default='esc', choices=['esc','cima','cb'],
                        help='One of {esc, cima, cb}.')
    parser.add_argument('--system', type=str, default='vicuna', choices=['vicuna','chatgpt','llama2', 'chatglm'],
                        help='One of {vicuna, chatgpt, llama2}.')
    parser.add_argument('--user', type=str, default='vicuna', choices=['vicuna','chatgpt','llama2', 'chatglm'],
                        help='One of {vicuna, chatgpt, llama2}.')
    parser.add_argument('--critic', type=str, default='vicuna', choices=['vicuna','chatgpt','llama2', 'chatglm'],
                        help='One of {vicuna, chatgpt, llama2}.')
    parser.add_argument('--planner', type=str, default='vicuna', choices=['vicuna','chatgpt','llama2', 'chatglm'],
                        help='One of {vicuna, chatgpt, llama2}.')
    parser.add_argument('--sft_dir', default='sft', #../pretrain/outputs/best_pretrain.pt
                        type=str, help="Pretrain model path.")
    parser.add_argument('--max_turn', type=int, default=8, help='max conversation turn')
    parser.add_argument('--mode', type=str, default='train', help='the mode in [train, test]')
    parser.add_argument('--load_rl_epoch', type=int, default=0, help='load agent from epoch')


    parser.add_argument("--cache_dir", default='/storage_fast/ydeng/plm', type=str, help="The cache directory.")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--model_path", type=str, default="/storage_fast/ydeng/llm/vicuna_hf/7B")
    parser.add_argument("--model_name", type=str, default="roberta")
    parser.add_argument("--model_name_or_path", default='roberta-large', type=str, help="model name or path")

    parser.add_argument("--do_lower_case", action='store_false', help="Set this flag if you are using an uncased model.")

    parser.add_argument('--start_step', type=int, default=0, help='max training steps')
    parser.add_argument('--max_steps', type=int, default=10, help='max training steps')
    parser.add_argument('--sample_times', type=int, default=100, help='the epoch of sampling')
    parser.add_argument('--eval_num', type=int, default=1, help='the number of steps to evaluate RL model and metric')
    parser.add_argument('--save_num', type=int, default=1, help='the number of steps to save RL model and metric')
    
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--temperature', type=float, default=0.7)

    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval.")
    
    parser.add_argument("--output_dir", default='sft', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--eval_sample_times', type=int, default=10)
    parser.add_argument('--eval_start_index', type=int, default=0)
    
    parser.add_argument('--entropy_bound', type=float, default=1.67)
    parser.add_argument('--mcts_applied_ratio', type=float, default=0.0)
    parser.add_argument('--sub_value', type=float, default=0.5)
    parser.add_argument('--use_mcts_sys_resp', action='store_true')
    parser.add_argument('--use_mcts_usr_resp', action='store_true')
    
    parser.add_argument("--dropout", default=0.05, type=float)
    
    parser.add_argument('--zero_shot', action="store_true")
    parser.add_argument('--gen_sentences', type=int, default=-1, help='number of sentences to generate from the llm. Longer ones will be truncated by nltk.')
    parser.add_argument('--num_mcts_sims', type=int, default=20, help='number of mcts simulations')
    parser.add_argument('--max_realizations', type=int, default=1, help='number of realizations per mcts state')
    parser.add_argument('--Q_0', type=float, default=0.0, help='initial Q value for unitialized states. to control exploration')
    parser.add_argument('--num_dialogs', type=int, default=20, help='number of dialogs to test MCTS on')
    parser.add_argument("--max_conv_turns", type=int, default=9)
    parser.add_argument("--max_hist_num_turns", type=int, default=8)
    parser.add_argument("--resp_max_new_tokens", type=int, default=64)
    parser.add_argument("--reward_max_new_tokens", type=int, default=16)
    parser.add_argument("--action_temperature", type=float, default=1.0)
    parser.add_argument("--resp_temperature", type=float, default=0.7)
    parser.add_argument("--reward_temperature", type=float, default=1.1)
    parser.add_argument("--action_num_return_sequences", type=int, default=15)
    parser.add_argument("--reward_num_return_sequences", type=int, default=10)
    
    parser.add_argument('--rl_dir', default='ppdpp', #../pretrain/outputs/best_pretrain.pt
                        type=str, help="Trained model path.")
    parser.add_argument("--do_epoch_eval", action='store_true')
    parser.add_argument("--success_base", type=float, default=0.1)
    parser.add_argument("--critic_loss_w", type=float, default=1.0)
    parser.add_argument("--target_update_count", type=int, default=3)
    parser.add_argument("--remark", type=str, default='None')
    parser.add_argument("--use_policy_prior", action='store_true')
    
    parser.add_argument('--use_wandb', action='store_true')

    add_model_args(parser)
    args = parser.parse_args()
    
    return args


def main(args):
    #os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.raw_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = torch.device('cuda:{}'.format(args.device_id)) if torch.cuda.is_available() else 'cpu'
    if args.do_train:
        trained = 'trained'
    elif args.do_eval:
        trained = 'untrained'
    elif args.do_epoch_eval:
        trained =  'epoch_eval'
    else:
        raise NotImplemented
    args.output_dir = os.path.join(args.output_dir, args.data_name, trained, 
                                   "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(args.sft_dir, args.system, args.user, args.critic, args.max_steps, 
                                                                                      args.sample_times, args.eval_sample_times, args.learning_rate, args.weight_decay, args.mcts_applied_ratio, 
                                                                                      args.dropout, args.critic_loss_w, args.eps, args.remark))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
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
    
    if args.use_wandb:
        wandb.init(
            # set the wandb project where this run will be logged
            project="pdp",
            
            # track hyperparameters and run metadata
            config=vars(args)
        )

    dataset = load_dataset(args.data_name)
    if args.do_train or args.do_eval:
        filename = '{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}'.format(args.data_name, args.sft_dir, args.system, args.user, args.critic, args.learning_rate, args.weight_decay, 
                                                                   args.max_steps, args.dropout, args.critic_loss_w, args.eps, args.remark)
    elif args.do_epoch_eval:
        filename = '{}-{}-{}'.format(args.data_name, args.rl_dir, args.mcts_applied_ratio)
    else:
        raise NotImplementedError

    config = cfg[args.model_name].from_pretrained(args.model_name_or_path)
    tokenizer = tok[args.model_name].from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)

    if args.sft_dir:
        args.sft_dir = os.path.join('sft', args.data_name, args.sft_dir, 'best_checkpoint')
    if not os.path.exists(args.sft_dir) :
        logger.info("no sft model, randomly initialize policy model")
        args.sft_dir = None

    if args.do_train or args.do_eval:
        train(args, config, dataset, filename, tokenizer)
    elif args.do_epoch_eval:
        epoch_evaluate(args, config, dataset, filename, tokenizer)
    if args.use_wandb:
        wandb.finish()

if __name__ == '__main__':
    main(parse_args())
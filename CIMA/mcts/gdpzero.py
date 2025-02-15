import numpy as np
import logging, os, sys
import pickle
import argparse
import random

from tqdm.auto import tqdm
from mcts.core.gen_models import (
	LocalModel, OpenAIModel, OpenAIChatModel, ChatGLM3Model
)
from mcts.core.esc_players import (
	TherapistModel, PatientModel, ESCSystemPlanner,
	TherapistChatModel, PatientChatModel, ESCChatSystemPlanner
)
from mcts.core.cima_players import (
	TeacherChatModel, StudentChatModel, CIMAChatSystemPlanner
)
from mcts.core.game import EmotionalSupportGame, CIMAGame
from mcts.core.mcts import OpenLoopMCTS
from mcts.core.helpers import DialogSession
from mcts.utils.utils import dotdict
from mcts.utils.prompt_examples import ESConv_EXP_DIALOG, CIMA_EXP_DIALOG


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class GDPZero:
    def __init__(self, cmd_args, success_base):
        self.args = cmd_args
        
        ModelMap = {
            'esc': {'sys': TherapistChatModel, 'usr': PatientChatModel, 'planner': ESCChatSystemPlanner, 'game': EmotionalSupportGame},
            'cima': {'sys': TeacherChatModel, 'usr': StudentChatModel, 'planner': CIMAChatSystemPlanner, 'game': CIMAGame}
        }
        
        game_ontology = ModelMap[cmd_args.data_name]['game'].get_game_ontology()
        sys_da = game_ontology['system']['dialog_acts']
        user_da = game_ontology['user']['dialog_acts']
        system_name = ModelMap[cmd_args.data_name]['game'].SYS
        user_name = ModelMap[cmd_args.data_name]['game'].USR

        if cmd_args.data_name.lower() == 'esc':
            exp_1 = DialogSession(system_name, user_name).from_history(ESConv_EXP_DIALOG)
            self.usr_da_map = {
                EmotionalSupportGame.U_FeelWorse: "Feel worse",
                EmotionalSupportGame.U_FeelTheSame: "Feel the same",
                EmotionalSupportGame.U_FeelBetter: "Feel better",
                EmotionalSupportGame.U_Solved: "Solved",
            }
        elif cmd_args.data_name.lower() == 'cima':
            exp_1 = DialogSession(system_name, user_name).from_history(CIMA_EXP_DIALOG)
            self.usr_da_map = {
                CIMAGame.U_Incorrect: "Made an incorrect translation",
                CIMAGame.U_DidNotTry: "Did not try to translate",
                CIMAGame.U_OnlyPart: "Only correctly translated a part of",
                CIMAGame.U_Correct: "Correctly translated whole",
            }
        else:
            raise NotImplementedError
        
        llm_dict = {
            'chatgpt': OpenAIChatModel,
            'chatglm': ChatGLM3Model,
        }
        
        self.system_model = llm_dict[cmd_args.system](cmd_args, cmd_args.system)
        self.user_model = llm_dict[cmd_args.user](cmd_args, cmd_args.user)
        self.planner_model = llm_dict[cmd_args.planner](cmd_args, cmd_args.planner)
        
        SysModel = ModelMap[cmd_args.data_name]['sys']
        UsrModel = ModelMap[cmd_args.data_name]['usr']
        SysPlanner = ModelMap[cmd_args.data_name]['planner']
        
        self.system = SysModel(
            sys_da,
            self.system_model, 
            max_hist_num_turns=cmd_args.max_hist_num_turns,
            conv_examples=[exp_1],
            inference_args={
                "max_new_tokens": cmd_args.resp_max_new_tokens,
                "temperature": cmd_args.resp_temperature,
                "do_sample": True,  # for MCTS open loop
                "return_full_text": False,
            },
            zero_shot=cmd_args.zero_shot
        )
        self.user = UsrModel(
            user_da,
            inference_args={
                "max_new_tokens": cmd_args.resp_max_new_tokens,
                "temperature": cmd_args.resp_temperature,
                "repetition_penalty": 1.0,
                "do_sample": True,  # for MCTS open loop
                "return_full_text": False,
            },
            backbone_model=self.user_model, 
            conv_examples=[exp_1],
            max_hist_num_turns=cmd_args.max_hist_num_turns,
            zero_shot=cmd_args.zero_shot
        )
        self.planner = SysPlanner(
            dialog_acts=self.system.dialog_acts,
            max_hist_num_turns=self.system.max_hist_num_turns,
            user_dialog_acts=self.user.dialog_acts,
            user_max_hist_num_turns=self.user.max_hist_num_turns,
            generation_model=self.planner_model,
            conv_examples=[exp_1],
            zero_shot=cmd_args.zero_shot,
            use_policy_prior=cmd_args.use_policy_prior,
            action_temperature=cmd_args.action_temperature,
            action_num_return_sequences=cmd_args.action_num_return_sequences,
            eval_temperature=cmd_args.reward_temperature,
            eval_num_return_sequences=cmd_args.reward_num_return_sequences
        )
        self.game = ModelMap[cmd_args.data_name]['game'](self.system, self.user, self.planner, cmd_args.zero_shot, 
                                                         cmd_args.max_conv_turns, success_base)
        
        logger.info(f"System dialog acts: {self.system.dialog_acts}")
        logger.info(f"User dialog acts: {self.user.dialog_acts}")

    def _init_state(self, emotion_type, problem_type):
        state = self.game.init_dialog(emotion_type, problem_type)
        return state
    
    def _build_planner(self, args):
        dialog_planner = OpenLoopMCTS(self.game, self.planner, args)
        return dialog_planner
    
    def _collect_da_action(self, dialog_planner, args, state, policy, ent_bound, agent_state, transition_dict=None):
        state_str = ""
        for (role, action, uttr) in state:
            state_str += "({}, {}, {})\t".format(role, action, uttr)
        logger.info('Start from MCTS searching state {}'.format(state_str))
        self.game.system_agent.backbone_model.apply_chatgpt_times = 0
        self.game.user_agent.backbone_model.apply_chatgpt_times = 0
        self.game.planner.generation_model.apply_chatgpt_times = 0
        for i in range(args.num_MCTS_sims):
            logger.info("Searching for {}th simulation".format(i))
            dialog_planner.search(state, policy, ent_bound, agent_state, -0.5, transition_dict)
            logger.info("*****************************")
        
        mcts_policy = dialog_planner.get_action_prob(state)
        # mcts_policy_next_da = self.system.dialog_acts[np.argmax(mcts_policy)]
        # # # fetch the generated utterance from simulation
        # mcts_pred_rep = dialog_planner.get_best_realization(state, np.argmax(mcts_policy))
        
        state, _, reward, _ = dialog_planner._get_next_state(state, np.argmax(mcts_policy), agent_state)
        full_history = dialog_planner.traverse_valid_path(state, rewards=[])
        apply_chatgpt_times = self.game.system_agent.backbone_model.apply_chatgpt_times \
            + self.game.user_agent.backbone_model.apply_chatgpt_times + self.game.planner.generation_model.apply_chatgpt_times
        return state, reward, full_history, transition_dict, apply_chatgpt_times


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default="ESConv", choices=['ESConv', 'P4G'], help='dataset file')
    parser.add_argument('--n_train', type=int, default=2000) #
    parser.add_argument('--n_val', type=int, default=500) #
    parser.add_argument('--n_test', type=int, default=500) #    
    parser.add_argument('--output', type=str, default="outputs/gdpzero.pkl", help='output file')
    parser.add_argument('--log_dir', type=str, default="log", help='log file')
    parser.add_argument('--llm', type=str, default="code-davinci-002", choices=["code-davinci-002", "chatgpt", "gpt-3.5-turbo", "chatglm", 
                                                                                "qwen-7b-chat", "meta/llama-2-7b-chat", "meta/llama-2-13b-chat",
                                                                                "meta/llama-2-70b-chat"], 
                        help='OpenAI model name')
    parser.add_argument('--gen_sentences', type=int, default=-1, help='number of sentences to generate from the llm. Longer ones will be truncated by nltk.')
    parser.add_argument('--num_mcts_sims', type=int, default=20, help='number of mcts simulations')
    parser.add_argument('--max_realizations', type=int, default=3, help='number of realizations per mcts state')
    parser.add_argument('--Q_0', type=float, default=0.0, help='initial Q value for unitialized states. to control exploration')
    parser.add_argument('--num_dialogs', type=int, default=20, help='number of dialogs to test MCTS on')
    parser.add_argument('--max_step', type=int, default=8, help='number of steps')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.parse_args()
    cmd_args = parser.parse_args()
    if not os.path.exists(os.path.dirname(cmd_args.output)):
        os.makedirs(os.path.dirname(cmd_args.output))
    logger.info("saving to {}".format(cmd_args.output))
    
    with open("./data/{}/data/train.txt".format(cmd_args.dataset.lower()), 'r', encoding='utf-8') as file:
        train_data = []
        for line in file:
            train_data.append(eval(line))
    train_data = train_data[:cmd_args.n_train]
    
    gdp = GDPZero(cmd_args)
    gdp.collect_dialogues(train_data, cmd_args.max_step)
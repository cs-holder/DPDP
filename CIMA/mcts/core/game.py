import numpy as np
import logging

from mcts.core.gen_models import DialogModel
from mcts.core.helpers import DialogSession, EmotionSupportDialogSession, CIMADialogSession
from abc import ABC, abstractmethod
from typing import List
from collections import defaultdict as ddict


logger = logging.getLogger(__name__)


class DialogGame(ABC):
	def __init__(self, 
            dataset: str,
			system_name:str, system_agent: DialogModel, 
			user_name: str, user_agent: DialogModel, 
			planner, zero_shot: bool,
			success_base: float
   ):
		self.dataset = dataset
		self.SYS = system_name
		self.system_agent = system_agent
		self.USR = user_name
		self.user_agent = user_agent
		self.planner = planner
		self.zero_shot = zero_shot
		self.success_base = success_base
		return

	@staticmethod
	@abstractmethod
	def get_game_ontology() -> dict:
		"""returns game related information such as dialog acts, slots, etc.
		"""
		raise NotImplementedError

	def init_dialog(self) -> DialogSession:
		# [(sys_act, sys_utt, user_act, user_utt), ...]
		return DialogSession(self.SYS, self.USR)
	
	def get_next_state_batched(self, state:DialogSession, action, batch=3) -> List[DialogSession]:
		all_next_states = [state.copy() for _ in range(batch)]

		sys_utts = self.system_agent.get_utterance_batched(state.copy(), action, batch)  # action is DA
		sys_da = self.system_agent.dialog_acts[action]
		for i in range(batch):
			all_next_states[i].add_single(state.SYS, sys_da, sys_utts[i])
		
		# state in user's perspective
		user_das, user_resps = self.user_agent.get_utterance_w_da_from_batched_states(all_next_states, None)  # user just reply
		for i in range(batch):
			all_next_states[i].add_single(state.USR, user_das[i], user_resps[i])
		return all_next_states

	def display(self, state:DialogSession):
		string_rep = state.to_string_rep(keep_sys_da=True, keep_user_da=True)
		print(string_rep)
		return

	@abstractmethod
	def get_dialog_ended(self, state) -> float:
		"""returns 0 if not ended, then (in general) 1 if system success, -1 if failure 
		"""
		raise NotImplementedError


class EmotionalSupportGame(DialogGame):
	SYS = "Therapist"
	USR = "Patient"

	S_Question = "Question"
	S_SelfDisclosure = "Self-disclosure"
	S_AffirmationAndReassurance = "Affirmation and Reassurance"
	S_ReflectionOfFeelings = "Reflection of feelings"
	S_ProvidingSuggestions = "Providing Suggestions"
	S_Information = "Information"
	S_RestatementOrParaphrasing = "Restatement or Paraphrasing"
	S_Others = "Others"

	U_FeelWorse = "Feel worse"
	U_FeelTheSame = "Feel the same"
	U_FeelBetter = "Feel better"
	U_Solved = "Solved"

	def __init__(self, system_agent:DialogModel, user_agent:DialogModel, 
              planner, zero_shot, max_conv_turns=15, success_base=0.1):
		super().__init__('esc', EmotionalSupportGame.SYS, system_agent, EmotionalSupportGame.USR, user_agent, planner, zero_shot, success_base)
		self.max_conv_turns = max_conv_turns
		return

	@staticmethod
	def get_game_ontology() -> dict:
     	# ['Affirmation and Reassurance', 'Information', 'Others', 'Providing Suggestions', 'Question', 'Reflection of feelings', 'Restatement or Paraphrasing', 'Self-disclosure']
		return {
			"system": {
				"dialog_acts": [
					EmotionalSupportGame.S_AffirmationAndReassurance, EmotionalSupportGame.S_Information, EmotionalSupportGame.S_Others, 
     				EmotionalSupportGame.S_ProvidingSuggestions, EmotionalSupportGame.S_Question, EmotionalSupportGame.S_ReflectionOfFeelings, 
					EmotionalSupportGame.S_RestatementOrParaphrasing, EmotionalSupportGame.S_SelfDisclosure
				],
			},
			"user": {
				"dialog_acts": [
					EmotionalSupportGame.U_FeelWorse, EmotionalSupportGame.U_FeelTheSame, 
     				EmotionalSupportGame.U_FeelBetter, EmotionalSupportGame.U_Solved
				]
			}
		}

	def map_user_action(self, v, sampled_das):
		if v > self.success_base:
			return EmotionalSupportGame.U_Solved
		da_dict = ddict(int)	
		for sample_da in sampled_das:
			if sample_da != 'Solved':
				da_dict[sample_da] += 1
		max_freq_da = max(da_dict, key=lambda x: da_dict[x])
		return max_freq_da

	def get_dialog_ended(self, state) -> float:
		# terminate if there is a <donate> action in persudee resp
		# allow only 10 turns
		for (_, da, _) in state:
			if da == EmotionalSupportGame.U_Solved:
				logger.info("Dialog ended with being solved")
				return 1.0
		if len(state) >= self.max_conv_turns:
			logger.info("Dialog ended with treat failure for reaching maximum turns")
			return -1.0
		return 0.0

	def init_dialog(self, emotion_type, problem_type) -> EmotionSupportDialogSession:
    	# [(sys_act, sys_utt, user_act, user_utt), ...]
		return EmotionSupportDialogSession(self.SYS, self.USR, emotion_type, problem_type)

	def get_next_state(self, state:DialogSession, action, agent_state: list = None, mode: str = 'train') -> DialogSession:
		next_state = state.copy()
		next_agent_state = agent_state.copy()

		sys_utt = self.system_agent.get_utterance(next_state, action)  # action is DA
		sys_da = self.system_agent.dialog_acts[action]
		next_state.add_single(state.SYS, sys_da, sys_utt)
		next_agent_state.append({'role': state.SYS, 'content': sys_utt})

		# state in user's perspective
		if not self.zero_shot:
			user_da, user_resp = self.user_agent.get_utterance_w_da(next_state, None, mode)  # user just reply
			next_state.add_single(state.USR, user_da, user_resp)
			v = None
		else:
			user_resp = self.user_agent.get_utterance(next_state, None, mode)  # user just reply
			next_state.add_single(state.USR, None, user_resp)
			# v, sampled_das = 0.1, ["No, better"] * 10
			# if len(state) == 8:
			# 	v = 0.19
			# 	sampled_das[-1] = "Yes, Solved"
			v, sampled_das = self.planner.heuristic(next_state)
			user_da = self.map_user_action(v, sampled_das)
			next_state[-1][1] = user_da
		next_agent_state.append({'role': state.USR, 'content': user_resp})
		return next_state, next_agent_state, v


class CIMAGame(DialogGame):
	SYS = "Teacher"
	USR = "Student"

	S_Confirmation = "Confirmation"
	S_Correction = "Correction"
	S_Hint = "Hint"
	S_Others = "Others"
	S_Question = "Question"

	U_Incorrect = "Made an incorrect translation"
	U_DidNotTry = "Did not try to translate"
	U_OnlyPart = "Only correctly translated a part of"
	U_Correct = "Correctly translated whole"

	def __init__(self, system_agent:DialogModel, user_agent:DialogModel, 
              planner, zero_shot, max_conv_turns=15, success_base=0.1):
		super().__init__('esc', CIMAGame.SYS, system_agent, CIMAGame.USR, user_agent, planner, zero_shot, success_base)
		self.max_conv_turns = max_conv_turns
		return

	@staticmethod
	def get_game_ontology() -> dict:
     	# []
		return {
			"system": {
				"dialog_acts": [
					CIMAGame.S_Confirmation, CIMAGame.S_Correction, CIMAGame.S_Hint, 
					CIMAGame.S_Others, CIMAGame.S_Question,
				],
			},
			"user": {
				"dialog_acts": [
					CIMAGame.U_Incorrect, CIMAGame.U_DidNotTry, 
     				CIMAGame.U_OnlyPart, CIMAGame.U_Correct
				]
			}
		}

	def map_user_action(self, v, sampled_das):
		if v >= self.success_base:
			return CIMAGame.U_Correct
		da_dict = ddict(int)	
		for sample_da in sampled_das:
			if sample_da != "Correctly translated whole":
				da_dict[sample_da] += 1
		max_freq_da = max(da_dict, key=lambda x: da_dict[x])
		return max_freq_da

	def get_dialog_ended(self, state) -> float:
		# terminate if there is a <donate> action in persudee resp
		# allow only 10 turns
		for (_, da, _) in state:
			if da == CIMAGame.U_Correct:
				logger.info("Dialog ended with being taught")
				return 1.0
		if len(state) >= self.max_conv_turns:
			logger.info("Dialog ended with treat failure for reaching maximum turns")
			return -1.0
		return 0.0

	def init_dialog(self, sentence, target, history) -> CIMADialogSession:
    	# [(sys_act, sys_utt, user_act, user_utt), ...]
		return CIMADialogSession(self.SYS, self.USR, sentence, target, history)

	def get_next_state(self, state:DialogSession, action, agent_state: list = None, mode: str = 'train') -> DialogSession:
		next_state = state.copy()
		next_agent_state = agent_state.copy()

		sys_utt = self.system_agent.get_utterance(next_state, action)  # action is DA
		sys_da = self.system_agent.dialog_acts[action]
		next_state.add_single(state.SYS, sys_da, sys_utt)
		next_agent_state.append({'role': state.SYS, 'content': sys_utt})

		# state in user's perspective
		if not self.zero_shot:
			user_da, user_resp = self.user_agent.get_utterance_w_da(next_state, None, mode)  # user just reply
			next_state.add_single(state.USR, user_da, user_resp)
			v = None
		else:
			user_resp = self.user_agent.get_utterance(next_state, None, mode)  # user just reply
			next_state.add_single(state.USR, None, user_resp)
			# v, sampled_das = 0.1, ["No, better"] * 10
			# if len(state) == 8:
			# 	v = 0.19
			# 	sampled_das[-1] = "Yes, Solved"
			# if len(next_state) == self.max_conv_turns:
			# 	v = 1.0
			# 	sampled_das = ["Correctly translated whole" for _ in range(10)]
			# else:
			# 	v = 0.9
			# 	sampled_das = ['Only correctly translated a part of' for _ in range(10)]
			v, sampled_das = self.planner.heuristic(next_state)
			user_da = self.map_user_action(v, sampled_das)
			next_state[-1][1] = user_da
		next_agent_state.append({'role': state.USR, 'content': user_resp})
		return next_state, next_agent_state, v
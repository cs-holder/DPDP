import numpy as np
import logging
import math

from mcts.core.helpers import DialogSession
from mcts.core.game import DialogGame
from mcts.core.esc_players import DialogPlanner
from mcts.core.game import EmotionalSupportGame


logger = logging.getLogger(__name__)


class MCTS():
	def __init__(self, dataset, game:DialogGame, player:DialogPlanner, configs) -> None:
		self.dataset = dataset
		self.game = game
		self.player = player
		self.configs = configs
		# U(s,a) = Q(s,a) + c * P(s,a) * (\sqrt{ \sum_{a'} N(s,a')}) / (1+N(s,a))
		self.Ns: dict = {}  # saves compute
		self.Nsa: dict = {}
		self.Q: dict = {}
		self.P: dict = {}
		# utility
		self.valid_moves: dict = {}
		self.terminals: dict = {}
		# debugging / more information
		self.Vs: dict = {}
		return

	def _to_string_rep(self, state:DialogSession):
		# for tree search, keep all dialog turns
		return state.to_string_rep(keep_sys_da=True, keep_user_da=True, max_turn_to_display=-1)

	def _init_node(self, state:DialogSession):
		hashable_state = self._to_string_rep(state)
		allowed_actions = self.player.get_valid_moves(state)
		self.valid_moves[hashable_state] = allowed_actions.nonzero()[0]

		self.Ns[hashable_state] = 0
		self.Nsa[hashable_state] = {action: 0 for action in self.valid_moves[hashable_state]}
		self.Q[hashable_state] = {action: self.configs.Q_0 for action in self.valid_moves[hashable_state]}

		prior, v = self.player.predict(state)
		self.Vs[state.to_string_rep(keep_sys_da=True, keep_user_da=True)] = v  # for debugging
		self.P[hashable_state] = prior * allowed_actions
		# renormalize
		if np.sum(self.P[hashable_state]) == 0:
			self.P[hashable_state] = allowed_actions / np.sum(allowed_actions)
			logger.warning("This should never happen")
		else:
			self.P[hashable_state] /= np.sum(self.P[hashable_state])
		return v

	def search(self, state:DialogSession):
		hashable_state = self._to_string_rep(state)
		
		is_leaf_node = False
		v = 0.0
		if hashable_state not in self.terminals:
			# selected leaf node, expand
			self.terminals[hashable_state] = self.game.get_dialog_ended(state)
			v = self._init_node(state)
			is_leaf_node = True
		# if this leaf node is terminal, return the value
		if self.terminals[hashable_state] > 0:
			# terminal node
			logger.debug("ended")
			return self.terminals[hashable_state]
		# otherwise, return v
		if is_leaf_node:
			return v
		
		# existing, continue selection
		# go next state by picking best according to U(s,a)
		best_uct = -float('inf')
		best_action = -1
		for a in self.valid_moves[hashable_state]:
			Ns = self.Ns[hashable_state]
			if Ns == 0:
				Ns = 1e-8
			uct = self.Q[hashable_state][a] + self.configs.cpuct * self.P[hashable_state][a] * math.sqrt(Ns) / (1 + self.Nsa[hashable_state][a])
			if uct > best_uct:
				best_uct = uct
				best_action = a
		# transition
		next_state = self.game.get_next_state(state, best_action)
		
		# 1. if not leaf, continue traversing, and state=s will get the value from the leaf node
		# 2. if leaf, we will expand it and return the value for backpropagation
		v = self.search(next_state)

		# update stats
		# add in new estimate and average
		self.Q[hashable_state][best_action] = (self.Nsa[hashable_state][best_action] * self.Q[hashable_state][best_action] + v) / (self.Nsa[hashable_state][best_action] + 1)
		self.Ns[hashable_state] += 1
		self.Nsa[hashable_state][best_action] += 1
		
		# now we are single player, hence just v instead of -v
		return v

	def get_action_prob(self, state:DialogSession):
		hashable_state = self._to_string_rep(state)
		if hashable_state not in self.Ns:
			# selected leaf node, expand
			logging.warn("querying a state that has not been visited")
			self._init_node(state)
		# get the counts for all moves
		# convert to prob
		prob = np.zeros(self.player.get_valid_moves(state).shape)
		for a in self.valid_moves[hashable_state]:
			prob[a] = self.Nsa[hashable_state][a]
		prob /= prob.sum()
		return prob


class OpenLoopMCTS(MCTS):
	def __init__(self, dataset, game, player, configs, success_base) -> None:
		super().__init__(dataset, game, player, configs)
		self.realizations: dict = {}  # state -> list of real DialogSessions
		self.realizations_Vs: dict = {}  # state -> {realization: V(realization)}
		self.realizations_Ns: dict = {}  # state -> {realization: N(realization)}
		self.max_realizations = configs.max_realizations
		self.success_base = success_base
		return

	def _to_string_rep(self, state:DialogSession):
		# for tree search, keep all dialog turns
		das = []
		for (speaker, da, _) in state:
			if speaker == state.SYS:
				das.append(da)
		return "__".join(das)

	def _init_node(self, state:DialogSession, policy, ent_bound: float, agent_state: list, reward: float):
		hashable_state = self._to_string_rep(state)
		allowed_actions = self.player.get_valid_moves(state)
		self.valid_moves[hashable_state] = allowed_actions.nonzero()[0]

		self.Ns[hashable_state] = 0
		self.Nsa[hashable_state] = {action: 0 for action in self.valid_moves[hashable_state]}
		self.Q[hashable_state] = {action: self.configs.Q_0 for action in self.valid_moves[hashable_state]}
		self.realizations[hashable_state] = [(state.copy(), agent_state.copy())]
		self.Vs[hashable_state] = [reward]

		prior = self.player.predict(state, policy, ent_bound, agent_state)
		# self.Vs[state.to_string_rep(keep_sys_da=True, keep_user_da=True)] = v  # for debugging
		self.P[hashable_state] = prior * allowed_actions
		# renormalize
		if np.sum(self.P[hashable_state]) == 0:
			self.P[hashable_state] = allowed_actions / np.sum(allowed_actions)
			logger.warning("This should never happen")
		else:
			self.P[hashable_state] /= np.sum(self.P[hashable_state])
		logger.info('Initiate a new node, hashable state: {}, prior probability: {}'.format(hashable_state, self.P[hashable_state]))
		return reward

	def _sample_realization(self, hashable_state):
		rand_i = np.random.randint(len(self.realizations[hashable_state]))
		next_state, next_agent_state = self.realizations[hashable_state][rand_i]
		v = self.Vs[hashable_state][rand_i]
		return next_state, next_agent_state, v

	def _add_new_realizations(self, state, agent_state, reward):
		hashable_state = self._to_string_rep(state)
		if hashable_state not in self.realizations:
			self.realizations[hashable_state] = []
		if hashable_state not in self.Vs:
			self.Vs[hashable_state] = []
		if (state, agent_state) in self.realizations[hashable_state]:
			return
		
		self.realizations[hashable_state].append((state.copy(), agent_state.copy()))
		self.Vs[hashable_state].append(reward)
		if len(self.realizations[hashable_state]) > self.max_realizations:
			# should never happen
			logger.warning(f"len(self.realizations[hashable_state])={len(self.realizations[hashable_state])}")
			self.realizations[hashable_state].pop(0)
			self.Vs[hashable_state].pop(0)
		return

	def _get_next_state(self, state, best_action, agent_state, transition_dict=None):
		prefetch_state = self._to_string_rep(state) + "__" + self.player.dialog_acts[best_action]
		if prefetch_state in self.realizations and len(self.realizations[prefetch_state]) == self.max_realizations:
			# use the cached realization
			next_state, next_agent_state, v = self._sample_realization(prefetch_state)
			return next_state, next_agent_state, v, transition_dict
		
		# otherwise, generate a new realization
		next_state, next_agent_state, v = self.game.get_next_state(state, best_action, agent_state)
		if transition_dict is None:
			transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
		transition_dict['states'].append(agent_state)
		transition_dict['actions'].append(int(best_action))
		transition_dict['next_states'].append(next_agent_state)
		transition_dict['rewards'].append(v)
		transition_dict['dones'].append(1.0 if v >= self.success_base else 0.0)
		return next_state, next_agent_state, v, transition_dict
	
	def _update_realizations_Vs(self, state: DialogSession, v: float):
		hashable_state = self._to_string_rep(state)
		if hashable_state not in self.realizations_Vs:
			self.realizations_Vs[hashable_state] = {}
			self.realizations_Ns[hashable_state] = {}
		sys_utt = state.get_turn_utt(
			turn=-1,
			role=state.SYS,
		)
		if sys_utt not in self.realizations_Vs[hashable_state]:
			self.realizations_Vs[hashable_state][sys_utt] = 0
			self.realizations_Ns[hashable_state][sys_utt] = 0
		# update
		self.realizations_Ns[hashable_state][sys_utt] += 1
		self.realizations_Vs[hashable_state][sys_utt] += (v - self.realizations_Vs[hashable_state][sys_utt]) / self.realizations_Ns[hashable_state][sys_utt]
		return

	def update_agent_state(self, agent_state: list, sys_utt: str, usr_utt: str):
		agent_state.append({'role': EmotionalSupportGame.SYS, 'content': sys_utt})
		agent_state.append({'role': EmotionalSupportGame.USR, 'content': usr_utt})
 
	def search(self, state:DialogSession, policy, ent_bound: float, agent_state: list, reward: float, transition_dict: dict):
		hashable_state = self._to_string_rep(state)
		# check everytime since state is stochastic, does not map to hashable_state
		
		# otherwise, if is nontermial leaf node, we initialize and return v
		if hashable_state not in self.P:
			# selected leaf node, expand it
			# first visit V because v is only evaluated once for a hashable_state
			reward = self._init_node(state, policy, ent_bound, agent_state, reward)
			return reward
		else:
			# add only when it is new
			self._add_new_realizations(state, agent_state, reward)

		terminated_v = self.game.get_dialog_ended(state)
		# check if it is terminal node
		if terminated_v != 0.0:
			logger.info("ended")
			return reward
		
		# existing, continue selection
		# go next state by picking best according to U(s,a)
		best_uct = -float('inf')
		best_action = -1
		for a in self.valid_moves[hashable_state]:
			Ns = self.Ns[hashable_state]
			if Ns == 0:
				Ns = 1e-8
			# a variant of PUCT
			uct = self.Q[hashable_state][a] + self.configs.cpuct * self.P[hashable_state][a] * math.sqrt(Ns) / (1 + self.Nsa[hashable_state][a])
			if uct > best_uct:
				best_uct = uct
				best_action = a
		# transition. For open loop, first sample from an existing realization
		state, agent_state, _ = self._sample_realization(hashable_state)
		next_state, next_agent_state, next_v, transition_dict = self._get_next_state(state, best_action, agent_state, transition_dict)
		logger.info('Choose best system action "{}" for current state "{}"'.format(self.game.system_agent.dialog_acts[best_action], hashable_state))
		logger.info('{}: "{}"'.format(next_state[-2][0], next_state[-2][-1]))
		logger.info('{}: "{}"'.format(next_state[-1][0], next_state[-1][-1]))
		logger.info("{}'s reaction: '{}'".format(next_state[-1][0], next_state[-1][1]))
		
		# 1. if not leaf, continue traversing, and state=s will get the value from the leaf node
		# 2. if leaf, we will expand it and return the value for backpropagation
		leaf_v = self.search(next_state, policy, ent_bound, next_agent_state, next_v, transition_dict)

		# update stats
		# add in new estimate and average
		used_leaf_v = 1.0 if leaf_v >= self.success_base else leaf_v
		self.Q[hashable_state][best_action] = (self.Nsa[hashable_state][best_action] * self.Q[hashable_state][best_action] + used_leaf_v) / (self.Nsa[hashable_state][best_action] + 1)
		self.Ns[hashable_state] += 1
		self.Nsa[hashable_state][best_action] += 1

		# update v to realizations for NLG at inference
		self._update_realizations_Vs(next_state, used_leaf_v)
		# now we are single player, hence just v instead of -v
		return leaf_v
	
	def get_best_realization(self, state:DialogSession, action: int):
		prefetch_state = self._to_string_rep(state) + "__" + self.player.dialog_acts[action]
		if prefetch_state not in self.realizations_Vs:
			raise Exception("querying a state that has no realizations sampled before")
		# get the counts for all moves
		# convert to prob
		curr_best_v = -float('inf')
		curr_best_realization = None
		for sys_utt, v in self.realizations_Vs[prefetch_state].items():
			if v > curr_best_v:
				curr_best_v = v
				curr_best_realization = sys_utt
		return curr_best_realization
	
	def traverse_valid_path(self, state, rewards):
		hashable_state = self._to_string_rep(state)
		if self.Vs[hashable_state][0] >= self.success_base:
			rewards.append(self.Vs[hashable_state][0])
			return {'state': state.history, 'rewards': rewards}
		if sum(self.Q[hashable_state]) == 0.:
			return None
		rewards.append(self.Vs[hashable_state][0])
		best_action = max(self.Q[hashable_state], key=lambda x: self.Q[hashable_state][x])
		prefetch_state = self._to_string_rep(state) + "__" + self.player.dialog_acts[best_action]
		if prefetch_state in self.realizations and len(self.realizations[prefetch_state]) == self.max_realizations:
			# use the cached realization
			next_state, _, _ = self._sample_realization(prefetch_state)
			return self.traverse_valid_path(next_state, rewards)
		else:
			return None
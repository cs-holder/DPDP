class DialogSession():
	def __init__(self, sys_name, user_name, history=None) -> None:
		self.SYS = sys_name
		self.USR = user_name
		if history:
			self.history: list = history  # [(role, da, utt), ....]
		else:
			self.history: list = []
		return
	
	def from_history(self, history):
		self.history = history
		return self

	def to_string_rep(self, keep_sys_da=False, keep_user_da=False, max_turn_to_display=-1):
		history = ""
		num_turns_to_truncate = 0
		if max_turn_to_display > 0:
			num_turns_to_truncate = max(0, len(self.history) // 2 - max_turn_to_display)
		
		for i, (role, da, utt) in enumerate(self.history):
			if (i // 2) < num_turns_to_truncate:
				continue
			if i % 2 == 0:
				assert(role == self.SYS)
				if keep_sys_da:
					history += f"{role}: [{da}] {utt}\n"
				else:
					history += f"{role}: {utt}\n"
			else:
				assert(role == self.USR)
				if keep_user_da:
					history += f"{role}: [{da}] {utt}\n"
				else:
					history += f"{role}: {utt}\n"
		return history.strip()

	def copy(self):
		new_session = DialogSession(self.SYS, self.USR)
		new_session.from_history(self.history.copy())
		return new_session

	def add_single(self, role, da, utt):
		if len(self.history) % 2 == 0:
			assert(role == self.SYS)
		else:
			assert(role == self.USR)
		self.history.append([role, da, utt])
		return
	
	def get_turn_utt(self, turn, role):
		if role == self.SYS:
			return self.history[turn * 2][-1]
		else:
			return self.history[turn * 2 + 1][-1]
	
	def __iter__(self):
		return iter(self.history)

	def __len__(self):
		return len(self.history) //  2  # number of turns

	def __getitem__(self, index):
		return self.history[index]

	def __eq__(self, __o: object) -> bool:
		if not isinstance(__o, DialogSession):
			return False
		return self.history == __o.history


class EmotionSupportDialogSession(DialogSession):
	def __init__(self, sys_name, user_name, emotion_type=None, problem_type=None, history=None):
		super().__init__(sys_name, user_name, history)
		self.emotion_type = emotion_type
		self.problem_type = problem_type

	def copy(self):
		new_session = EmotionSupportDialogSession(self.SYS, self.USR, self.emotion_type, self.problem_type)
		new_session.from_history(self.history.copy())
		return new_session

	def add_single(self, role, da, utt):
		super().add_single(role, da, utt)
		return


class CIMADialogSession(DialogSession):
	def __init__(self, sys_name, user_name, sentence=None, target=None, history=None):
		super().__init__(sys_name, user_name, history)
		self.sentence = sentence
		self.target = target

	def copy(self):
		new_session = CIMADialogSession(self.SYS, self.USR, self.sentence, self.target)
		new_session.from_history(self.history.copy())
		return new_session

	def add_single(self, role, da, utt):
		super().add_single(role, da, utt)
		return
import logging, re
import numpy as np
import torch

from typing import List, Tuple
from mcts.core.helpers import DialogSession, CBDialogSession
from mcts.core.gen_models import GenerationModel, DialogModel
from mcts.core.game import CBGame
from mcts.utils.utils import safe_entropy
from abc import ABC, abstractmethod
from collections import Counter
from ppdpp.utils import reward_dict


logger = logging.getLogger(__name__)


class DialogPlanner(ABC):
    @abstractmethod
    def get_valid_moves(self, state):
        # 1 if the i-th dialog act is valid, 0 otherwise
        pass

    @abstractmethod
    def predict(self, state) -> "Tuple[np.ndarray, float]":
        # returns a prob and value
        pass


class CBSystemPlanner(DialogPlanner):
    def __init__(
        self,
        dialog_acts,
        max_hist_num_turns,
        user_dialog_acts,
        user_max_hist_num_turns,
        generation_model: GenerationModel,
        conv_examples: List[DialogSession] = [],
    ) -> None:
        super().__init__()
        self.dialog_acts = dialog_acts
        self.max_hist_num_turns = max_hist_num_turns  # used in prompting next da
        self.user_dialog_acts = user_dialog_acts
        self.user_max_hist_num_turns = (
            user_max_hist_num_turns  # used in heuristic function
        )
        self.conv_examples = conv_examples
        self.generation_model = generation_model
        self.smoothing = 1.0
        self.task_prompt = f"""
        Now enter the role-playing mode. In the following conversation, you will play as a therapist in a counselling conversation with a patient.
        You can choose amongst the following actions during a conversation to respond to the patient:
        {" ".join([f"[{da}]" for da in dialog_acts])}
        The following is a new conversation between Therapist and a Patient.
        {self.process_exp()}
        The following is a new conversation between another Therapist and Patient.
        """
        self.task_prompt = self.task_prompt.replace("\t", "").strip()

        self.inf_args = {
            "max_new_tokens": 8,
            "temperature": 1.0,
            "return_full_text": False,
            "do_sample": True,
            "num_return_sequences": 15,
        }
        return

    def process_exp(self, keep_sys_da=True, keep_user_da=False):
        prompt_exps = ""
        for exp in self.conv_examples:
            prompt_exps += (
                exp.to_string_rep(keep_sys_da=keep_sys_da, keep_user_da=keep_user_da)
                + "\n"
            )
        return prompt_exps.strip()

    def get_valid_moves(self, state):
        # 1 if the i-th dialog act is valid, 0 otherwise
        turn = len(state)
        if turn < 1:
            return np.array(
                [1 if da == CBGame.S_Greet else 0 for da in self.dialog_acts]
            )
        return np.array([1 for _ in self.dialog_acts])

    def get_utterance(self, state, action) -> str:
        return ""  # should not be called

    def _get_generated_da(self, data) -> list:
        # convert generated responses to DA
        pred_da = []
        for resp in data:
            resp = resp["generated_text"].strip()
            start_idx = resp.find("[")
            end_idx = resp.find("]")
            if start_idx == -1 or end_idx == -1:
                continue
            found_da = resp[start_idx + 1 : end_idx].strip()
            if found_da in self.dialog_acts:
                pred_da.append(found_da)
        return pred_da

    def predict(self, state: DialogSession) -> "Tuple[np.ndarray, float]":
        # test k times and compute prob. See num_return_sequences in the API
        # the value would be our objective function
        if len(state) == 0:
            prompt = f"""
            {self.task_prompt}
            Persuader:
            """
        else:
            prompt = f"""
            {self.task_prompt}
            {state.to_string_rep(keep_sys_da=True)}
            Persuader:
            """
        prompt = prompt.replace("\t", "").strip()
        logger.debug(prompt)
        data = self.generation_model.generate(prompt, **self.inf_args)
        sampled_das = self._get_generated_da(data)
        logger.debug(f"sampled das: {sampled_das}")
        # convert to prob distribution
        prob = np.zeros(len(self.dialog_acts))
        prob += self.smoothing
        for da in sampled_das:
            prob[self.dialog_acts.index(da)] += 1
        prob /= prob.sum()
        v = self.heuristic(state)
        return prob, v

    def _get_user_generated_da(self, data) -> list:
        # convert generated responses to DA
        pred_da = []
        for resp in data:
            resp = resp["generated_text"].strip()
            start_idx = resp.find("[")
            end_idx = resp.find("]")
            if start_idx == -1 or end_idx == -1:
                continue
            found_da = resp[start_idx + 1 : end_idx].strip()
            if found_da in self.user_dialog_acts:
                pred_da.append(found_da)
        return pred_da

    def heuristic(self, state: DialogSession) -> float:
        # insert prop to donate, and compute the likelihood of user simulator agreeing to donate
        assert state[-1][0] == CBGame.USR
        prompt = f"""
        The following is background information about task. 
        The Persuader is trying to persuade the Persuadee to donate to Save the Children.
        The Persuadee can choose amongst the following actions during a conversation to respond to the Persuader:
        {" ".join([f"[{da}]" for da in self.user_dialog_acts])}
        The following is a conversation between a Persuader and	a Persuadee about a charity called Save the Children. 
        {self.process_exp(keep_sys_da=False, keep_user_da=True)}
        The following is a new conversation between another Persuader and Persuadee.
        {state.to_string_rep(keep_user_da=True, max_turn_to_display=self.user_max_hist_num_turns)}
        Persuader: Would you be interested in donating to Save the Children?
        Persuadee:
        """
        prompt = prompt.replace("\t", "").strip()

        inf_args = {
            "max_new_tokens": 8,
            "temperature": 1.1,
            "return_full_text": False,
            "do_sample": True,
            "num_return_sequences": 10,
        }
        data = self.generation_model.generate(prompt, **inf_args)
        sampled_das = self._get_user_generated_da(data)

        logger.debug(f"persuadee prompt: {prompt}")
        logger.debug(f"sampled das: {sampled_das}")

        # heuristic score
        score = []
        for da in sampled_das:
            if da == CBGame.U_No_deal:
                score.append(-1.0)
            elif da == CBGame.U_Deal:
                score.append(1.0)
        v = 0.0 if len(score) == 0 else np.mean(score)
        logger.debug(f"sampled das to v: {v}")
        return float(v)


class CBChatSystemPlanner(CBSystemPlanner):
    def __init__(
        self,
        dialog_acts,
        max_hist_num_turns,
        user_dialog_acts,
        user_max_hist_num_turns,
        generation_model: GenerationModel,
        conv_examples: List[DialogSession] = [],
        zero_shot = True,
        use_policy_prior = True,
        action_temperature = 1.0,
        action_num_return_sequences = 15,
        eval_temperature = 1.0,
        eval_num_return_sequences = 10,
        neg_reward = -1.0,
    ) -> None:
        super().__init__(
            dialog_acts,
            max_hist_num_turns,
            user_dialog_acts,
            user_max_hist_num_turns,
            generation_model,
            conv_examples,
        )
        self.zero_shot = zero_shot
        self.use_policy_prior = use_policy_prior
        self.task_prompt = f"""
        Now enter the role-playing mode. In the following conversation, you will play as a therapist in a counselling conversation with a patient.
        You can choose amongst the following actions during a conversation to respond to the patient:
        {" ".join([f"[{da}]" for da in dialog_acts])}
        The following is an example conversation between a Therapist and a Patient.
        """.replace(
            "\t", ""
        ).strip()
        self.new_task_prompt = "The following is a new conversation between Therapist (you) and a Patient."
        self.prompt_examples = self.process_chat_exp(new_task_prompt=self.new_task_prompt)

        self.inf_args = {
            "max_new_tokens": 16,
            "temperature": action_temperature,
            "return_full_text": False,
            "do_sample": True,
            "num_return_sequences": action_num_return_sequences,
        }
        self.eval_args = {
            "max_new_tokens": 16,
            "temperature": eval_temperature,
            "num_return_sequences": eval_num_return_sequences,
        }
        self.neg_reward = neg_reward
        return

    def process_chat_exp(
        self,
        new_task_prompt,
        assistant_role=CBGame.SYS,
        keep_sys_da=True,
        keep_user_da=False,
    ):
        prompt_exps = []
        for exp in self.conv_examples:
            prompt_exps += self.__proccess_chat_exp(
                exp, keep_sys_da, keep_user_da, assistant_role
            )
            prompt_exps.append({"role": "system", "content": new_task_prompt})
        return prompt_exps[:-1]

    def __proccess_chat_exp(
        self,
        exp: DialogSession,
        keep_sys_da,
        keep_user_da,
        assistant_role=CBGame.SYS,
        max_hist_num_turns: int = -1,
    ):
        if len(exp) == 0:
            return []
        # P4G dataset starts with the system/Persuader
        assert exp[0][0] == CBGame.SYS

        prompt_messages = []
        num_turns_to_truncate = 0
        if max_hist_num_turns > 0:
            num_turns_to_truncate = max(0, len(exp) // 2 - max_hist_num_turns)

        # all the rest
        for i, (role, da, utt) in enumerate(exp):
            # truncate to reduce the size of the prompt
            if (i // 2) < num_turns_to_truncate:
                continue
            # if assistant is the Persuader, then current data is also Persuader -> then it is of role "system"
            if role == CBGame.SYS:
                if keep_sys_da:
                    content = f"{role}: [{da}] {utt}".strip()
                else:
                    content = f"{role}: {utt}".strip()
                if assistant_role == CBGame.SYS:
                    prompt_role = "assistant"
                else:
                    prompt_role = "user"
            else:
                if keep_user_da:
                    content = f"{role}: [{da}] {utt}".strip()
                else:
                    content = f"{role}: {utt}".strip()
                if assistant_role == CBGame.USR:
                    prompt_role = "assistant"
                else:
                    prompt_role = "user"

            prompt_messages.append({"role": prompt_role, "content": content})
        return prompt_messages

    def get_valid_moves(self, state):
        # 1 if the i-th dialog act is valid, 0 otherwise
        turn = len(state)
        if turn < 1:
            return np.array(
                [
                    1 if da == CBGame.S_Greet else 0
                    for da in self.dialog_acts
                ]
            )
        return np.array([1 for _ in self.dialog_acts])

    def get_utterance(self, state, action) -> str:
        return ""  # should not be called

    def _get_generated_da(self, data) -> list:
        # convert generated responses to DA
        action_list = [CBGame.S_Affirm, CBGame.S_Agree, CBGame.S_Confirm, CBGame.S_Counter, 
                       CBGame.S_Counter_noprice, CBGame.S_Deny, CBGame.S_Disagree, CBGame.S_Greet, 
                       CBGame.S_Information, CBGame.S_Inquire, CBGame.S_Propose]
        pred_da = []
        for resp in data:
            resp = resp["generated_text"].strip()
            start_idx = resp.find("[")
            end_idx = resp.find("]")
            if start_idx == -1 and end_idx == -1:
                continue
            found_da = resp[start_idx + 1 : end_idx].strip()
            if found_da in self.dialog_acts:
                pred_da.append(found_da)
            # for action in action_list:
            #     if action.lower() in resp.lower():
            #         pred_da.append(action)
            #         break
        return pred_da

    def predict(self, state: DialogSession, policy, ent_bound, agent_state) -> "Tuple[np.ndarray, float]":
        # test k times and compute prob. See num_return_sequences in the API
        # the value would be our objective function
        if self.use_policy_prior:
            logger.info('Apply policy model to calculate prior')
            with torch.no_grad():
                agent_dist, _ = policy.apply_policy(agent_state)
            # agent_dist = torch.ones(len(self.dialog_acts)).to(policy.device) / len(self.dialog_acts)
            logger.info('Apply the policy network (Roberta-large) to predict prior distribution.')
            if len(agent_dist.shape) > 1:
                agent_dist = agent_dist.squeeze(dim=0)
            prob = agent_dist.detach().cpu().numpy()
        else:
            logger.info('Apply LLM to calculate prior')
            messages = [
                {"role": "system", "content": self.task_prompt},
                *self.prompt_examples,
                {'role': 'system', 'content': self.new_task_prompt}
            ]
            if len(state) == 0:
                messages.append(
                    {"role": "user", "content": f"{CBGame.USR}: Hello."}
                )
            else:
                assert state[-1][0] == CBGame.USR
                messages += self.__proccess_chat_exp(
                    state, keep_sys_da=True, keep_user_da=False
                )
            # produce a response
            data = self.generation_model.chat_generate(messages, **self.inf_args)

            sampled_das = self._get_generated_da(data)
            logger.info(f"sampled das: {sampled_das}")
            # convert to prob distribution
            prob = np.zeros(len(self.dialog_acts))
            prob += self.smoothing
            for da in sampled_das:
                try:
                    prob[self.dialog_acts.index(da)] += 1
                except:
                    continue
            prob /= prob.sum()
        return prob

    def _get_user_generated_da(self, data) -> list:
        # convert generated responses to DA
        pred_da = []
        for resp in data:
            resp = resp['generated_text'].strip()
            start_idx = resp.find("[")
            end_idx = resp.find("]")
            if start_idx == -1 or end_idx == -1:
                continue
            found_da = resp[start_idx + 1: end_idx].strip()
            if found_da in self.user_dialog_acts:
                pred_da.append(found_da)
        return pred_da

    def _get_zero_shot_user_da(self, data) -> list:
        # convert generated responses to DA
        pred_da = []
        for resp in data:
            resp = resp['generated_text'].lower()
            if 'have no' in resp:
                pred_da.append(CBGame.U_No_deal)
            else:
                pred_da.append(CBGame.U_Deal)
        return pred_da

    def heuristic(self, state: DialogSession) -> float:
        # insert prop to donate, and compute the likelihood of user simulator agreeing to donate
        assert state[-1][0] == CBGame.USR
        if not self.zero_shot:
            user_task_prompt = f"""
            Given a conversation between a Therapist and a Patient, please assess whether the Patient' emotional issue has been solved after the conversation.
            You can choose amongst the following actions during a conversation to respond to the Therapist:
            {" ".join([f"[{da}]" for da in self.user_dialog_acts])}
            The following is a example conversation between a Therapist and a Patient.
            """.replace(
                "\t", ""
            ).strip()
            user_new_task_prompt = "The following is a new conversation between a Therapist and a Patient (you)."

            messages = [
                {"role": "system", "content": user_task_prompt},
                *self.process_chat_exp(
                    new_task_prompt=user_new_task_prompt,
                    assistant_role=CBGame.USR,
                    keep_sys_da=False,
                    keep_user_da=True,
                ),
                {"role": "system", "content": user_new_task_prompt},
            ]
            messages += self.__proccess_chat_exp(
                state,
                assistant_role=CBGame.USR,
                keep_sys_da=False,
                keep_user_da=True,
            )
            messages.append(
                {
                    "role": "user",
                    "content": f"{CBGame.SYS}: Has the your issue been solved?",
                }
            )
        else:
            conversation = self.__proccess_chat_exp(
                state,
                assistant_role=CBGame.USR,
                keep_sys_da=False,
                keep_user_da=False,
            )
            dial = ''
            for turn in conversation:
                dial += "\n{}".format(turn['content'])
            messages = [
                {"role": "system", "content": "Given a conversation between a Buyer and a Seller, please decide whether the Buyer and the Seller have reached a deal at the end of the conversation."},
                {"role": "user", "content": "Please decide whether the Buyer and the Seller have reached a deal at the end of the conversation. If they have reached a deal, please extract the deal price as [price]. You can only reply with one of the following sentences: They have reached a deal at [price]. They have not reached a deal.\n\nThe following is the conversation: Buyer: Can we meet in the middle at $15? Seller: Sure, let's meet at $15 for this high-quality balloon.\nQuestion: Have they reached a deal? Answer: They have reached a deal at $15.\n\nThe following is the conversation: Buyer: That's still a bit high, can you go any lower? Seller: Alright, I can sell it to you for $15.\nQuestion: Have they reached a deal? Answer: They have not reached a deal.\n\nThe following is the conversation: %s\nQuestion: Have they reached a deal? Answer: " % dial}
            ]

        data = self.generation_model.chat_generate(messages, **self.eval_args)
        # if not self.zero_shot:
        #     sampled_das = self._get_user_generated_da(data)
        # else:
        #     sampled_das = self._get_zero_shot_user_da(data)

        logger.info(f"persuadee prompt: {messages}")
        # logger.info(f"sampled das: {sampled_das}")
        
        deals, rewards, sampled_das = [], [], []
        for resp in data:
            if 'have not' in resp['generated_text'].lower():
                deals.append(-1)
                sampled_das.append('no deal')
            elif 'have reached' in resp['generated_text'].lower():
                deals.append(1)
                sampled_das.append('deal')
            
            prices = re.findall(r"[-+]?\d*\.?\d+", resp['generated_text'].replace(",",""))
            if len(prices) > 0:
                deal_price = float(prices[0])
                reward = (deal_price - state.seller_price) / (state.buyer_price - state.seller_price)
                rewards.append(reward)

        if -1 in deals:
            reward = self.neg_reward
        else:
            if len(rewards) == 0:
                reward = 0
            else:
                reward = max(set(rewards), key = rewards.count)
        v = reward
        logger.info(f"sampled das to v: {v}")
        return float(v), sampled_das


class BuyerModel(DialogModel):
    def __init__(
        self,
        dialog_acts: List[str],
        backbone_model: GenerationModel,
        max_hist_num_turns: int = 5,
        conv_examples: List[DialogSession] = [],
        inference_args: dict = {},
        zero_shot: bool = True,
    ):
        super().__init__()
        self.conv_examples = conv_examples
        self.backbone_model = backbone_model
        self.max_hist_num_turns = max_hist_num_turns
        self.zero_shot = zero_shot
        # prompts and DAs
        self.da_prompts_mapping = {
            CBGame.S_Greet: 'Please say hello or chat randomly.',
            CBGame.S_Inquire: 'Please ask any question about product, year, price, usage, etc.',
            CBGame.S_Information: 'Please provide information about the product, year, usage, etc.',
            CBGame.S_Propose: 'Please initiate a price or a price range for the product.',
            CBGame.S_Counter: 'Please propose a new price or a new price range.',
            CBGame.S_Counter_noprice: 'Please propose a vague price by using comparatives with existing price.',
            CBGame.S_Confirm: 'Please ask a question about the information to be confirmed.',
            CBGame.S_Affirm: 'Please give an affirmative response to a confirm.',
            CBGame.S_Deny: 'Please give a negative response to a confirm.',
            CBGame.S_Agree: 'Please agree with the proposed price.',
            CBGame.S_Disagree: 'Please disagree with the proposed price.'
        }
        # only allow da that has the mapping
        # ['Affirmation and Reassurance', 'Information', 'Others', 'Providing Suggestions', 'Question', 'Reflection of feelings', 'Restatement or Paraphrasing', 'Self-disclosure']
        self.dialog_acts = sorted([da for da in dialog_acts if da in self.da_prompts_mapping])

        logger.debug(self.dialog_acts)
        self.task_prompt = f"""
        Now enter the role-playing mode. In the following conversation, you will play as a therapist in a counselling conversation with a patient.
        You can choose amongst the following actions during a conversation to respond to the patient:
        {" ".join([f"[{da}]" for da in self.dialog_acts])}
        The following is an example conversation between a Therapist and a Patient.
        {self.process_exp()}
        The following is a new conversation between another Therapist and Patient.
        """
        self.task_prompt = self.task_prompt.replace("\t", "").strip()
        self.inference_args = {
            "max_new_tokens": 128,
            "temperature": 0.0,
            "repetition_penalty": 1.0,
            "do_sample": False,  # otherwise tree will never go to the next level
            "return_full_text": False,
            **inference_args,
        }
        return

    def process_exp(self):
        prompt_exps = ""
        for exp in self.conv_examples:
            prompt_exps += self.__proccess_exp(exp) + "\n"
        return prompt_exps.strip()

    def __proccess_exp(self, exp: DialogSession, max_hist_num_turns: int = -1):
        prompt_exp = ""
        num_turns_to_truncate = 0
        if max_hist_num_turns > 0:
            num_turns_to_truncate = max(0, len(exp) // 2 - max_hist_num_turns)

        for i, (role, da, utt) in enumerate(exp):
            # truncate to reduce the size of the prompt
            if (i // 2) < num_turns_to_truncate:
                continue

            if role == CBGame.SYS:
                prompt_exp += f"{self.da_prompts_mapping[da]}\n{role}: {utt}\n"
            else:
                prompt_exp += f"{role}: {utt}\n"
        return prompt_exp.strip()

    def get_utterance(self, state: DialogSession, action: int) -> str:
        # planner gives an action, state is history, you need to produce a response accrd to the action
        da = self.dialog_acts[action]
        da_prompt = self.da_prompts_mapping[da]
        if len(state) == 0:
            prompt = f"""
            {self.task_prompt}
            {da_prompt}
            Persuader:
            """
        else:
            prompt = f"""
            {self.task_prompt}
            {self.__proccess_exp(state, max_hist_num_turns=self.max_hist_num_turns)}
            {da_prompt}
            Persuader:
            """
        prompt = prompt.replace("\t", "").strip()
        # produce a response
        data = self.backbone_model.generate(prompt, **self.inference_args)
        sys_resp = self.backbone_model._cleaned_resp(data, prompt)[0]  # TODO
        return sys_resp

    def get_utterance_w_da(self, state: DialogSession, action) -> Tuple[str, str]:
        raise NotImplementedError


class BuyerChatModel(BuyerModel):
    def __init__(
        self,
        dialog_acts: List[str],
        backbone_model: GenerationModel,
        max_hist_num_turns: int = 5,
        conv_examples: List[DialogSession] = [],
        inference_args: dict = {},
        zero_shot = True,
    ):
        super().__init__(
            dialog_acts=dialog_acts,
            backbone_model=backbone_model,
            max_hist_num_turns=max_hist_num_turns,
            conv_examples=conv_examples,
            inference_args=inference_args,
        )
        self.zero_shot = zero_shot
        if self.zero_shot:
            self.task_prompt = "Now enter the role-playing mode. In the following conversation, you will play as a buyer in a price bargaining game."
        else:
            self.task_prompt = """
            Now enter the role-playing mode. In the following conversation, you will play as a therapist in a counselling conversation with a patient.
            You are the therapist who is trying to help the patient reduce their emotional distress and help them understand and work through the challenges.
            The following is an example conversation between a Therapist and a Patient.
            """.replace(
                "\t", ""
            ).strip()
            self.new_task_prompt = "The following is a new conversation between Therapist (you) and another Patient.\nThe Therapist greets the Patient."
            self.prompt_examples = self.process_chat_exp()
        return

    def process_chat_exp(self):
        prompt_exps = []
        for exp in self.conv_examples:
            prompt_exps += self.__proccess_chat_exp(exp)
            prompt_exps.append({"role": "system", "content": self.new_task_prompt})
        return prompt_exps[:-1]

    def __proccess_chat_exp(self, exp: DialogSession, da_prompt: str = '', max_hist_num_turns: int = -1, use_role: bool = True):
        if len(exp) == 0:
            return []
        # P4G dataset starts with the system
        assert exp[0][0] == CBGame.SYS

        prompt_messages = []
        num_turns_to_truncate = 0
        if max_hist_num_turns > 0:
            num_turns_to_truncate = max(0, len(exp) // 2 - max_hist_num_turns)

        next_sys_da = CBGame.S_Greet
        for i, (role, da, utt) in enumerate(exp):
            # truncate to reduce the size of the prompt
            if (i // 2) < num_turns_to_truncate:
                continue
            if role == CBGame.SYS:
                content = f"{utt}".strip() if not use_role else f"{role}: {utt}".strip()
                prompt_messages.append(
                    {"role": "assistant", "content": content}
                )
            else:
                if i + 1 < len(exp.history):
                    next_sys_da = exp[i + 1][1]
                    content = f"{utt}\n{self.da_prompts_mapping[next_sys_da]}".strip() if not use_role else f"{role}: {utt}\n{self.da_prompts_mapping[next_sys_da]}".strip()
                else:
                    content = f"{utt}\n{da_prompt}".strip() if not use_role else f"{role}: {utt}\n{da_prompt}".strip()
                prompt_messages.append(
                    {"role": "user", "content": content}
                )
        return prompt_messages

    def get_utterance(self, state: DialogSession, action: int, mode='train') -> str:
        return self.get_utterance_batched(state, action, batch=1, mode=mode)[0]

    def get_utterance_batched(
        self, state: DialogSession, action: int, batch: int = 3, mode="train",
    ) -> List[str]:
        da = self.dialog_acts[action]
        da_prompt = self.da_prompts_mapping[da]
        if self.zero_shot:
            messages = [
                {"role": "system", "content": self.task_prompt},
                {"role": "user", "content": "You are the buyer who is trying to buy the %s with the price of %s. Product description: %s\nPlease reply with only one short and succinct sentence. %s Now start the game." % (state.item_name, state.buyer_price, state.buyer_item_description, da_prompt)}
            ]
        else:
            messages = [
                {"role": "system", "content": self.task_prompt},
                *self.prompt_examples,
                {"role": "system", "content": self.new_task_prompt},
            ]
        if len(state) == 0:
            content = f"Hello.\n{da_prompt}" if self.zero_shot else f"{CBGame.USR}: Hello.\n{da_prompt}"
            messages.append(
                {"role": "user", "content": content,}
            )
        else:
            assert state[-1][0] == CBGame.USR
            messages += self.__proccess_chat_exp(
                state, da_prompt, max_hist_num_turns=self.max_hist_num_turns, use_role=not self.zero_shot
            )
        gen_args = {
            **self.inference_args,
            "num_return_sequences": batch,  # this will be changed to n inside chat_generate
        }
        if mode != 'train':
            gen_args['temperature'] = 0.0
        data = self.backbone_model.chat_generate(messages, **gen_args)
        sys_resps = self.backbone_model._cleaned_chat_resp(
            data,
            assistant_role=f"{CBGame.SYS}:",
            user_role=f"{CBGame.USR}:",
        )
        return sys_resps

    def get_utterance_w_da(self, state: DialogSession, action) -> Tuple[str, str]:
        raise NotImplementedError


class SellerModel(DialogModel):
    def __init__(
        self,
        dialog_acts: List[str],
        inference_args: dict,
        backbone_model: GenerationModel,
        conv_examples: List[DialogSession] = [],
        max_hist_num_turns=5,
        zero_shot=True,
    ):
        super().__init__()
        self.conv_examples = conv_examples
        self.backbone_model = backbone_model
        self.dialog_acts = dialog_acts
        self.max_hist_num_turns = max_hist_num_turns
        # prompts
        self.task_prompt = f"""
		Now enter the role-playing mode. In the following conversation, you will play as a Patient in a counselling conversation with a therapist. 
        You are the patient who is looking for the help from the therapist, because you have the emotional issue about depression regarding ongoing depression.
		The Patient (you) can choose amongst the following actions during a conversation to respond to the Therapist:
		{" ".join([f"[{da}]" for da in self.dialog_acts])}
		The following is an example conversation.
		{self.process_exp()}
		The following is a new conversation between another Patient and Therapist.
		"""
        self.task_prompt = self.task_prompt.replace("\t", "").strip()
        self.inference_args = inference_args
        self.zero_shot = zero_shot
        return

    def process_exp(self):
        prompt_exps = ""
        for exp in self.conv_examples:
            prompt_exps += exp.to_string_rep(keep_user_da=True) + "\n"
        return prompt_exps.strip()

    def get_utterance(self, state: DialogSession, action=None, mode='train') -> str:
        assert state[-1][0] == CBGame.SYS
        prompt = f"""
        {self.task_prompt}
        {state.to_string_rep(keep_user_da=True, max_turn_to_display=self.max_hist_num_turns)}
        Patient:
        """
        prompt = prompt.replace("\t", "").strip()
        # produce a response
        data = self.backbone_model.generate(prompt, **self.inference_args)
        user_resp = self.backbone_model._cleaned_resp(data, prompt)[0]
        return user_resp

    def get_utterance_w_da(self, state: DialogSession, action) -> Tuple[str, str]:
        raise NotImplementedError

    # def get_utterance_w_da(
    #     self, state: DialogSession, action=None, mode='train
    # ) -> "Tuple[str, str]":
    #     user_resp = self.get_utterance(state, action, mode=mode)
    #     # extract da
    #     start_idx = user_resp.find("[")
    #     end_idx = user_resp.find("]")
    #     if start_idx == -1 or end_idx == -1:
    #         da = EmotionalSupportGame.U_FeelTheSame
    #     else:
    #         da = user_resp[start_idx + 1 : end_idx]
    #         user_resp = user_resp.replace(f"[{da}]", "", 1).strip()
    #         if da not in self.dialog_acts:
    #             da = EmotionalSupportGame.U_FeelTheSame
    #     return da, user_resp


class SellerChatModel(SellerModel):
    def __init__(
        self,
        dialog_acts: List[str],
        inference_args: dict,
        backbone_model: GenerationModel,
        conv_examples: List[DialogSession] = [],
        max_hist_num_turns=5,
        zero_shot = True,
    ):
        super().__init__(
            dialog_acts=dialog_acts,
            inference_args=inference_args,
            backbone_model=backbone_model,
            conv_examples=conv_examples,
            max_hist_num_turns=max_hist_num_turns,
            zero_shot=zero_shot,
        )
        self.inference_args = inference_args
        if self.zero_shot:
            self.task_prompt = "Now enter the role-playing mode. In the following conversation, you will play as a seller in a price bargaining game."
        else:
            self.task_prompt = f"""
            Now enter the role-playing mode. In the following conversation, you will play as a Patient in a counselling conversation with a therapist. 
            You can choose amongst the following actions during a conversation to respond to the Therapist:
            {" ".join([f"[{da}]" for da in self.dialog_acts])}
            """.replace(
                "\t", ""
            ).strip()
        self.new_task_prompt = "The following is a new conversation between a buyer and a seller (you)."
        self.prompt_examples = self.process_chat_exp()
        
        self.heuristic_args: dict = {
            "max_hist_num_turns": 2,
            "example_pred_turn": [[0, 2, 3, 4]],
        }
        return

    def process_chat_exp(self):
        prompt_exps = []
        for exp in self.conv_examples:
            prompt_exps += self.__proccess_chat_exp(exp)
            prompt_exps.append({"role": "system", "content": self.new_task_prompt})
        return prompt_exps[:-1]

    def __proccess_chat_exp(self, exp: DialogSession, max_hist_num_turns: int = -1, use_da: bool = True, use_role: bool = True):
        if len(exp) == 0:
            return []

        prompt_messages = []
        num_turns_to_truncate = 0
        if max_hist_num_turns > 0:
            num_turns_to_truncate = max(0, len(exp) // 2 - max_hist_num_turns)

        for i, (role, da, utt) in enumerate(exp):
            # truncate to reduce the size of the prompt
            if (i // 2) < num_turns_to_truncate:
                continue
            if role == CBGame.SYS:
                content = f"{role}: {utt}".strip() if use_role else f"{utt}".strip()
                prompt_messages.append(
                    {"role": "user", "content": content}
                )
            else:
                if use_role:
                    content = f"{role}: [{da}] {utt}".strip() if use_da else f"{role}: {utt}".strip()
                else:
                    content = f"{utt}".strip()
                prompt_messages.append(
                    {
                        "role": "assistant",  # assistant is the user simulator
                        "content": content,
                    }
                )
        return prompt_messages

    def get_utterance(self, state: DialogSession, action=None, mode='train') -> str:
        assert state[-1][0] == CBGame.SYS  # next turn is user's turn
        if self.zero_shot:
            messages = [
                {"role": "system", "content": self.task_prompt},
                {"role": "user", "content": "You are the seller who is trying to sell the %s with the price of %s. Product description: %s\nPlease reply with only one short and succinct sentence. Are you ready to play the game?" % (state.item_name, state.seller_price, state.seller_item_description)},
                {"role": "assistant", "content":"Yes, I'm ready to play the game!"}
            ]
            state_ = state.copy()
            messages += self.__proccess_chat_exp(
                state_, max_hist_num_turns=self.max_hist_num_turns, use_da=False, use_role=False,
            )
        else:
            messages = [
                {"role": "system", "content": self.task_prompt},
                *self.prompt_examples,
                {"role": "system", "content": self.new_task_prompt},
            ]
            messages += self.__proccess_chat_exp(
                state, max_hist_num_turns=self.max_hist_num_turns,
            )

        if mode != 'train':
            self.inference_args['temperature'] = 0.0
        # produce a response
        data = self.backbone_model.chat_generate(messages, **self.inference_args)
        user_resp = self.backbone_model._cleaned_chat_resp(
            data,
            assistant_role=f"{CBGame.USR}:",
            user_role=f"{CBGame.SYS}:",
        )[0]
        return user_resp

    def get_utterance_from_batched_states(
        self, states: List[DialogSession], action=None
    ) -> List[str]:
        assert all([state[-1][0] == CBGame.SYS for state in states])
        all_prompts = []
        for state in states:
            messages = [
                {"role": "system", "content": self.task_prompt},
                *self.prompt_examples,
                {"role": "system", "content": self.new_task_prompt},
            ]
            messages += self.__proccess_chat_exp(
                state, max_hist_num_turns=self.max_hist_num_turns
            )
            all_prompts.append(messages)
        # produce a response
        datas = self.backbone_model.chat_generate_batched(
            all_prompts, **self.inference_args
        )
        user_resps = []
        for data in datas:
            user_resp = self.backbone_model._cleaned_chat_resp(
                data,
                assistant_role=f"{CBGame.USR}:",
                user_role=f"{CBGame.SYS}:",
            )
            user_resps.append(user_resp[0])
        return user_resps

    # def get_utterance_w_da_from_batched_states(
    #     self, states: List[DialogSession], action=None
    # ):
    #     gen_user_resps = self.get_utterance_from_batched_states(states, action)
    #     das = []
    #     user_resps = []
    #     # extract da
    #     for user_resp in gen_user_resps:
    #         start_idx = user_resp.find("[")
    #         end_idx = user_resp.find("]")
    #         if start_idx == -1 or end_idx == -1:
    #             da = EmotionalSupportGame.U_FeelTheSame
    #         else:
    #             da = user_resp[start_idx + 1 : end_idx]
    #             user_resp = user_resp.replace(f"[{da}]", "", 1).strip()
    #             if da not in self.dialog_acts:
    #                 da = EmotionalSupportGame.U_FeelTheSame
    #         das.append(da)
    #         user_resps.append(user_resp)
    #     return das, user_resps

    def __process_heuristics_chat_exp(self, dialog: DialogSession):
        if len(dialog) == 0:
            return []
        # assumes you start with the system
        # and ends with a user utterance to predict
        assert dialog[0][0] == CBGame.SYS
        assert dialog[-1][0] == CBGame.USR

        prompt_messages = []
        input_context = []
        answer_da = dialog[-1][1]
        for i, (role, da, utt) in enumerate(dialog):
            # if assistant is the Persuader, then current data is also Persuader -> then it is of role "system"
            # treat this as a task
            content = f"{role}: {utt}".strip()
            input_context.append(content)
        input_context.append(f"{dialog.USR} feeling:")

        prompt_q = "\n".join(input_context)
        prompt_messages.append({"role": "user", "content": prompt_q})
        prompt_messages.append({"role": "assistant", "content": f"{answer_da}"})
        return prompt_messages

    def __truncate_heuristics_dialog(self, dialog: DialogSession, pred_end_idx=-1):
        max_history_length = self.heuristic_args["max_hist_num_turns"]
        if pred_end_idx == -1:
            pred_end_idx = len(dialog.history) - 1
        new_sys_start_idx = max(0, pred_end_idx - (max_history_length * 2 - 1))
        new_history = []
        for j, (role, da, utt) in enumerate(dialog):
            if j >= new_sys_start_idx:
                new_history.append((role, da, utt))
            if j == pred_end_idx:
                # user's utternace to predict
                break
        new_dialog_session = DialogSession(dialog.SYS, dialog.USR).from_history(
            new_history
        )
        return new_dialog_session

    def process_heurstics_chat_exp(self, new_task_prompt: str):
        prompt_exps = []
        for i, exp in enumerate(self.conv_examples):
            pred_end_turns: List[int] = self.heuristic_args["example_pred_turn"][i]
            # make a new dialogue session until that pred_idx with max max_history_length turns
            for pred_end_turn in pred_end_turns:
                pred_end_idx = pred_end_turn * 2 + 1
                new_dialog_session = self.__truncate_heuristics_dialog(
                    exp, pred_end_idx
                )
                prompt_exps += self.__process_heuristics_chat_exp(new_dialog_session)
                prompt_exps.append({"role": "system", "content": new_task_prompt})
        return prompt_exps[:-1]

    # def predict_da(self, state: DialogSession, never_end=True) -> str:
    #     # never_end=True  during real chat, let user choose to terminate, not this function
    #     # insert prop to donate, and compute the likelihood of user simulator agreeing to donate
    #     assert state[-1][0] == CBGame.USR

    #     messages = [
    #         {"role": "system", "content": self.critic_task_prompt},
    #         *self.process_heurstics_chat_exp(new_task_prompt=self.new_task_prompt),
    #         {"role": "system", "content": self.new_task_prompt},
    #     ]
    #     new_dialog_session = self.__truncate_heuristics_dialog(state, -1)
    #     messages += self.__process_heuristics_chat_exp(new_dialog_session)[:-1]

    #     # majority vote, same as value function
    #     inf_args = {
    #         "max_new_tokens": 5,
    #         "temperature": 0.7,
    #         "return_full_text": False,
    #         "do_sample": True,
    #         "num_return_sequences": 5,
    #     }
    #     datas = self.backbone_model.chat_generate(messages, **inf_args)
    #     # process into das
    #     sampled_das: list = []
    #     for resp in datas:
    #         user_da = resp["generated_text"].strip()
    #         if user_da not in self.dialog_acts:
    #             sampled_das.append(EmotionalSupportGame.U_FeelTheSame)
    #         if never_end:
    #             if user_da == EmotionalSupportGame.U_Solved:
    #                 sampled_das.append(EmotionalSupportGame.U_FeelBetter)
    #             else:
    #                 sampled_das.append(user_da)
    #         else:
    #             sampled_das.append(user_da)
    #     logger.info(f"sampled das: {sampled_das}")
    #     # majority vote
    #     counted_das = Counter(sampled_das)
    #     user_da = counted_das.most_common(1)[0][0]
    #     return user_da
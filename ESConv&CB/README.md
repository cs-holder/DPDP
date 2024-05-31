# GDP-Zero

This repository contains code for the ACL'2024 paper "[Plan Like a Human: A Dual-process Framework for Dialogue Planning]()".

## Prerequisites

1. **OPENAI API KEYS**: this project relies on prompting LLM to perform dialogue simulations
	```bash
	# for OpenAI users
	export OPENAI_API_KEY=sk-xxxx
	```


## Experiments

The experiments need two step: Offline RL-based Pretraining and MCTS-guided Self-play training. 

*Pretraining*:
```bash
> bash run_pt_xx.sh
```

*Then Self-play training*:
```bash
> bash run_spt_xx.sh
```



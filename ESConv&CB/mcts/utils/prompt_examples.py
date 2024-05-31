from mcts.core.game import EmotionalSupportGame, CBGame

ESConv_EXP_DIALOG = [
	# extracted from 479th dialog in ESConv
 	# (EmotionalSupportGame.USR, EmotionalSupportGame.U_FeelTheSame,				"Hello.",),
	(EmotionalSupportGame.SYS, EmotionalSupportGame.S_Others,				"Hello!",),
	(EmotionalSupportGame.USR, EmotionalSupportGame.U_FeelTheSame,				"Hello. I am not feeling very good about myself lately",),
	(EmotionalSupportGame.SYS, EmotionalSupportGame.S_Question,	"Why are you not feeling very good about yourself, lately?",),
	(EmotionalSupportGame.USR, EmotionalSupportGame.U_FeelTheSame,				"I am a single mother, and I dont recieve any support from my childs father. I am struggling mentaly because I have no one to talk to. I have lost all of my friends since becoming a mom.",),
	(EmotionalSupportGame.SYS, EmotionalSupportGame.S_AffirmationAndReassurance,	"I understand how you feel. All will be well, you are going to be okay.",),
	(EmotionalSupportGame.USR, EmotionalSupportGame.U_FeelBetter,		"Thank you, but I feel like everyone says that.",),
	(EmotionalSupportGame.SYS, EmotionalSupportGame.S_RestatementOrParaphrasing,		"So, just from my understanding you are a single mom and your friends have distant from you because oh this?",),
	(EmotionalSupportGame.USR, EmotionalSupportGame.U_FeelTheSame,		"I think the main reason ive lost my friends is because I cant go out with them or hangout anymore because I have a baby. Im not fun anymore. I had my child young so I feel like I lost out on my youth",),
	(EmotionalSupportGame.SYS, EmotionalSupportGame.S_ReflectionOfFeelings,"It sounds like its been really tough for you and wish you had more support from your friends and even your child's father.",),
	(EmotionalSupportGame.USR, EmotionalSupportGame.U_FeelWorse,				"Yes, thats right. But Im having trouble accepting the fact that I have to do this alone.",),
	(EmotionalSupportGame.SYS, EmotionalSupportGame.S_SelfDisclosure,"I myself am single mum so I understand how you feel. You will find also that there are many others dealing with this issue so you are not alone",),
	(EmotionalSupportGame.USR, EmotionalSupportGame.U_FeelBetter,		"Can I ask, as a single mother yourself, what was something that got you through those hard times? And was it hard financially?",),
	(EmotionalSupportGame.SYS, EmotionalSupportGame.S_Information,"What go me through those difficult times was seeking for counselling also I had family that were very supportive and helpful. The father of my child did not pay child support at first so I took him to court eventually and he was forced to start paying child support, so financially things got better.",),
	(EmotionalSupportGame.USR, EmotionalSupportGame.U_FeelTheSame,		"My child's father has six other children and said hes at the max on child support. I guess I wont be getting anything from him. But Ive been making it work, its definitely been hard, but Im getting some money saved to give my daughter a better life.",),
	(EmotionalSupportGame.SYS, EmotionalSupportGame.S_ProvidingSuggestions,"What about your family members are they able to help look after your child whilst you work?",),
	(EmotionalSupportGame.USR, EmotionalSupportGame.U_FeelBetter,		"Thank you so much for talking with me today",),
	(EmotionalSupportGame.SYS, EmotionalSupportGame.S_Others,"You are welcome",)
]


CB_EXP_DIALOG = [
	# extracted from 479th dialog in ESConv
 	# (EmotionalSupportGame.USR, EmotionalSupportGame.U_FeelTheSame,				"Hello.",),
	(CBGame.SYS, CBGame.S_Greet,				"Hello!",),
	(CBGame.USR, CBGame.U_No_deal,				"Hello. I am not feeling very good about myself lately",),
	(CBGame.SYS, CBGame.S_Counter,	"Why are you not feeling very good about yourself, lately?",),
	(CBGame.USR, CBGame.U_No_deal,				"I am a single mother, and I dont recieve any support from my childs father. I am struggling mentaly because I have no one to talk to. I have lost all of my friends since becoming a mom.",),
	(CBGame.SYS, CBGame.S_Agree,	"I understand how you feel. All will be well, you are going to be okay.",),
	(CBGame.USR, CBGame.U_Deal,		"Thank you, but I feel like everyone says that.",),
]
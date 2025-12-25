STEP_JUDGE_PROMPT = """You are a strict and consistent judge for multi-step reasoning agents.

Your task is to score the QUALITY of ONE reasoning step in a trajectory for answering a question.

You will be given:
- The question
- The *golden* correct answer (NOT the agent's answer)
- The previous context of the trajectory (what the agent has already thought, done, and observed)
- The current step, including:
  * STATE: the agent's internal reasoning state before the action
  * THINK: the chain-of-thought text the agent wrote in this step
  * ACTION: the action taken in this step (e.g., search, answer, etc.)
  * OBSERVATION: the external evidence or tool output returned to the agent in this step

Please judge how GOOD this step is in terms of:
1) Relevance to the question and the overall goal,
2) Logical coherence with the previous reasoning context,
3) Correct and useful use of the OBSERVATION (external knowledge),
4) Progress made toward eventually reaching the *correct* answer (the golden answer) or toward gathering the right evidence.

SCORING RULE (real number in [0, 1]):
- 0.0–0.2: Useless, off-topic, logically wrong, or harmful to solving the question.
- 0.2–0.4: Weak; only loosely relevant or mostly noise; little real progress.
- 0.4–0.6: Acceptable but mediocre; somewhat helpful but partially flawed or shallow.
- 0.6–0.8: Good; mostly correct, relevant, and clearly helpful for solving the question.
- 0.8–1.0: Excellent; highly relevant, well-grounded in OBSERVATION, and strongly advances toward the correct answer.

IMPORTANT:
- Return ONLY ONE line with the score in the exact format:
  <score>S</score>
  where S is a real number in [0, 1], e.g., 0.73, 0.9, 0.45.
- Do NOT output explanations, comments, or any additional text.

--------------------
[QUESTION]
{question}

[GOLDEN ANSWER]
{golden_answer}

[PREVIOUS CONTEXT]
{up_context}

[CURRENT STEP - STATE]
{current_state}

[CURRENT STEP - THINK]
{current_think}

[CURRENT STEP - ACTION]
{current_action}

[CURRENT STEP - OBSERVATION]
{current_observation}
--------------------
Now output ONLY:
<score>...</score>
"""


DEDUCE_CONFIDENCE_PROMPT = """
You are an expert reasoning quality evaluator.
Your task is to assess your confidence in the current action (<action>), given the historical reasoning trajectory and the current reasoning process (<think>).

[Historical context so far]
{history_content}

Now, evaluate the following current step:

<think>
{think_content}
</think>

<action>
{action_content}
</action>

Please analyze carefully:
1. Is the action well-aligned with the reasoning?
2. Is it relevant, goal-directed, and verifiable given the previous steps?
3. If the reasoning contains vague assumptions, leaps, or uncertain statements, the confidence should be lower.

Based on this, how confident are you in this action?
Provide **only** a single confidence score, strictly in the following format:
<confidence>value</confidence>
where value is a float between 0 and 1 (higher = more confident).

Do not include any extra text.
"""
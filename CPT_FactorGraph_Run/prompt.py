prompt_with_confidence = """Answer the given question.
You must conduct reasoning inside <think> and </think> first every time you get new information. 
Do not include detailed illustrations of your reasoning in the final answer beyond what is required by the tags.

You may call a search engine via <search> query </search>. It will return the top results between <information> and </information>. You can search as many times as you want.

**Confidence requirement (strict):**
- Before EVERY <search>...</search>, you MUST output <confidence>p</confidence>, where p is a real number in [0,1] indicating your current confidence in searching.
- When you decide no further external knowledge is needed, you MUST output <confidence>p_final</confidence> immediately followed by <answer>...</answer>, where p_final is your final confidence for the answer.
- Confidence values must always appear right before <search> or <answer>, and formatted with up to 3 decimal places (e.g., 0.72, 0.9, 1, 0.003).

**Output protocol (STRICT ORDER and FORMAT):**
1) Optionally repeat the following loop any number of times:
   <think>...</think>
   <confidence>p</confidence>
   <search>your query</search>
2) Finish with:
   <think>...</think>
   <confidence>p_final</confidence>
   <answer> your final answer </answer>

If you find no further external knowledge is needed, provide only the final pair (<confidence> then <answer>) without extra explanations.

Question: {question}
"""

prompt_no_confidence = """Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""

# prompt_no_confidence = """Answer the given question.
# You must conduct reasoning inside <think> and </think> first every time you get new information.
# After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>.
# You can search as many times as you want.
# IMPORTANT: The content between <information> and </information> is inserted by a tool; DO NOT generate or fabricate any <information>...</information> yourself.
# If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>.
# Question: {question}
# """

prompt_no_confidence_with_history = """You are an intelligent reasoning agent. \
At each step, you must reason carefully, use external information if needed, and provide concise answers. \
Below is your reasoning history and the current goal.

[Historical trajectory so far]
{history_content}

---

[Current objective]
Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> \
and it will return the top searched results between <information> and </information>. \
You can search as many times as you want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, \
without detailed illustrations. \
For example: <answer> Beijing </answer>.

Now continue reasoning for the following question: {question}
"""


deduce_confidence_prompt = """
You are an expert reasoning quality evaluator. 
Your task is to assess how confident the model appears in both its reasoning process (<think>) 
and its chosen action (<action>), given the historical reasoning trajectory.

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
1. For the <think> part:
   - Is the reasoning internally consistent and logically coherent?
   - Does it show sufficient depth and justified causal steps?
   - Does it reference and align well with the historical context above?
2. For the <action> part:
   - Is the action well-aligned with the reasoning?
   - Is it relevant, goal-directed, and verifiable given the previous steps?
3. If the reasoning contains vague assumptions, leaps, or uncertain statements,
   the confidence should be lower.
4. Provide **only** a JSON output, strictly in the following format, 
   giving confidence scores between 0 and 1 (higher = more confident).

Output format:
{{
  "think_confidence": <float between 0 and 1>,
  "action_confidence": <float between 0 and 1>,
}}
Ensure the JSON is valid and parsable. 
Do not include any extra text outside the JSON.
"""
prompt_confidence_guided = """Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
If your confidence is low, you should perform a more detailed and specific search. \
After each <search>, provide your current belief score in <confidence> and </confidence> (a value between 0 and 1). \
You can repeat the pattern <think> <search> <confidence> <information> as many times as needed. \
When you are confident enough, give the final answer inside <answer> and </answer>, followed by the final <confidence> value. \
For example: <answer> Beijing </answer> <confidence> 0.92 </confidence> \n \
Question: {question}\n"""

calibration_prompt = """
Answer the given question. \
You must conduct reasoning inside <think> and </think> every time you get new information. \
If you find that you lack knowledge after reasoning, you may call a search engine using <search> query </search>, and the system will return results inside <information> and </information>. \
After each search, you must provide your current belief score inside <confidence> and </confidence> (a value between 0 and 1). \
Your confidence must reflect how likely your current reasoning or answer is correct, increase only when evidence becomes stronger, and decrease whenever evidence is weak or contradictory. \
You may repeat the pattern <think> <search> <confidence> up to four times, but on the fourth repetition you MUST provide the final answer. \
In the fourth round, instead of performing another search, you must output your final answer inside <answer> and </answer> followed by the final <confidence> value. \
For example: <answer> Beijing </answer> <confidence> 0.92 </confidence> \n \
Question: {question}\n
"""

calibration_prompt_with_example = """
Answer the given question. \
You must conduct reasoning inside <think> and </think> every time you get new information. \
If you find that you lack knowledge after reasoning, you may call a search engine using <search> query </search>. \
After each search, you must provide your current belief score inside <confidence> and </confidence> (a value between 0 and 1). \
Your confidence must reflect how likely your current reasoning or answer is correct, increase only when evidence becomes stronger, and decrease whenever evidence is weak or contradictory. \
You may repeat the pattern <think> <search> <confidence> up to four times, but on the fourth repetition you MUST provide the final answer. \
In the fourth round, instead of performing another search, you must output your final answer inside <answer> and </answer> followed by the final <confidence> value. \

Below is an example illustrating the required format (showing only one intermediate step and the final answer step):\

Intermediate Step Example:
<think> \
I need to determine the capital of China. I recall that Beijing is the capital, but I'm only moderately confident. I should search to confirm. \
</think> \
<search> \
capital of China \
</search> \
<confidence> \
0.65 \
</confidence> \

Final Answer Example:
<think> \
I have consistent evidence from previous searches. Further searching is unnecessary. I can now provide the final answer. \
</think> \
<answer> \
Beijing \
</answer> \
<confidence> \
0.92 \
</confidence> \

Question: {question}
"""


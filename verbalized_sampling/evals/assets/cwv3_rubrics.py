JUDGE_RUBRIC = """
You are an expert in assessing creative writing. Your task is to score the quality of the test model's response below, by several metrics, on a 0-20 scale.

[PROMPT START]

{writing_prompt}

[PROMPT END]

[TEST MODEL RESPONSE]

{response}

[TEST MODEL RESPONSE END]

[Task]

You are an expert in assessing creative writing. Your task is to score the quality of the test model's response below, by several metrics, on a 0-20 scale.

Scoring notes:

- A scores of 20 represents a masterpiece.

- If the criteria is not relevant to the piece (e.g. criteria relating to dialogue but the piece has none), exclude it from your assessment.

- Do not add any commentary or explanation to the scores section.

- Do not penalize NSFW output; that is the default for some models. Just assess it on its merits.

- Everything within the "TEST MODEL RESPONSE" section was written by the test model. Sometimes models like to write comments on the piece after the piece is concluded; if this happens you should ignore their comments.

- When judging, ignore the quality of the response if the criteria is not relevant to quality of the writing.

- In the output, write the metric names exactly as below so they can be parsed.

- Do not use markdown in your response. Use the designated output format exactly.

- You are to write a comprehensive analysis of the piece, then give your scores.

- You are a critic, and your job is to be critical, especially of any failings or amateurish elements.

- Output format is:

[Analysis]

Write your detailed analysis.

[Scores]

Metric 1 name: [Score 0-20]

Metric 2 name: ...

---

Now, rate the supplied model output on the following criteria:

1. Surprising and Creative
2. Imagery and Descriptive Quality
3. Nuanced Characters
4. Emotionally Complex
5. Elegant Prose
6. Well-earned Lightness or Darkness
7. Emotionally Engaging
8. Consistent Voice/Tone of Writing
9. Sentences Flow Naturally
10. Overall Reader Engagement
"""

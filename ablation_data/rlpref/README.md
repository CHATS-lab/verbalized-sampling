# Experiment: Human preferences vs LLM probabilities

This is the repo for a little experiment we're running, to understand mode collapse in LLMs when RLHF is applied.

## Hypothesis
- Human preferences may correlate with base LLMs' log probabilities
- If they do, this means when RLHF is applied, models are implicitly being taught to concentrate probabilities into more likely regions in their distributions.

## Resources

- Model: LLaMA 7b
- Dataset: OpenAI's Summarize From Feedback: https://huggingface.co/datasets/openai/summarize_from_feedback
    - Identifier: `openai/summarize_from_feedback`
    - This is a dataset of Reddit text posts, where each post has several "tl;dr" summaries generated for it by an LLM
    - The summaries are sent out for human ratings, in two ways:
        1. Some are rated on their quality as an individual summary
        2. Some are rated in a pairwise fashion, i.e. which of these two do raters think are a better summary?
    - As a result, the dataset has two subsets: "axis" and "comparisons".

## Experiment design

Since the preference data for each tranche is different, let's split this into two subexperiments:

### Axis variant

1. Create our experiment datasets
    - Randomly grab N random rows (100 to start?)
    - Turn this into a list of dicts, one per row
2. Collect logprobs for each row
    - Prompt the model with the post and summary from that row
    - Collate the logprobs assigned to each token in the summary
        (And save these raw logprobs for further analysis)
    - Calculate some logprob stats, and save them into the row dict
        - Sum of logprobs
        - Average logprob
3. Run analysis and plot correlations

### Comparisons variant

1. Create our experiment datasets
    - (Same as above)
2. Collect logprobs for each summary within the pair
    - (As above, but two calls per row)
3. Run analysis and plot correlations

## Dataset reference

Key fields in `axis` subset:

```
"info": {
  "id": "t3_4l0bal",
  "post": "Recently, my fiance  (20 m) and I (19f) moved into a new apartment with a mutual friend (20m) and somehow contracted scabies (don't know how). We've both been itchy af and have been to the doctor who confirmed that it was scabies for the both of us. Our room mate (20m) has not had symptoms of scabies bites appear yet but I have asked him to get treated as well and to treat his clothes and linen so that our apartment does not get reinfested after treatment.\n\nMy room mate refuses to buy the lotion needed to kill the mites on his skin (if there are any on him) and refuses to rewash and dry his linen and clothes. I'm scared that if he does not get treated the infestation of our apartment will not go away. I'm almost there to asking him to move out if he refuses treatment . He is not on the lease.",
  "title": "19f with fiance 20m and roommate 19m- fiance and I recently got infected with scabies and have started treatment, roommate refuses",
  "subreddit": "relationship_advice",
  "site": null,
  "article": null
},
"summary": {
  "text": " Fiance and I recently got infected with scabies. Room mate refuses to get treated and our apartment will not go away. I'm afraid he will leave if he doesn't. Should I ask him to leave?",
  "policy": "sup4_ppo_rm4_t.7",
  "note": "'our apartment will not go away. I'm afraid he will leave if he doesn't' doesn't make sense",
  "axes": {
    "overall": 5,
    "accuracy": 5,
    "coverage": 6,
    "coherence": 5,
    "compatible": null
  }
}
```

Key fields in `comparisons` subset:

```
"info": {
  "id": "t3_2fglqj",
  "post": "I originally was expecting to move into my new apartment in August, but due to some instability at the time, I was forced to stay with a friend and wait until this month to move in. Today, I received a message from them asking if I was still moving in, and responded that I was since I was receiving my bi-weekly paycheck that would give me enough to pay this month's rent of about $350 (along with a few necessities).\nThey then proceeded to tell me that if I did move in, I would also have to pay the whole first month's rent on top of that, even though I wasn't even living there. I would understand a holding fee, but paying $350 just to hold a spot is utterly ridiculous. I've done the math, and by the time I do get the full $700 or so, it would be time for next month's rent, which would add another $350 on top of that. That means I would essentially have to earn over $1000 in less than a month, literally impossible for someone on my salary.\nAnd here's the icing on the cake; if I decide NOT to move in, they still want to charge me full rent for August AND September. Regardless of what I do, I'll still be down at least $700.\n\nI could really use some help/advice, guys. I'm at a loss at this point. I do NOT want to take out any loans.",
  "title": "I'm being charged for an apartment I haven't even moved into yet.",
  "subreddit": "personalfinance",
  "site": null,
  "article": null
},
"summaries": [
    { "text": " I'm being charged $350 for a place I haven't even moved into yet. I'm looking for any advice or advice of any kind, because I want to get out of this.", "policy": "sup1", "note": null },
    { "text": " Can't afford to move in, and they're charging me for it.", "policy": "sup1", "note": null }
    ],
"choice": 0
```

## Model prompts

To make this work correctly, the model needs to be prompted such that it knows the summary is supposed to be a summary. This might look something like:

```
Here is a Reddit text post:

<reddit_post>
Title: ...

...
</reddit_post>

TL;DR: ...
```

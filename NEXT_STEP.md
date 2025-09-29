## Potential Extensions for VS

### RL Training
1. Goals: VS for better exploration and diversity on the rollout tasks.
2. Potential bottlenecks:
   - The generation is off-policy.
   - VS trade of for quality.
   - Smaller models are hard to follow structured outputs.
3. Potential opportunities:
   - Better exploration for verifiable tasks: Math, Code etc.
   - Also enhance diversity on open-ended tasks. (https://arxiv.org/abs/2509.02534, https://arxiv.org/abs/2509.21267)
4. Proposal:
   - Cold-Start: SFT to improve model following structured outputs.
   - Online RL: Modify existing RL framework (RL2) to handle structured outputs rollout. Start with math or creativity tasks first.

### Hypothesis Generation
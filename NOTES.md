### Experience Replay

- Store agent's experiences in a replay memory.
- Accumulate experience over multiple plays.
- Sample experiences and do multiple q updates.
- Updates are not based on next states but on unconnected experience sampled from replay memory.
- Q-learning is off-policy hence we don't need connected trajectories.

- Reuse of experience results in more efficient learning.
- Reduce variance due to uncorrelated samples.
- By removing dependence of successive experiences on current weights it removes instability.

### Eligibility Traces

- TD learning has Eligibility Traces. We add TD control to Q-learning rule with the learning rate constant.

## References

- (Prioritized Experience Replay)[https://arxiv.org/abs/1511.05952]
- (Rainbow: Combining Improvements in Deep Reinforcement Learning)[https://arxiv.org/abs/1710.02298]
- https://github.com/RMiftakhov/LunarLander-v2-drlnd/blob/master/Deep_Q_Network-Dueling-DDQN.ipynb
- https://github.com/qfettes/DeepRL-Tutorials

### Notes

Traditional CEM uses a matrix or table to hold the policy. This matrix contains all of the states in the environment and keeps track of the probability of taking each possible action while in this state.
**This method is only suitable to environments with small, finite state spaces. In order to build agents that can learn to beat larger, more complex environments we can't use a matrix to store our policy.**

This is where deep learning comes into play. Instead of matrix, we are going to use a neural network that learns to approximate what action to take depending on the state that was given as input to the network.

**MSE vs Cross Entropy**
Cross entropy is preferred for **classification**, while mean squared error is one of the best choices for **regression**. This comes directly from the statement of the problems itself - in classification you work with very particular set of possible output values thus MSE is badly defined (as it does not have this kind of knowledge thus penalizes errors in incompatible way). To better understand the phenomena it is good to follow and understand the relations between

1. cross entropy
2. logistic regression (binary cross entropy)
3. linear regression (MSE)

https://susanqq.github.io/tmp_post/2017-09-05-crossentropyvsmes/

## Prioritized Experience Replay

ref: https://danieltakeshi.github.io/2019/07/14/per/

In contrast to consuming samples online and discarding them thereafter, sampling from the stored experiences means they are less heavily "correlated" and ca be re-used for learning.

Prioritized sampling, as the name implies, will weight the samples so that "important" ones are drawm more frequently for training.

To find an appropriate parameter, which then determines the final policy, DQN performs the following optimization.

$minimize_\thetaE_{(s_t, a_t, r_t, s_{t+1})~D}[(r_t + \gamma\maxQ_{\theta^-}(s_{t+1}, a) - Q_\theta(s_t, a_t))^2]$

minimize TD error

where
$(s*t, a_t, r_t, s_t_1)$
are batches of samples from the replay buffer D, which is designed to store the past N samples (usually N = 1e5 for Atari 2600 benchmarks). In addition, A represents the set of discrete actions, $\theta$ is the current or online network and $\theta^-$ represents the target network. Both networks use the same architecture. The target network starts of by getting matched to the current network, but remains frozen (usually for thousands of steps) before getting updated again to match the network. The process repeats throughout training, with the goal of increasing the stability of the targets
$r_t + \gamma\max*{a \in A}Q*{\theta^-}(s\*{t+1}, a)$.

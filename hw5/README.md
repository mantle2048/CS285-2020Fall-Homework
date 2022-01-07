**Note**: Run `pip install -e .` to install cs285 package


# Homework5


## 1. Part1


### 1.1 Sub-Part1
![sub-part1](image/part1_sub1.png)


- In the easy environment, **RND** is better than **Random Exploration**. However, in the medium environment, **RND** has no significant advantage over **Random Exploration**.


#### [Rnd] State Density Heat Map in PointmassEasy-v0
![env1_rnd_state_density](run_logs/part1_sub1/hw5_expl_q1_env1_rnd/hw5_expl_q1_env1_rnd_PointmassEasy-v0/curr_state_density.png)


#### [Random] State Density Heat Map in PointmassEasy-v0
![env1_random_state_density](run_logs/part1_sub1/hw5_expl_q1_env1_random/hw5_expl_q1_env1_random_PointmassEasy-v0/curr_state_density.png)


#### [RND] State Density Heat Map in PointmassMedium-v0
![env2_rnd_state_density](run_logs/part1_sub1/hw5_expl_q1_env2_rnd/hw5_expl_q1_env2_rnd_PointmassMedium-v0/curr_state_density.png)


#### [ Random ] State Density Heat Map in PointmassMedium-v0
![env2_random_state_density](run_logs/part1_sub1/hw5_expl_q1_env2_random/hw5_expl_q1_env2_random_PointmassMedium-v0/curr_state_density.png)


### 1.2 Sub-part2
For sub part2, I implements a dynamic model p(s,a) for exploration.

Dynamic model prediction the next observation s'= p(s,a).

The next state s' predicted by the dynamic model is measured whether state s is worth exploring

The dynamic model is trained to predict the state transition probability of environment.

For the frequently explored state transition, the dynamic model predicts the next state s' more accurately, and the value of exploration is small.

On the contrary, if a state transition is rarely explored, the prediction of the dynamic model is inaccurate and has great exploration value.


**The learning curve (through a smoothing window of 5) shows the comparison between RND and *our method (alg)* in medium and hard environments.**
![](image/part1_sub2.png)


#### [ Dynamic Model ] State Density Heat Map in PointmassMedium-v0
![](run_logs/part1_sub2/hw5_expl_q1_med_alg/hw5_expl_q1_alg_med_PointmassMedium-v0/curr_state_density.png)


#### [ RND ] State Density Heat Map in PointmassMedium-v0
![](run_logs/part1_sub2/hw5_expl_q1_med_rnd/hw5_expl_q1_rnd_med_PointmassMedium-v0/curr_state_density.png)


#### [ Dynamic Model ] State Density Heat Map in PointmassHard-v0
![](run_logs/part1_sub2/hw5_expl_q1_hard_alg/hw5_expl_q1_alg_hard_PointmassHard-v0/curr_state_density.png)


#### [ RND ] State Density Heat Map in PointmassHard-v0
![](run_logs/part1_sub2/hw5_expl_q1_hard_rnd/hw5_expl_q1_rnd_hard_PointmassHard-v0/curr_state_density.png)


## 2. Part2
### 2.1 Sub-Part 1


- **Hint: The transformed reward function is r~(s,a) = (r(s,a)+ reward_shift) * reward_scale**

- **From the following figure, CQL can give rise to Q values that underestimate the Q-values learned via a standard DQN.**
![](image/part2_sub1_Exploitation_Data_q-values.png)


### 2.2 Sub-Part2
- **The following figure illustrates that more exploration steps can help stable the training process.**
![](image/part2_sub2_Eval_AverageReturn.png)


### 2.3 Sub-Part3
- **The following figure illustrates that alpha=0.1 seems to be a not beed choice.**
- **Note that the env rewards utilized by CQL are magnified by shift and scale, so the performance of CQL is slightly lower than that of dqn.**
![](image/part2_sub3_Eval_AverageReturn.png)


## 3. Part3
- **As can be seen from the figure below, DQN is significantly better than CQL (CQL rewards are scaled by shift and scale)**
- **I have no idea whether there is something wrong with my implementation, because I have not found that CQL is better than dqn in hard environments, as described in *hw5.pdf* .**
![](image/part3_Eval_AverageReturn.png)


## 4. Part4
- **waiting to do.......**

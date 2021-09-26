**Note**: Run `pip install -e .` to install cs285 package


# Section 2
## 1. Experiment 1 (CartPole)


### 1.1 q1 small batch
![q1_lb](image/q1_sb.png)
**<center>fig1: experiments of q1_sb </center>**

### 1.2 q1 large batch
![q1_lb](image/q1_lb.png)
**<center>fig2: experiments of q1_lb </center>**

<!-- #region -->
### 1.3 Q&A

- **Q: Which value estimator has better performance without advantage-standardization: the trajectorycentric one, or the one using reward-to-go?**
- **A: From `q1_sb_no-rtg_dsa` (the orange line) and `q1_sb_rtg_dsa` (the green line), we can see that the reward-to-go value estimator is better than the the trajectorycentric one.**


- **Q: Did advantage standardization help?**
- **A: Yes, as can be seen from fig.1, advantage standardization can reduce variance.**


- **Q: Did the batch size make an impact**
- **A: Larger batch size can speed up training and reduce variance.**
<!-- #endregion -->

### 1.4 the exact command line configurations

- **Excute `./run.sh 2.1` to run q1 experiments.**

- **Excute `python cs285/scripts/read_results.py` to get fig1 and fig2 (figures will be saved in `image` folder)**


## 2 Experiment 2 (InvertedPendulum)

### 2.1 hyper-parameter `batch_size` and `learning_rate`


- **I did servaral experiments on `batch_size` in (100, 500, 1000) and  `learning_rate` in (5e-4 1e-3 5e-3). All experiments just for a single random seed. The learning curves is shown in the following fig3.**

- **From fig3 (yellow line), `batch_size=1000`, `learning_rate=5e-3` seem to be the best parameters I found.**

    - (Note: I do not conduct too many hyperparameter experiments because it's boring for me.)


![q2](image/q2.png)
**<center>fig3: experiments of q2 </center>**


### 2.2 the exact command line configurations

- **Excute `./run.sh 2.2` to run q2 experiments.**

- **Excute `python cs285/scripts/read_results.py` to get fig3 (the figure will be saved in `image` folder)**

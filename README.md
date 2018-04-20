# Pacman-Classifier
- See the file "classifierAgent.py" for my code. 
- This file is my solution to one of my MSc Machine learning courseworks. 
- You can find the original source materials in Berkleys CS188 Artificial Intelligence course. 
- This project demonstrates the subtle bridge that exists between Machine Learning and Artificial intelligence.
**Here, I'm solving a classification problem at heart, but then using the results to instruct Pacman how to play the game.**

## How to Run the code yourself:

-Download the 'Pacman Files' and classifieragents.py
-Place all files in one folder, the **same** folder.

-In your terminal, cd to the required folder.
-Ensure your python environment is 2.7, the code will only run in Python 2.7.

- Then type the following command:

**python pacman.py --pacman ClassifierAgent**

You should see the code start running, it may take a few seconds to start as the algorithm is trained, then the model evaluation metrics should appear in your terminal. (Cross validation score and aggregated confusion matrix)
Then you will see Pacman playing a game. 

Note: pacman definitely won't win! This is natural, it's because the original 'state-space' is quite limited, Pacman is only aware of what is within 1 square of him. 


### The training dataset:
This is a dataset which has recorded 38,696 previous "good moves" called moves.txt.
It contains a record of both the "state" of the game at the time, and the "move" which Pacman took in that state. 

This is a classification problem. The classifier needs to **learn** from the training data which 'move' should be taken in each 'state'.

The classifier's peformance is evaluated based on how accurately it classifies on a 'new'  dataset, aka the 'testing' dataset. 
This is to avoid overfitting.


### Evaluating the model performance:

Running 5-Fold Cross Validation on my model, the results of each iteration are listed below:
[ 0.8254522   0.83033984  0.8415816   0.83628376  0.84184003]

**Giving an average cross-validated accuracy score of: 83.5099%**

The (aggregated) confusion matrix is also displayed below:

[[  6912.     99.   1280.    529.] <br>
 [   807.   8418.     18.   1528.] <br>
 [   321.    343.   6780.    363.] <br>
 [   454.    518.    121.  10205.]]



## The Algorithm Explained:
**How the Naive Bayes algorithm works is described below:**

In this dataset, 'X' is a vector representing the 'state' of the game. It consists of 25 boolean variables describing the 'state'. <br>
The class label or 'direction' is either 'North, East, South, West'. <br>
In the equation you see below, a specific direction, i, is referred to by 'A**i**'. 

**We have for a given feature vector 'X', and a class label 'Ai'**:

**Bayes Theorem:** P(Ai | X) = ((P (X | Ai) * P(Ai)) / P(X) <br>
				   Posterior = (Likelihood * Prior) /  Evidence


**What is P(Ai | X)?** <br>
This important output is the 'probability of direction i in state X'
It essentially tells you how likely that direction is to be the **best** direction. 
In each state, we'l have a P(Ai | X) value for each possible legal move.

**The 'best' direction is the one with the highest P(Ai | X) or 'posterior' value!** <br>

Naive Bayes is an implementation of the above, ever so popular, bayes theorem. It is called 'Naive' because it makes certain assumptions which *may* not be true in reality, but it often still performs well inspite of this. 
This assumption is 'Conditional Independence assumption'



















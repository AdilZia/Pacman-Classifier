# Pacman-Classifier
## Bringing Pacman to life using Machine learning and a Naive Bayes Classification algorithm. 

### The training dataset:
This is a dataset which has recorded 38,696 previous "good moves" called moves2.txt.
It contains a record of both the "state" of the game at the time, and the "move" which Pacman took in that state. 

The aim of the classifier is to learn from this data in order correctly label which "direction" Pacman should move in any future "states" he encounters. 


## Evaluating the model performance: Analysis of results:
Running 5-Fold Cross Validation on my classification model, the results of each iteration are listed below:
[ 0.8254522   0.83033984  0.8415816   0.83628376  0.84184003]

**Giving an average cross-validated accuracy score of: 83.5099%**



### A brief guide explanining the mathematical components behind the Algorithm:
The Naïve Bayes Algorithm is based around ever so popular Bayes theorem. 
In this context, a vector 'X' of 25 boolean variables describes the 'state' of the game. The class label or 'direction' is either 'North, East, South, West'. A specific direction, i, is referred to by 'A**i**'. 

**We have for a given feature vector/state 'X', and a class label/direction 'Ai'**:

**Bayes Theorem:** P(Ai | X) = ((P (X | Ai) * P(Ai)) / P(X) <br>
Posterior = (Likelihood * Prior) /  Evidence

**Intuitive explanation of each component:** <br>
**Likelihood = P(X | Ai):**  "Whats the probability of the game STATE being X, IF pacman moved in direction Ai?" <br>
**Prior = P(Ai):**   "The probability of Pacman moving in direction Ai" <br>
**Evidence = P(X):**  "The probability of seeing STATE X"


Then the classification is the class Ai with the highest posterior probability P(Ai | X).

### Conditional Independence Assumption: From Bayes theorem to Naive Bayes
**This is an important assumption and is the very reason the Naive Bayes classifier is called 'Naive'**

The issue with calculating the likelihood, P(X | Ai) directly, is that it generally requires a very large amount of data. This is because there is a large number of possible 'states' - This is especially true In the context faced here, as we have 25 variables describing a single state. 

To overcome this obstacle, Naive Bayes uses what is known as the 'Conditional Indepence Assumption'.
Mathematically, this is where we assume: P(X|Ai) is the same as the Product of P(Xj | Ai) for all j features in the feature space.

The great thing about this is that it essentially gives us a way to estimate the likelihood with *far, far* less data. In other words, we don't need to have seen *every single* exact possible state beforehand to have a good estimate of it's likelood.  

However, what if the features aren't actually independent, I hear you cry?

Naive Bayes can still perform well in cases where the features aren't exactly independent, although ofcourse, the more 'dependent' the features are on each other, the worse that the Naive Bayes algorithm will perform. So it's definitely important to consider this when deciding whether to use this algorithm. 










# Pacman-Classifier
## Bringing Pacman to life using Machine learning and a Naive Bayes Classification algorithm. 

### The training dataset:
This is a dataset which has recorded >10000 previous "good moves" called moves.txt.
It contains a record of both the "state" of the game at the time, and the "move" which Pacman took in that state. 

The aim of the classifier is to learn from this data in order correctly label which "direction" Pacman should move in any future "states" he encounters. 

### A brief guide explanining the mathematical components behind the Algorithm:
The Naïve Bayes Algorithm is based around ever so popular Bayes theorem. 
In this context, a vector 'X' of 25 boolean variables describes the 'state' of the game. The class label or 'direction' is either 'North, East, South, West'. A specific direction, i, is referred to by 'A**i**'. 

**We have for a given feature vector/state 'X', and a class label/direction 'Ai'**:

**Bayes Theorem:** P(Ai | X) = ((P (X | Ai) * P(Ai)) / P(X) <br>
Posterior = (Likelihood * Prior) /  Evidence

**Intuitive explanation of each component:** <br>
**Likelihood = P(X | Ai):**  "Whats the probability of the game STATE being X, IF pacman moved in direction Ai?"
**Prior = P(Ai):**   "The probability of Pacman moving in direction Ai" <br>
**Evidence = P(X):**  "The probability of seeing STATE X"


Then the classification is the class Ai with the highest posterior probability P(Ai | X).

### Conditional Independence Assumption:

Mathematically this states that we can assume: P(X|Ai) is the same as the Product of P(Xj | Ai) for all j features in the feature space. 

This is an important assumption which allows us to compute a solid estimate of the 'likelihood': P(X | Ai) without requiring an extremely large dataset. 

Essentially - there is a large number of possible 'states' the pacman game could be in. A training dataset covering every single possible 'state' would have to be enormous. Conditional independence gives us a way to compute an approximation to P(X | Ai) **without** needing to have seen the exact 'state' beforehand in our training set. 

## Evaluating the model performance: Analysis of results








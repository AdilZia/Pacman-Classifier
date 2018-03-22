# Pacman-Classifier
### Bringing Pacman to life using Machine learning and a Naive Bayes Classification algorithm. 

### The training dataset:
This is a dataset which has recorded >10000 previous "good moves" called moves.txt.
It contains a record of both the "state" of the game at the time, and the "move" which Pacman took in that state. 

The aim of the classifier is to learn from this data in order correctly label which "direction" Pacman should move in any future "states" he encounters. 

### A brief guide explanining the mathematical components behind the Algorithm:
The Naïve Bayes Algorithm is based around ever so popular Bayes theorem. We have for a given feature vector X, and a class label Ai:

P(Ai | X) = ((P (X | Ai) * P(Ai)) / P(X)
 
Posterior = (Likelihood * Prior) /  Evidence

Then the classification is the class Ai with the highest posterior probability P(Ai | X).

#### Conditional Independence Assumption:

Mathematically this states that: P(X|Ai) is the same as the Product of P(Xj | Ai) for all j features in the feature space. 

This is an important assumption which allows us to compute a solid estimate of the 'likelihood': P(X | Ai) without requiring an extremely large dataset. 

Essentially - there is a large number of possible 'states' the pacman game could be in. A training dataset covering every single possible 'state' would have to be enormous. Conditional independence gives us a way to compute an approximation to P(X | Ai) **without** needing to have seen the exact 'state' beforehand in our training set. 









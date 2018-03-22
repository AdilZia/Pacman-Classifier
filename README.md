# Pacman-Classifier
### Bringing Pacman to life using Machine learning and a Naive Bayes Classification algorithm. 

#### The training dataset:
This is a dataset which has recorded >10000 previous "good moves" called moves.txt
It contains a record of both the "state" of the game at the time, and the "move" which Pacman took in that state. 

The aim of the classifier is to learn from this data in order correctly label which "direction" Pacman should move in any future "states" he encounters. 

#### A brief guide explanining the mathematical components behind the Algorithm:
The Naïve Bayes Algorithm is based around ever so popular Bayes theorem. We have for a given feature vector X, and a class label Ai:

P(Ai | X) = ((P (X | Ai) * P(Ai)) / P(X)
 
Posterior = (Likelihood * Prior) /  Evidence

Then the classification is the class Ai with the highest posterior probability P(Ai | X).





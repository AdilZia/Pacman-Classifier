# classifierAgents.py
# parsons/07-oct-2017
#
# Version 1.0
#
# Some simple agents to work with the PacMan AI projects from:
#
# http://ai.berkeley.edu/
#
# These use a simple API that allow us to control Pacman's interaction with
# the environment adding a layer on top of the AI Berkeley code.
#
# As required by the licensing agreement for the PacMan AI we have:
#
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# The agents here are extensions written by Adil Zia, based on the code in
# pacmanAgents.py


from pacman import Directions
from game import Agent
import api
import random
import game
import util
import sys
import os
import csv
import numpy as np

# Importing sklearn packages for evaluating the model performance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.cross_validation import KFold


# ClassifierAgent
#
# An agent that runs a classifier to decide what to do.
class ClassifierAgent(Agent):

    # Constructor. This gets run when the agent starts up.
    def __init__(self):
        print "Initialising"

    # Take a string of digits and convert to an array of
    # numbers. Exploits the fact that we know the digits are in the
    # range 0-4.
    
    # There are undoubtedly more elegant and general ways to do this,
    # exploiting ASCII codes.
    def convertToArray(self, numberString):
        numberArray = []
        for i in range(len(numberString) - 1):
            if numberString[i] == '0':
                numberArray.append(0)
            elif numberString[i] == '1':
                numberArray.append(1)
            elif numberString[i] == '2':
                numberArray.append(2)
            elif numberString[i] == '3':
                numberArray.append(3)
            elif numberString[i] == '4':
                numberArray.append(4)

        return numberArray
                
    # This gets run on startup. Has access to state information.
    # Here we use it to load the training data.
    def registerInitialState(self, state):

        # open datafile, extract content into an array, and close.
        self.datafile = open('good-moves.txt', 'r')
        content = self.datafile.readlines()
        self.datafile.close()

        # Now extract data, which is in the form of strings, into an
        # array of numbers, and separate into matched data and target
        # variables.
        self.data = []
        self.target = []
        
        # Turn content into nested lists
        for i in range(len(content)):
            lineAsArray = self.convertToArray(content[i])
            dataline = []
            for j in range(len(lineAsArray) - 1):
                dataline.append(lineAsArray[j])

            self.data.append(dataline)
            targetIndex = len(lineAsArray) - 1
            self.target.append(lineAsArray[targetIndex])
        
        # Convert data and target into numpy arrays 
        self.X = np.array(self.data)
        self.y = np.array(self.target).reshape(-1,1)

        # Intitialise self.PConditional and self.Priors
        self.PConditional = None
        self.Priors = None
        
    
        # X and y are both arrays of arbitrary length.
        # X is an array of arrays of integers (0 or 1) indicating state.
        # y is an array of imtegers 0-3 indicating the action taken in that state.

        # *********************************************
        # CODE WHICH OUTPUTS MY MODEL EVALUATION METRICS
        # Displays results in terminal on running
        # *********************************************

        print
        print '5-fold cross validation performance!'
        # This uses 5 fold cross validation to measure the accuracy
        scores,mean,aggregate_confusion = self.KFold_Validation(5)
        print scores
        print 'Average score:', mean
        print
        print 'Aggregate Confusion Matrix:'
        print aggregate_confusion
        print 
        print 'FULL Training set evaluation!'
        # This trains the model on the full dataset
        # This will also set self.PConditional and self.Priors 
        PConditional, Priors = self.NB_TrainModel()
        y_predict = self.NB_PredictionVector(self.X,PConditional,Priors)
        accuracy = accuracy_score(self.y,y_predict)
        print 'Accuracy:', accuracy
        confusion2 = confusion_matrix(self.y,y_predict)
        print 'Confusion Matrix:'
        print confusion2
        report = classification_report(self.y,y_predict)
        print
        print report
        print 'Done! Pacman will start playing now'
        

           
    # ***************************************************************    
    # NB_TrainModel Trains the model given predictors X and target y 
    # It computes the prior probabilities of classes
    # And it computes the conditional probabilities P(Xj | Ai)
    # Optional: Specify the X, y data. By default will take self.X & self.y
    
    
    def NB_TrainModel(self,X_train = None,y_train = None):
        # Establishing X_train & y_train
        # If not provided in function arguements, default to self.X and self.y
        if X_train is None:
            X_train = self.X
        if y_train is None:
            y_train = self.y
        
        # (1) Calculate PRIORS based on training set
        n = len(y_train)   
        action, class_counts = np.unique(y_train,return_counts = True)
        Priors = class_counts / float(n)
    
        # (2) Calculate probability each Xj occurs given Ai using training set
        PConditional = np.empty([4,25])
        # PConditional will store conditional Probabilities each Xj occurs given action i
        # Assumptions, Conditional Independence 
        # Assumption 2: Laplace Smoothing to avoid problematic '0' posterior probabilities
        
        y_transpose = y_train.transpose()[0]
        # Transposed to enable dimension matching for boolean indexing
        
        # (3) Append values one at a time, to the appropriate position in PConditional
        for i in range(0,4):
            for j in range(0,25):
                # count_Xj is a Conditional COUNT of SINGLE feature given action i
                count_Xj = sum(X_train[y_transpose == i][:,j])  
                # PXjAi is conditional PROBABILITY, acquired by dividng by COUNT of ai
                PXjAi = (float(count_Xj))/ (float(class_counts[i]))  
                #Laplace smoothing applied to numerator and denominator to prevent '0' probabilities  
                # Assign value of PXjAi, to the appropriate position in PConditional
                PConditional[i,j] = PXjAi                


        # If this is learnt on the full dataset, set the class variables so they are fixed
        # This means we don't have to keep re-learning these. These tables are now stored.
        if X_train.all() == self.X.all():
            self.PConditional = PConditional
            self.Priors = Priors
            
        # Also, Output PCondtional,Priors as a temporary output that can be used how you like. 
        
        return (PConditional,Priors)
    
     
        
    # This function calculates the posterior probabilities and classifies a new feature
    # by assuming conditional independence and finding the product of P(Xj | Ai)
    # Outputs Action Number
    
    def NB_ClassifyFeature(self,feature,PConditional,Priors):
        
        # Convert feature into an array to enable BOOLEAN INDEXING
        feature = np.array(feature)      
        # Filter PConditional where the feature vector takes values = 1      
        ToMultiply = PConditional[:,feature == 1]
        
        # (1) Calculate Likelihoods 
        # Index (i) of likelihoods = likelihood of feature given action (i)
        likelihoods = [1,1,1,1]
        for i in range(0,4):
            for j in range(len(ToMultiply[0])):
                    likelihoods[i] = likelihoods[i] * ToMultiply[i,j]
        
        # (2) Calculate Likelihoods * Priors      
        # Index (i) of numerators corresponds with action (i)            
        numerators = [1,1,1,1]
        for i in range(0,4):
            numerators[i] = likelihoods[i] * Priors[i]
        
        # (3) Calculate PX by taking sum of numerators
        PX = float(sum(numerators))
        
        # (4) Calculate Posteriors 
        # Index (i) of Posteriors = Posterior of action (i) 
        Posteriors = [1,1,1,1]
        for i in range(0,4):
            Posteriors[i] = numerators[i] / PX
        
        # CLASSIFY THE POSTERIORS - Finds the action Number
        actionNumber = np.argmax(Posteriors)
        
        # CONVERTS actionNumber into the DIRECTION 
        action = self.convertNumberToMove(actionNumber)
           
        return {'Posteriors': Posteriors, 'actionNumber': actionNumber,'action': action}        
        
    # K-FOLD FUNCTION
    # Performs, at each iteration of KFold:
        # Filter self.X and self.y for the train and test portions of the data
        # Make predictions on X_test 
        # Calculate testing accuracy and a confusion matrix
    # Output average accuracy and aggregated confusion matrix
        
    def KFold_Validation(self,num_folds):
        
        # Generate the n-folds of training and testing splits
        kf = KFold(self.X.shape[0], n_folds=num_folds)
        scores = []
        aggregate_confusion = np.zeros((4,4))
    
        for train, test in kf:
            
            # Filter X and y for the training and testing set
            X_train = self.X[train,:] 
            y_train = self.y[train]    
            X_test = self.X[test,:]
            y_test = self.y[test]
            
            # Train the Model
            PConditional,Priors = self.NB_TrainModel(X_train,y_train)
            
            # Make predictions on the testing set 
            y_predict = self.NB_PredictionVector(X_test,PConditional,Priors)
            
            score = accuracy_score(y_test,y_predict)
            scores.append(score)
            
            confusion = confusion_matrix(y_test,y_predict)
            aggregate_confusion += confusion
        scores = np.array(scores)
        mean = scores.mean()
        
        return (scores,mean,aggregate_confusion)
                    


    # Makes predictions on the given data X
    # Requires specifying the training data: X_train and y_train
    
    # Potential coding flaw? model is re-trained at each call of Prediction Vector? 

    # Which model is prediction vector going to use ???
    def NB_PredictionVector(self,X,PConditional,Priors):
        Predictions = []
        for i in X:
            Prediction = self.NB_ClassifyFeature(i,PConditional,Priors)['actionNumber']
            Predictions.append(Prediction)
        return Predictions


    # Tidy up when Pacman dies
    def final(self, state):

        print "I'm done!"
        
        # *********************************************
        #
        # Any code you want to run at the end goes here.
        #
        # *********************************************

    # Turn the numbers from the feature set into actions:
    def convertNumberToMove(self,number):
        if number == 0:
            return Directions.NORTH
        elif number == 1:
            return Directions.EAST
        elif number == 2:
            return Directions.SOUTH
        elif number == 3:
            return Directions.WEST

    # Here we just run the classifier to decide what to do
    def getAction(self, state):

        # How we access the features.
        features = api.getFeatureVector(state)
        
        # *****************************************************
        # Call Classifier to decide which move to make 
        action = self.NB_ClassifyFeature(features,self.PConditional,self.Priors)['action']
        # *******************************************************

        # Get the actions we can try.
        legal = api.legalActions(state)
        
        if action in legal:
            return api.makeMove(action,legal)
        else:
            print 'Classifier predicted an illegal move, picking random!'
            return api.makeMove(random.choice(legal),legal)

        # getAction has to return a move. Here we pass "STOP" to the
        # API to ask Pacman to stay where they are. We need to pass
        # the set of legal moves to teh API so it can do some safety
        # checking.
        
        

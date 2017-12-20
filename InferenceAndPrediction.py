import pandas
from pgmpy.models import BayesianModel
import numpy
import itertools
from pgmpy.inference import VariableElimination
import re

#####################         Function Definitions:       #####################  
def intToValue(variable,valueIndex):
    # A function to obtain a value name from an index a variable given.
    # This "translates" this index into the variable that is accessed by this 
    # index within the order of the Bayesian Network. 
    # INPUT: 
    # variable              string: name of the variable the value belongs to
    # valueIndex            the index as an integer
    # OUTPUT:
    # value                 the value accessed by the index 
    
    valueNames = data.sort_values(variable)[variable].unique()
    value = valueNames[valueIndex]
    return value


def valueToInt(variable,value):
    # A function to compute an an integer index from a value of a variable given. 
    # This index corresponds to how the values can be accessed when performing
    # inference with the Bayesian Network.   
    # INPUT: 
    # variable              string: name of the variable the value belongs to
    # value                 the value accessed by the index   
    # OUTPUT: 
    # valueIndex            the index as an integer
    
    valueNames = data.sort_values(variable)[variable].unique()
    valueIndex = numpy.argwhere(valueNames==value).item()
    return int(valueIndex)


def infer50kValue(listofInputVariables, inferer):
    # A function to compute the most probable value of the value '50k' given all 
    # possible combinations of values of variables in a list as evidence. These
    # values will be printed out for each combination.
    # INPUT: 
    # listofInputVariables  a list containing all variables whose values should 
    #                       be used as evidence for the 50k outcome
    # inferer               an instance of VariableElimination of a Bayesian
    #                       Network
    
    possible_values = [] 
    for variable in listofInputVariables:
        possible_values.append(range(0,len(data[variable].unique())))   
        
    # Create list of all possible combinations of values
    combinations = list(itertools.product(*possible_values)) 
    
    # Append all combinations to the evidence and perform inference
    for comb in combinations:
        evidence = dict(zip(listofInputVariables,(comb)))
        print([(key, intToValue(key,evidence[key])) for key in evidence])
        print(inferer.query(['50k'], evidence) ['50k'])
        
        
def predictMissingValuesWithBN(datawithNaNs,inferer):
    # A function to predict missing values in a DataFrame using a Bayesian Network. 
    # INPUT: 
    # datawithNaNs          a panda DataFrame that contains unknown values, 
    #                       marked by 'nan'
    # inferer               an instance of VariableElimination of a Bayesian
    #                       Network
    # OUTPUT: 
    # datawithNaNs          the input DataFrame with 'nan' replaced by predicted
    #                       variable values
    
    # Loop through all rows and check for occurences of 'UNKNOWN'
    for index, row in datawithNaNs.iterrows():
        
        # We will store both the evidence and the values that need to be predicted
        evidence = {}
        toPredict = []
        for column in columnNames:
            
            # If something is nan, it needs to be predicted
            if(pandas.isnull(row[column])):
                toPredict.append(column)
                
            # Otherwise, it counts as evidence    
            else:
                evidence[column] = valueToInt(column,re.sub(r'[\s]', '', row[column]))
        
        # When everything is stored, perform the predictions:  
        predictions = inferer.map_query(toPredict, evidence)
        
        # Now, replace each of those predictions
        for prediction in predictions:
            datawithNaNs.loc[index,prediction] = intToValue(prediction,predictions[prediction])
    return datawithNaNs    
 
def compareMissingValuePrediction_BN_vs_MostFrequent(data_UNKNOWN,data,inferer):
    # A function to compare two approaches of missing value prediction on a 
    # given dataframe: The predictions made by a Bayesian networks and the 
    # predictions made when always choosing the most frequent value.
    # INPUT: 
    # data_UNKNOWN          a panda DataFrame that contains unknown values, 
    #                       marked by 'nan'
    # data                  a panda DataFrame that contains only known values,
    #                       with variables and possible values being the same 
    #                       as those of data_UNKOWN       
    # inferer               an instance of VariableElimination of the 
    #                       BayesianModel that has been fit to data        
    # OUTPUT: 
    
    # Copy the dataset for each of the two approaches:
    data_mostfrequent = data_UNKNOWN.copy()
    data_filledByNetwork = data_UNKNOWN.copy()   
    
    # Fill in missing values according to most frequent approach:
    for column in columnNames: 
        data_mostfrequent[column] = data_mostfrequent[column].replace({'UNKNOWN':data[column].mode()[0]}) 
        
    # Fill in missing values according to prediction
    data_filledByNetwork = data_filledByNetwork.replace('UNKNOWN',numpy.nan)
    data_filledByNetwork = predictMissingValuesWithBN(data_filledByNetwork,inferer)
    
    # Check the predictive power: 
    # Step 1: Remove labels
    data_filledByNetworkPrediction = data_filledByNetwork.copy()
    data_filledByNetworkPrediction['50k'] = numpy.nan
    data_mostfrequentPrediction = data_mostfrequent.copy()
    data_mostfrequentPrediction['50k'] = numpy.nan
    
    # Step 2: Predict the missing 50k states:                              
    data_filledByNetworkPrediction = predictMissingValuesWithBN(data_filledByNetworkPrediction,inferer)
    data_mostfrequentPrediction = predictMissingValuesWithBN(data_mostfrequentPrediction,inferer)
    
    # Step 3: Compare to performance of the 
    correctlyPredicted_ByNetwork = 0;
    correctlyPredicted_mostfrequent = 0;
    for index, row in data_UNKNOWN.iterrows():    
        if(data_mostfrequentPrediction.loc[index,'50k'] == re.sub(r'[\s]', '', data_UNKNOWN.loc[index,'50k'])):
            correctlyPredicted_mostfrequent += 1
        if(data_filledByNetworkPrediction.loc[index,'50k'] == re.sub(r'[\s]', '', data_UNKNOWN.loc[index,'50k'])):
            correctlyPredicted_ByNetwork += 1
    return [correctlyPredicted_ByNetwork,correctlyPredicted_mostfrequent, data_mostfrequentPrediction,data_filledByNetworkPrediction]        
                                                    
     
#####################              Load Data:             #####################    
# Read in binned data
data = pandas.read_csv('adult_dataset_removedunkown_binned.csv')
data_UNKNOWN = pandas.read_csv('adult_dataset-UNKNOWN.csv')

# Name the colums
columnNames = ['Age','Workclass','Education', 'Marital-and-spouse-status','Occupation','Race','Sex','Capital-change','Hours-per-week', 'Native-continent','50k']
data.columns = columnNames
data_UNKNOWN.columns = columnNames

#####################           Create Network:           #####################
print('Starting network creation')

# Define the network structure:
adultDataModel = BayesianModel([('Sex','Occupation'),
                                ('Sex','Workclass'),
                                ('Sex','Hours-per-week'),
                                ('Sex','50k'),
                                ('Age','Occupation'),
                                ('Age','Hours-per-week'),
                                ('Age','Marital-and-spouse-status'),
                                ('Age','Capital-change'),
                                ('Age','Education'),
                                ('Native-continent','Occupation'),
                                ('Native-continent','Education'),
                                ('Native-continent','Race'),
                                ('Race','Occupation'),
                                ('Race','Education'),
                                ('Race','50k'),
                                ('Occupation','Workclass'),
                                ('Occupation','Hours-per-week'),
                                ('Education','Workclass'),
                                ('Education','Marital-and-spouse-status'),
                                ('Education','Capital-change'),
                                ('Workclass','50k'),
                                ('Workclass','Hours-per-week'),
                                ('Workclass','Capital-change'),
                                ('Marital-and-spouse-status','Hours-per-week'),
                                ('Hours-per-week','50k'),
                                ('Capital-change','50k')])

# Compute the CPD for each node:
adultDataModel.fit(data)     
print('Network created')

#####################         Inference problems:         ##################### 
print('Starting inference')     

#Create an instance VariableElimination                                    
inferer = VariableElimination(adultDataModel)  

# Call the inference function:
infer50kValue(['Sex'], inferer)
infer50kValue(['Race'], inferer)
infer50kValue(['Sex','Race'], inferer)
print('Inference done!')                               

#####################           Missing values:           #####################
print('Starting missing value analysis')

# Perform prediction of the missing values using two techniques (most frequent value from column vs MAP using the Bayesian network)
# Compute the overall score for both       
[correctlyPredicted_ByNetwork,correctlyPredicted_mostfrequent,data_mostfrequentPrediction,data_filledByNetworkPrediction] = compareMissingValuePrediction_BN_vs_MostFrequent(data_UNKNOWN,data,inferer)    
BNscore = correctlyPredicted_ByNetwork/float(len(data_UNKNOWN))
Frscore = correctlyPredicted_mostfrequent/float(len(data_UNKNOWN))
print(correctlyPredicted_ByNetwork)
print(correctlyPredicted_mostfrequent)
print('The Bayesian Network prediction was '+ str(BNscore*100) + '% correct')
print('The most frequent approach prediction was '+ str(Frscore*100) + '% correct')
print('Missing value analysis done!')

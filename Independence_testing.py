import pandas
from scipy.stats import chisquare
import numpy as np
import itertools

####################
# Needed functions #
####################

def test_conditional_independence(data, X, Y, Zs=[]):
    """Chi-square conditional independence test.
    Tests the null hypothesis that X is independent from Y given Zs.

    This is done by comparing the observed frequencies with the expected
    frequencies if X,Y were conditionally independent, using a chisquare
    deviance statistic. The expected frequencies given independence are
    `P(X,Y,Zs) = P(X|Zs)*P(Y|Zs)*P(Zs)`. The latter term can be computed
    as `P(X,Zs)*P(Y,Zs)/P(Zs).

    Parameters
    ----------
    X: int, string, hashable object
        A variable name contained in the data set
    Y: int, string, hashable object
        A variable name contained in the data set, different from X
    Zs: list of variable names 
        A list of variable names contained in the data set, different from X and Y.
        This is the separating set that (potentially) makes X and Y independent.
        Default: []

    Returns
    -------
    For every combination of values from the evidence, return:
    chi2: float
        The chi2 test statistic.
    p_value: float
        The p_value, i.e. the probability of observing the computed chi2
        statistic (or an even higher value), given the null hypothesis
        that X _|_ Y | Zs.
    """
    variables = list(data.columns.values)
    state_names = {var: collect_state_names(data, var) for var in variables}
    
    if isinstance(Zs, (frozenset, list, set, tuple,)):
        Zs = list(Zs)
    else:
        Zs = [Zs]
    
    # compute actual frequency/state_count table:
    # = P(X,Y,Zs)
    XYZ_state_counts = pandas.crosstab(index=data[X],
                                   columns=[data[Y]] + [data[Z][data[Z] == value] for Z,value in Zs])
    
    # reindex to add missing rows & columns (if some values don't appear in data)
    row_index = state_names[X]
    column_index = pandas.MultiIndex.from_product(
                        [state_names[Y]] + [[value] for Z,value in Zs], names=[Y]+[value for Z,value in Zs])
    XYZ_state_counts = XYZ_state_counts.reindex(index=row_index,    columns=column_index).fillna(0)
    #print(XYZ_state_counts)
    
    # compute the expected frequency/state_count table if X _|_ Y | Zs:
    # = P(X|Zs)*P(Y|Zs)*P(Zs) = P(X,Zs)*P(Y,Zs)/P(Zs)
    if Zs:
        XZ_state_counts = XYZ_state_counts.sum(axis=1, level=[value for Z,value in Zs])  # marginalize out Y
        YZ_state_counts = XYZ_state_counts.sum().unstack([value for Z,value in Zs])      # marginalize out X
    else:
        XZ_state_counts = XYZ_state_counts.sum(axis=1)
        YZ_state_counts = XYZ_state_counts.sum()
    Z_state_counts = YZ_state_counts.sum()  # marginalize out both
    
    XYZ_expected = pandas.DataFrame(index=XYZ_state_counts.index, columns=XYZ_state_counts.columns)
    for X_val in XYZ_expected.index:
        if Zs:
            for Y_val in XYZ_expected.columns.levels[0]:
                XYZ_expected.loc[X_val, Y_val] = (XZ_state_counts.loc[X_val] *
                                                  YZ_state_counts.loc[Y_val] /
                                                  Z_state_counts).values
        else:
            for Y_val in XYZ_expected.columns:
                XYZ_expected.loc[X_val, Y_val] = (XZ_state_counts.loc[X_val] *
                                                  YZ_state_counts.loc[Y_val] /
                                                  float(Z_state_counts))

    observed = XYZ_state_counts.values.flatten()
    expected = XYZ_expected.fillna(0).values.flatten()
    if(len(Zs) > 0):
        expected = np.concatenate(expected).ravel().tolist()
    
    # remove elements where the expected value is 0;
    # this also corrects the degrees of freedom for chisquare
    observed, expected = zip(*((o, e) for o, e in zip(observed, expected) if not e == 0))

    chi2, significance_level = chisquare(observed, expected)

    return (chi2, significance_level)
    
def collect_state_names(data, variable):
    "Return a list of states that the variable takes in the data"
    states = sorted(list(data.ix[:, variable].dropna().unique()))
    return states


########################
# TESTING INDEPENDENCE #
########################

# Read in binned data and name columns
data = pandas.read_csv('adult_dataset_removedunkown_binned.csv')
data.columns = ['Age','Workclass','Education', 'Marital-and-spouse-status','Occupation',
                'Race','Sex','Capital-change','Hours-per-week', 'Native-continent','50K']

# Make estimator for chi^2 test and make list of variables to test (in the form of a list of [variable1, variable2, [(evidence1,value),(evidence2,value) etc.]])
tests = [#['Age', 'Sex', []],
         #['Age', 'Native-continent', []], 
         #['Native-continent', 'Sex', []],
         ['Occupation', 'Age', []],
         ['Occupation', 'Native-continent', []],
         ['Occupation', 'Race', []],
         ['Education', 'Sex', []],
         ['Hours-per-week', 'Native-continent', []],
         ['Occupation', 'Education', []],
         ['Occupation', 'Hours-per-week', ['Sex']],
         ['Workclass', 'Hours-per-week', ['Occupation']],
         #['Occupation', 'Hours-per-week', ['Sex','Workclass']], #EXTRA TEST
         #['Workclass', 'Hours-per-week', ['Sex','Occupation']], #EXTRA TEST
         #['Occupation', 'Hours-per-week', ['Sex','Age']], #EXTRA TEST
         ['Marital-and-spouse-status', 'Education', ['Age']],
         ['Marital-and-spouse-status', 'Capital-change', ['Age']],
         ['Education', 'Race', ['Native-continent']],
         ['Education', 'Capital-change', ['Age','Workclass']],
         ['Race', 'Capital-change', ['Native-continent']],
         ['Workclass', 'Sex', ['Occupation']],
         ['Hours-per-week', 'Age', ['Marital-and-spouse-status']]
         #['Hours-per-week', 'Age', ['Marital-and-spouse-status','Workclass','Occupation']], #EXTRA TEST
         #['Hours-per-week', 'Capital-change', ['Age','Workclass','Education']], #EXTRA TEST
        ]

# Calculate RMSE for the tests
for variables in tests:
    
    # Create tasks for each value in the conditional set
    pos_val = []
    for ev in variables[2]:
        pos_val.append(list(data[ev].unique()))
    combinations = list(itertools.product(*pos_val))
    
    for comb in combinations:
        # Create tuples of variable and value
        evidence = [(a,b) for a,b in zip(variables[2],comb)]
        # Determine chi^2 for variable[0] with variable[1] given the variables in variable[2]
        chi2,p = test_conditional_independence(data, variables[0], variables[1], evidence)
        #print(chi2)

        # Determine degrees of freedom for variable[0], variable[1] and the variables in variable[2]
        df_1 = len(data.groupby(variables[0]).size())
        df_2 = len(data.groupby(variables[1]).size())
        df = (df_1-1)*(df_2-1)
        #print(df)

        # Determine length of data
        evidenced_data = data
        for var,value in evidence:
            evidenced_data = evidenced_data[evidenced_data[var] == value]
        N = evidenced_data.shape[0]
        #print(N)
        
        # print the test with the RMSE scores
        if(chi2 > df):
            RMSE = np.sqrt((chi2 - df)/(N-1))
        else:
            RMSE = 0
        print("The root mean square error for the hypothesis that " + variables[0] + " is independent of " + variables[1] + 
              " given " + str(evidence) + " is: " + str(RMSE))
    print('\n')

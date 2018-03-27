import pandas as pd
import numpy as np
np.random.seed(42)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as LR
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.metrics import roc_auc_score
from sklearn.tree import _tree

def f_gini(y_true, y_pred):
    gini = 200 * roc_auc_score(y_true, y_pred) - 100
    return gini
    
def f(X_train, X_test, y_train, y_test, low_gini_cols, model = LR(tol = 1e-8), iterations = 8, verbose = False):    
    all_cols = list(X_train.columns)
    
    def print_if(txt, verbose = verbose):
        if verbose:
            print(txt)
        else:
            pass

    current_cols = []
    #low_gini_cols = []

    max_gini_train = 0
    max_gini_test = 0
    max_gini_ft = None

    for i in range(1, iterations + 1):
        print('\nITERATION {}'.format(i))
        
        max_gini_train = 0
        max_gini_test = 0
        max_gini_ft = None
        
        for col_temp in all_cols:
            if col_temp in current_cols \
                     or 'PAYMENT' in col_temp \
                     or  col_temp in low_gini_cols:
                continue

            X_train_temp = X_train.loc[:, current_cols + [col_temp]]
            X_test_temp  = X_test.loc[:, current_cols + [col_temp]]

            model.fit(X_train_temp, y_train)

            y_pred_train = model.predict_proba(X_train_temp)[:, 1]
            y_pred_test  = model.predict_proba(X_test_temp)[:, 1]

            gini_train = 200 * roc_auc_score(y_train, y_pred_train) - 100
            gini_test  = 200 * roc_auc_score(y_test, y_pred_test) - 100

            if gini_test < 10:
                low_gini_cols.append(col_temp)
                print_if('{} goes to low_gini_cols list\n'.format(col_temp))
            else:
                print_if('{} ({})\ngini_train: {}, gini_test: {}\n'.format(col_temp, i, gini_train, gini_test))  
                if max_gini_test < gini_test:
                    max_gini_test = gini_test
                    max_gini_ft = col_temp
                    max_gini_train = gini_train


        current_cols.append(max_gini_ft)
        print()
        print('current_cols: {}, \nmax_gini (train / test): {} / {}\n'.format(current_cols, max_gini_train, max_gini_test))
    
    #final model train
    X_train_temp = X_train.loc[:, current_cols]
    model.fit(X_train_temp, y_train)
    
    return model
    
def tree_to_code(model, feature_names):
    from sklearn.tree import _tree
    
    tree_ = model.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    print('def tree({}):'.format(", ".join(feature_names)))

    def recurse(node, depth):
        indent = "    " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print("{}if {} <= {}:".format(indent, name, threshold))
            recurse(tree_.children_left[node], depth + 1)
            print("{}else:  # if {} > {}".format(indent, name, threshold))
            recurse(tree_.children_right[node], depth + 1)
        else:
            print("{}return {}".format(indent, tree_.value[node]))

    recurse(0, 1)

    
def make_info_table(df):
    info_table = pd.concat([100 * df.sum() / df.shape[0], df.sum()], axis = 1)
    info_table.columns = ['pcnt', 'qty']
    info_table.loc['qty_all', ['pcnt', 'qty']] = info_table.loc[:, ['pcnt', 'qty']].sum()
    info_table['qty'] = info_table['qty'].astype(np.int64)
    return info_table


def create_bins_for_column(df1, col, q_qty = 5):
    '''
    Make a dataframe with q_qty bins for col in df1
    '''
    #q_qty can't be more than all unique_values 
    q_qty = np.min([q_qty, df1[col].nunique()]) #(NaN doesn't count)
    
    while True:
        #create dataframe for bins
        df2 = df1.loc[:, [col]].copy()

        #deal with NaNs
        df2[col + '=NaN'] = (df1[col].isnull()).astype(np.int64)

        #create bins for non-NaNs
        q_qty_temp = q_qty
        values_qty = df1[col].value_counts()
        freq_values = []
        intervals_cols1 = None

        for i in range(q_qty):
            try:
                if q_qty_temp > 0:
                    intervals_col = pd.qcut(df1.loc[(~df1[col].isin(freq_values))
                                                    & (df1[col].notnull()), col], q_qty_temp)
                    #print(freq_values)
                    intervals_cols1 = pd.get_dummies(intervals_col, columns = col, prefix = col, prefix_sep = '=')
                    #print(intervals_cols1)
                break
            except:
                q_qty_temp -= 1
                current_frequent_value = values_qty.index[i]
                #current_frequent_qty = values_qty.values[i]
                df2[col + '=' + str(current_frequent_value)] = (df1[col] == current_frequent_value).astype(np.int64)
                freq_values.append(current_frequent_value)
            
        if intervals_cols1 is not None and intervals_cols1.shape[0] > 2:
            df2 = pd.concat([df2, intervals_cols1], axis = 1).fillna(0)
        for item in df2.columns:
            if item != col:
                df2[item] = df2[item].astype(np.int64)
                
        if df2.drop(col, axis = 1).sum().sum() == df2.shape[0]:
            break
        else:
            q_qty -= 1

    assert df2.drop(col, axis = 1).sum().sum() == df2.shape[0]
    
    #Eleminate the problem, when there is a single value can be placed into one of the intervals
    for val in freq_values:
        for col1 in df2.columns:
            s = col1
            if s == col or s.find('(') < 0 or s.find(']') < 0:
                continue
            equal_position = s.find('=')
            s = s[(equal_position + 1):]
            for item in ('[', ']', '(', ')', ' '):
                s = s.replace(item, '')
            mn, mx = s.split(',')
            mn = float(mn)
            mx = float(mx)
            if mn < val and val <= mx:
                df2[col + '=(' + str(mn) + ', ' + str(val) + ')'] = ((df2[col] > mn) & (df2[col] < val)).astype(np.int64)
                df2[col + '=(' + str(val) + ', ' + str(mx) + ']'] = ((df2[col] > val) & (df2[col] <= mx)).astype(np.int64)
                df2 = df2.drop(col1, axis = 1)

    
    df2_bins = df2.drop(col, axis = 1).sort_index()
    
    #Sort columns
    def clear_str(s):
        equal_position = s.find('=')
        s = s[(equal_position + 1):]
        for item in ('[', ']', '(', ')', ' '):
            s = s.replace(item, '')
        s = s.split(',')[0]
        s = float(s)
        return s

    columns_no_order = list(df2_bins.columns)
    columns_sorted = []
    nan_columns = columns_no_order.pop(columns_no_order.index(col + '=NaN'))
    columns_sorted.append(nan_columns)
    columns_no_order.sort(key=lambda x: clear_str(x), reverse=False)
    columns_sorted.extend(columns_no_order)
    
    #order columns and sort indexes
    df2 = df2.loc[:, columns_sorted].sort_index()
    df2_bins = df2_bins.loc[:, columns_sorted].sort_index()
    
    #info_table
    info_table = make_info_table(df2_bins)

    return df2_bins, df2, info_table, columns_sorted


def bins_transfer(df_other, col, columns_sorted):
    '''
    Transfer bins from test to train,
    so bins for test become the same as for train.
    
    df2 - test (or valid) dataframe that is need to be binned,
    columns_sorted - a list of bin columns from create_bins_for_column function,
    col - a feature that is going to be binned.
    '''
    assert col == columns_sorted[0][:columns_sorted[0].find('=')]
    
    df_other_binned = df_other.loc[:, [col]]
    
    len_columns_sorted = len(columns_sorted)
    
    for i, bin_col in enumerate(columns_sorted):
        bc = bin_col[bin_col.find('=') + 1:]
        #print(bc)
        if bc == 'NaN':
            df_other_binned[bin_col] = (df_other_binned[col].isnull()).astype(np.int64)
            continue
        for item in ['(', ')', '[', ']']:
            if bin_col.find(item) >= 0:
                left_boundary = bc[0]
                right_boundary = bc[-1]
                for item2 in ['(', ')', '[', ']', ' ']:
                    bc = bc.replace(item2, '')
                    
                mn = float(bc.split(',')[0])
                mx = float(bc.split(',')[1])
                #print(bc, mn, mx)
                
                if i == 1: # first interval after NaN
                    if right_boundary == ')':
                        df_other_binned[bin_col] = (df_other_binned[col] < mx).astype(np.int64)
                    elif right_boundary == ']':
                        df_other_binned[bin_col] = (df_other_binned[col] <= mx).astype(np.int64)
                elif i == len_columns_sorted - 1: # first interval after NaN
                    if left_boundary == '(':
                        df_other_binned[bin_col] = (mn < df_other_binned[col]).astype(np.int64)
                    elif right_boundary == ']':
                        df_other_binned[bin_col] = (mn <= df_other_binned[col]).astype(np.int64)
                else:
                    if left_boundary == '(' and right_boundary == ')':
                        df_other_binned[bin_col] = ((mn < df_other_binned[col]) & (df_other_binned[col] < mx)).astype(np.int64)
                    elif left_boundary == '[' and right_boundary == ']':
                        df_other_binned[bin_col] = ((mn <= df_other_binned[col]) & (df_other_binned[col] <= mx)).astype(np.int64)
                    elif left_boundary == '[' and right_boundary == ')':
                        df_other_binned[bin_col] = ((mn <= df_other_binned[col]) & (df_other_binned[col] < mx)).astype(np.int64)
                    elif left_boundary == '(' and right_boundary == ']':
                        df_other_binned[bin_col] = ((mn < df_other_binned[col]) & (df_other_binned[col] <= mx)).astype(np.int64)
       
                break
            else:
                if i == 1: # first interval after NaN
                    df_other_binned[bin_col] = (df_other_binned[col] <= float(bc)).astype(np.int64)
                elif i == len_columns_sorted - 1: #last interval
                    df_other_binned[bin_col] = (df_other_binned[col] >= float(bc)).astype(np.int64)
                else:
                    df_other_binned[bin_col] = (df_other_binned[col] == float(bc)).astype(np.int64)
                    
    
    assert df_other_binned.drop(col, axis = 1).sum().sum() == df_other_binned.shape[0]
    
    df_other_binned_short = df_other_binned.drop(col, axis = 1).sort_index()
    
    #order columns
    df_other_binned = df_other_binned.loc[:, columns_sorted].sort_index()
    df_other_binned_short = df_other_binned_short.loc[:, columns_sorted].sort_index()
    
    #info_table
    info_table = make_info_table(df_other_binned_short)

    return df_other_binned_short, df_other_binned, info_table



    
    
    
    
    
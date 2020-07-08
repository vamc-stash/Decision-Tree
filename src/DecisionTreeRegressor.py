import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import math 
import operator


eps = np.finfo(float).eps

def rmse_score(y_true, y_pred):

	"""	rmse score = sqrt((sum[i=0 to n](y_true - y_pred)) / len(y_true)) """

	return np.sqrt((np.subtract(y_pred, y_true) ** 2).sum()/len(y_true))

def train_test_split(x, y, test_size = 0.25, random_state = None):

	""" partioning the data into train and test sets """

	x_test = x.sample(frac = test_size, random_state = random_state)
	y_test = y[x_test.index]

	x_train = x.drop(x_test.index)
	y_train = y.drop(y_test.index)

	return x_train, x_test, y_train, y_test




class DecisionTreeRegressor:

	def __init__(self, max_depth = None, min_sample_leaf = 3):

		self.depth = 0 #Depth of the tree
		self.max_depth = max_depth	#Maximum depth of the tree
		self.min_sample_leaf = min_sample_leaf	#Minimum number of samples for each node
		self.coefficient_of_variation = 10 	#Stopping Criterion

		self.features = list
		self.X_train = np.array
		self.y_train = np.array
		self.num_feats = int 
		self.train_size = int 

	def fit(self, X, y):

		self.X_train = X 
		self.y_train = y
		self.features = list(X.columns)
		self.train_size = X.shape[0]
		self.num_feats = X.shape[1]

		df = X.copy()
		df['target'] = y.copy()

		#Builds Decision Tree
		self.tree = self._build_tree(df)

		print("\nDecision Tree(depth = {}) : \n {}".format(self.depth, self.tree))

	def _build_tree(self, df, tree = None):

		"""
			Args:
				df: current number of rows available for splitting(decision making)

		"""

		#Get feature with minimum score
		feature, cutoff = self._find_best_split(df)

		if cutoff is None:
			return tree

		#Initialization of tree
		if tree is None:
			tree = {}
			tree[feature] = {}

		#Left Child
		new_df = self._split_rows(df, feature, cutoff, operator.le)
		
		target_coef_of_var = self._coef_ov(new_df['target'])

		self.depth += 1

		if(target_coef_of_var < self.coefficient_of_variation or len(new_df) <= self.min_sample_leaf): #pure group
			tree[feature]['<=' + str(cutoff)] = new_df['target'].mean()
		else:
			if self.max_depth is not None and self.depth >= self.max_depth:
				tree[feature]['<=' + str(cutoff)] = new_df['target'].mean()
			else:
				tree[feature]['<=' + str(cutoff)] = self._build_tree(new_df)


		#Right Child
		new_df = self._split_rows(df, feature, cutoff, operator.gt)

		target_coef_of_var = self._coef_ov(new_df['target'])

		if(target_coef_of_var < self.coefficient_of_variation or len(new_df) <= self.min_sample_leaf): #pure group
			tree[feature]['>' + str(cutoff)] = new_df['target'].mean()
		else:
			if self.max_depth is not None and self.depth >= self.max_depth:
				tree[feature]['>' + str(cutoff)] = new_df['target'].mean()
			else:
				tree[feature]['>' + str(cutoff)] = self._build_tree(new_df)

		return tree

	def _coef_ov(self, y):

		""" calculates coefficient of variation:

		    COV = (Mean of y / Standard Deviation of y) * 100

		"""
		if(y.std() == 0):
			return 0
		coef_of_var = (y.mean()/y.std()) * 100

		return coef_of_var

	def _split_rows(self, df, feature, feat_val, operation ):

		""" split rows based on given criterion """

		return df[operation(df[feature], feat_val)].reset_index(drop = True)

	def _find_best_split(self, df):

		"""
			Finds the column to split on first.

		"""

		best_feature = str
		cutoff = None
		best_score = float('inf')


		for feature in list(df.columns[:-1]):

			score, threshold = self._find_feature_split(feature, df)

			if score < best_score:
				best_feature = feature
				best_score = score
				cutoff = threshold
		
		return best_feature, cutoff 

	def _find_feature_split(self, feature, df):

		best_score = float('inf')
		cutoff = float

		for val in df[feature]:
			left_child = df[feature][df[feature] <= val] 
			right_child = df[feature][df[feature] > val]

			if(len(left_child) > 0 and len(right_child) > 0):
				score = self._find_score(df, left_child, right_child)

				if score < best_score:
					best_score = score
					cutoff = val

		return best_score, cutoff


	def _find_score(self, df, lhs, rhs):

		y = df['target']

		lhs_std = y.iloc[lhs.index].std()
		rhs_std = y.iloc[rhs.index].std()

		if(np.isnan(lhs_std)):
			lhs_std = 0
		if(np.isnan(rhs_std)):
			rhs_std = 0

		return lhs_std * lhs.sum() + rhs_std * rhs.sum()

	def _predict_target(self, feature_lookup, x, tree):

		for node in tree.keys():
			val = x[node]
			if type(val) == str:
				tree = tree[node][val]
			else:
				cutoff = str(list(tree[node].keys())[0]).split('<=')[1]

				if(val <= float(cutoff)):	#Left Child
					tree = tree[node]['<='+cutoff]
				else:						#Right Child
					tree = tree[node]['>'+cutoff]

			prediction = str

			if type(tree) is dict:
				prediction = self._predict_target(feature_lookup, x, tree)
			else:
				predicton = tree 
				return predicton

		return prediction   


	def predict(self, X):

		results = []
		feature_lookup = {key: i for i, key in enumerate(list(X.columns))}
		
		for index in range(len(X)):

			results.append(self._predict_target(feature_lookup, X.iloc[index], self.tree))

		return np.array(results)





if __name__ == '__main__':

	#Loading Dataset
	print('\n--------------Real Estate Dataset------------------------')

	data = pd.read_csv('../Data/datasets_Real_Estate.csv')
	#print(df)

	#Split Features and target
	X, y = data.drop([data.columns[0], data.columns[-1]], axis = 1), data[data.columns[-1]]

	#Split data into Training and Testing Sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

	dt_reg = DecisionTreeRegressor()
	dt_reg.fit(X, y)
	
	print("\nTrain RMSE : {}".format(rmse_score(y_train, dt_reg.predict(X_train))))
	print("\nTest RMSE: {}".format(rmse_score(y_test, dt_reg.predict(X_test))))


	dt_reg = DecisionTreeRegressor(max_depth = 150)
	dt_reg.fit(X, y)
	
	print("\nTrain RMSE : {}".format(rmse_score(y_train, dt_reg.predict(X_train))))
	print("\nTest RMSE: {}".format(rmse_score(y_test, dt_reg.predict(X_test))))

	
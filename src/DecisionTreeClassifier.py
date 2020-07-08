import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import math 
import operator


eps = np.finfo(float).eps

def accuracy_score(y_true, y_pred):

	"""	score = (y_true - y_pred) / len(y_true) """

	return round(float(sum(y_pred == y_true))/float(len(y_true)) * 100 ,2)

def train_test_split(x, y, test_size = 0.25, random_state = None):

	""" partioning the data into train and test sets """

	x_test = x.sample(frac = test_size, random_state = random_state)
	y_test = y[x_test.index]

	x_train = x.drop(x_test.index)
	y_train = y.drop(y_test.index)

	return x_train, x_test, y_train, y_test




class DecisionTreeClassifier:

	def __init__(self, max_depth = None, min_sample_leaf = None):

		self.depth = 0 #Depth of the tree
		self.max_depth = max_depth	#Maximum depth of the tree
		self.min_sample_leaf = min_sample_leaf	#Minimum number of samples for each node

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

		#Get feature with maximum information gain
		feature, cutoff = self._find_best_split(df)

		#Initialization of tree
		if tree is None:
			tree = {}
			tree[feature] = {}

		if df[feature].dtypes == object:
			"""
				- to handle columns with Categorical Values(object type)
 				- parent have only one child

			"""
			for feat_val in np.unique(df[feature]):

				new_df = self._split_rows(df, feature, feat_val, operator.eq)
				targets, count = np.unique(new_df['target'], return_counts = True)

				if(len(count) == 1): #pure group 
					tree[feature][feat_val] = targets[0]
				else:
					self.depth += 1
					if self.max_depth is not None and self.depth >= self.max_depth:
						tree[feature][feat_val] = targets[np.argmax(count)]
					else:
						tree[feature][feat_val] = self._build_tree(new_df)
					
		else:
			"""
				- to handle columns with Numerical Values(int, float....)
 				- parent have two child
 				- Left Child: rows with <= cutoff
 				- Right Child:  rows with > cutoff

			"""
			#Left Child
			new_df = self._split_rows(df, feature, cutoff, operator.le)
			targets, count = np.unique(new_df['target'], return_counts = True)

			self.depth += 1

			if(len(count) == 1): #pure group
				tree[feature]['<=' + str(cutoff)] = targets[0]
			else:
				if self.max_depth is not None and self.depth >= self.max_depth:
					tree[feature]['<=' + str(cutoff)] = targets[np.argmax(count)]
				else:
					tree[feature]['<=' + str(cutoff)] = self._build_tree(new_df)


			#Right Child
			new_df = self._split_rows(df, feature, cutoff, operator.gt)
			targets, count = np.unique(new_df['target'], return_counts = True)

			if(len(count) == 1): #pure group
				tree[feature]['>' + str(cutoff)] = targets[0]
			else:
				if self.max_depth is not None and self.depth >= self.max_depth:
					tree[feature]['>' + str(cutoff)] = targets[np.argmax(count)]
				else:
					tree[feature]['>' + str(cutoff)] = self._build_tree(new_df)

		return tree


	def _split_rows(self, df, feature, feat_val, operation ):

		""" split rows based on given criterion """

		return df[operation(df[feature], feat_val)].reset_index(drop = True)

	def _find_best_split(self, df):

		"""
			Finds the column to split on first using 'Information Gain' Metric.

			Information Gain(IG) = Entropy(parent) - Sum of Entropy(Children)
						IG(T, a) = H(T) - H(T|a)

			Entropy(parent) H(T) = (Sum[i=1 to J](- Pi * log(Pi)))
			Sum of Entropy(children) H(T|a) = Sum(P(a) * Sum[i=1 to J](- P(i|a) * log(P(i|a)))

			Returns:
				Feature With Maximum Information Gain

		"""

		ig = []
		thresholds = []

		for feature in list(df.columns[:-1]):

			entropy_parent = self._get_entropy(df) #H(T)
			entropy_feature_split, threshold = self._get_entropy_feature(df, feature) #H(T|a)

			info_gain = entropy_parent - entropy_feature_split #IG(T, a)

			ig.append(info_gain)
			thresholds.append(threshold)

		
		return df.columns[:-1][np.argmax(ig)], thresholds[np.argmax(ig)] #Returns feature with max information gain 

	def _get_entropy(self, df):

		""" Finds Entropy of parent ie., H(T) """

		entropy = 0
		for target in np.unique(df['target']):
			fraction = df['target'].value_counts()[target] / len(df['target'])
			entropy += -fraction * np.log2(fraction)

		return entropy

	def _get_entropy_feature(self, df, feature):

		""" Finds Sum of entropy of children ie., H(T|a) """

		entropy = 0
		threshold = None

		if(df[feature].dtypes == object):

			#sum of entropies of children(all distinct features)
			for feat_val in np.unique(df[feature]):
				entropy_feature = 0

				#entropy for each distinct feature value
				for target in np.unique(df['target']):
					num = len(df[feature][df[feature] == feat_val][df['target'] == target])
					den = len(df[feature][df[feature] == feat_val])

					fraction = num / (den+eps)
					entropy_feature += -fraction * np.log2(fraction + eps)

				weightage = den/len(df)
				entropy += weightage * entropy_feature
		else:
			entropy = 1 #Max Value

			prev = 0
			for feat_val in np.unique(df[feature]):
				cur_entropy = 0
				cutoff = (feat_val + prev)/2

				#sum of entropies of left child(<= cutoff) and right child(> cutoff)
				for operation in [operator.le, operator.gt]:
					entropy_feature = 0

					for target in np.unique(df['target']):
						num = len(df[feature][operation(df[feature], cutoff)][df['target'] == target])
						den = len(df[feature][operation(df[feature], cutoff)])

						fraction = num / (den + eps)
						entropy_feature += -fraction * np.log2(fraction + eps)

					weightage = den/len(df)
					cur_entropy += weightage * entropy_feature

				if cur_entropy < entropy:
					entropy = cur_entropy
					threshold = cutoff
				prev = feat_val

		return entropy, threshold

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
	print('\n--------------Weather Dataset------------------------')

	data = pd.read_table('../Data/weather.txt')
	#print(df)

	#Split Features and target
	X, y = data.drop([data.columns[-1]], axis = 1), data[data.columns[-1]]

	dt_clf = DecisionTreeClassifier()
	dt_clf.fit(X, y)

	print("\nTrain Accuracy: {}".format(accuracy_score(y, dt_clf.predict(X))))


	############################################################################################################

	print('\n--------------Fruit Dataset------------------------')
	training_data = [['Green', 3, 'Apple'],
				     ['Yellow', 3, 'Apple'],
				     ['Red', 1, 'Grape'],
				     ['Red', 1, 'Grape'],
				     ['Yellow', 2, 'Lemon']]

	data = pd.DataFrame(training_data, columns = ['Color', 'Diameter', 'Label'])

	#Split Features and target
	X, y = data.drop([data.columns[-1]], axis = 1), data[data.columns[-1]]

	dt_clf = DecisionTreeClassifier()
	dt_clf.fit(X, y)

	print("\nTrain Accuracy: {}".format(accuracy_score(y, dt_clf.predict(X))))

	############################################################################################################

	print('\n--------------Gender Dataset------------------------')

	data = pd.read_csv('../Data/gender.csv')
	#print(df)

	#Split Features and target
	X, y = data.drop([data.columns[0]], axis = 1), data[data.columns[0]]

	dt_clf = DecisionTreeClassifier()
	dt_clf.fit(X, y)

	print("\nTrain Accuracy: {}".format(accuracy_score(y, dt_clf.predict(X))))

	############################################################################################################

	print('\n--------------Iris Dataset------------------------')

	data = pd.read_csv('../Data/iris.csv')
	#print(df)

	#Split Features and target
	X, y = data.drop([data.columns[-1]], axis = 1), data[data.columns[-1]]

	#Split data into Training and Testing Sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

	dt_clf = DecisionTreeClassifier()
	dt_clf.fit(X_train, y_train)

	print("Train Accuracy: {}".format(accuracy_score(y_train, dt_clf.predict(X_train))))
	print("Test Accuracy: {}".format(accuracy_score(y_test, dt_clf.predict(X_test))))


	dt_clf = DecisionTreeClassifier(max_depth = 4)
	dt_clf.fit(X_train, y_train)

	print("Train Accuracy: {}".format(accuracy_score(y_train, dt_clf.predict(X_train))))
	print("Test Accuracy: {}".format(accuracy_score(y_test, dt_clf.predict(X_test))))


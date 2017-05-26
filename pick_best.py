from pprint import pprint

features = {}

with open('trees.txt','r') as file:
	lines = file.readlines()
	for line in lines:
		stripped = line.strip()
		if '|' in stripped:
			split = stripped.split('   ')
			for part in split:
				if len(part) > 0 and part != '|':
					feature = part.split(' ')[0]
					if feature in features.keys():
						features[feature] += 1
					else:
						features[feature] = 1
					break


# pprint(features, width=1)

best_features = []

for i in range(20):
	best = list(features.keys())[list(features.values()).index(max(features.values()))]
	best_features.append(best)
	del features[best]

pprint(best_features, width=1)
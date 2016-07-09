# coding: utf-8
import pickle
from pprint import pprint

# Uses the list composition to make the key value pairs over a dictionary.
dict2list = lambda dic: [ (k, v) for(k, v) in dic.iteritems()]

# Use the built in dictionary constructor to convert the list.
list2dict = lambda lis: dict(lis)

judgenan = lambda item: ( item[0], judge( item[1] ) ) # make NaN as 1 and if actual numbers "0"

fetch_key = lambda item: item[0]
fetch_value = lambda item: item[1]

def judge( value ):
	if value == "NaN":
		return 1
	else:
		return 0


def walk( x, y ):
	for n in range( len( x ) ):
		key = fetch_key( x[n] )
		value = fetch_value( x[n] ) + fetch_value( y[n] ) # why + isnt it key + value?
		x[n] = ( key, value )
	return x

# finding the number of pois for each parameter

 
with open("final_project_dataset.pkl", "r")as data_file:    
    data_dict = pickle.load( data_file )
    data_base = dict2list( data_dict )
    data_list = lambda item: ( dict2list( item[1] ) )# we are taking only the second part (leave out the names)
    add = lambda item: map( judgenan, item  ) # make NaN as 1 and if actual numbers "0"
    calc = lambda x, y: walk( x, y )
    poi_count = list2dict( reduce( calc, map( add, map( data_list, data_base ) ) ) )
    pprint( poi_count )
    
# plot
    
    
import pylab as pl
import numpy as np


X = np.arange(len(poi_count))
pl.bar(X, poi_count.values(), align='center', width=0.5)
pl.xticks(X, poi_count.keys())
locs, labels = pl.xticks()
pl.setp(labels, rotation=90)
ymax = max(poi_count.values()) + 1
pl.xlabel('Features')
pl.ylabel('NaN count')
pl.title('The number of NaN values in each feature in the dataset')
pl.ylim(0, ymax)
pl.show()

# references
# http://stackoverflow.com/questions/10998621/rotate-axis-text-in-python-matplotlib
# http://stackoverflow.com/questions/16892072/histogram-in-pylab-from-a-dictionary
# http://badpopcorn.com/blog/2006/03/16/map-filter-and-reduce-over-python-dictionaries/
    
    





 


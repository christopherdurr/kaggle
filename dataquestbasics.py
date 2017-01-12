
###############################################
#
# LEARNING NumPy - THE BASICS
#
#
###############################################

import numpy

world_alcohol = numpy.genfromtxt("world_alcohol.csv", delimiter = ",")
print(type(world_alcohol))

vector = numpy.array([10, 20, 30])
matrix = numpy.array([[5, 10, 15], [20, 25, 30], [35, 40, 45]])

vector_shape = vector.shape
matrix_shape = matrix.shape

world_alcohol_dtype = world_alcohol.dtype

# READ THE DATA PROPERLY
world_alcohol = numpy.genfromtxt("world_alcohol.csv", delimiter=",", dtype = "U75", skip_header = 1)
print(world_alcohol)

countries = world_alcohol[:,2]
alcohol_consumption = world_alcohol[:,4]

first_two_columns = world_alcohol[:,0:2]
first_ten_years = world_alcohol[0:10,0]
first_ten_rows = world_alcohol[0:10,:]

first_twenty_regions = world_alcohol[:20, 1:3]

###############################################
#
# LEARNING PANDAS - THE BASICS
#
#
###############################################
### CHAPTER 1
import pandas as pd
pd.set_option('display.mpl_style', 'default') # Make the graphs a bit prettier
figsize(15, 5)

fixed_df = pd.read_csv('../data/bikes.csv', sep=';', encoding='latin1', parse_dates=['Date'], dayfirst=True, index_col='Date')
# Look at the first 3 rows
broken_df[:3]

# Plotting a column

fixed_df["Berri 1"].plot()

fixed_df.plot(figsize=(15,10))

### CHAPTER 2

complaints[['Complaint Type', 'Borough']]

complaint_counts = complaints['Complaint Type'].value_counts()

complaint_counts[:10].plot(kind='bar')







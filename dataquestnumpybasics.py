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

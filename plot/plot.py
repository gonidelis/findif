# accept a search.csv file with three columns: mismatch ratio, 
# time of cub_findif, time of thrust_findif, time of thrust_countif
# and plot the three curves in one figure with y axis as time and x axis as mismatch ratio.


import matplotlib.pyplot as plt
import numpy as np
import csv

def read_csv(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    return data

# plot and save the figure in the same directory
# show x ticks and y ticks
def plot_search(data):
    data = np.array(data)
    data = data.astype(float)
    plt.plot(data[:,0], data[:,1], label='cub_findif')
    plt.plot(data[:,0], data[:,2], label='thrust_findif')
    plt.plot(data[:,0], data[:,3], label='thrust_countif')
    plt.xlabel('mismatch ratio')
    plt.ylabel('time')
    plt.legend()
    locs, labels = plt.xticks()  # Get the current locations and labels.
    plt.xticks(np.arange(0, 1.1, step=0.03), rotation='vertical')  # Set label locations.
    plt.grid()
    plt.title('Search operation in A6000')
    plt.savefig('search.png')
    plt.show()
    plt.close()  # Close the figure to free up memory


# plot a new figure in a new file with the same directory
# but only this time the data is named equal.csv and the plot
# equal.png

def plot_equal(data):
    data = np.array(data)
    data = data.astype(float)
    plt.plot(data[:,0], data[:,1], label='cub_findif')
    plt.plot(data[:,0], data[:,2], label='thrust_findif')
    plt.plot(data[:,0], data[:,3], label='thrust_countif')
    plt.xlabel('mismatch ratio')
    plt.ylabel('time')
    plt.legend()
    locs, labels = plt.xticks()  # Get the current locations and labels.
    plt.xticks(np.arange(0, 100, step=3), rotation='vertical')  # Set label locations.
    plt.grid()
    plt.title('Equal operation in A6000')
    plt.savefig('equal.png')
    plt.show()

def main():
    search_data = read_csv('search.csv')
    plot_search(search_data)
    equal_data = read_csv('equal.csv')
    plot_equal(equal_data)

if __name__ == '__main__':
    main()
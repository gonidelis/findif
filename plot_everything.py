import json
import matplotlib.pyplot as plt
import numpy as np

def extract_values(json_file, benchmark_name):
    with open(json_file, 'r') as file:
        data = json.load(file)
        
    mismatch_at_values = []
    gpu_time_values = []
    gpu_time_dict = {}
    
    # Iterate through the benchmarks
    for benchmark in data['benchmarks']:
        if benchmark['name'] == benchmark_name:
            # Iterate through each state
            for state in benchmark['states']:
                mismatch_at = None
                gpu_time = None
                
                # Extract MismatchAt value and convert to percentage
                for axis in state['axis_values']:
                    if axis['name'] == 'MismatchAt':
                        mismatch_at = float(axis['value']) * 100
                
                # Extract GPU Time from summaries
                for summary in state['summaries']:
                    if summary['name'] == 'GPU Time':
                        gpu_time_value = summary['data'][0]['value']
                        gpu_time = float(gpu_time_value)
                
                if mismatch_at is not None and gpu_time is not None:
                    if mismatch_at not in gpu_time_dict:
                        gpu_time_dict[mismatch_at] = []
                    gpu_time_dict[mismatch_at].append(gpu_time)
    
    # Calculate average GPU time for each MismatchAt value
    for mismatch_at, times in gpu_time_dict.items():
        avg_gpu_time = sum(times) / len(times)
        mismatch_at_values.append(mismatch_at)
        gpu_time_values.append(avg_gpu_time)
    
    return mismatch_at_values, gpu_time_values

def plot_results(cub_data, thrust_data, countif_data):
    plt.figure(figsize=(10, 6))
    
    # Plot CUB data if available
    if cub_data[0]:
        plt.plot(cub_data[0], cub_data[1], label='CUB', color='blue')
    
    # Plot Thrust data if available
    if thrust_data[0]:
        plt.plot(thrust_data[0], thrust_data[1], label='Thrust', color='red')
    
    # Plot CountIf data if available
    if countif_data[0]:
        plt.plot(countif_data[0], countif_data[1], label='CountIf', color='green')
    
    plt.title('Elapsed Time vs MismatchAt')
    plt.xlabel('Mismatch in first % of the data')
    plt.ylabel('Elapsed time [s]')
    
    # Determine max mismatch_at for setting x-ticks
    all_mismatch_at = cub_data[0] + thrust_data[0] + countif_data[0]
    if all_mismatch_at:
        max_mismatch_at = max(all_mismatch_at)
        plt.xticks(np.arange(0, max_mismatch_at, 0.315), rotation=90)
    
    plt.grid(True)
    
    # Add legend to distinguish between datasets
    plt.legend()
    
    # Save the plot as a PNG image
    plt.savefig('plot_everything.png')
    
    # Optionally display the plot
    plt.show()

# Use the function with your JSON files
cub_file = 'cub.json'
thrust_file = 'thrust.json'
countif_file = 'countif.json'

cub_data = extract_values(cub_file, 'find_if')
thrust_data = extract_values(thrust_file, 'find_if')
countif_data = extract_values(countif_file, 'thrust::count_if')

plot_results(cub_data, thrust_data, countif_data)

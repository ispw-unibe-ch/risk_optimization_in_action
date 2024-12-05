__author__ = "Martin Widmer, Stephan Zahno"
__email__ = "martin.widmer2@unibe.ch, stephan.zahno@unibe.ch"


# use numpy for stuff that requires numpy even if cupy is available
import os
import numpy
import argparse
import pandas as pd
# use np when it can be substituted by cupy. This will usually be more performant, but in some case it might require back conversion to numpy depending on the _USE_GPU variable, E.G. when creating a plot with matplotlib. Search document for _USE_GPU to see how this is done
# install cuda on computer and try to run pip install cupy-cuda12x in terminal
try:
    import cupy as np
    print("CuPy available, calculating on gpu ...")
    _USE_GPU = True
except ModuleNotFoundError:
    import numpy as np
    print("CuPy not available, falling back to cpu ...")
    _USE_GPU = False
import matplotlib.pyplot as plt
import seaborn as sns
from monte_carlo_utils import get_sim_endpoints, monte_carlo



def main(args):

    inputdata = pd.read_excel(args.inputfile, sheet_name='data_clean', header=0)
    all_data = pd.DataFrame()
    summary_data = pd.DataFrame(columns=['Vpn', 'Penalty_condition', 'Distance_condition', 'optimal_aimpoint_x'])

    subjects = inputdata.groupby('Vpn')

    #example_subject = subjects.get_group('v04')
    #example_subject_cov = example_subject[['Ball_impact_shift_x', 'Ball_impact_y']].cov()
    #example_subject_cov = np.array(example_subject_cov)

    #plot_simulation_distribution(example_subject_cov, example_subject)

    # separating the dataframe into the different conditions
    for subject_key, subject in subjects:
        subject_cov = subject[['Ball_impact_shift_x', 'Ball_impact_y']].cov()
        subject_cov = np.array(subject_cov)

        distance_conditions = subject.groupby('Distance_condition')
        for distance_key , distance_condition in distance_conditions:
            penalty_circle_location_x = 0 - distance_key
            penalty_conditions = distance_condition.groupby('Penalty_condition')
            for penalty_key, penalty_condition in penalty_conditions:
                penalty_condition_cov = penalty_condition[['Ball_impact_shift_x', 'Ball_impact_y']].cov()
                penalty_condition_cov = np.array(penalty_condition_cov)

                # optionally use endpoint variance data from the entire subject for simulation
                if args.use_vp_endpoint_variance:
                    data_cov = subject_cov
                else:
                    data_cov = penalty_condition_cov
                print(f"Processing Subject: {subject_key} | distance: {distance_key} | penalty: {penalty_key}")

                proportion_red, proportion_green, expected_gain, optimal_aimpoint_x = monte_carlo(data_cov=data_cov,
                                                                                                  num_endpoints=args.num_endpoints, 
                                                                                                  gain_penalty=penalty_key,
                                                                                                  penalty_circle_location_x=penalty_circle_location_x)
                #make_plots(proportion_red, proportion_green, expected_gain, optimal_aimpoint_x) # Attention: Will open window with plot for 120 times in a row. Very annoying but great for debugging and seeing what is going on with different conditions. Use CTRL + C in terminal and click on the plot-window to exit this.
                
                # convert gpu arrays into cpu array if we use gpu
                if _USE_GPU:
                    proportion_red = proportion_red.get()
                    proportion_green = proportion_green.get()
                    expected_gain = expected_gain.get()
                    optimal_aimpoint_x = optimal_aimpoint_x.get()

                # append optimal aimpoint to the imported dataframe (instead of adding a new collumn there, we copy the inputdataframe piece by piece and form a new one with the same data. This simplifies the logic significantly)
                temp_df_all_data = penalty_condition.copy()
                temp_df_all_data['optimal_aimpoint_x'] = optimal_aimpoint_x
                all_data = pd.concat([all_data, temp_df_all_data])

                # generate one summary dataframe that can later be exported as xlsx
                summary_row = pd.DataFrame([{
                    'Vpn': subject_key,
                    'Penalty_condition': penalty_key,
                    'Distance_condition': distance_key,
                    'optimal_aimpoint_x': optimal_aimpoint_x  # Assuming this is the optimal aimpoint
                }])
                summary_data = pd.concat([summary_data, summary_row])
                
                # make a dataframe for each condition.
                aimpoints_x = numpy.arange(len(proportion_red))
                condition_df = pd.DataFrame({
                    'aimpoints_x': aimpoints_x,
                    'proportion_red': proportion_red,
                    'proportion_green': proportion_green,
                    'expected_gain': expected_gain
                })                    

                condition_df.to_excel(f"{args.outputfolder}expected_gains_{subject_key}_distance_{distance_key}_{penalty_key * -1}.xlsx", index=False)



    all_data.to_excel(f"{args.outputfolder}{os.path.basename(args.inputfile)}", index=False)
    summary_data.to_excel(f"{args.outputfolder}optimal_aimpoints.xlsx", index=False)




def make_plots(proportion_red, proportion_green, expected_gain, optimal_aimpoint_x):
    if _USE_GPU:
        # converting gpu arrays to cpu array for use in matplotlib
        proportion_red = proportion_red.get()
        proportion_green = proportion_green.get()
        expected_gain = expected_gain.get()
        optimal_aimpoint_x = optimal_aimpoint_x.get()
    

    plt.figure(figsize=(10, 5))

    # Plotting proportions on the primary y-axis
    plt.plot(proportion_red, label='Proportion Red', marker='o', linestyle='-')
    plt.plot(proportion_green, label='Proportion Green', marker='x', linestyle='-')

    # Create a second y-axis for the expected gain
    ax = plt.gca()  # Get current axes
    ax2 = ax.twinx()  # Create another y-axis that shares the same x-axis

    # Plotting expected gain on the secondary y-axis
    ax2.plot(expected_gain, label='Expected Gain', marker='+', linestyle='-', color='tab:red')
    
    # Highlight the maximum expected gain point
    optimal_value = expected_gain[optimal_aimpoint_x]
    ax2.plot(optimal_aimpoint_x, optimal_value, 'ko')  # 'ko' stands for black circle marker

    # Annotating the point
    ax2.annotate(f'Max Gain: {optimal_value:.2f}',
                xy=(optimal_aimpoint_x, optimal_value),
                xytext=(optimal_aimpoint_x, optimal_value + 0.5),  # slightly offset text for clarity
                arrowprops=dict(facecolor='black', shrink=0.05),
                horizontalalignment='center')

    # Setting labels for y-axes
    ax.set_ylabel('Proportion')
    ax2.set_ylabel('Expected Gain', color='tab:red')
    ax.set_xlabel('aimpoint_x')

    # To make the legend handle all lines, include lines from both axes
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.show()

def plot_simulation_distribution(data_cov, actual_data, num_endpoints=100000, radius=30, 
                                 penalty_circle_location_x=-30, target_circle_location_x=0, 
                                 aimpoint_x=0):

    sim_endpoints = get_sim_endpoints(num_endpoints, data_cov, aimpoint_x)

    print(data_cov)
    print(np.cov(sim_endpoints, rowvar=False))

    if _USE_GPU:
        sim_endpoints = sim_endpoints.get()

    # Create the heatmap for the simulated data
    plt.figure(figsize=(10, 5))
    sns.kdeplot(x=sim_endpoints[:, 0], y=sim_endpoints[:, 1], cmap="Blues", fill=True, thresh=0, levels=100, label='Simulated Data')
    
    # Add actual data points
    plt.scatter(actual_data['Ball_impact_shift_x'], actual_data['Ball_impact_y'], color='orange', s=10, alpha=0.5, label='Actual Data')

    # Plot the red circle
    circle_red = plt.Circle((penalty_circle_location_x, 0), radius, color='red', fill=False, linewidth=2, linestyle='--')
    plt.gca().add_patch(circle_red)

    # Plot the green circle
    circle_green = plt.Circle((target_circle_location_x, 0), radius, color='green', fill=False, linewidth=2, linestyle='--')
    plt.gca().add_patch(circle_green)

    # Plot the aimpoint location
    plt.scatter([aimpoint_x], [0], color='black', zorder=5, label='Aimpoint')

    # Set the aspect ratio of the plot to be equal
    plt.gca().set_aspect('equal', adjustable='box')

    # Add titles and labels
    plt.title('Distribution of Simulated and Actual Endpoints')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.legend()

    # Display the plot
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A template script that reads command-line arguments.")

   
    # Optional arguments
    parser.add_argument("-i", "--inputfile", type=str, default='./in/data_exp1.xlsx', help="The input file path (default: in/data_exp1.xlsx)")
    parser.add_argument("-o", "--outputfolder", type=str, default='./out/', help="The output folder path, all outputfiles will be put in this folder (default: out/)")
    parser.add_argument("--use_vp_endpoint_variance", type=bool, default=False, help="Uses the enpoint variance from the entire data of a subject for simulation, instead of just the corresponding block's data (default: False)")
    parser.add_argument("--num_endpoints", type=int, default=1000000, help="Amount of endpoints that should be simulated per aimpoint. (default: 1000000)")

    args = parser.parse_args()
    main(args)
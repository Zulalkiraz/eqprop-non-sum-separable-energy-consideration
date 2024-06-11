"""
Starting point analysis for analog neural networks using Equilibrium Propagation (EqProp)

It implements the training and analysis of an XOR task using an analog neural network with resistors and diodes with the EqProp algorithm to update the conductances of the resistors based on the power computation method (PCM) and traditional voltage drop method (VDM).
"""

"""
- Conducts multiple randomizations to evaluate the robustness and convergence of the learning process for both of the methods.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import datetime
import PySpice.Logging.Logging as Logging
import eqprop_module_xor_pcm_vdm as EPxor
logger = Logging.setup_logging()
from PySpice.Spice.Netlist import Circuit, SubCircuit
from PySpice.Unit import *
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
####################################################################################################

method = "VDM" # options: "PCM" for power computation method and "VDM" for voltage drop method
eps_ratio =0.01 # the ratio of the perturbation on conductance array 
N_Neuron = 2
N_EPOCH = 100   
N_randomize = 100
N_res = 8 # number of resistors

x_in  = ['1','2','3','4','5','6']
h_in  = ['7','8']
h_out = ['9','10']
y     = ['11']


g_limits = dict.fromkeys(['min', 'max'])
g_limits['min']  = 1e-4@u_S
g_limits['max']  = 1e-1@u_S

factors = dict.fromkeys(['beta', 'alpha', 'lambda'])
factors['beta']  = 5e-6
factors['alpha'] = 5e-5
factors['lambda'] = 1

####### XOR TABLE ########
XOR_table                = {'in': np.zeros((4,4)), 'out': np.zeros((4,1))}
XOR_table['in'][:]       = np.array([[0,0,0,0],[0,0,1,1],[1,1,0,0],[1,1,1,1]])
XOR_table['out'][:]      = np.array([[0],[1],[1],[0]])

XOR_voltage_table        = {'in': np.zeros((4,4)), 'out': np.zeros((4,1))}
XOR_voltage_table['in']  = XOR_table['in']*4-2
XOR_voltage_table['out'] = XOR_table['out']

bias_input_voltage       = np.array([1@u_V, 1@u_V])
####################################################################################################

# rng = np.random.default_rng()
np.random.seed(27)
G_array = np.random.uniform(low=g_limits['min'].value,
                high=g_limits['max'].value, size=(N_res+1))
R_array = 1/G_array


G_array[0] = np.nan
R_array[0] = np.nan 
####################################################################################################


input_voltages = np.array([-2@u_V, -2@u_V, -2@u_V, -2@u_V, 1@u_V, 1@u_V])

output_voltage_desired = 0.0@u_V


# Initialize circuits
circuit_free, circuit_nudge, simulator_free_phase, simulator_nudge_phase = \
    EPxor.initialize_circuit(factors,x_in,h_in,h_out,y,R_array,input_voltages,simulator_free='ngspice-subprocess',simulator_nudge='ngspice-subprocess')

all_errors = np.zeros((N_randomize, N_EPOCH))


# Specify the shape of the 2D array (rows x columns)
shape = (N_randomize, N_res+1)  # 15 rows and 11 columns

# Initialize an empty 2D NumPy array with NaN values
R_init_array = np.empty(shape)
R_init_array[:] = np.nan

best_case_index_vdm = [45]
best_case_index_pcm = [45]

best_mask_vdm = np.zeros(all_errors.shape[0], dtype=bool)
best_mask_vdm[best_case_index_vdm] = True

best_mask_pcm = np.zeros(all_errors.shape[0], dtype=bool)
best_mask_pcm[best_case_index_pcm] = True

# Setting up the plot
plt.figure() 
plt.grid()
plt.xlabel('Epoch')
plt.ylabel(r'$10\log_{10}(\mathrm{Error})(dB)$')




####################################################################################################
# Learning phase
####################################################################################################

for nRand in range(N_randomize):
    print("number of random: ",nRand)
    seed_num = nRand + 3*(nRand*nRand) + 5
    print("seed number = ", seed_num) 
    np.random.seed(nRand + 3*(nRand*nRand) + 5)
    G_array = np.random.uniform(low=g_limits['min'].value,
                    high=g_limits['max'].value, size=(N_res+1))
    R_array = 1/G_array

    print("We start with R_array = ", R_array)

    Power_diodes_free = np.zeros(shape=(1))
    Power_diodes_nudge = np.zeros(shape=(1))
    Power_resistors_free = np.zeros(shape=(1))
    Power_resistors_nudge = np.zeros(shape=(1))

    signs = np.ones(len(R_array)-1)
    G_array_signed = G_array

    # Initialize circuits
    circuit_free, circuit_nudge, simulator_free_phase, simulator_nudge_phase = \
        EPxor.initialize_circuit(factors,x_in,h_in,h_out,y,R_array,input_voltages,simulator_free='ngspice-subprocess',simulator_nudge='ngspice-subprocess')
    


####################################################################################################
# Learning phase
####################################################################################################
    Error_array_2D = np.zeros(shape = N_EPOCH)
    
    epoch_array = np.zeros(N_EPOCH)
    for epoch in range(N_EPOCH):
        epoch_array[epoch] = epoch

        Loss_sum = 0 # To calculate the weighted Loss    
        for case in range(4):

            input_voltages = np.hstack((XOR_voltage_table['in'][case,:],bias_input_voltage))
            output_voltage_desired = XOR_voltage_table['out'][case][0]@u_V 

            # reset the circuit and the gains of the bidirectional amplifiers
            circuit_free, circuit_nudge, simulator_free_phase, simulator_nudge_phase = \
            EPxor.initialize_circuit(factors,x_in,h_in,h_out,y,R_array,input_voltages,simulator_free='ngspice-subprocess',simulator_nudge='ngspice-subprocess') 

            # If there is a negative resitance, the input voltage and the gain of the biamps will be minues 
            R_array,G_array,input_voltages,signs = EPxor.sign_abs(R_array,G_array_signed, input_voltages,circuit_free,circuit_nudge, N_Neuron,signs)

            # Two phases: Free and Nudge
            # simulation, feedback currents injection, loss calculation, circuit parameter's updates all done in this function
            try: 
                Loss, dc_free_phase, dc_nudge_phase,circuit_free,circuit_nudge = EPxor.two_phases(circuit_free,
                circuit_nudge,input_voltages,output_voltage_desired,R_array,factors)

                Loss_sum = Loss_sum + Loss
            except NameError as identifier:
                print('Two phase simulation failed at iteration: ',epoch)
                print('Failed R_array=',R_array)
                print('Stop learning and continue with the next randomization realization')
                break

####################################################################################################
            if method == 'VDM': 

                G_array,norm = EPxor.compute_and_update_conductances_xor(dc_free_phase,dc_nudge_phase,x_in,h_in,h_out,y,factors,G_array_signed,signs)
                R_array = 1/G_array
                
                
            else:
            # Compute all passive elements' power dissapations.
                #diodes
                Power_diodes_free = EPxor.power_diss_diodes(dc_free_phase,N_Neuron,h_in)
                Power_diodes_nudge = EPxor.power_diss_diodes(dc_nudge_phase,N_Neuron,h_in)
                #resistors
                Power_resistors_free = EPxor.power_diss_resistors(dc_free_phase,x_in,h_in,h_out,y,G_array)
                Power_resistors_nudge = EPxor.power_diss_resistors(dc_nudge_phase,x_in,h_in,h_out,y,G_array)
                #Dependent Sources
                Power_cccs_vcvs_free = EPxor.power_vcvs_cccs(dc_free_phase,N_Neuron,Gain = 4,h_in = h_in, h_out=h_out)
                Power_cccs_vcvs_nudged = EPxor.power_vcvs_cccs(dc_nudge_phase,N_Neuron,Gain = 4,h_in = h_in, h_out=h_out)


                Power_all_components_free_phase = Power_diodes_free + Power_resistors_free + Power_cccs_vcvs_free
                Power_all_components_nudge_phase = Power_diodes_nudge + Power_resistors_nudge + Power_cccs_vcvs_nudged


    ####################################################################################################

                # Create another G_array for the perturbation

                # Create two dimensional array from G_array 
                #  
                G_array_not_perturbated = np.repeat([G_array], len(R_array)-1, axis=0)

                # Initialize the perturbated array
                G_array_perturbated = np.repeat([G_array], len(R_array)-1, axis=0)
                epsilon_array_for_each_cond = np.zeros(shape=(N_res))

                # Start perturbation
                # add a perturbation to each conductance with an epsilon ratio
                for conductance in range(1,len(R_array)):
                    G_array_perturbated[conductance-1][conductance] += G_array_not_perturbated[conductance-1][conductance]*(eps_ratio)
                    # compute the difference between perturbated array and the G_array
                    epsilon_array_for_each_cond[conductance-1] =  G_array_perturbated[conductance-1][conductance] - G_array_not_perturbated[conductance-1][conductance]
                
                #convert resistors' perturbated array
                R_array_perturbated = 1/G_array_perturbated
    ####################################################################################################

                # initialize all the lists and arrays for power computations and dc analyses

                Loss_array_after_perturbation = np.zeros(shape=len(R_array_perturbated))
                dc_free_phase_list_after_perturbation = []
                dc_nudge_phase_list_after_perturbation = []
                Power_diodes_free_array_after_perturbation = np.zeros(shape=len(R_array_perturbated))
                Power_diodes_nudge_array_after_perturbation = np.zeros(shape=len(R_array_perturbated))
                Power_resistors_free_array_after_perturbation = np.zeros(shape=len(R_array_perturbated))
                Power_resistors_nudge_array_after_perturbation = np.zeros(shape=len(R_array_perturbated))
                Power_cccs_vcvs_free_array_after_perturbation = np.zeros(shape=len(R_array_perturbated))
                Power_cccs_vcvs_nudged_array_after_perturbation = np.zeros(shape=len(R_array_perturbated))

    ####################################################################################################

            # Perform two phases for each perturbated G_array conductance values

                try:
                    for res_arr_num in range(len(R_array_perturbated)):

                        Loss_array_after_perturbation[res_arr_num], dc_free_phase, dc_nudge_phase, circuit_free, circuit_nudge = EPxor.two_phases(circuit_free, circuit_nudge, input_voltages, output_voltage_desired, R_array_perturbated[res_arr_num], factors)
                        dc_free_phase_list_after_perturbation.append(dc_free_phase)
                        dc_nudge_phase_list_after_perturbation.append(dc_nudge_phase)

                except NameError as identifier:
                    print('Two Phase simulation for perturbated arrays failed at iteration: ',epoch)
                    print('Stop learning and continue with the next randomization realization')
                    break

                for num_cond in range(len(G_array_perturbated)):

                    Power_diodes_free_array_after_perturbation[num_cond] = EPxor.power_diss_diodes(dc_free_phase_list_after_perturbation[num_cond],N_Neuron,h_in)
                    Power_diodes_nudge_array_after_perturbation[num_cond] = EPxor.power_diss_diodes(dc_nudge_phase_list_after_perturbation[num_cond],N_Neuron,h_in)
                    Power_resistors_free_array_after_perturbation[num_cond] = EPxor.power_diss_resistors(dc_free_phase_list_after_perturbation[num_cond],x_in,h_in,h_out,y,G_array_perturbated[num_cond])
                    Power_resistors_nudge_array_after_perturbation[num_cond] = EPxor.power_diss_resistors(dc_nudge_phase_list_after_perturbation[num_cond],x_in,h_in,h_out,y,G_array_perturbated[num_cond])
                    Power_cccs_vcvs_free_array_after_perturbation[num_cond] = EPxor.power_vcvs_cccs(dc_free_phase_list_after_perturbation[num_cond],N_Neuron,Gain = 4,h_in = h_in, h_out=h_out)
                    Power_cccs_vcvs_nudged_array_after_perturbation[num_cond] = EPxor.power_vcvs_cccs(dc_nudge_phase_list_after_perturbation[num_cond],N_Neuron,Gain = 4,h_in = h_in, h_out=h_out)
                # sum powers of diodes and resistors power diss.

                Power_all_components_free_phase_after_perturbation = Power_diodes_free_array_after_perturbation + Power_resistors_free_array_after_perturbation + Power_cccs_vcvs_free_array_after_perturbation
                Power_all_components_nudge_phase_after_perturbation = Power_diodes_nudge_array_after_perturbation + Power_resistors_nudge_array_after_perturbation + Power_cccs_vcvs_nudged_array_after_perturbation


                len_R_array_pert = len(R_array_perturbated)

                G_array_signed,norm_p = EPxor.compute_differentiation_of_power_update_conductances(len_R_array=len_R_array_pert,power_free = Power_all_components_free_phase,
                power_free_after_pertub_array= Power_all_components_free_phase_after_perturbation, power_nudge = Power_all_components_nudge_phase, power_nudge_after_pertub_array= Power_all_components_nudge_phase_after_perturbation,epsilon = epsilon_array_for_each_cond, G_array_signed= G_array_signed,factors = factors, signs = signs)
                #Pseudo Power Calculation

        

        Mean_Loss = Loss_sum/4


        print("Loss=",Mean_Loss) 
        Error_array_2D[epoch] = Mean_Loss
        all_errors[nRand,epoch] = Mean_Loss

    if (10*np.log10(Mean_Loss)) <= -9: 
        print("the best case nRand = ", nRand )
        the_best = nRand

    colors = cm.jet(np.linspace(0, 1, N_randomize))
    plt.figure(1)
    plt.plot(epoch_array, 10*np.log10(Error_array_2D))   

print("at last the best", the_best)
av_er = np.mean(all_errors, axis=0)
np.save('av_'+method+'.npy' + 'alpha='+str(factors['alpha'])+ 'beta='+str(factors['beta']), av_er)
if method == 'VDM':
    best_case = all_errors[best_mask_vdm].flatten()
    np.save('best-case_'+method+'.npy'+ 'alpha='+str(factors['alpha'])+ 'beta='+str(factors['beta']), best_case)
else: 
    best_case = all_errors[best_mask_pcm].flatten()
    np.save('best-case_'+method+'.npy'+ 'alpha='+str(factors['alpha'])+ 'beta='+str(factors['beta']), best_case)

now = datetime.datetime.now()
date_str = now.strftime('%Y%m%d_%H%M%S')  # This formats the date and time as YYYYMMDD_HHMMSS

displayed_epochs = np.arange(0, max(epoch_array) + 1, 10)

common_ylim = (-18, 18)
common_ylim2 = (-10, -10)

plt.tight_layout() 
plt.xticks(displayed_epochs)  # Set xlim to the full range of epochs
plt.ylim(common_ylim)
plt.savefig(method+'_Error_rnd_asp_xor_alpha='+str(factors['alpha'])+'beta='+str(factors['beta'])+ '_'+ date_str +'.pdf', bbox_inches='tight')
plt.show()


plt.figure
plt.grid()
plt.xlabel('Epoch')
plt.ylabel(r'$10\log_{10}(\mathrm{Error})(dB)$')
plt.tight_layout(pad=3.0) 
plt.xticks(displayed_epochs)  # Set xlim to the full range of epochs
plt.ylim(common_ylim)
plt.plot(epoch_array, 10*np.log10(av_er), label='All Cases', color='black')  # plot average error evolution in black

plt.plot(epoch_array, 10 * np.log10(best_case), label='Best Case', color='green')

plt.legend(loc='upper left')

plt.savefig(method +'_Avr_Error_asp_XOR_alpha='+str(factors['alpha'])+'beta='+str(factors['beta'])+'_'+date_str+'.pdf')
plt.show()


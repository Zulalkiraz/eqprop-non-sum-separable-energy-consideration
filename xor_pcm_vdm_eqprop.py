"""Main code for analyzing analog neural networks using Equilibrium Propagation (EqProp) with voltage drop method (VDM) and new power computation method (PCM). It implements the training of the XOR task using a ened-to-end nonlinear analog neural network with resistors and diodes. """


import datetime
import numpy as np
import matplotlib.pyplot as plt
import PySpice.Spice.Netlist as netlist
import PySpice.Logging.Logging as Logging
import eqprop_module_xor_pcm_vdm as EPxor
logger = Logging.setup_logging()
from PySpice.Spice.Netlist import Circuit, SubCircuit
from PySpice.Unit import * 


seed_num = 18 
method = 'VDM'
eps_ratio =0.01 # the ratio of the perturbation on conductance array 
N_Neuron = 2
N_EPOCH = 50
x_in  = ['1','2','3','4','5','6']
h_in  = ['7','8']
h_out = ['9','10']
y     = ['11']

N_res = 8 

g_limits = dict.fromkeys(['min', 'max'])
g_limits['min']  = 1e-4@u_S
g_limits['max']  = 1e-1@u_S

factors = dict.fromkeys(['beta', 'alpha', 'lambda'])
factors['beta']  = 1e-6
factors['alpha'] = 5e-3
factors['lambda'] = 0.78426420453 
# rng = np.random.default_rng()
np.random.seed(seed_num)
G_array = np.random.uniform(low=g_limits['min'].value,
                high=g_limits['max'].value, size=(N_res+1))
R_array = 1/G_array


G_array[0] = np.nan
R_array[0] = np.nan  # Do not use the first array element
R_array_init = R_array[1:]
print("R_array_initial",R_array_init)


# initialize a dict with 'in' and 'out' keys
XOR_table                = {'in': np.zeros((4,4)), 'out': np.zeros((4,1))}
XOR_table['in'][:]       = np.array([[0,0,0,0],[0,0,1,1],[1,1,0,0],[1,1,1,1]])
XOR_table['out'][:]      = np.array([[0],[1],[1],[0]])

XOR_voltage_table        = {'in': np.zeros((4,4)), 'out': np.zeros((4,1))}
XOR_voltage_table['in']  = XOR_table['in']*4-2
XOR_voltage_table['out'] = XOR_table['out']

bias_input_voltage       = np.array([1@u_V, 1@u_V])


#voltage values
input_voltages = np.array([-2@u_V, -2@u_V, 2@u_V, 2@u_V, 1@u_V, 1@u_V])

output_voltage_desired = 1.0@u_V




####################################################################################################
# Initialize circuits
####################################################################################################

Error_array, Power_diodes_free, Power_diodes_nudge, Power_resistors_free, Power_resistors_nudge, sum_power_all_free, sum_power_resistors_free  = EPxor.pow_init(N_EPOCH)

# Setup the network and the netlist for free and nudged phase
circuit_free, circuit_nudge, simulator_free_phase, simulator_nudge_phase = \
    EPxor.initialize_circuit(factors,x_in,h_in,h_out,y,R_array,input_voltages,simulator_free='ngspice-subprocess',simulator_nudge='ngspice-subprocess')


####################################################################################################
# Learning phase
####################################################################################################
epoch_array = np.zeros(N_EPOCH)
iteration_array = np.zeros(N_EPOCH*4)
All_Loss_Array = np.zeros(N_EPOCH*4)
iteration = 0
Norm_power_2D = np.zeros(shape = N_EPOCH)
mask1 = np.array([False, False, False, False, False, False]) # Which Input Sources should be minused for the first layer 
mask2 = np.array([False, False]) # For the second layer

signs = np.ones(len(R_array)-1)
G_array_signed = G_array
R_array_evol_2d = np.zeros(shape=(N_EPOCH,len(R_array)))

for epoch in range(N_EPOCH): 

    epoch_array[epoch] = epoch

    Loss_sum = 0 # To calculate the weighted Loss                                 
    sum_norm = 0 
    for case in range(4):


           
        input_voltages = np.hstack((XOR_voltage_table['in'][case,:],bias_input_voltage))
        output_voltage_desired = XOR_voltage_table['out'][case][0]@u_V 


        # Setup the network and the netlist for free and nudged phase
        circuit_free, circuit_nudge, simulator_free_phase, simulator_nudge_phase = \
            EPxor.initialize_circuit(factors,x_in,h_in,h_out,y,R_array,input_voltages,simulator_free='ngspice-subprocess',simulator_nudge='ngspice-subprocess') 

        R_array,G_array,input_voltages,signs = EPxor.sign_abs(R_array,G_array_signed, input_voltages,circuit_free,circuit_nudge, N_Neuron,signs)

        
####################################################################################################
        # Two phases: Free and Nudge
        # simulation, feedback currents injection, loss calculation, circuit parameter's updates all done in this function
        Loss, dc_free_phase, dc_nudge_phase,circuit_free,circuit_nudge = EPxor.two_phases(circuit_free,
        circuit_nudge,input_voltages,output_voltage_desired,R_array,factors)
        
        All_Loss_Array[iteration] = Loss
        iteration_array[iteration] = iteration
        iteration = iteration + 1 
        Loss_sum = Loss_sum + Loss
####################################################################################################
        if method == 'VDM': 
            # # 
            # VDM sign under dev.
            G_array_signed,norm = EPxor.compute_and_update_conductances_xor(dc_free_phase,dc_nudge_phase,x_in,h_in,h_out,y,factors,G_array_signed,signs)
            print("Epoch and G_array_signed = ", epoch, G_array_signed)
            
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

            sum_power_all_free[epoch] = Power_all_components_free_phase
            sum_power_resistors_free[epoch] = Power_resistors_free
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
            for res_arr_num in range(len(R_array_perturbated)):

                Loss_array_after_perturbation[res_arr_num], dc_free_phase, dc_nudge_phase, circuit_free, circuit_nudge = EPxor.two_phases(circuit_free, circuit_nudge, input_voltages, output_voltage_desired, R_array_perturbated[res_arr_num], factors)
                dc_free_phase_list_after_perturbation.append(dc_free_phase)
                dc_nudge_phase_list_after_perturbation.append(dc_nudge_phase)

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

            length = len(Power_all_components_free_phase_after_perturbation)
            # print("len",length)
            len_R_array_pert = len(R_array_perturbated)

            G_array_signed,norm = EPxor.compute_differentiation_of_power_update_conductances(len_R_array=len_R_array_pert,power_free = Power_all_components_free_phase,
            power_free_after_pertub_array= Power_all_components_free_phase_after_perturbation, power_nudge = Power_all_components_nudge_phase, power_nudge_after_pertub_array= Power_all_components_nudge_phase_after_perturbation,epsilon = epsilon_array_for_each_cond, G_array_signed= G_array_signed,factors = factors, signs = signs)
            #Pseudo Power Calculation

            # there is a negative sign in this array (G_array_signed)

        sum_norm = sum_norm + norm

    R_array_evol_2d[epoch] = R_array

    Mean_Norm = sum_norm /4      
    Norm_power_2D[epoch] = Mean_Norm
    print("first norm of the gradient vector = ", Mean_Norm)
    Mean_Loss = Loss_sum/4
    print("Loss=",Mean_Loss) 
    Error_array[epoch] = Mean_Loss


now = datetime.datetime.now()
date_str = now.strftime('%Y%m%d_%H%M%S')  # This formats the date and time as YYYYMMDD_HHMMSS


print("Final R_array = ", R_array)
now = datetime.datetime.now()
date_str = now.strftime('%Y%m%d_%H%M%S')  # This formats the date and time as YYYYMMDD_HHMMSS

#PLOTTING
displayed_epochs = np.arange(0, max(epoch_array) + 1, 10)
plt.plot(epoch_array.astype(int),10*np.log10(Error_array), marker='o')
# plt.plot(epoch_array,Error_array, marker='o')
plt.grid()
plt.xlabel('Epoch')
plt.ylabel('10log10(Error)')
plt.title('Alpha: {}, Beta: {}\nSeed: {}'.format(factors['alpha'], factors['beta'], seed_num))
plt.suptitle('Error Evolution ' + method +' for XOR With Single Sources')
plt.savefig('results/ErrorEvolution_vdm_XOR_SingleS=' + str(factors['alpha']) + '_beta=' + str(factors['beta']) + date_str+ '.pdf', bbox_inches='tight')
plt.xticks(displayed_epochs)
plt.tight_layout()
plt.show()

displayed_epochs = np.arange(0, max(epoch_array) + 1, 10)
plt.plot(epoch_array.astype(int),R_array_evol_2d, marker='o')
# plt.plot(epoch_array,Error_array, marker='o')
plt.grid()
plt.xlabel('Epoch')
plt.ylabel('10log10(Error)')
plt.title('Alpha: {}, Beta: {}\nR_array: {}'.format(factors['alpha'], factors['beta'], R_array_init))
plt.suptitle('Resistance Evolution ' + method +' for XOR With Single Sourcess')
# plt.tight_layout(pad=3.0) 
plt.savefig('results/ErrorEvolution_vdm_XOR_SingleS=' + str(factors['alpha']) + '_beta=' + str(factors['beta']) + date_str+ '.pdf', bbox_inches='tight')
plt.xticks(displayed_epochs)
plt.tight_layout()
plt.show()


plt.plot(epoch_array.astype(int),Norm_power_2D, marker='o')
# plt.plot(epoch_array,Error_array, marker='o')
plt.grid()
plt.xlabel('Epoch')
plt.ylabel('Norm')
plt.title('Alpha: {}, Beta: {}\nR_array: {}'.format(factors['alpha'], factors['beta'], R_array_init))
plt.suptitle('Norms of the Update Vector XOR ' + method)
# plt.tight_layout(pad=3.0) 
plt.savefig('results/Norms_'+method+'_XOR_SiSo=' + str(factors['alpha']) + '_beta=' + str(factors['beta']) + date_str + '.pdf', bbox_inches='tight')
plt.xticks(displayed_epochs)
plt.tight_layout()
plt.show()

plt.plot(iteration_array.astype(int),10*np.log10(All_Loss_Array), marker='o')
# plt.plot(epoch_array,Error_array, marker='o')
plt.grid()
plt.xlabel('4*Epoch')
plt.ylabel('10log10(Error)')
plt.title('Alpha: {}, Beta: {}\nR_array: {}'.format(factors['alpha'], factors['beta'], R_array_init))
plt.suptitle("Error Evolution per Case for XOR Single Sources " + method)
# # plt.tight_layout(pad=3.0) 
plt.savefig('results/ErrorEvol_' + method +'_XOR_SiSo=' + str(factors['alpha']) + '_beta=' + str(factors['beta']) + date_str+ '.pdf', bbox_inches='tight')
plt.xticks(epoch_array*4)
plt.tight_layout()
plt.show()



"""The main functionalities include setting up the circuit, initializing parameters, performing the free and nudge phases, and updating conductances based on the power computation (PCM) and traditional voltage drop methods (VDM). We use the PySpice library to simulate the circuit behavior and evaluate performance on the XOR problem. """

"""Key Functions: - sign_abs: Adjusts input voltage signs and amplifier gains to represent negative weights. 
- two_phases: Executes the free-phase and nudge-phase simulations and computes loss. 
- power_diss_diodes: Calculates power dissipation in diodes. 
- power_vcvs_cccs: Calculates power dissipation in voltage-controlled voltage sources (VCVS) and current-controlled current sources (CCCS). 
- power_diss_resistors: Calculates power dissipation in resistors. 
- setup_circuit: Sets up the initial analog circuit with given parameters. 
- update_circuit_resistors: Updates resistor values in the circuit. 
- compute_differentiation_of_power_update_conductances: (For PCM) Computes gradient and updates conductances based on power dissipation differences. 
- compute_and_update_conductances_xor: (For VDM) Computes and updates conductances specifically for the XOR task. 
- initialize_R_and_G_array: Initializes resistance and conductance arrays based on specified initialization type. 
- initialize_circuit: Initializes the free and nudge phase circuits for simulation."""

"""
Author: Zülal F. Kiraz
2023-2024
"""
import numpy as np
import matplotlib.pyplot as plt
####################################################################################################

import PySpice.Logging.Logging as Logging

from PySpice.Spice.Netlist import Circuit, SubCircuit
from PySpice.Unit import *
logger = Logging.setup_logging(logging_level='WARNING')
####################################################################################################


# initialize a dict with 'in' and 'out' keys
XOR_table                = {'in': np.zeros((4,2)), 'out': np.zeros((4,1))}
XOR_table['in'][:]       = np.array([[0,0],[0,1],[1,0],[1,1]])
XOR_table['out'][:]      = np.array([[0],[1],[1],[0]])

XOR_voltage_table        = {'in': np.zeros((4,2)), 'out': np.zeros((4,1))}
XOR_voltage_table['in']  = XOR_table['in']*4-2
XOR_voltage_table['out'] = XOR_table['out']

bias_input_voltage       = 1.0@u_V


def pow_init(N_Epoch): 
    Error_arr = np.zeros(shape = N_Epoch)

    pow_dio_free = np.zeros(shape=(1))
    pow_dio_nudge = np.zeros(shape=(1))
    pow_res_free = np.zeros(shape=(1))
    pow_res_nudge = np.zeros(shape=(1))
    sum_pow_all_free = np.zeros(shape = N_Epoch)
    sum_pow_res_free = np.zeros(shape = N_Epoch)
    return Error_arr, pow_dio_free, pow_dio_nudge, pow_res_free, pow_res_nudge, sum_pow_all_free, sum_pow_res_free

def sign_abs(R_array,G_array_signed, input_voltages,circuit_free,circuit_nudge, N_Neuron,signs):

# Create slices for the first and second layer
    G_array_signed_1 = G_array_signed[1:7]
    G_array_signed_2 = G_array_signed[7:9]   
    signs1 = np.sign(G_array_signed_1).astype(int)  
    signs2 = np.sign(G_array_signed_2).astype(int)
    signs = np.concatenate((signs1,signs2))


    # For the First Layer
    input_voltages[:6] = signs1*(input_voltages[:6]) 

    # For the Second Layer

    subcircuits_list_free = list(circuit_free.subcircuits)
    subcircuits_list_nudge = list(circuit_nudge.subcircuits)
    # Create a list with a size of numbers of bidirectonal amplifiers

    elements_free = [None] * (N_Neuron) # a list of None's, to be replaced with elements of subcircuits
    elements_nudge = [None] * (N_Neuron)

    for bam in range(1,N_Neuron+1): 
        elements_free[bam-1] = list(subcircuits_list_free[bam].elements)
        elements_nudge[bam-1] = list(subcircuits_list_nudge[bam].elements) # store all bidirectional amplifiers' elements
        #here we have the same number of birectional amplifiers with neuron number. XOR: single output

    elements_free = np.array(elements_free) 
    elements_nudge = np.array(elements_nudge) 
    # print("elements",elements)

    for i in range(len(signs2)):
        # elements[i][0] is always the current source
        # elements[i][1] is always the voltage source
        elements_free[i][1].voltage_gain = signs2[i]*elements_free[i][1].voltage_gain 
        elements_nudge[i][1].voltage_gain = signs2[i]*elements_nudge[i][1].voltage_gain 
        # if it is truly changing my netlist
        ### TO-DO:
        # if you'll update you have to update the circuit nudge too

    # # absolute value
    G_array = abs(G_array_signed)
    R_array = 1 /G_array


    G_array[0] = np.nan
    R_array[0] = np.nan 

    return R_array,G_array,input_voltages,signs

def two_phases(circuit_free,circuit_nudge,input_voltages,output_voltage_desired,R_array,factors):
    # Set the current sources to the feedback currents 
    # Setup input voltages
    circuit_free  = update_circuit_voltages(circuit_free, input_voltages)
    circuit_nudge = update_circuit_voltages(circuit_nudge, input_voltages)

    # Set the current sources to the feedback currents 
    # print("circuit free before update")
    # print(str(circuit_free))
    # print("R_array which should be integrated on the netlist",R_array)
    circuit_free  = update_circuit_resistors(circuit_free, R_array)
    # print("circuit free after update")
    # print(str(circuit_free))

    circuit_nudge = update_circuit_resistors(circuit_nudge, R_array)

    simulator_free_phase = circuit_free.simulator(simulator='ngspice-subprocess',temperature=25 )
    simulator_nudge_phase = circuit_nudge.simulator(simulator='ngspice-subprocess',temperature=25)
    # run the circuit in free phase
    # print("circuit",circuit_free)
    # run the circuit in free phase
    try:
    # while try except break... success or not variable
        dc_free_phase = simulator_free_phase.operating_point()
    except Exception as e:
        print("free circuit is failed")
        print(str(circuit_free))
        raise e

    # Get voltage outputs
    voltage_outputs = np.array(dc_free_phase['11'][0])
    output_voltage_obtained = voltage_outputs.item()

    # compute error
    Loss = (output_voltage_desired.value - output_voltage_obtained.value)**2

    # set feedback currents
    I_p = factors['beta']*(output_voltage_desired.value - output_voltage_obtained.value)


    #### Nudge phase
    # Set the current sources to the feedback currents
    circuit_nudge.I1.dc_value = I_p
   
    # run the circuit in nudge phase

    try:
    # while try except break... success or not variable
        dc_nudge_phase = simulator_nudge_phase.operating_point()
    except Exception as e:
        print("nudge circuit is failed")
        print(str(circuit_nudge))
        raise e
    return  Loss, dc_free_phase, dc_nudge_phase,circuit_free, circuit_nudge

# Define a helper function to get the resistor voltage drops
def get_resistor_voltage_drops(dc_analysis, n_plus_nodes, n_minus_nodes):
    nodes_dict = dc_analysis.nodes
    n_p_nodes_voltages = np.array(list(map(lambda node: nodes_dict.get(node)[0].value, n_plus_nodes)))
    n_m_nodes_voltages = np.array(list(map(lambda node: nodes_dict.get(node)[0].value, n_minus_nodes)))

    voltage_drops = np.subtract.outer(n_p_nodes_voltages,n_m_nodes_voltages)

    return voltage_drops


def power_diss_diodes(dc_analysis,N_Neuron,h_in):

    nodes_dict = dc_analysis.nodes
    branch_dict = dc_analysis.branches

    neuron_currents_array = np. zeros(shape=(N_Neuron,2))
    diode_voltage_drops_array = np. zeros(shape=(N_Neuron,2)) #we have two diodes in each neuron
    shift_voltage = np. zeros(shape=(N_Neuron,2))
    h_in_voltage =  np. zeros(N_Neuron)
    h_in_voltage = np.array(list(map(lambda node: nodes_dict.get(node)[0].value, h_in)))
    for neuron in range(N_Neuron):
        neuron_str = str(neuron+1) 
        
        #get branch currents of shift voltage sources serially connected to the diodes
        neuron_currents_array[neuron][0] = branch_dict.get('v.x'+ neuron_str + '.vshift1')[0].value
        neuron_currents_array[neuron][1]= branch_dict.get('v.x'+ neuron_str + '.vshift2')[0].value
        
        #get shift voltages for each voltage sources serially connected to the diodes

        shift_voltage[neuron][0] = dc_analysis['x'+ neuron_str + '.ndshft1'].as_ndarray()
        shift_voltage[neuron][1] = dc_analysis['x'+ neuron_str + '.ndshft2'].as_ndarray()
        
        #calculate voltages through each diode
        diode_voltage_drops_array[neuron][0] = np.subtract.outer(h_in_voltage[neuron],shift_voltage[neuron][0])
        diode_voltage_drops_array[neuron][1] = np.subtract.outer(h_in_voltage[neuron],shift_voltage[neuron][1])
        #for the fist diodes in each neuron  
    #power calculation
    neuron_currents_array_shaped = neuron_currents_array.flatten('C')
    neuron_currents_array_shaped_transpose = np.transpose(neuron_currents_array_shaped) 
    diode_voltage_drops_shaped = diode_voltage_drops_array.flatten('C')
    diodes_power_sum = (neuron_currents_array_shaped_transpose @ diode_voltage_drops_shaped)

    return diodes_power_sum 

def power_vcvs_cccs(dc_analysis,N_Neuron,Gain,h_in,h_out):

    vcvs_currents = np.zeros(N_Neuron)
    cccs_currents = np.zeros(N_Neuron)
    cccs_voltages = np.zeros(N_Neuron)
    vcvs_voltages = np.zeros(N_Neuron)

    nodes_dict = dc_analysis.nodes
    branch_dict = dc_analysis.branches

    cccs_voltages[0] = np.array(list(map(lambda node: nodes_dict.get(node)[0].value, h_in[0])))

    cccs_voltages[1] = np.array(list(map(lambda node: nodes_dict.get(node)[0].value, h_in[1])))

    vcvs_voltages = np.array(list(map(lambda node: nodes_dict.get(node)[0].value, h_out)))

    vcvs_currents[0] = branch_dict.get('e.x3.e1')[0].value
    vcvs_currents[1] = branch_dict.get('e.x4.e1')[0].value

    cccs_currents =  vcvs_currents * (1/Gain)

    power_vcvs = np.transpose(vcvs_currents) @ vcvs_voltages # matrix product
    power_cccs = np.transpose(cccs_currents) @ cccs_voltages


    power_cccs_vcvs = power_cccs + power_vcvs

    return power_cccs_vcvs
      
def power_diss_resistors(dc_analysis,x_in,h_in,h_out,y,G_array):

    v_x_h_even = get_resistor_voltage_drops(dc_analysis, x_in[::2],h_in[::2] )  
    v_x_h_odd = get_resistor_voltage_drops(dc_analysis, x_in[1::2],h_in[1::2] )
    v_h_y_even = get_resistor_voltage_drops(dc_analysis, h_out[::2],y[::2] ) # only 1 output y 
    v_h_y_odd = get_resistor_voltage_drops(dc_analysis, h_out[1::2],y[::2] )


    v_x_h_even_shaped = v_x_h_even.flatten('C')
    v_x_h_odd_shaped = v_x_h_odd.flatten('C')
    v_h_y_even_shaped = v_h_y_even.flatten('C')
    v_h_y_odd_shaped = v_h_y_odd.flatten('C')

    v_x_y_shaped = np.concatenate((v_x_h_even_shaped, v_x_h_odd_shaped, v_h_y_even_shaped,v_h_y_odd_shaped))

    all_resistors_power_sum = np.transpose(G_array[1:]) @ np.square(v_x_y_shaped)


    return all_resistors_power_sum

# Definition of the subcircuit
class DiodesAmp(SubCircuit):
    """
    A subcircuit that contains the diodes and the amplifier.
    """
    __nodes__ = ('subinput', 'suboutput', 'gnd')

    def __init__(self, name, vshift1=0.3@u_V, vshift2=-0.3@u_V, A_gain=4.0):
        SubCircuit.__init__(self, name, *self.__nodes__)
        self.D(1,          'subinput',    'ndshft1',     model='simple_diode')
        self.V('shift1',   'ndshft1',     'gnd',         vshift1)
        self.D(2,          'ndshft2',     'subinput',    model='simple_diode')
        self.V('shift2',   'ndshft2',     'gnd',         vshift2)
        self.CCCS(1,       'subinput',    'gnd',         'E1',           current_gain=1/A_gain)
        self.VCVS(1,       'suboutput',   'gnd',         'subinput', 'gnd', voltage_gain=A_gain)

class Diodes(SubCircuit):
    """
    A subcircuit that contains the diodes
    """
    __nodes__ = ('subport', 'gnd')

    def __init__(self, name, vshift1=0.3@u_V, vshift2=-0.3@u_V):
        SubCircuit.__init__(self, name, *self.__nodes__)
        self.D(1,          'subport',    'ndshft1',     model='simple_diode')
        self.V('shift1',   'ndshft1',     'gnd',         vshift1)
        self.D(2,          'ndshft2',     'subport',    model='simple_diode')
        self.V('shift2',   'ndshft2',     'gnd',         vshift2)

class BiDirectionalAmp(SubCircuit):
    """
    A subcircuit that contains the diodes
    """
    __nodes__ = ('subinput', 'suboutput', 'gnd')

    def __init__(self, name, A_Vgain=4.0, A_Igain = 1/4):
        SubCircuit.__init__(self, name, *self.__nodes__)
        self.A_Vgain = A_Vgain
        self.A_Igain = A_Igain
        self.CCCS(1,       'subinput',    'gnd',         'E1',           current_gain=A_Igain)
        self.VCVS(1,       'suboutput',   'gnd',         'subinput', 'gnd', voltage_gain=A_Vgain)
    def get_params(self):
        return {'A_Vgain': self.A_Vgain, 'A_Igain': self.A_Igain}
    
def setup_circuit(circuit,x_in , h_in, h_out, y, input_voltages, R_array):
    """
    docstring
    """
    # Model and subcircuit statements
    circuit.model('simple_diode', 'D', Is=1e-6, N=2)
    circuit.subcircuit(Diodes('dio'))


    A_Vgain = 4
    for i in range(7,9):
        circuit.subcircuit(BiDirectionalAmp(f'bam_{i}', A_Vgain)) # to change the sign of the gains sep, we need to create the subcircuits bams with different names


    # Netlist of the circuit
    circuit.V('x1',      x_in[0], circuit.gnd, input_voltages[0])
    circuit.V('x2',      x_in[1], circuit.gnd, input_voltages[1])
    circuit.V('x3',      x_in[2], circuit.gnd, input_voltages[2])
    circuit.V('x4',      x_in[3], circuit.gnd, input_voltages[3])


    circuit.V('x_bias1', x_in[4], circuit.gnd, input_voltages[4])
    circuit.V('x_bias2', x_in[5], circuit.gnd, input_voltages[5])


    circuit.R('1',       x_in[0],  h_in[0],  R_array[1])
    circuit.R('2',       x_in[1],  h_in[1],  R_array[2])
    circuit.R('3',       x_in[2],  h_in[0],  R_array[3])

    circuit.R('4',       x_in[3],  h_in[1],  R_array[4])
    circuit.R('5',       x_in[4],  h_in[0],  R_array[5])
    circuit.R('6',       x_in[5],  h_in[1],  R_array[6])

    circuit.X(1, 'dio'  ,  h_in[0],  circuit.gnd)
    circuit.X(3, 'bam_7',  h_in[0], h_out[0],  circuit.gnd)


    circuit.X(2, 'dio'   ,  h_in[1],  circuit.gnd)
    circuit.X(4, 'bam_8',  h_in[1], h_out[1],  circuit.gnd)


    circuit.R('7',      h_out[0], y[0],     R_array[7])
    circuit.R('8',      h_out[1], y[0],     R_array[8])

    return circuit
def update_circuit_resistors(circuit, R_array):
    """
    docstring
    """
    circuit.R1.resistance = R_array[1]
    circuit.R2.resistance = R_array[2]
    circuit.R3.resistance = R_array[3]
    circuit.R4.resistance = R_array[4]
    circuit.R5.resistance = R_array[5]
    circuit.R6.resistance = R_array[6]
    circuit.R7.resistance = R_array[7]
    circuit.R8.resistance = R_array[8]  
    return circuit


def compute_differentiation_of_power_update_conductances(len_R_array,power_free,power_free_after_pertub_array,power_nudge,power_nudge_after_pertub_array,epsilon,G_array_signed,factors,signs):
    
    differentiation_wrt_conductance_free = np.zeros(shape=(len_R_array))
    differentiation_wrt_conductance_nudge = np.zeros(shape=(len_R_array))
    differentiation_wrt_conductances = np.zeros(shape=(len_R_array))

    #power free is scalar so you don't have to use loop, onmy matrix computation. 
    differentiation_wrt_conductance_free= (power_free-power_free_after_pertub_array)/epsilon
    differentiation_wrt_conductance_nudge = (power_nudge-power_nudge_after_pertub_array)/epsilon 
    differentiation_wrt_conductances = (differentiation_wrt_conductance_nudge - differentiation_wrt_conductance_free)

    update  = (factors['alpha']/factors['beta'])*differentiation_wrt_conductances.astype(float)
    norm_p = np.linalg.norm(update)

    
    G_array_signed[1:] += signs *(factors['alpha']/factors['beta'])*differentiation_wrt_conductances.astype(float)

    
    G_array_signed[0] = np.nan
    

    return G_array_signed, norm_p


def compute_and_update_conductances_xor(dc_free_phase,dc_nudge_phase,x_in,h_in,h_out,y,factors,G_array_signed,signs):
    """
    docstring
    """
    # For VDM 
    # Free Phase Voltage Drops
    v_x_h_even_free = get_resistor_voltage_drops(dc_free_phase, x_in[::2],h_in[::2] )  
    v_x_h_odd_free = get_resistor_voltage_drops(dc_free_phase, x_in[1::2],h_in[1::2] )
    v_h_y_even_free = get_resistor_voltage_drops(dc_free_phase, h_out[::2],y[::2] )  
    v_h_y_odd_free = get_resistor_voltage_drops(dc_free_phase, h_out[1::2],y[::2] )



    v_x_h_even_free_shaped = v_x_h_even_free.flatten('C')
    v_x_h_odd_free_shaped = v_x_h_odd_free.flatten('C')
    v_h_y_even_free_shaped = v_h_y_even_free.flatten('C')
    v_h_y_odd_free_shaped = v_h_y_odd_free.flatten('C')

    v_x_y_free_shaped = np.concatenate((v_x_h_even_free_shaped, v_x_h_odd_free_shaped, v_h_y_even_free_shaped,v_h_y_odd_free_shaped))
    
    # Nudge Phase Voltage Drops

    v_x_h_even_nudge = get_resistor_voltage_drops(dc_nudge_phase, x_in[::2],h_in[::2] )  
    v_x_h_odd_nudge = get_resistor_voltage_drops(dc_nudge_phase, x_in[1::2],h_in[1::2] )
    v_h_y_even_nudge = get_resistor_voltage_drops(dc_nudge_phase, h_out[::2],y[::2] )  
    v_h_y_odd_nudge = get_resistor_voltage_drops(dc_nudge_phase, h_out[1::2],y[::2] )


    v_x_h_even_nudge_shaped = v_x_h_even_nudge.flatten('C')
    v_x_h_odd_nudge_shaped = v_x_h_odd_nudge.flatten('C')
    v_h_y_even_nudge_shaped = v_h_y_even_nudge.flatten('C')
    v_h_y_odd_nudge_shaped = v_h_y_odd_nudge.flatten('C')

    v_x_y_nudge_shaped = np.concatenate((v_x_h_even_nudge_shaped, v_x_h_odd_nudge_shaped, v_h_y_even_nudge_shaped,v_h_y_odd_nudge_shaped))
    

    # compute co-content 
    x_y_cocontent = v_x_y_nudge_shaped**2-v_x_y_free_shaped**2 # [3x2]

    update = ((factors['alpha']*factors['lambda'])/factors['beta'])*x_y_cocontent.flatten('C')
    norm_v = np.linalg.norm(update)

    # compute conductances update and update the resistor array
    G_array_signed[1:] += signs * ((factors['alpha']*factors['lambda'])/factors['beta'])*x_y_cocontent.flatten('C')
    G_array_signed[0] = np.nan

    return G_array_signed, norm_v

def update_circuit_voltages(circuit,input_voltages):
    """
    docstring
    """

    circuit.Vx1.dc_value     = input_voltages[0]
    circuit.Vx2.dc_value     = input_voltages[1]
    circuit.Vx3.dc_value     = input_voltages[2]
    circuit.Vx4.dc_value     = input_voltages[3]
    circuit.Vx_bias1.dc_value = input_voltages[4]
    circuit.Vx_bias2.dc_value = input_voltages[5]
    return circuit

def initialize_R_and_G_array(initialization_type,NRES):
    g_limits = dict.fromkeys(['min', 'max'])
    g_limits['min']  = 1e-4@u_S
    g_limits['max']  = 1e-1@u_S

    """
    docstring
    initialization_type = 'random-fix' or 'random' or 'kendallfig3' or 'ones'
    """
    if initialization_type == 'random-fix':
        # Setup random generator to have reproducible results
        rng = np.random.default_rng(0)
    elif initialization_type == 'random':
        # Setup random generator to have fully random results
        rng = np.random.default_rng()
    else:
        # For other initialization types, we don't need a random generator
        rng = []

    # if initialization_type begins with 'random' (includes 'random' and 'random-fix')
    if initialization_type.startswith('random'):
        # Initialize conductance array with iid samples uniformly distributed between g_limits['min'] and g_limits['max']
        G_array = rng.uniform(low=g_limits['min'].value, 
                        high=g_limits['max'].value, size=(NRES+1))
        G_array[0] = np.nan # Do not use the first array element
    elif initialization_type == 'ones':
        G_array = np.ones(NRES+1)
        G_array[0] = np.nan
    elif 'fig3' in initialization_type:
        R_array = np.array([np.nan,
                            10.13@u_Ω,
                            16.34@u_Ω,
                            9.63@u_Ω,
                            20.22@u_Ω,
                            79279.53@u_Ω,
                            17.14@u_Ω,
                            11.98@u_Ω,
                            177.3@u_Ω])
        G_array = 1/R_array
    else:
        raise ValueError('Invalid initialization type')

    R_array = 1/G_array

    return G_array,R_array

def initialize_circuit(factors, x_in, h_in, h_out, y, R_array, input_voltages,simulator_free='ngspice-shared',simulator_nudge='ngspice-shared'):
    """
    docstring
    """
    # Free phase
    circuit_free = Circuit('Free circuit')

    # Setup the network and the netlist
    circuit_free = setup_circuit(circuit_free, x_in, h_in, h_out, y, input_voltages, R_array)

    simulator_free_phase = circuit_free.simulator(simulator=simulator_free, temperature=25, nominal_temperature=25)

    # Nudge phase
    circuit_nudge = Circuit('Nudge circuit')

    # Setup the network and the netlist
    circuit_nudge = setup_circuit(circuit_nudge, x_in, h_in, h_out, y, input_voltages, R_array)

    # Add feedback current sources
    circuit_nudge.I('1',      y[0],          circuit_nudge.gnd, factors['beta']*1.0@u_uA)
    # circuit_nudge.I('2',      y[1],          circuit_nudge.gnd, factors['beta']*1.0@u_uA)

    simulator_nudge_phase = circuit_nudge.simulator(simulator=simulator_nudge, temperature=25, nominal_temperature=25)

    return circuit_free, circuit_nudge, simulator_free_phase, simulator_nudge_phase
from matplotlib import pyplot as plt
import numpy as np
import SM as sm
import Measurements as me
import funcs_SDP as SDP
import funcs_SDP_Kraus as Kraus
from tqdm import tqdm
import pandas as pd

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------OPTIMIZING SDP FOR DIFFERENT SETS OF MEASUREMENTS-----------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def verts(vert_label, z_weight): #Defining vertices and z component
    if vert_label == 'p':
        return me.vert_custom(me.vert_p, z_weight)
    elif vert_label == 's1':
        return me.vert_custom(me.vert_s1, z_weight)
    elif vert_label == 't1':
        return me.vert_custom(me.vert_t1, z_weight)
    elif vert_label == 's2':
        return me.vert_custom(me.vert_s2, z_weight)
    elif vert_label == 't2':
        return me.vert_custom(me.vert_t2, z_weight)
    elif vert_label == 's3':
        return me.vert_custom(me.vert_s3, z_weight)
    elif vert_label == 't3':
        return me.vert_custom(me.vert_t3, z_weight)

def plot_q_by_z(rho, rho_sep, vert_label, steps , data_label, plot = False): #Creating data to plot optimal noise for local states, as function of z component of the measurements
    z_list = np.linspace(1/(steps+1),1-1/(steps+1),steps)
    SDP_list = []
    Kraus_list = []
    ratio_list = []
    for z in tqdm(z_list):
        q_Kraus = Kraus.SDP_LHS(rho, rho_sep, verts(vert_label, z), plot)[3]
        q_SDP = SDP.SDP_LHS(rho,rho_sep,verts(vert_label,z), plot)[3]
        ratio = q_Kraus/q_SDP
        SDP_list.append(q_SDP)
        Kraus_list.append(q_Kraus)
        ratio_list.append(ratio)

    dict = {'z':z_list, 'SDP':SDP_list, 'KRAUS':Kraus_list, 'Ratio':ratio_list}
    df = pd.DataFrame(dict)
    df.to_csv(data_label)

    max_SDP, max_Kraus, max_ratio = np.max(SDP_list), np.max(Kraus_list), np.max(ratio_list)
    max_z_SDP, max_z_Kraus, max_z_ratio = z_list[np.argmax(SDP_list)], z_list[np.argmax(Kraus_list)], z_list[np.argmax(ratio_list)]
    print(f'MAX SDP: {max_SDP}, z = {max_z_SDP}')
    print(f'MAX Kraus: {max_Kraus}, z = {max_z_Kraus}')
    print(f'MAX ratio: {max_ratio}, z = {max_z_ratio}')
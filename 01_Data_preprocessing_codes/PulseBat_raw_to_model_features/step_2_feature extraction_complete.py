# This code extracts turning point features from the '工步层' or the workstep layer file.
# Complete Case: 5-90% SOC, all kinds of Pulse time or pulse width, U1-U41.
# This version is implemented for completeness.
# 20240809 at TBSI.

import os
import pandas as pd

# Replace according to battery type !!!
cap_mat = '10Ah LMO'
# Replace according to your file path !!!
# Create the folder manually in advance !!!                      # To be created     # To be created                    # 3 Subfolders to be created
source_folder = 'D:/BaiduSyncdisk/实验数据集合/力景数据代码上传/' + 'ProcessingData All/' + 'step_1_extract workstep sheet/' + cap_mat + '/'
# Replace according to your file path !!!
# Create the folder manually in advance !!!                                        # To be created                           # 3 Subfolders to be created
save_folder = 'D:/BaiduSyncdisk/实验数据集合/力景数据代码上传/' + 'ProcessingData All/' + 'step_2_feature extraction_complete/' + cap_mat + '/'

# Must be consistent with the settings in step_3_feature collection.py !!!
soc_to_extract = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90] # Complete Case: 5-90% SOC
pt_to_extract = [0.03, 0.05, 0.07, 0.1, 0.3, 0.5, 0.7, 1, 3, 5] # Completeness Case
U_to_extract = range(1,41 +1) # Completeness Case: U1-U41
    # U1: steady state open cicrcuit voltage (OCV) after 10 mins rest
    # U2-U9: voltage at the beginning and end of 0.5C positive pulse, rest, 0.5C negative pulse and rest.
    # U10-U17: 1C. # U18-U25: 1.5C. # U26-U33: 2C. # U34-U41: 2.5C.

# ONLY files in .xlsx format will be read in.
# Temporary files in .xlsx format will NOT be read in.
xlsx_files = [f for f in os.listdir(source_folder) if f.endswith('.xlsx') and not f.startswith('~$')]
xlsx_file_num = len(xlsx_files)

i = 0

item = ['File_Name','Mat','No.','ID','Qn','Q','SOH','Pt','SOC','SOCR'] + ['U' + str(i) for i in U_to_extract]

for f in xlsx_files:
    # Progress bar
    # This process may be time consuming.
    # Please be patient.
    i = i + 1
    print(str(i) + '/' + str(xlsx_file_num) + ' ' + f + ' processing')

    # Read in
    df = pd.read_excel(source_folder + f).values

    ft = [None] * len(item)
    # Correspond to the item list
    ft[0] = f                                   # File_Name
    ft[1] = f.split('_')[0]                     # Mat
    ft[2] = int(f.split('_')[4])                # No.
    ft[3] = f.split('_')[10].split('.xlsx')[0]  # ID
    ft[4] = int(f.split('_')[2])                # Qn
    Q = - df[3][16]                             
    ft[5] = Q                                   # Q
    ft[6] = Q/ft[4]                             # SOH

    data = []

    # train
    if len(f.split('_')[6].split('-')) == 2:
        # Feature extraction
        for soc_num in soc_to_extract:
            soc_ith = int(soc_num/5 -1)

            ft[8] = soc_num # SOC, state-of-charge.
                # The assumed value in ideal experiment.
                # Actually it is a category here, not a very accurate value.

            for pt_num in pt_to_extract:
                pt_ith = [0.03, 0.05, 0.07, 0.1, 0.3, 0.5, 0.7, 1, 3, 5].index(pt_num)

                ft[7] = pt_num  # Pt, i.e. pulse time or pulse width.

                soc_row_num = 5-1 + (10*5*4+2)*soc_ith + 2 + 5*4*pt_ith
                # 5-1: 5 steps SOH or Capacity measurement by CCCV charge - CC discharge, -1 due to start from 0 in python
                    # rest, CCCV charge, rest, CC discharge, rest
                # 2: 2 steps to condition to assigned SOC
                    # CC charge, rest
                # 10: 10 kinds of pulse time or pulse width
                # 5: 5 kinds of pulse current intensity
                # 4: 4 steps per pulse current intensity per pulse width
                    # CC charge or positive pulse, rest, CC discharge or negative pulse, rest
                ft[9] = sum(df[5:(soc_row_num+1),18]) / ft[4]    # SOCR, state-of-charge in real at U1.
                    # A more accurate value based on accumulated net charged capacity from statistics in the '工步层' or the workstep layer.

                ft[10:len(item)] = [None] * (len(item)-10)

                for U_num in U_to_extract:
                    U_ith = U_to_extract.index(U_num) + 1

                    # Ensure that it will not read more than the actual number of rows or steps
                    U_row_num = 5-1 + (10*5*4+2)*soc_ith + 2 + 5*4*pt_ith + U_num // 2
                        # Due to security concern and voltage protection, some batteries failed to complete all planed pulse tests with different pulse current amplitude at same SOC level and with same pulse time.
                        # If the experiemnt stop at step i corresponding to U(j) and U(j+1) with certain SOC, pulse time and pulse current amplitude:
                        # In this version for completeness:
                            # U(1), U(2), ..., U(j+1) at this SOC, pulse time and pulse current amplitude will be recorded.
                            # U(j+2), ..., U(41) at this SOC, pulse time and pulse current amplitude will be NOT recorded.

                    # Ensure that it will not read more than the actual number of rows or steps
                    if U_row_num < df.shape[0]:
                        ft_ith = 9 + U_ith # ft[0]-ft[9]

                        if U_num == 1:
                            ft[ft_ith] = df[U_row_num][12]  # U1.
                        elif U_num % 2 == 0:
                            ft[ft_ith] = df[U_row_num][10]  # Beginning point: U2, U4, ..., U40.
                        elif U_num % 2 == 1:
                            ft[ft_ith] = df[U_row_num][12]  # End point: U3, U5, ..., U41.
                        
                if ft[10]:
                    data.append(list(ft))

    # test
    elif len(f.split('_')[6].split('-')) == 1:
        # Feature extraction
        soc_num = float(f.split('_')[6].split('-')[0])
        soc_ith  = 0

        ft[8] = soc_num # SOC, state-of-charge.
            # The assumed value in ideal experiment.
            # Actually it is a category here, not a very accurate value.

        for pt_num in pt_to_extract:
            pt_ith = [0.03, 0.05, 0.07, 0.1, 0.3, 0.5, 0.7, 1, 3, 5].index(pt_num)

            ft[7] = pt_num  # Pt, i.e. pulse time or pulse width.

            soc_row_num = 5-1 + (10*5*4+2)*soc_ith + 2 + 5*4*pt_ith
            # 5-1: 5 steps SOH or Capacity measurement by CCCV charge - CC discharge, -1 due to start from 0 in python
                # rest, CCCV charge, rest, CC discharge, rest
            # 2: 2 steps to condition to assigned SOC
                # CC charge, rest
            # 10: 10 kinds of pulse time or pulse width
            # 5: 5 kinds of pulse current intensity
            # 4: 4 steps per pulse current intensity per pulse width
                # CC charge or positive pulse, rest, CC discharge or negative pulse, rest
            ft[9] = sum(df[5:(soc_row_num+1),18]) / ft[4]    # SOCR, state-of-charge in real at U1.
                # A more accurate value based on accumulated net charged capacity from statistics in the '工步层' or the workstep layer.

            ft[10:len(item)] = [None] * (len(item)-10)

            for U_num in U_to_extract:
                U_ith = U_to_extract.index(U_num) + 1

                # Ensure that it will not read more than the actual number of rows or steps
                U_row_num = 5-1 + (10*5*4+2)*soc_ith + 2 + 5*4*pt_ith + U_num // 2
                    # Due to security concern and voltage protection, some batteries failed to complete all planed pulse tests with different pulse current amplitude at same SOC level and with same pulse time.
                    # If the experiemnt stop at step i corresponding to U(j) and U(j+1) with certain SOC, pulse time and pulse current amplitude:
                    # In this version for completeness:
                        # U(1), U(2), ..., U(j+1) at this SOC, pulse time and pulse current amplitude will be recorded.
                        # U(j+2), ..., U(41) at this SOC, pulse time and pulse current amplitude will be NOT recorded.

                # Ensure that it will not read more than the actual number of rows or steps
                if U_row_num < df.shape[0]:
                    ft_ith = 9 + U_ith # ft[0]-ft[9]

                    if U_num == 1:
                        ft[ft_ith] = df[U_row_num][12]  # U1.
                    elif U_num % 2 == 0:
                        ft[ft_ith] = df[U_row_num][10]  # Beginning point: U2, U4, ..., U40.
                    elif U_num % 2 == 1:
                        ft[ft_ith] = df[U_row_num][12]  # End point: U3, U5, ..., U41.
                    
            if ft[10]:
                data.append(list(ft))


    # Save
    save_data = pd.DataFrame(data)
    save_data.to_excel(save_folder + f, index=False, header=item)

# Progress bar
print('Finished.')
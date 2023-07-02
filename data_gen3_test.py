# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 08:03:33 2021

@author: Hwang
"""

import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pickle

def data_parsing(txtfile_path):
    txt_file = open(txtfile_path)
    txt_file_lines = txt_file.readlines()
    all_data = []
    #mz_range = np.arange(0, 2001, 0.2)
    intlist =[]
    mzlist = []
    min_ = 0.0
    for line in txt_file_lines[3:]:
        lines = line[:-1].split('\t')
        if float(lines[1]) < 600:
            if float(lines[1]) - min_ > 0.2:
                all_data.append([mzlist[intlist.index(max(intlist))], max(intlist)])
                min_ = float(lines[1])
                intlist = []
                mzlist = []
            else:
                intlist.append(float(lines[2]))
                mzlist.append(float(lines[1]))

    
    all_data_df = pd.DataFrame(all_data, columns = ['m/z', 'intensity'])
    
    return all_data_df


def search(path):
    res = []
    for root, dirs, files in os.walk(path):
        rootpath = os.path.join(os.path.abspath(path), root)
        
        for file in files:
            filepath = os.path.join(rootpath, file)
            
            res.append(filepath)
        #for ress in res:
        #    print ress
    return res


def figure (data_df, filename):
    fig = plt.figure(figsize=(24,6), dpi=600)
    sns.lineplot(data = data_df, x = 'm/z', y = 'intensity')
    plt.xlabel('m/z')
    plt.ylabel('Intensity')
    plt.title(filename)    
    plt.ylim(0,)
    plt.xlim(0, 601)
    plt.grid(True)
    fig.tight_layout()
#   
    fig.savefig(filename + '.png')    

if __name__ == "__main__":
    nowdir = os.getcwd()
    file_list = search(nowdir)
    
    
    
    #training_AR = []
    #training_Bi3 = []
    
    test_ = []
    
    file_path_dic = {}
    for file_path in file_list:
        if ".txt" in file_path or ".TXT" in file_path:
            print(file_path)
            
            file_path_ = file_path.split('\\')
            folder_name = '_'.join(file_path_[-5:-1])
            try: 
                file_path_dic [folder_name] =  file_path_dic [folder_name] + [file_path]
            except:
                file_path_dic [folder_name] =  [file_path]
    
    set_list = list(file_path_dic.keys())
    
    for set_ in set_list:
        file_path_list = file_path_dic [set_]
        filenames = []
        total_df = pd.DataFrame()
        for file_path in file_path_list:
            
            
            
            
            try:
                file_paths =  file_path.split('\\')
                file_name = file_paths[-1]
                file_names = file_name.split('#')
                file_names_ = file_names[-1].split('.')
                number = int(file_names_[0])
                
                data_df = data_parsing(file_path)
                
                test_.append([data_df, file_path])
                #if number > 6:
                    # if "Depth_Ar cluster only_20210908" in file_path:
                    #     #data_df = data_parsing(file_path)
                    #     test_AR.append([data_df, file_path])
                        
                    # elif "Depth_Bi3 only_20210908" in file_path:
                    #     #data_df = data_parsing(file_path)
                    #     test_Bi3.append([data_df, file_path])
    
    
                    
                # else:
                    
                #if "Depth_Ar cluster only_20210908" in file_path:
                #    #data_df = data_parsing(file_path)
                #    training_AR.append([data_df, file_path])
                    
                #elif "Depth_Bi3 only_20210908" in file_path:
                #    #data_df = data_parsing(file_path)
                #    training_Bi3.append([data_df, file_path])
                
                    #try:
                #data_df = data_parsing(file_path)
                
                file_path_list = file_path.split('\\')
                filename = file_path_list[-1][:-4]
                
                #figure (data_df, file_path[:-4])
                
                
                filenames.append(filename)
                total_df = pd.concat([total_df, data_df ['intensity']], axis = 1)
            except:
                pass
        total_df.columns = filenames           
        
        total_df.to_excel(set_ + '_alldata_intensity.xlsx')
        
        # training_set = []
        # training_Y = []
        # for i in range(0, 11, 1):
        #     for j in range(len(training_AR)):
        #         for k in range(len(training_Bi3)):
                    
        #             ii = i * 0.1
        #             AR = training_AR[j][0]
        #             Bi3 = training_Bi3[k][0]
                    
        #             AR_int = np.array(list(AR.intensity))
        #             Bi3_int = np.array(list(Bi3.intensity))
                    
        #             Rel_AR_int = AR_int / max(AR_int)
        #             Rel_Bi3_int = Bi3_int / max(Bi3_int)
                    
        #             training_set.append(Rel_AR_int * (1 - ii) + Rel_Bi3_int * (ii))
                    
                                 
        #             Y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        #             Y [i] = 1
        #             training_Y.append(Y)
                    
        test_set = []
        test_Y = []
        #for i in range(0, 11, 1):
        #    for j in range(len(test_AR)):
        #        for k in range(len(test_Bi3)):
        total_df2 = pd.DataFrame()
        test_path_list = []
        for i in range(len(test_)):
            data_df = test_[i][0]
            test_path = test_[i][1]
    
            int_list = np.array(list(data_df ['intensity']))
            max_int = max(int_list)
            rel_list = int_list / max_int
            
            test_set.append(rel_list[:2964])
            
            data_df ['intensity'] = rel_list
            
            file_path_list = test_path.split('\\')
            filename = 'REl_' + file_path_list[-1][:-4]
            
            #figure (data_df, filename)
            test_path_list.append(test_path)
            
            #filenames.append(filename)
            
            total_df2 = pd.concat([total_df2, data_df ['intensity']], axis = 1)        
            
                   
        total_df2.columns = test_path_list
        total_df2.to_excel(set_ + '_test_set.xlsx')            
        # for i in range(len(training_set)):
        #     if i % 500 == 0:
        #         data_df ['intensity'] = training_set[i]
                
                
        #         figure (data_df, 'retrain_set' + str(i) + str(training_Y[i]))
        #training_set_file = open('training_set2.dmp', 'wb')
        #training_Y_file = open('training_Y2.dmp', 'wb')
        test_set_file = open(set_+ '_test_set2.dmp', 'wb')
        # test_Y_file = open('test_Y.dmp', 'wb')
        
        
        #pickle.dump(training_set, training_set_file)
        #pickle.dump(training_Y, training_Y_file)
        pickle.dump(test_set, test_set_file)
        # pickle.dump(test_Y, test_Y_file)
                    
                #except:
                #    pass
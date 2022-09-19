#!/bin/bash

#Experiment1 (Table 1)
#alpha=0.4
python3 main.py --bandit_type "FPE_AI" --params_path "AI_K=6_M=3_N=1_datatype=personalizedsynthetic_ninstances=1_alpha=0.400000_delta=0.100000_beta=heuristic"
python3 main.py --bandit_type "PFLUCB_BAI" --params_path "AI_K=6_M=3_N=1_datatype=personalizedsynthetic_ninstances=1_alpha=0.400000_delta=0.100000_beta=heuristic"
#alpha=0.5
python3 main.py --bandit_type "FPE_AI" --params_path "AI_K=6_M=3_N=1_datatype=personalizedsynthetic_ninstances=1_alpha=0.500000_delta=0.100000_beta=heuristic"
python3 main.py --bandit_type "PFLUCB_BAI" --params_path "AI_K=6_M=3_N=1_datatype=personalizedsynthetic_ninstances=1_alpha=0.500000_delta=0.100000_beta=heuristic"
#alpha=0.6
python3 main.py --bandit_type "FPE_AI" --params_path "AI_K=6_M=3_N=1_datatype=personalizedsynthetic_ninstances=1_alpha=0.600000_delta=0.100000_beta=heuristic"
python3 main.py --bandit_type "PFLUCB_BAI" --params_path "AI_K=6_M=3_N=1_datatype=personalizedsynthetic_ninstances=1_alpha=0.600000_delta=0.100000_beta=heuristic"
#alpha=0.7
python3 main.py --bandit_type "FPE_AI" --params_path "AI_K=6_M=3_N=1_datatype=personalizedsynthetic_ninstances=1_alpha=0.700000_delta=0.100000_beta=heuristic"
python3 main.py --bandit_type "PFLUCB_BAI" --params_path "AI_K=6_M=3_N=1_datatype=personalizedsynthetic_ninstances=1_alpha=0.700000_delta=0.100000_beta=heuristic"
#alpha=0.8
python3 main.py --bandit_type "FPE_AI" --params_path "AI_K=6_M=3_N=1_datatype=personalizedsynthetic_ninstances=1_alpha=0.800000_delta=0.100000_beta=heuristic"
python3 main.py --bandit_type "PFLUCB_BAI" --params_path "AI_K=6_M=3_N=1_datatype=personalizedsynthetic_ninstances=1_alpha=0.800000_delta=0.100000_beta=heuristic"

#Experiment2 (Table 2)
#delta=0.1
python3 main.py --bandit_type "FPE_AI" --params_path "AI_K=6_M=3_N=1_datatype=personalizedsynthetic_ninstances=1_alpha=0.500000_delta=0.100000_beta=theoretical"
python3 main.py --bandit_type "ORACLE_AI" --params_path "AI_K=6_M=3_N=1_datatype=personalizedsynthetic_ninstances=1_alpha=0.500000_delta=0.100000_beta=theoretical"
#delta=0.05
python3 main.py --bandit_type "FPE_AI" --params_path "AI_K=6_M=3_N=1_datatype=personalizedsynthetic_ninstances=1_alpha=0.500000_delta=0.050000_beta=theoretical"
python3 main.py --bandit_type "ORACLE_AI" --params_path "AI_K=6_M=3_N=1_datatype=personalizedsynthetic_ninstances=1_alpha=0.500000_delta=0.050000_beta=theoretical"
#delta=0.01
python3 main.py --bandit_type "FPE_AI" --params_path "AI_K=6_M=3_N=1_datatype=personalizedsynthetic_ninstances=1_alpha=0.500000_delta=0.010000_beta=theoretical"
python3 main.py --bandit_type "ORACLE_AI" --params_path "AI_K=6_M=3_N=1_datatype=personalizedsynthetic_ninstances=1_alpha=0.500000_delta=0.010000_beta=theoretical"
#delta=0.001
python3 main.py --bandit_type "FPE_AI" --params_path "AI_K=6_M=3_N=1_datatype=personalizedsynthetic_ninstances=1_alpha=0.500000_delta=0.001000_beta=theoretical"
python3 main.py --bandit_type "ORACLE_AI" --params_path "AI_K=6_M=3_N=1_datatype=personalizedsynthetic_ninstances=1_alpha=0.500000_delta=0.001000_beta=theoretical"
#delta=0.0001
python3 main.py --bandit_type "FPE_AI" --params_path "AI_K=6_M=3_N=1_datatype=personalizedsynthetic_ninstances=1_alpha=0.500000_delta=0.000100_beta=theoretical"
python3 main.py --bandit_type "ORACLE_AI" --params_path "AI_K=6_M=3_N=1_datatype=personalizedsynthetic_ninstances=1_alpha=0.500000_delta=0.000100_beta=theoretical"
#delta=0.00001
python3 main.py --bandit_type "FPE_AI" --params_path "AI_K=6_M=3_N=1_datatype=personalizedsynthetic_ninstances=1_alpha=0.500000_delta=0.000010_beta=theoretical"
python3 main.py --bandit_type "ORACLE_AI" --params_path "AI_K=6_M=3_N=1_datatype=personalizedsynthetic_ninstances=1_alpha=0.500000_delta=0.000010_beta=theoretical"

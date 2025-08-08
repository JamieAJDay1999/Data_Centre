\* Sustain_Flex_Duration *\
Maximize
OBJ: IsActive_1 + IsActive_10 + IsActive_11 + IsActive_12 + IsActive_13
 + IsActive_14 + IsActive_15 + IsActive_16 + IsActive_17 + IsActive_18
 + IsActive_19 + IsActive_2 + IsActive_20 + IsActive_21 + IsActive_22
 + IsActive_23 + IsActive_24 + IsActive_25 + IsActive_26 + IsActive_27
 + IsActive_28 + IsActive_29 + IsActive_3 + IsActive_30 + IsActive_31
 + IsActive_32 + IsActive_33 + IsActive_34 + IsActive_35 + IsActive_36
 + IsActive_37 + IsActive_38 + IsActive_39 + IsActive_4 + IsActive_40
 + IsActive_41 + IsActive_42 + IsActive_43 + IsActive_44 + IsActive_45
 + IsActive_46 + IsActive_47 + IsActive_48 + IsActive_5 + IsActive_6
 + IsActive_7 + IsActive_8 + IsActive_9
Subject To
Continuity_10: IsActive_10 - IsActive_9 <= 0
Continuity_11: - IsActive_10 + IsActive_11 <= 0
Continuity_12: - IsActive_11 + IsActive_12 <= 0
Continuity_13: - IsActive_12 + IsActive_13 <= 0
Continuity_14: - IsActive_13 + IsActive_14 <= 0
Continuity_15: - IsActive_14 + IsActive_15 <= 0
Continuity_16: - IsActive_15 + IsActive_16 <= 0
Continuity_17: - IsActive_16 + IsActive_17 <= 0
Continuity_18: - IsActive_17 + IsActive_18 <= 0
Continuity_19: - IsActive_18 + IsActive_19 <= 0
Continuity_2: - IsActive_1 + IsActive_2 <= 0
Continuity_20: - IsActive_19 + IsActive_20 <= 0
Continuity_21: - IsActive_20 + IsActive_21 <= 0
Continuity_22: - IsActive_21 + IsActive_22 <= 0
Continuity_23: - IsActive_22 + IsActive_23 <= 0
Continuity_24: - IsActive_23 + IsActive_24 <= 0
Continuity_25: - IsActive_24 + IsActive_25 <= 0
Continuity_26: - IsActive_25 + IsActive_26 <= 0
Continuity_27: - IsActive_26 + IsActive_27 <= 0
Continuity_28: - IsActive_27 + IsActive_28 <= 0
Continuity_29: - IsActive_28 + IsActive_29 <= 0
Continuity_3: - IsActive_2 + IsActive_3 <= 0
Continuity_30: - IsActive_29 + IsActive_30 <= 0
Continuity_31: - IsActive_30 + IsActive_31 <= 0
Continuity_32: - IsActive_31 + IsActive_32 <= 0
Continuity_33: - IsActive_32 + IsActive_33 <= 0
Continuity_34: - IsActive_33 + IsActive_34 <= 0
Continuity_35: - IsActive_34 + IsActive_35 <= 0
Continuity_36: - IsActive_35 + IsActive_36 <= 0
Continuity_37: - IsActive_36 + IsActive_37 <= 0
Continuity_38: - IsActive_37 + IsActive_38 <= 0
Continuity_39: - IsActive_38 + IsActive_39 <= 0
Continuity_4: - IsActive_3 + IsActive_4 <= 0
Continuity_40: - IsActive_39 + IsActive_40 <= 0
Continuity_41: - IsActive_40 + IsActive_41 <= 0
Continuity_42: - IsActive_41 + IsActive_42 <= 0
Continuity_43: - IsActive_42 + IsActive_43 <= 0
Continuity_44: - IsActive_43 + IsActive_44 <= 0
Continuity_45: - IsActive_44 + IsActive_45 <= 0
Continuity_46: - IsActive_45 + IsActive_46 <= 0
Continuity_47: - IsActive_46 + IsActive_47 <= 0
Continuity_48: - IsActive_47 + IsActive_48 <= 0
Continuity_5: - IsActive_4 + IsActive_5 <= 0
Continuity_6: - IsActive_5 + IsActive_6 <= 0
Continuity_7: - IsActive_6 + IsActive_7 <= 0
Continuity_8: - IsActive_7 + IsActive_8 <= 0
Continuity_9: - IsActive_8 + IsActive_9 <= 0
_C1: - 0.001 P_Chiller_HVAC_Watts_1 - 0.001 P_Chiller_TES_Watts_1
 + P_Cooling_kW_1 = 0
_C10: P_UPS_Discharge_1 - 100 Z_Discharge_1 >= 0
_C100: P_UPS_Charge_5 - 40 Z_Charge_5 >= 0
_C1000: Z_Charge_44 + Z_Discharge_44 <= 1
_C1001: - 6 P_Chiller_HVAC_Watts_44 + Q_Cool_Watts_44
 - Q_Discharge_TES_Watts_44 = 0
_C1002: - 6 P_Chiller_TES_Watts_44 + Q_Charge_TES_Watts_44 = 0
_C1003: 9.94579541499e-06 Q_Cool_Watts_44 + T_Aisle_In_44 - T_HotAisle_43 = 0
_C1004: - 0.0503355704698 P_IT_Total_44 + 4.48616573826 T_IT_43 + T_IT_44
 - 5.48616573826 T_Rack_43 = 0
_C1005: - 3.84820916835 T_ColdAisle_43 - 5.4436888589 T_IT_43
 + 8.29189802725 T_Rack_43 + T_Rack_44 = 0
_C1006: - 29.7271551724 T_Aisle_In_44 + 30.4570824572 T_ColdAisle_43
 + T_ColdAisle_44 = 38.0584002656
_C1007: 58.4543103448 T_HotAisle_43 + T_HotAisle_44 - 59.4543103448 T_Rack_43
 = 0
_C1008: - E_TES_43 + E_TES_44 - 0.000225 Q_Charge_TES_Watts_44
 + 0.000277777777778 Q_Discharge_TES_Watts_44 = 0
_C1009: - Q_Discharge_TES_Watts_43 + Q_Discharge_TES_Watts_44 <= 1000000
_C101: P_UPS_Discharge_5 - 2700 Z_Discharge_5 <= 0
_C1010: - Q_Charge_TES_Watts_43 + Q_Charge_TES_Watts_44 <= 1000000
_C1011: P_Chiller_HVAC_Watts_44 + P_Chiller_TES_Watts_44 <= 400000
_C1012: - 1000 P_IT_Total_44 + Q_Cool_Watts_44 >= 0
_C1013: - 0.001 P_Chiller_HVAC_Watts_45 - 0.001 P_Chiller_TES_Watts_45
 + P_Cooling_kW_45 = 0
_C1014: - P_Grid_IT_45 + P_IT_Total_45 - P_UPS_Discharge_45 = 0
_C1015: P_Grid_Other_45 - 0.1 P_IT_Total_45 = 0
_C1016: - P_Cooling_kW_45 - P_Grid_IT_45 - P_Grid_Other_45 + P_Grid_Total_45
 - P_UPS_Charge_45 = 0
_C1017: 10000 IsActive_45 + P_Grid_Total_45 <= 10924.33931
_C1018: - E_UPS_44 + E_UPS_45 - 0.205 P_UPS_Charge_45
 + 0.271739130435 P_UPS_Discharge_45 = 0
_C1019: P_UPS_Charge_45 - 270 Z_Charge_45 <= 0
_C102: P_UPS_Discharge_5 - 100 Z_Discharge_5 >= 0
_C1020: P_UPS_Charge_45 - 40 Z_Charge_45 >= 0
_C1021: P_UPS_Discharge_45 - 2700 Z_Discharge_45 <= 0
_C1022: P_UPS_Discharge_45 - 100 Z_Discharge_45 >= 0
_C1023: Z_Charge_45 + Z_Discharge_45 <= 1
_C1024: - 6 P_Chiller_HVAC_Watts_45 + Q_Cool_Watts_45
 - Q_Discharge_TES_Watts_45 = 0
_C1025: - 6 P_Chiller_TES_Watts_45 + Q_Charge_TES_Watts_45 = 0
_C1026: 9.94579541499e-06 Q_Cool_Watts_45 + T_Aisle_In_45 - T_HotAisle_44 = 0
_C1027: - 0.0503355704698 P_IT_Total_45 + 4.48616573826 T_IT_44 + T_IT_45
 - 5.48616573826 T_Rack_44 = 0
_C1028: - 3.84820916835 T_ColdAisle_44 - 5.4436888589 T_IT_44
 + 8.29189802725 T_Rack_44 + T_Rack_45 = 0
_C1029: - 29.7271551724 T_Aisle_In_45 + 30.4570824572 T_ColdAisle_44
 + T_ColdAisle_45 = 38.0584002656
_C103: Z_Charge_5 + Z_Discharge_5 <= 1
_C1030: 58.4543103448 T_HotAisle_44 + T_HotAisle_45 - 59.4543103448 T_Rack_44
 = 0
_C1031: - E_TES_44 + E_TES_45 - 0.000225 Q_Charge_TES_Watts_45
 + 0.000277777777778 Q_Discharge_TES_Watts_45 = 0
_C1032: - Q_Discharge_TES_Watts_44 + Q_Discharge_TES_Watts_45 <= 1000000
_C1033: - Q_Charge_TES_Watts_44 + Q_Charge_TES_Watts_45 <= 1000000
_C1034: P_Chiller_HVAC_Watts_45 + P_Chiller_TES_Watts_45 <= 400000
_C1035: - 1000 P_IT_Total_45 + Q_Cool_Watts_45 >= 0
_C1036: - 0.001 P_Chiller_HVAC_Watts_46 - 0.001 P_Chiller_TES_Watts_46
 + P_Cooling_kW_46 = 0
_C1037: - P_Grid_IT_46 + P_IT_Total_46 - P_UPS_Discharge_46 = 0
_C1038: P_Grid_Other_46 - 0.1 P_IT_Total_46 = 0
_C1039: - P_Cooling_kW_46 - P_Grid_IT_46 - P_Grid_Other_46 + P_Grid_Total_46
 - P_UPS_Charge_46 = 0
_C104: - 6 P_Chiller_HVAC_Watts_5 + Q_Cool_Watts_5 - Q_Discharge_TES_Watts_5
 = 0
_C1040: 10000 IsActive_46 + P_Grid_Total_46 <= 10924.33931
_C1041: - E_UPS_45 + E_UPS_46 - 0.205 P_UPS_Charge_46
 + 0.271739130435 P_UPS_Discharge_46 = 0
_C1042: P_UPS_Charge_46 - 270 Z_Charge_46 <= 0
_C1043: P_UPS_Charge_46 - 40 Z_Charge_46 >= 0
_C1044: P_UPS_Discharge_46 - 2700 Z_Discharge_46 <= 0
_C1045: P_UPS_Discharge_46 - 100 Z_Discharge_46 >= 0
_C1046: Z_Charge_46 + Z_Discharge_46 <= 1
_C1047: - 6 P_Chiller_HVAC_Watts_46 + Q_Cool_Watts_46
 - Q_Discharge_TES_Watts_46 = 0
_C1048: - 6 P_Chiller_TES_Watts_46 + Q_Charge_TES_Watts_46 = 0
_C1049: 9.94579541499e-06 Q_Cool_Watts_46 + T_Aisle_In_46 - T_HotAisle_45 = 0
_C105: - 6 P_Chiller_TES_Watts_5 + Q_Charge_TES_Watts_5 = 0
_C1050: - 0.0503355704698 P_IT_Total_46 + 4.48616573826 T_IT_45 + T_IT_46
 - 5.48616573826 T_Rack_45 = 0
_C1051: - 3.84820916835 T_ColdAisle_45 - 5.4436888589 T_IT_45
 + 8.29189802725 T_Rack_45 + T_Rack_46 = 0
_C1052: - 29.7271551724 T_Aisle_In_46 + 30.4570824572 T_ColdAisle_45
 + T_ColdAisle_46 = 38.0584002656
_C1053: 58.4543103448 T_HotAisle_45 + T_HotAisle_46 - 59.4543103448 T_Rack_45
 = 0
_C1054: - E_TES_45 + E_TES_46 - 0.000225 Q_Charge_TES_Watts_46
 + 0.000277777777778 Q_Discharge_TES_Watts_46 = 0
_C1055: - Q_Discharge_TES_Watts_45 + Q_Discharge_TES_Watts_46 <= 1000000
_C1056: - Q_Charge_TES_Watts_45 + Q_Charge_TES_Watts_46 <= 1000000
_C1057: P_Chiller_HVAC_Watts_46 + P_Chiller_TES_Watts_46 <= 400000
_C1058: - 1000 P_IT_Total_46 + Q_Cool_Watts_46 >= 0
_C1059: - 0.001 P_Chiller_HVAC_Watts_47 - 0.001 P_Chiller_TES_Watts_47
 + P_Cooling_kW_47 = 0
_C106: 9.94579541499e-06 Q_Cool_Watts_5 + T_Aisle_In_5 - T_HotAisle_4 = 0
_C1060: - P_Grid_IT_47 + P_IT_Total_47 - P_UPS_Discharge_47 = 0
_C1061: P_Grid_Other_47 - 0.1 P_IT_Total_47 = 0
_C1062: - P_Cooling_kW_47 - P_Grid_IT_47 - P_Grid_Other_47 + P_Grid_Total_47
 - P_UPS_Charge_47 = 0
_C1063: 10000 IsActive_47 + P_Grid_Total_47 <= 10924.33931
_C1064: - E_UPS_46 + E_UPS_47 - 0.205 P_UPS_Charge_47
 + 0.271739130435 P_UPS_Discharge_47 = 0
_C1065: P_UPS_Charge_47 - 270 Z_Charge_47 <= 0
_C1066: P_UPS_Charge_47 - 40 Z_Charge_47 >= 0
_C1067: P_UPS_Discharge_47 - 2700 Z_Discharge_47 <= 0
_C1068: P_UPS_Discharge_47 - 100 Z_Discharge_47 >= 0
_C1069: Z_Charge_47 + Z_Discharge_47 <= 1
_C107: - 0.0503355704698 P_IT_Total_5 + 4.48616573826 T_IT_4 + T_IT_5
 - 5.48616573826 T_Rack_4 = 0
_C1070: - 6 P_Chiller_HVAC_Watts_47 + Q_Cool_Watts_47
 - Q_Discharge_TES_Watts_47 = 0
_C1071: - 6 P_Chiller_TES_Watts_47 + Q_Charge_TES_Watts_47 = 0
_C1072: 9.94579541499e-06 Q_Cool_Watts_47 + T_Aisle_In_47 - T_HotAisle_46 = 0
_C1073: - 0.0503355704698 P_IT_Total_47 + 4.48616573826 T_IT_46 + T_IT_47
 - 5.48616573826 T_Rack_46 = 0
_C1074: - 3.84820916835 T_ColdAisle_46 - 5.4436888589 T_IT_46
 + 8.29189802725 T_Rack_46 + T_Rack_47 = 0
_C1075: - 29.7271551724 T_Aisle_In_47 + 30.4570824572 T_ColdAisle_46
 + T_ColdAisle_47 = 38.0584002656
_C1076: 58.4543103448 T_HotAisle_46 + T_HotAisle_47 - 59.4543103448 T_Rack_46
 = 0
_C1077: - E_TES_46 + E_TES_47 - 0.000225 Q_Charge_TES_Watts_47
 + 0.000277777777778 Q_Discharge_TES_Watts_47 = 0
_C1078: - Q_Discharge_TES_Watts_46 + Q_Discharge_TES_Watts_47 <= 1000000
_C1079: - Q_Charge_TES_Watts_46 + Q_Charge_TES_Watts_47 <= 1000000
_C108: - 3.84820916835 T_ColdAisle_4 - 5.4436888589 T_IT_4
 + 8.29189802725 T_Rack_4 + T_Rack_5 = 0
_C1080: P_Chiller_HVAC_Watts_47 + P_Chiller_TES_Watts_47 <= 400000
_C1081: - 1000 P_IT_Total_47 + Q_Cool_Watts_47 >= 0
_C1082: - 0.001 P_Chiller_HVAC_Watts_48 - 0.001 P_Chiller_TES_Watts_48
 + P_Cooling_kW_48 = 0
_C1083: - P_Grid_IT_48 + P_IT_Total_48 - P_UPS_Discharge_48 = 0
_C1084: P_Grid_Other_48 - 0.1 P_IT_Total_48 = 0
_C1085: - P_Cooling_kW_48 - P_Grid_IT_48 - P_Grid_Other_48 + P_Grid_Total_48
 - P_UPS_Charge_48 = 0
_C1086: 10000 IsActive_48 + P_Grid_Total_48 <= 10924.33931
_C1087: - E_UPS_47 + E_UPS_48 - 0.205 P_UPS_Charge_48
 + 0.271739130435 P_UPS_Discharge_48 = 0
_C1088: P_UPS_Charge_48 - 270 Z_Charge_48 <= 0
_C1089: P_UPS_Charge_48 - 40 Z_Charge_48 >= 0
_C109: - 29.7271551724 T_Aisle_In_5 + 30.4570824572 T_ColdAisle_4
 + T_ColdAisle_5 = 38.0584002656
_C1090: P_UPS_Discharge_48 - 2700 Z_Discharge_48 <= 0
_C1091: P_UPS_Discharge_48 - 100 Z_Discharge_48 >= 0
_C1092: Z_Charge_48 + Z_Discharge_48 <= 1
_C1093: - 6 P_Chiller_HVAC_Watts_48 + Q_Cool_Watts_48
 - Q_Discharge_TES_Watts_48 = 0
_C1094: - 6 P_Chiller_TES_Watts_48 + Q_Charge_TES_Watts_48 = 0
_C1095: 9.94579541499e-06 Q_Cool_Watts_48 + T_Aisle_In_48 - T_HotAisle_47 = 0
_C1096: - 0.0503355704698 P_IT_Total_48 + 4.48616573826 T_IT_47 + T_IT_48
 - 5.48616573826 T_Rack_47 = 0
_C1097: - 3.84820916835 T_ColdAisle_47 - 5.4436888589 T_IT_47
 + 8.29189802725 T_Rack_47 + T_Rack_48 = 0
_C1098: - 29.7271551724 T_Aisle_In_48 + 30.4570824572 T_ColdAisle_47
 + T_ColdAisle_48 = 38.0584002656
_C1099: 58.4543103448 T_HotAisle_47 + T_HotAisle_48 - 59.4543103448 T_Rack_47
 = 0
_C11: Z_Charge_1 + Z_Discharge_1 <= 1
_C110: 58.4543103448 T_HotAisle_4 + T_HotAisle_5 - 59.4543103448 T_Rack_4 = 0
_C1100: - E_TES_47 + E_TES_48 - 0.000225 Q_Charge_TES_Watts_48
 + 0.000277777777778 Q_Discharge_TES_Watts_48 = 0
_C1101: - Q_Discharge_TES_Watts_47 + Q_Discharge_TES_Watts_48 <= 1000000
_C1102: - Q_Charge_TES_Watts_47 + Q_Charge_TES_Watts_48 <= 1000000
_C1103: P_Chiller_HVAC_Watts_48 + P_Chiller_TES_Watts_48 <= 400000
_C1104: - 1000 P_IT_Total_48 + Q_Cool_Watts_48 >= 0
_C111: - E_TES_4 + E_TES_5 - 0.000225 Q_Charge_TES_Watts_5
 + 0.000277777777778 Q_Discharge_TES_Watts_5 = 0
_C112: - Q_Discharge_TES_Watts_4 + Q_Discharge_TES_Watts_5 <= 1000000
_C113: - Q_Charge_TES_Watts_4 + Q_Charge_TES_Watts_5 <= 1000000
_C114: P_Chiller_HVAC_Watts_5 + P_Chiller_TES_Watts_5 <= 400000
_C115: - 1000 P_IT_Total_5 + Q_Cool_Watts_5 >= 0
_C116: - 0.001 P_Chiller_HVAC_Watts_6 - 0.001 P_Chiller_TES_Watts_6
 + P_Cooling_kW_6 = 0
_C117: - P_Grid_IT_6 + P_IT_Total_6 - P_UPS_Discharge_6 = 0
_C118: P_Grid_Other_6 - 0.1 P_IT_Total_6 = 0
_C119: - P_Cooling_kW_6 - P_Grid_IT_6 - P_Grid_Other_6 + P_Grid_Total_6
 - P_UPS_Charge_6 = 0
_C12: - 6 P_Chiller_HVAC_Watts_1 + Q_Cool_Watts_1 - Q_Discharge_TES_Watts_1
 = 0
_C120: 10000 IsActive_6 + P_Grid_Total_6 <= 10924.33931
_C121: - E_UPS_5 + E_UPS_6 - 0.205 P_UPS_Charge_6
 + 0.271739130435 P_UPS_Discharge_6 = 0
_C122: P_UPS_Charge_6 - 270 Z_Charge_6 <= 0
_C123: P_UPS_Charge_6 - 40 Z_Charge_6 >= 0
_C124: P_UPS_Discharge_6 - 2700 Z_Discharge_6 <= 0
_C125: P_UPS_Discharge_6 - 100 Z_Discharge_6 >= 0
_C126: Z_Charge_6 + Z_Discharge_6 <= 1
_C127: - 6 P_Chiller_HVAC_Watts_6 + Q_Cool_Watts_6 - Q_Discharge_TES_Watts_6
 = 0
_C128: - 6 P_Chiller_TES_Watts_6 + Q_Charge_TES_Watts_6 = 0
_C129: 9.94579541499e-06 Q_Cool_Watts_6 + T_Aisle_In_6 - T_HotAisle_5 = 0
_C13: - 6 P_Chiller_TES_Watts_1 + Q_Charge_TES_Watts_1 = 0
_C130: - 0.0503355704698 P_IT_Total_6 + 4.48616573826 T_IT_5 + T_IT_6
 - 5.48616573826 T_Rack_5 = 0
_C131: - 3.84820916835 T_ColdAisle_5 - 5.4436888589 T_IT_5
 + 8.29189802725 T_Rack_5 + T_Rack_6 = 0
_C132: - 29.7271551724 T_Aisle_In_6 + 30.4570824572 T_ColdAisle_5
 + T_ColdAisle_6 = 38.0584002656
_C133: 58.4543103448 T_HotAisle_5 + T_HotAisle_6 - 59.4543103448 T_Rack_5 = 0
_C134: - E_TES_5 + E_TES_6 - 0.000225 Q_Charge_TES_Watts_6
 + 0.000277777777778 Q_Discharge_TES_Watts_6 = 0
_C135: - Q_Discharge_TES_Watts_5 + Q_Discharge_TES_Watts_6 <= 1000000
_C136: - Q_Charge_TES_Watts_5 + Q_Charge_TES_Watts_6 <= 1000000
_C137: P_Chiller_HVAC_Watts_6 + P_Chiller_TES_Watts_6 <= 400000
_C138: - 1000 P_IT_Total_6 + Q_Cool_Watts_6 >= 0
_C139: - 0.001 P_Chiller_HVAC_Watts_7 - 0.001 P_Chiller_TES_Watts_7
 + P_Cooling_kW_7 = 0
_C14: 9.94579541499e-06 Q_Cool_Watts_1 + T_Aisle_In_1 = 32.807087
_C140: - P_Grid_IT_7 + P_IT_Total_7 - P_UPS_Discharge_7 = 0
_C141: P_Grid_Other_7 - 0.1 P_IT_Total_7 = 0
_C142: - P_Cooling_kW_7 - P_Grid_IT_7 - P_Grid_Other_7 + P_Grid_Total_7
 - P_UPS_Charge_7 = 0
_C143: 10000 IsActive_7 + P_Grid_Total_7 <= 10924.33931
_C144: - E_UPS_6 + E_UPS_7 - 0.205 P_UPS_Charge_7
 + 0.271739130435 P_UPS_Discharge_7 = 0
_C145: P_UPS_Charge_7 - 270 Z_Charge_7 <= 0
_C146: P_UPS_Charge_7 - 40 Z_Charge_7 >= 0
_C147: P_UPS_Discharge_7 - 2700 Z_Discharge_7 <= 0
_C148: P_UPS_Discharge_7 - 100 Z_Discharge_7 >= 0
_C149: Z_Charge_7 + Z_Discharge_7 <= 1
_C15: - 0.0503355704698 P_IT_Total_1 + T_IT_1 = 0.818376698609
_C150: - 6 P_Chiller_HVAC_Watts_7 + Q_Cool_Watts_7 - Q_Discharge_TES_Watts_7
 = 0
_C151: - 6 P_Chiller_TES_Watts_7 + Q_Charge_TES_Watts_7 = 0
_C152: 9.94579541499e-06 Q_Cool_Watts_7 + T_Aisle_In_7 - T_HotAisle_6 = 0
_C153: - 0.0503355704698 P_IT_Total_7 + 4.48616573826 T_IT_6 + T_IT_7
 - 5.48616573826 T_Rack_6 = 0
_C154: - 3.84820916835 T_ColdAisle_6 - 5.4436888589 T_IT_6
 + 8.29189802725 T_Rack_6 + T_Rack_7 = 0
_C155: - 29.7271551724 T_Aisle_In_7 + 30.4570824572 T_ColdAisle_6
 + T_ColdAisle_7 = 38.0584002656
_C156: 58.4543103448 T_HotAisle_6 + T_HotAisle_7 - 59.4543103448 T_Rack_6 = 0
_C157: - E_TES_6 + E_TES_7 - 0.000225 Q_Charge_TES_Watts_7
 + 0.000277777777778 Q_Discharge_TES_Watts_7 = 0
_C158: - Q_Discharge_TES_Watts_6 + Q_Discharge_TES_Watts_7 <= 1000000
_C159: - Q_Charge_TES_Watts_6 + Q_Charge_TES_Watts_7 <= 1000000
_C16: T_Rack_1 = 30.111489634
_C160: P_Chiller_HVAC_Watts_7 + P_Chiller_TES_Watts_7 <= 400000
_C161: - 1000 P_IT_Total_7 + Q_Cool_Watts_7 >= 0
_C162: - 0.001 P_Chiller_HVAC_Watts_8 - 0.001 P_Chiller_TES_Watts_8
 + P_Cooling_kW_8 = 0
_C163: - P_Grid_IT_8 + P_IT_Total_8 - P_UPS_Discharge_8 = 0
_C164: P_Grid_Other_8 - 0.1 P_IT_Total_8 = 0
_C165: - P_Cooling_kW_8 - P_Grid_IT_8 - P_Grid_Other_8 + P_Grid_Total_8
 - P_UPS_Charge_8 = 0
_C166: 10000 IsActive_8 + P_Grid_Total_8 <= 10924.33931
_C167: - E_UPS_7 + E_UPS_8 - 0.205 P_UPS_Charge_8
 + 0.271739130435 P_UPS_Discharge_8 = 0
_C168: P_UPS_Charge_8 - 270 Z_Charge_8 <= 0
_C169: P_UPS_Charge_8 - 40 Z_Charge_8 >= 0
_C17: - 29.7271551724 T_Aisle_In_1 + T_ColdAisle_1 = -631.997413793
_C170: P_UPS_Discharge_8 - 2700 Z_Discharge_8 <= 0
_C171: P_UPS_Discharge_8 - 100 Z_Discharge_8 >= 0
_C172: Z_Charge_8 + Z_Discharge_8 <= 1
_C173: - 6 P_Chiller_HVAC_Watts_8 + Q_Cool_Watts_8 - Q_Discharge_TES_Watts_8
 = 0
_C174: - 6 P_Chiller_TES_Watts_8 + Q_Charge_TES_Watts_8 = 0
_C175: 9.94579541499e-06 Q_Cool_Watts_8 + T_Aisle_In_8 - T_HotAisle_7 = 0
_C176: - 0.0503355704698 P_IT_Total_8 + 4.48616573826 T_IT_7 + T_IT_8
 - 5.48616573826 T_Rack_7 = 0
_C177: - 3.84820916835 T_ColdAisle_7 - 5.4436888589 T_IT_7
 + 8.29189802725 T_Rack_7 + T_Rack_8 = 0
_C178: - 29.7271551724 T_Aisle_In_8 + 30.4570824572 T_ColdAisle_7
 + T_ColdAisle_8 = 38.0584002656
_C179: 58.4543103448 T_HotAisle_7 + T_HotAisle_8 - 59.4543103448 T_Rack_7 = 0
_C18: T_HotAisle_1 = 30.0437101095
_C180: - E_TES_7 + E_TES_8 - 0.000225 Q_Charge_TES_Watts_8
 + 0.000277777777778 Q_Discharge_TES_Watts_8 = 0
_C181: - Q_Discharge_TES_Watts_7 + Q_Discharge_TES_Watts_8 <= 1000000
_C182: - Q_Charge_TES_Watts_7 + Q_Charge_TES_Watts_8 <= 1000000
_C183: P_Chiller_HVAC_Watts_8 + P_Chiller_TES_Watts_8 <= 400000
_C184: - 1000 P_IT_Total_8 + Q_Cool_Watts_8 >= 0
_C185: - 0.001 P_Chiller_HVAC_Watts_9 - 0.001 P_Chiller_TES_Watts_9
 + P_Cooling_kW_9 = 0
_C186: - P_Grid_IT_9 + P_IT_Total_9 - P_UPS_Discharge_9 = 0
_C187: P_Grid_Other_9 - 0.1 P_IT_Total_9 = 0
_C188: - P_Cooling_kW_9 - P_Grid_IT_9 - P_Grid_Other_9 + P_Grid_Total_9
 - P_UPS_Charge_9 = 0
_C189: 10000 IsActive_9 + P_Grid_Total_9 <= 10924.33931
_C19: E_TES_1 - 0.000225 Q_Charge_TES_Watts_1
 + 0.000277777777778 Q_Discharge_TES_Watts_1 = 0
_C190: - E_UPS_8 + E_UPS_9 - 0.205 P_UPS_Charge_9
 + 0.271739130435 P_UPS_Discharge_9 = 0
_C191: P_UPS_Charge_9 - 270 Z_Charge_9 <= 0
_C192: P_UPS_Charge_9 - 40 Z_Charge_9 >= 0
_C193: P_UPS_Discharge_9 - 2700 Z_Discharge_9 <= 0
_C194: P_UPS_Discharge_9 - 100 Z_Discharge_9 >= 0
_C195: Z_Charge_9 + Z_Discharge_9 <= 1
_C196: - 6 P_Chiller_HVAC_Watts_9 + Q_Cool_Watts_9 - Q_Discharge_TES_Watts_9
 = 0
_C197: - 6 P_Chiller_TES_Watts_9 + Q_Charge_TES_Watts_9 = 0
_C198: 9.94579541499e-06 Q_Cool_Watts_9 + T_Aisle_In_9 - T_HotAisle_8 = 0
_C199: - 0.0503355704698 P_IT_Total_9 + 4.48616573826 T_IT_8 + T_IT_9
 - 5.48616573826 T_Rack_8 = 0
_C2: - P_Grid_IT_1 + P_IT_Total_1 - P_UPS_Discharge_1 = 0
_C20: Q_Discharge_TES_Watts_1 <= 1000000
_C200: - 3.84820916835 T_ColdAisle_8 - 5.4436888589 T_IT_8
 + 8.29189802725 T_Rack_8 + T_Rack_9 = 0
_C201: - 29.7271551724 T_Aisle_In_9 + 30.4570824572 T_ColdAisle_8
 + T_ColdAisle_9 = 38.0584002656
_C202: 58.4543103448 T_HotAisle_8 + T_HotAisle_9 - 59.4543103448 T_Rack_8 = 0
_C203: - E_TES_8 + E_TES_9 - 0.000225 Q_Charge_TES_Watts_9
 + 0.000277777777778 Q_Discharge_TES_Watts_9 = 0
_C204: - Q_Discharge_TES_Watts_8 + Q_Discharge_TES_Watts_9 <= 1000000
_C205: - Q_Charge_TES_Watts_8 + Q_Charge_TES_Watts_9 <= 1000000
_C206: P_Chiller_HVAC_Watts_9 + P_Chiller_TES_Watts_9 <= 400000
_C207: - 1000 P_IT_Total_9 + Q_Cool_Watts_9 >= 0
_C208: - 0.001 P_Chiller_HVAC_Watts_10 - 0.001 P_Chiller_TES_Watts_10
 + P_Cooling_kW_10 = 0
_C209: - P_Grid_IT_10 + P_IT_Total_10 - P_UPS_Discharge_10 = 0
_C21: Q_Charge_TES_Watts_1 <= 1000000
_C210: P_Grid_Other_10 - 0.1 P_IT_Total_10 = 0
_C211: - P_Cooling_kW_10 - P_Grid_IT_10 - P_Grid_Other_10 + P_Grid_Total_10
 - P_UPS_Charge_10 = 0
_C212: 10000 IsActive_10 + P_Grid_Total_10 <= 10924.33931
_C213: E_UPS_10 - E_UPS_9 - 0.205 P_UPS_Charge_10
 + 0.271739130435 P_UPS_Discharge_10 = 0
_C214: P_UPS_Charge_10 - 270 Z_Charge_10 <= 0
_C215: P_UPS_Charge_10 - 40 Z_Charge_10 >= 0
_C216: P_UPS_Discharge_10 - 2700 Z_Discharge_10 <= 0
_C217: P_UPS_Discharge_10 - 100 Z_Discharge_10 >= 0
_C218: Z_Charge_10 + Z_Discharge_10 <= 1
_C219: - 6 P_Chiller_HVAC_Watts_10 + Q_Cool_Watts_10
 - Q_Discharge_TES_Watts_10 = 0
_C22: P_Chiller_HVAC_Watts_1 + P_Chiller_TES_Watts_1 <= 400000
_C220: - 6 P_Chiller_TES_Watts_10 + Q_Charge_TES_Watts_10 = 0
_C221: 9.94579541499e-06 Q_Cool_Watts_10 + T_Aisle_In_10 - T_HotAisle_9 = 0
_C222: - 0.0503355704698 P_IT_Total_10 + T_IT_10 + 4.48616573826 T_IT_9
 - 5.48616573826 T_Rack_9 = 0
_C223: - 3.84820916835 T_ColdAisle_9 - 5.4436888589 T_IT_9 + T_Rack_10
 + 8.29189802725 T_Rack_9 = 0
_C224: - 29.7271551724 T_Aisle_In_10 + T_ColdAisle_10
 + 30.4570824572 T_ColdAisle_9 = 38.0584002656
_C225: T_HotAisle_10 + 58.4543103448 T_HotAisle_9 - 59.4543103448 T_Rack_9 = 0
_C226: E_TES_10 - E_TES_9 - 0.000225 Q_Charge_TES_Watts_10
 + 0.000277777777778 Q_Discharge_TES_Watts_10 = 0
_C227: Q_Discharge_TES_Watts_10 - Q_Discharge_TES_Watts_9 <= 1000000
_C228: Q_Charge_TES_Watts_10 - Q_Charge_TES_Watts_9 <= 1000000
_C229: P_Chiller_HVAC_Watts_10 + P_Chiller_TES_Watts_10 <= 400000
_C23: - 1000 P_IT_Total_1 + Q_Cool_Watts_1 >= 0
_C230: - 1000 P_IT_Total_10 + Q_Cool_Watts_10 >= 0
_C231: - 0.001 P_Chiller_HVAC_Watts_11 - 0.001 P_Chiller_TES_Watts_11
 + P_Cooling_kW_11 = 0
_C232: - P_Grid_IT_11 + P_IT_Total_11 - P_UPS_Discharge_11 = 0
_C233: P_Grid_Other_11 - 0.1 P_IT_Total_11 = 0
_C234: - P_Cooling_kW_11 - P_Grid_IT_11 - P_Grid_Other_11 + P_Grid_Total_11
 - P_UPS_Charge_11 = 0
_C235: 10000 IsActive_11 + P_Grid_Total_11 <= 10924.33931
_C236: - E_UPS_10 + E_UPS_11 - 0.205 P_UPS_Charge_11
 + 0.271739130435 P_UPS_Discharge_11 = 0
_C237: P_UPS_Charge_11 - 270 Z_Charge_11 <= 0
_C238: P_UPS_Charge_11 - 40 Z_Charge_11 >= 0
_C239: P_UPS_Discharge_11 - 2700 Z_Discharge_11 <= 0
_C24: - 0.001 P_Chiller_HVAC_Watts_2 - 0.001 P_Chiller_TES_Watts_2
 + P_Cooling_kW_2 = 0
_C240: P_UPS_Discharge_11 - 100 Z_Discharge_11 >= 0
_C241: Z_Charge_11 + Z_Discharge_11 <= 1
_C242: - 6 P_Chiller_HVAC_Watts_11 + Q_Cool_Watts_11
 - Q_Discharge_TES_Watts_11 = 0
_C243: - 6 P_Chiller_TES_Watts_11 + Q_Charge_TES_Watts_11 = 0
_C244: 9.94579541499e-06 Q_Cool_Watts_11 + T_Aisle_In_11 - T_HotAisle_10 = 0
_C245: - 0.0503355704698 P_IT_Total_11 + 4.48616573826 T_IT_10 + T_IT_11
 - 5.48616573826 T_Rack_10 = 0
_C246: - 3.84820916835 T_ColdAisle_10 - 5.4436888589 T_IT_10
 + 8.29189802725 T_Rack_10 + T_Rack_11 = 0
_C247: - 29.7271551724 T_Aisle_In_11 + 30.4570824572 T_ColdAisle_10
 + T_ColdAisle_11 = 38.0584002656
_C248: 58.4543103448 T_HotAisle_10 + T_HotAisle_11 - 59.4543103448 T_Rack_10
 = 0
_C249: - E_TES_10 + E_TES_11 - 0.000225 Q_Charge_TES_Watts_11
 + 0.000277777777778 Q_Discharge_TES_Watts_11 = 0
_C25: - P_Grid_IT_2 + P_IT_Total_2 - P_UPS_Discharge_2 = 0
_C250: - Q_Discharge_TES_Watts_10 + Q_Discharge_TES_Watts_11 <= 1000000
_C251: - Q_Charge_TES_Watts_10 + Q_Charge_TES_Watts_11 <= 1000000
_C252: P_Chiller_HVAC_Watts_11 + P_Chiller_TES_Watts_11 <= 400000
_C253: - 1000 P_IT_Total_11 + Q_Cool_Watts_11 >= 0
_C254: - 0.001 P_Chiller_HVAC_Watts_12 - 0.001 P_Chiller_TES_Watts_12
 + P_Cooling_kW_12 = 0
_C255: - P_Grid_IT_12 + P_IT_Total_12 - P_UPS_Discharge_12 = 0
_C256: P_Grid_Other_12 - 0.1 P_IT_Total_12 = 0
_C257: - P_Cooling_kW_12 - P_Grid_IT_12 - P_Grid_Other_12 + P_Grid_Total_12
 - P_UPS_Charge_12 = 0
_C258: 10000 IsActive_12 + P_Grid_Total_12 <= 10924.33931
_C259: - E_UPS_11 + E_UPS_12 - 0.205 P_UPS_Charge_12
 + 0.271739130435 P_UPS_Discharge_12 = 0
_C26: P_Grid_Other_2 - 0.1 P_IT_Total_2 = 0
_C260: P_UPS_Charge_12 - 270 Z_Charge_12 <= 0
_C261: P_UPS_Charge_12 - 40 Z_Charge_12 >= 0
_C262: P_UPS_Discharge_12 - 2700 Z_Discharge_12 <= 0
_C263: P_UPS_Discharge_12 - 100 Z_Discharge_12 >= 0
_C264: Z_Charge_12 + Z_Discharge_12 <= 1
_C265: - 6 P_Chiller_HVAC_Watts_12 + Q_Cool_Watts_12
 - Q_Discharge_TES_Watts_12 = 0
_C266: - 6 P_Chiller_TES_Watts_12 + Q_Charge_TES_Watts_12 = 0
_C267: 9.94579541499e-06 Q_Cool_Watts_12 + T_Aisle_In_12 - T_HotAisle_11 = 0
_C268: - 0.0503355704698 P_IT_Total_12 + 4.48616573826 T_IT_11 + T_IT_12
 - 5.48616573826 T_Rack_11 = 0
_C269: - 3.84820916835 T_ColdAisle_11 - 5.4436888589 T_IT_11
 + 8.29189802725 T_Rack_11 + T_Rack_12 = 0
_C27: - P_Cooling_kW_2 - P_Grid_IT_2 - P_Grid_Other_2 + P_Grid_Total_2
 - P_UPS_Charge_2 = 0
_C270: - 29.7271551724 T_Aisle_In_12 + 30.4570824572 T_ColdAisle_11
 + T_ColdAisle_12 = 38.0584002656
_C271: 58.4543103448 T_HotAisle_11 + T_HotAisle_12 - 59.4543103448 T_Rack_11
 = 0
_C272: - E_TES_11 + E_TES_12 - 0.000225 Q_Charge_TES_Watts_12
 + 0.000277777777778 Q_Discharge_TES_Watts_12 = 0
_C273: - Q_Discharge_TES_Watts_11 + Q_Discharge_TES_Watts_12 <= 1000000
_C274: - Q_Charge_TES_Watts_11 + Q_Charge_TES_Watts_12 <= 1000000
_C275: P_Chiller_HVAC_Watts_12 + P_Chiller_TES_Watts_12 <= 400000
_C276: - 1000 P_IT_Total_12 + Q_Cool_Watts_12 >= 0
_C277: - 0.001 P_Chiller_HVAC_Watts_13 - 0.001 P_Chiller_TES_Watts_13
 + P_Cooling_kW_13 = 0
_C278: - P_Grid_IT_13 + P_IT_Total_13 - P_UPS_Discharge_13 = 0
_C279: P_Grid_Other_13 - 0.1 P_IT_Total_13 = 0
_C28: 10000 IsActive_2 + P_Grid_Total_2 <= 10924.33931
_C280: - P_Cooling_kW_13 - P_Grid_IT_13 - P_Grid_Other_13 + P_Grid_Total_13
 - P_UPS_Charge_13 = 0
_C281: 10000 IsActive_13 + P_Grid_Total_13 <= 10924.33931
_C282: - E_UPS_12 + E_UPS_13 - 0.205 P_UPS_Charge_13
 + 0.271739130435 P_UPS_Discharge_13 = 0
_C283: P_UPS_Charge_13 - 270 Z_Charge_13 <= 0
_C284: P_UPS_Charge_13 - 40 Z_Charge_13 >= 0
_C285: P_UPS_Discharge_13 - 2700 Z_Discharge_13 <= 0
_C286: P_UPS_Discharge_13 - 100 Z_Discharge_13 >= 0
_C287: Z_Charge_13 + Z_Discharge_13 <= 1
_C288: - 6 P_Chiller_HVAC_Watts_13 + Q_Cool_Watts_13
 - Q_Discharge_TES_Watts_13 = 0
_C289: - 6 P_Chiller_TES_Watts_13 + Q_Charge_TES_Watts_13 = 0
_C29: - E_UPS_1 + E_UPS_2 - 0.205 P_UPS_Charge_2
 + 0.271739130435 P_UPS_Discharge_2 = 0
_C290: 9.94579541499e-06 Q_Cool_Watts_13 + T_Aisle_In_13 - T_HotAisle_12 = 0
_C291: - 0.0503355704698 P_IT_Total_13 + 4.48616573826 T_IT_12 + T_IT_13
 - 5.48616573826 T_Rack_12 = 0
_C292: - 3.84820916835 T_ColdAisle_12 - 5.4436888589 T_IT_12
 + 8.29189802725 T_Rack_12 + T_Rack_13 = 0
_C293: - 29.7271551724 T_Aisle_In_13 + 30.4570824572 T_ColdAisle_12
 + T_ColdAisle_13 = 38.0584002656
_C294: 58.4543103448 T_HotAisle_12 + T_HotAisle_13 - 59.4543103448 T_Rack_12
 = 0
_C295: - E_TES_12 + E_TES_13 - 0.000225 Q_Charge_TES_Watts_13
 + 0.000277777777778 Q_Discharge_TES_Watts_13 = 0
_C296: - Q_Discharge_TES_Watts_12 + Q_Discharge_TES_Watts_13 <= 1000000
_C297: - Q_Charge_TES_Watts_12 + Q_Charge_TES_Watts_13 <= 1000000
_C298: P_Chiller_HVAC_Watts_13 + P_Chiller_TES_Watts_13 <= 400000
_C299: - 1000 P_IT_Total_13 + Q_Cool_Watts_13 >= 0
_C3: P_Grid_Other_1 - 0.1 P_IT_Total_1 = 0
_C30: P_UPS_Charge_2 - 270 Z_Charge_2 <= 0
_C300: - 0.001 P_Chiller_HVAC_Watts_14 - 0.001 P_Chiller_TES_Watts_14
 + P_Cooling_kW_14 = 0
_C301: - P_Grid_IT_14 + P_IT_Total_14 - P_UPS_Discharge_14 = 0
_C302: P_Grid_Other_14 - 0.1 P_IT_Total_14 = 0
_C303: - P_Cooling_kW_14 - P_Grid_IT_14 - P_Grid_Other_14 + P_Grid_Total_14
 - P_UPS_Charge_14 = 0
_C304: 10000 IsActive_14 + P_Grid_Total_14 <= 10924.33931
_C305: - E_UPS_13 + E_UPS_14 - 0.205 P_UPS_Charge_14
 + 0.271739130435 P_UPS_Discharge_14 = 0
_C306: P_UPS_Charge_14 - 270 Z_Charge_14 <= 0
_C307: P_UPS_Charge_14 - 40 Z_Charge_14 >= 0
_C308: P_UPS_Discharge_14 - 2700 Z_Discharge_14 <= 0
_C309: P_UPS_Discharge_14 - 100 Z_Discharge_14 >= 0
_C31: P_UPS_Charge_2 - 40 Z_Charge_2 >= 0
_C310: Z_Charge_14 + Z_Discharge_14 <= 1
_C311: - 6 P_Chiller_HVAC_Watts_14 + Q_Cool_Watts_14
 - Q_Discharge_TES_Watts_14 = 0
_C312: - 6 P_Chiller_TES_Watts_14 + Q_Charge_TES_Watts_14 = 0
_C313: 9.94579541499e-06 Q_Cool_Watts_14 + T_Aisle_In_14 - T_HotAisle_13 = 0
_C314: - 0.0503355704698 P_IT_Total_14 + 4.48616573826 T_IT_13 + T_IT_14
 - 5.48616573826 T_Rack_13 = 0
_C315: - 3.84820916835 T_ColdAisle_13 - 5.4436888589 T_IT_13
 + 8.29189802725 T_Rack_13 + T_Rack_14 = 0
_C316: - 29.7271551724 T_Aisle_In_14 + 30.4570824572 T_ColdAisle_13
 + T_ColdAisle_14 = 38.0584002656
_C317: 58.4543103448 T_HotAisle_13 + T_HotAisle_14 - 59.4543103448 T_Rack_13
 = 0
_C318: - E_TES_13 + E_TES_14 - 0.000225 Q_Charge_TES_Watts_14
 + 0.000277777777778 Q_Discharge_TES_Watts_14 = 0
_C319: - Q_Discharge_TES_Watts_13 + Q_Discharge_TES_Watts_14 <= 1000000
_C32: P_UPS_Discharge_2 - 2700 Z_Discharge_2 <= 0
_C320: - Q_Charge_TES_Watts_13 + Q_Charge_TES_Watts_14 <= 1000000
_C321: P_Chiller_HVAC_Watts_14 + P_Chiller_TES_Watts_14 <= 400000
_C322: - 1000 P_IT_Total_14 + Q_Cool_Watts_14 >= 0
_C323: - 0.001 P_Chiller_HVAC_Watts_15 - 0.001 P_Chiller_TES_Watts_15
 + P_Cooling_kW_15 = 0
_C324: - P_Grid_IT_15 + P_IT_Total_15 - P_UPS_Discharge_15 = 0
_C325: P_Grid_Other_15 - 0.1 P_IT_Total_15 = 0
_C326: - P_Cooling_kW_15 - P_Grid_IT_15 - P_Grid_Other_15 + P_Grid_Total_15
 - P_UPS_Charge_15 = 0
_C327: 10000 IsActive_15 + P_Grid_Total_15 <= 10924.33931
_C328: - E_UPS_14 + E_UPS_15 - 0.205 P_UPS_Charge_15
 + 0.271739130435 P_UPS_Discharge_15 = 0
_C329: P_UPS_Charge_15 - 270 Z_Charge_15 <= 0
_C33: P_UPS_Discharge_2 - 100 Z_Discharge_2 >= 0
_C330: P_UPS_Charge_15 - 40 Z_Charge_15 >= 0
_C331: P_UPS_Discharge_15 - 2700 Z_Discharge_15 <= 0
_C332: P_UPS_Discharge_15 - 100 Z_Discharge_15 >= 0
_C333: Z_Charge_15 + Z_Discharge_15 <= 1
_C334: - 6 P_Chiller_HVAC_Watts_15 + Q_Cool_Watts_15
 - Q_Discharge_TES_Watts_15 = 0
_C335: - 6 P_Chiller_TES_Watts_15 + Q_Charge_TES_Watts_15 = 0
_C336: 9.94579541499e-06 Q_Cool_Watts_15 + T_Aisle_In_15 - T_HotAisle_14 = 0
_C337: - 0.0503355704698 P_IT_Total_15 + 4.48616573826 T_IT_14 + T_IT_15
 - 5.48616573826 T_Rack_14 = 0
_C338: - 3.84820916835 T_ColdAisle_14 - 5.4436888589 T_IT_14
 + 8.29189802725 T_Rack_14 + T_Rack_15 = 0
_C339: - 29.7271551724 T_Aisle_In_15 + 30.4570824572 T_ColdAisle_14
 + T_ColdAisle_15 = 38.0584002656
_C34: Z_Charge_2 + Z_Discharge_2 <= 1
_C340: 58.4543103448 T_HotAisle_14 + T_HotAisle_15 - 59.4543103448 T_Rack_14
 = 0
_C341: - E_TES_14 + E_TES_15 - 0.000225 Q_Charge_TES_Watts_15
 + 0.000277777777778 Q_Discharge_TES_Watts_15 = 0
_C342: - Q_Discharge_TES_Watts_14 + Q_Discharge_TES_Watts_15 <= 1000000
_C343: - Q_Charge_TES_Watts_14 + Q_Charge_TES_Watts_15 <= 1000000
_C344: P_Chiller_HVAC_Watts_15 + P_Chiller_TES_Watts_15 <= 400000
_C345: - 1000 P_IT_Total_15 + Q_Cool_Watts_15 >= 0
_C346: - 0.001 P_Chiller_HVAC_Watts_16 - 0.001 P_Chiller_TES_Watts_16
 + P_Cooling_kW_16 = 0
_C347: - P_Grid_IT_16 + P_IT_Total_16 - P_UPS_Discharge_16 = 0
_C348: P_Grid_Other_16 - 0.1 P_IT_Total_16 = 0
_C349: - P_Cooling_kW_16 - P_Grid_IT_16 - P_Grid_Other_16 + P_Grid_Total_16
 - P_UPS_Charge_16 = 0
_C35: - 6 P_Chiller_HVAC_Watts_2 + Q_Cool_Watts_2 - Q_Discharge_TES_Watts_2
 = 0
_C350: 10000 IsActive_16 + P_Grid_Total_16 <= 10924.33931
_C351: - E_UPS_15 + E_UPS_16 - 0.205 P_UPS_Charge_16
 + 0.271739130435 P_UPS_Discharge_16 = 0
_C352: P_UPS_Charge_16 - 270 Z_Charge_16 <= 0
_C353: P_UPS_Charge_16 - 40 Z_Charge_16 >= 0
_C354: P_UPS_Discharge_16 - 2700 Z_Discharge_16 <= 0
_C355: P_UPS_Discharge_16 - 100 Z_Discharge_16 >= 0
_C356: Z_Charge_16 + Z_Discharge_16 <= 1
_C357: - 6 P_Chiller_HVAC_Watts_16 + Q_Cool_Watts_16
 - Q_Discharge_TES_Watts_16 = 0
_C358: - 6 P_Chiller_TES_Watts_16 + Q_Charge_TES_Watts_16 = 0
_C359: 9.94579541499e-06 Q_Cool_Watts_16 + T_Aisle_In_16 - T_HotAisle_15 = 0
_C36: - 6 P_Chiller_TES_Watts_2 + Q_Charge_TES_Watts_2 = 0
_C360: - 0.0503355704698 P_IT_Total_16 + 4.48616573826 T_IT_15 + T_IT_16
 - 5.48616573826 T_Rack_15 = 0
_C361: - 3.84820916835 T_ColdAisle_15 - 5.4436888589 T_IT_15
 + 8.29189802725 T_Rack_15 + T_Rack_16 = 0
_C362: - 29.7271551724 T_Aisle_In_16 + 30.4570824572 T_ColdAisle_15
 + T_ColdAisle_16 = 38.0584002656
_C363: 58.4543103448 T_HotAisle_15 + T_HotAisle_16 - 59.4543103448 T_Rack_15
 = 0
_C364: - E_TES_15 + E_TES_16 - 0.000225 Q_Charge_TES_Watts_16
 + 0.000277777777778 Q_Discharge_TES_Watts_16 = 0
_C365: - Q_Discharge_TES_Watts_15 + Q_Discharge_TES_Watts_16 <= 1000000
_C366: - Q_Charge_TES_Watts_15 + Q_Charge_TES_Watts_16 <= 1000000
_C367: P_Chiller_HVAC_Watts_16 + P_Chiller_TES_Watts_16 <= 400000
_C368: - 1000 P_IT_Total_16 + Q_Cool_Watts_16 >= 0
_C369: - 0.001 P_Chiller_HVAC_Watts_17 - 0.001 P_Chiller_TES_Watts_17
 + P_Cooling_kW_17 = 0
_C37: 9.94579541499e-06 Q_Cool_Watts_2 + T_Aisle_In_2 - T_HotAisle_1 = 0
_C370: - P_Grid_IT_17 + P_IT_Total_17 - P_UPS_Discharge_17 = 0
_C371: P_Grid_Other_17 - 0.1 P_IT_Total_17 = 0
_C372: - P_Cooling_kW_17 - P_Grid_IT_17 - P_Grid_Other_17 + P_Grid_Total_17
 - P_UPS_Charge_17 = 0
_C373: 10000 IsActive_17 + P_Grid_Total_17 <= 10924.33931
_C374: - E_UPS_16 + E_UPS_17 - 0.205 P_UPS_Charge_17
 + 0.271739130435 P_UPS_Discharge_17 = 0
_C375: P_UPS_Charge_17 - 270 Z_Charge_17 <= 0
_C376: P_UPS_Charge_17 - 40 Z_Charge_17 >= 0
_C377: P_UPS_Discharge_17 - 2700 Z_Discharge_17 <= 0
_C378: P_UPS_Discharge_17 - 100 Z_Discharge_17 >= 0
_C379: Z_Charge_17 + Z_Discharge_17 <= 1
_C38: - 0.0503355704698 P_IT_Total_2 + 4.48616573826 T_IT_1 + T_IT_2
 - 5.48616573826 T_Rack_1 = 0
_C380: - 6 P_Chiller_HVAC_Watts_17 + Q_Cool_Watts_17
 - Q_Discharge_TES_Watts_17 = 0
_C381: - 6 P_Chiller_TES_Watts_17 + Q_Charge_TES_Watts_17 = 0
_C382: 9.94579541499e-06 Q_Cool_Watts_17 + T_Aisle_In_17 - T_HotAisle_16 = 0
_C383: - 0.0503355704698 P_IT_Total_17 + 4.48616573826 T_IT_16 + T_IT_17
 - 5.48616573826 T_Rack_16 = 0
_C384: - 3.84820916835 T_ColdAisle_16 - 5.4436888589 T_IT_16
 + 8.29189802725 T_Rack_16 + T_Rack_17 = 0
_C385: - 29.7271551724 T_Aisle_In_17 + 30.4570824572 T_ColdAisle_16
 + T_ColdAisle_17 = 38.0584002656
_C386: 58.4543103448 T_HotAisle_16 + T_HotAisle_17 - 59.4543103448 T_Rack_16
 = 0
_C387: - E_TES_16 + E_TES_17 - 0.000225 Q_Charge_TES_Watts_17
 + 0.000277777777778 Q_Discharge_TES_Watts_17 = 0
_C388: - Q_Discharge_TES_Watts_16 + Q_Discharge_TES_Watts_17 <= 1000000
_C389: - Q_Charge_TES_Watts_16 + Q_Charge_TES_Watts_17 <= 1000000
_C39: - 3.84820916835 T_ColdAisle_1 - 5.4436888589 T_IT_1
 + 8.29189802725 T_Rack_1 + T_Rack_2 = 0
_C390: P_Chiller_HVAC_Watts_17 + P_Chiller_TES_Watts_17 <= 400000
_C391: - 1000 P_IT_Total_17 + Q_Cool_Watts_17 >= 0
_C392: - 0.001 P_Chiller_HVAC_Watts_18 - 0.001 P_Chiller_TES_Watts_18
 + P_Cooling_kW_18 = 0
_C393: - P_Grid_IT_18 + P_IT_Total_18 - P_UPS_Discharge_18 = 0
_C394: P_Grid_Other_18 - 0.1 P_IT_Total_18 = 0
_C395: - P_Cooling_kW_18 - P_Grid_IT_18 - P_Grid_Other_18 + P_Grid_Total_18
 - P_UPS_Charge_18 = 0
_C396: 10000 IsActive_18 + P_Grid_Total_18 <= 10924.33931
_C397: - E_UPS_17 + E_UPS_18 - 0.205 P_UPS_Charge_18
 + 0.271739130435 P_UPS_Discharge_18 = 0
_C398: P_UPS_Charge_18 - 270 Z_Charge_18 <= 0
_C399: P_UPS_Charge_18 - 40 Z_Charge_18 >= 0
_C4: - P_Cooling_kW_1 - P_Grid_IT_1 - P_Grid_Other_1 + P_Grid_Total_1
 - P_UPS_Charge_1 = 0
_C40: - 29.7271551724 T_Aisle_In_2 + 30.4570824572 T_ColdAisle_1
 + T_ColdAisle_2 = 38.0584002656
_C400: P_UPS_Discharge_18 - 2700 Z_Discharge_18 <= 0
_C401: P_UPS_Discharge_18 - 100 Z_Discharge_18 >= 0
_C402: Z_Charge_18 + Z_Discharge_18 <= 1
_C403: - 6 P_Chiller_HVAC_Watts_18 + Q_Cool_Watts_18
 - Q_Discharge_TES_Watts_18 = 0
_C404: - 6 P_Chiller_TES_Watts_18 + Q_Charge_TES_Watts_18 = 0
_C405: 9.94579541499e-06 Q_Cool_Watts_18 + T_Aisle_In_18 - T_HotAisle_17 = 0
_C406: - 0.0503355704698 P_IT_Total_18 + 4.48616573826 T_IT_17 + T_IT_18
 - 5.48616573826 T_Rack_17 = 0
_C407: - 3.84820916835 T_ColdAisle_17 - 5.4436888589 T_IT_17
 + 8.29189802725 T_Rack_17 + T_Rack_18 = 0
_C408: - 29.7271551724 T_Aisle_In_18 + 30.4570824572 T_ColdAisle_17
 + T_ColdAisle_18 = 38.0584002656
_C409: 58.4543103448 T_HotAisle_17 + T_HotAisle_18 - 59.4543103448 T_Rack_17
 = 0
_C41: 58.4543103448 T_HotAisle_1 + T_HotAisle_2 - 59.4543103448 T_Rack_1 = 0
_C410: - E_TES_17 + E_TES_18 - 0.000225 Q_Charge_TES_Watts_18
 + 0.000277777777778 Q_Discharge_TES_Watts_18 = 0
_C411: - Q_Discharge_TES_Watts_17 + Q_Discharge_TES_Watts_18 <= 1000000
_C412: - Q_Charge_TES_Watts_17 + Q_Charge_TES_Watts_18 <= 1000000
_C413: P_Chiller_HVAC_Watts_18 + P_Chiller_TES_Watts_18 <= 400000
_C414: - 1000 P_IT_Total_18 + Q_Cool_Watts_18 >= 0
_C415: - 0.001 P_Chiller_HVAC_Watts_19 - 0.001 P_Chiller_TES_Watts_19
 + P_Cooling_kW_19 = 0
_C416: - P_Grid_IT_19 + P_IT_Total_19 - P_UPS_Discharge_19 = 0
_C417: P_Grid_Other_19 - 0.1 P_IT_Total_19 = 0
_C418: - P_Cooling_kW_19 - P_Grid_IT_19 - P_Grid_Other_19 + P_Grid_Total_19
 - P_UPS_Charge_19 = 0
_C419: 10000 IsActive_19 + P_Grid_Total_19 <= 10924.33931
_C42: - E_TES_1 + E_TES_2 - 0.000225 Q_Charge_TES_Watts_2
 + 0.000277777777778 Q_Discharge_TES_Watts_2 = 0
_C420: - E_UPS_18 + E_UPS_19 - 0.205 P_UPS_Charge_19
 + 0.271739130435 P_UPS_Discharge_19 = 0
_C421: P_UPS_Charge_19 - 270 Z_Charge_19 <= 0
_C422: P_UPS_Charge_19 - 40 Z_Charge_19 >= 0
_C423: P_UPS_Discharge_19 - 2700 Z_Discharge_19 <= 0
_C424: P_UPS_Discharge_19 - 100 Z_Discharge_19 >= 0
_C425: Z_Charge_19 + Z_Discharge_19 <= 1
_C426: - 6 P_Chiller_HVAC_Watts_19 + Q_Cool_Watts_19
 - Q_Discharge_TES_Watts_19 = 0
_C427: - 6 P_Chiller_TES_Watts_19 + Q_Charge_TES_Watts_19 = 0
_C428: 9.94579541499e-06 Q_Cool_Watts_19 + T_Aisle_In_19 - T_HotAisle_18 = 0
_C429: - 0.0503355704698 P_IT_Total_19 + 4.48616573826 T_IT_18 + T_IT_19
 - 5.48616573826 T_Rack_18 = 0
_C43: - Q_Discharge_TES_Watts_1 + Q_Discharge_TES_Watts_2 <= 1000000
_C430: - 3.84820916835 T_ColdAisle_18 - 5.4436888589 T_IT_18
 + 8.29189802725 T_Rack_18 + T_Rack_19 = 0
_C431: - 29.7271551724 T_Aisle_In_19 + 30.4570824572 T_ColdAisle_18
 + T_ColdAisle_19 = 38.0584002656
_C432: 58.4543103448 T_HotAisle_18 + T_HotAisle_19 - 59.4543103448 T_Rack_18
 = 0
_C433: - E_TES_18 + E_TES_19 - 0.000225 Q_Charge_TES_Watts_19
 + 0.000277777777778 Q_Discharge_TES_Watts_19 = 0
_C434: - Q_Discharge_TES_Watts_18 + Q_Discharge_TES_Watts_19 <= 1000000
_C435: - Q_Charge_TES_Watts_18 + Q_Charge_TES_Watts_19 <= 1000000
_C436: P_Chiller_HVAC_Watts_19 + P_Chiller_TES_Watts_19 <= 400000
_C437: - 1000 P_IT_Total_19 + Q_Cool_Watts_19 >= 0
_C438: - 0.001 P_Chiller_HVAC_Watts_20 - 0.001 P_Chiller_TES_Watts_20
 + P_Cooling_kW_20 = 0
_C439: - P_Grid_IT_20 + P_IT_Total_20 - P_UPS_Discharge_20 = 0
_C44: - Q_Charge_TES_Watts_1 + Q_Charge_TES_Watts_2 <= 1000000
_C440: P_Grid_Other_20 - 0.1 P_IT_Total_20 = 0
_C441: - P_Cooling_kW_20 - P_Grid_IT_20 - P_Grid_Other_20 + P_Grid_Total_20
 - P_UPS_Charge_20 = 0
_C442: 10000 IsActive_20 + P_Grid_Total_20 <= 10924.33931
_C443: - E_UPS_19 + E_UPS_20 - 0.205 P_UPS_Charge_20
 + 0.271739130435 P_UPS_Discharge_20 = 0
_C444: P_UPS_Charge_20 - 270 Z_Charge_20 <= 0
_C445: P_UPS_Charge_20 - 40 Z_Charge_20 >= 0
_C446: P_UPS_Discharge_20 - 2700 Z_Discharge_20 <= 0
_C447: P_UPS_Discharge_20 - 100 Z_Discharge_20 >= 0
_C448: Z_Charge_20 + Z_Discharge_20 <= 1
_C449: - 6 P_Chiller_HVAC_Watts_20 + Q_Cool_Watts_20
 - Q_Discharge_TES_Watts_20 = 0
_C45: P_Chiller_HVAC_Watts_2 + P_Chiller_TES_Watts_2 <= 400000
_C450: - 6 P_Chiller_TES_Watts_20 + Q_Charge_TES_Watts_20 = 0
_C451: 9.94579541499e-06 Q_Cool_Watts_20 + T_Aisle_In_20 - T_HotAisle_19 = 0
_C452: - 0.0503355704698 P_IT_Total_20 + 4.48616573826 T_IT_19 + T_IT_20
 - 5.48616573826 T_Rack_19 = 0
_C453: - 3.84820916835 T_ColdAisle_19 - 5.4436888589 T_IT_19
 + 8.29189802725 T_Rack_19 + T_Rack_20 = 0
_C454: - 29.7271551724 T_Aisle_In_20 + 30.4570824572 T_ColdAisle_19
 + T_ColdAisle_20 = 38.0584002656
_C455: 58.4543103448 T_HotAisle_19 + T_HotAisle_20 - 59.4543103448 T_Rack_19
 = 0
_C456: - E_TES_19 + E_TES_20 - 0.000225 Q_Charge_TES_Watts_20
 + 0.000277777777778 Q_Discharge_TES_Watts_20 = 0
_C457: - Q_Discharge_TES_Watts_19 + Q_Discharge_TES_Watts_20 <= 1000000
_C458: - Q_Charge_TES_Watts_19 + Q_Charge_TES_Watts_20 <= 1000000
_C459: P_Chiller_HVAC_Watts_20 + P_Chiller_TES_Watts_20 <= 400000
_C46: - 1000 P_IT_Total_2 + Q_Cool_Watts_2 >= 0
_C460: - 1000 P_IT_Total_20 + Q_Cool_Watts_20 >= 0
_C461: - 0.001 P_Chiller_HVAC_Watts_21 - 0.001 P_Chiller_TES_Watts_21
 + P_Cooling_kW_21 = 0
_C462: - P_Grid_IT_21 + P_IT_Total_21 - P_UPS_Discharge_21 = 0
_C463: P_Grid_Other_21 - 0.1 P_IT_Total_21 = 0
_C464: - P_Cooling_kW_21 - P_Grid_IT_21 - P_Grid_Other_21 + P_Grid_Total_21
 - P_UPS_Charge_21 = 0
_C465: 10000 IsActive_21 + P_Grid_Total_21 <= 10924.33931
_C466: - E_UPS_20 + E_UPS_21 - 0.205 P_UPS_Charge_21
 + 0.271739130435 P_UPS_Discharge_21 = 0
_C467: P_UPS_Charge_21 - 270 Z_Charge_21 <= 0
_C468: P_UPS_Charge_21 - 40 Z_Charge_21 >= 0
_C469: P_UPS_Discharge_21 - 2700 Z_Discharge_21 <= 0
_C47: - 0.001 P_Chiller_HVAC_Watts_3 - 0.001 P_Chiller_TES_Watts_3
 + P_Cooling_kW_3 = 0
_C470: P_UPS_Discharge_21 - 100 Z_Discharge_21 >= 0
_C471: Z_Charge_21 + Z_Discharge_21 <= 1
_C472: - 6 P_Chiller_HVAC_Watts_21 + Q_Cool_Watts_21
 - Q_Discharge_TES_Watts_21 = 0
_C473: - 6 P_Chiller_TES_Watts_21 + Q_Charge_TES_Watts_21 = 0
_C474: 9.94579541499e-06 Q_Cool_Watts_21 + T_Aisle_In_21 - T_HotAisle_20 = 0
_C475: - 0.0503355704698 P_IT_Total_21 + 4.48616573826 T_IT_20 + T_IT_21
 - 5.48616573826 T_Rack_20 = 0
_C476: - 3.84820916835 T_ColdAisle_20 - 5.4436888589 T_IT_20
 + 8.29189802725 T_Rack_20 + T_Rack_21 = 0
_C477: - 29.7271551724 T_Aisle_In_21 + 30.4570824572 T_ColdAisle_20
 + T_ColdAisle_21 = 38.0584002656
_C478: 58.4543103448 T_HotAisle_20 + T_HotAisle_21 - 59.4543103448 T_Rack_20
 = 0
_C479: - E_TES_20 + E_TES_21 - 0.000225 Q_Charge_TES_Watts_21
 + 0.000277777777778 Q_Discharge_TES_Watts_21 = 0
_C48: - P_Grid_IT_3 + P_IT_Total_3 - P_UPS_Discharge_3 = 0
_C480: - Q_Discharge_TES_Watts_20 + Q_Discharge_TES_Watts_21 <= 1000000
_C481: - Q_Charge_TES_Watts_20 + Q_Charge_TES_Watts_21 <= 1000000
_C482: P_Chiller_HVAC_Watts_21 + P_Chiller_TES_Watts_21 <= 400000
_C483: - 1000 P_IT_Total_21 + Q_Cool_Watts_21 >= 0
_C484: - 0.001 P_Chiller_HVAC_Watts_22 - 0.001 P_Chiller_TES_Watts_22
 + P_Cooling_kW_22 = 0
_C485: - P_Grid_IT_22 + P_IT_Total_22 - P_UPS_Discharge_22 = 0
_C486: P_Grid_Other_22 - 0.1 P_IT_Total_22 = 0
_C487: - P_Cooling_kW_22 - P_Grid_IT_22 - P_Grid_Other_22 + P_Grid_Total_22
 - P_UPS_Charge_22 = 0
_C488: 10000 IsActive_22 + P_Grid_Total_22 <= 10924.33931
_C489: - E_UPS_21 + E_UPS_22 - 0.205 P_UPS_Charge_22
 + 0.271739130435 P_UPS_Discharge_22 = 0
_C49: P_Grid_Other_3 - 0.1 P_IT_Total_3 = 0
_C490: P_UPS_Charge_22 - 270 Z_Charge_22 <= 0
_C491: P_UPS_Charge_22 - 40 Z_Charge_22 >= 0
_C492: P_UPS_Discharge_22 - 2700 Z_Discharge_22 <= 0
_C493: P_UPS_Discharge_22 - 100 Z_Discharge_22 >= 0
_C494: Z_Charge_22 + Z_Discharge_22 <= 1
_C495: - 6 P_Chiller_HVAC_Watts_22 + Q_Cool_Watts_22
 - Q_Discharge_TES_Watts_22 = 0
_C496: - 6 P_Chiller_TES_Watts_22 + Q_Charge_TES_Watts_22 = 0
_C497: 9.94579541499e-06 Q_Cool_Watts_22 + T_Aisle_In_22 - T_HotAisle_21 = 0
_C498: - 0.0503355704698 P_IT_Total_22 + 4.48616573826 T_IT_21 + T_IT_22
 - 5.48616573826 T_Rack_21 = 0
_C499: - 3.84820916835 T_ColdAisle_21 - 5.4436888589 T_IT_21
 + 8.29189802725 T_Rack_21 + T_Rack_22 = 0
_C5: 10000 IsActive_1 + P_Grid_Total_1 <= 10924.33931
_C50: - P_Cooling_kW_3 - P_Grid_IT_3 - P_Grid_Other_3 + P_Grid_Total_3
 - P_UPS_Charge_3 = 0
_C500: - 29.7271551724 T_Aisle_In_22 + 30.4570824572 T_ColdAisle_21
 + T_ColdAisle_22 = 38.0584002656
_C501: 58.4543103448 T_HotAisle_21 + T_HotAisle_22 - 59.4543103448 T_Rack_21
 = 0
_C502: - E_TES_21 + E_TES_22 - 0.000225 Q_Charge_TES_Watts_22
 + 0.000277777777778 Q_Discharge_TES_Watts_22 = 0
_C503: - Q_Discharge_TES_Watts_21 + Q_Discharge_TES_Watts_22 <= 1000000
_C504: - Q_Charge_TES_Watts_21 + Q_Charge_TES_Watts_22 <= 1000000
_C505: P_Chiller_HVAC_Watts_22 + P_Chiller_TES_Watts_22 <= 400000
_C506: - 1000 P_IT_Total_22 + Q_Cool_Watts_22 >= 0
_C507: - 0.001 P_Chiller_HVAC_Watts_23 - 0.001 P_Chiller_TES_Watts_23
 + P_Cooling_kW_23 = 0
_C508: - P_Grid_IT_23 + P_IT_Total_23 - P_UPS_Discharge_23 = 0
_C509: P_Grid_Other_23 - 0.1 P_IT_Total_23 = 0
_C51: 10000 IsActive_3 + P_Grid_Total_3 <= 10924.33931
_C510: - P_Cooling_kW_23 - P_Grid_IT_23 - P_Grid_Other_23 + P_Grid_Total_23
 - P_UPS_Charge_23 = 0
_C511: 10000 IsActive_23 + P_Grid_Total_23 <= 10924.33931
_C512: - E_UPS_22 + E_UPS_23 - 0.205 P_UPS_Charge_23
 + 0.271739130435 P_UPS_Discharge_23 = 0
_C513: P_UPS_Charge_23 - 270 Z_Charge_23 <= 0
_C514: P_UPS_Charge_23 - 40 Z_Charge_23 >= 0
_C515: P_UPS_Discharge_23 - 2700 Z_Discharge_23 <= 0
_C516: P_UPS_Discharge_23 - 100 Z_Discharge_23 >= 0
_C517: Z_Charge_23 + Z_Discharge_23 <= 1
_C518: - 6 P_Chiller_HVAC_Watts_23 + Q_Cool_Watts_23
 - Q_Discharge_TES_Watts_23 = 0
_C519: - 6 P_Chiller_TES_Watts_23 + Q_Charge_TES_Watts_23 = 0
_C52: - E_UPS_2 + E_UPS_3 - 0.205 P_UPS_Charge_3
 + 0.271739130435 P_UPS_Discharge_3 = 0
_C520: 9.94579541499e-06 Q_Cool_Watts_23 + T_Aisle_In_23 - T_HotAisle_22 = 0
_C521: - 0.0503355704698 P_IT_Total_23 + 4.48616573826 T_IT_22 + T_IT_23
 - 5.48616573826 T_Rack_22 = 0
_C522: - 3.84820916835 T_ColdAisle_22 - 5.4436888589 T_IT_22
 + 8.29189802725 T_Rack_22 + T_Rack_23 = 0
_C523: - 29.7271551724 T_Aisle_In_23 + 30.4570824572 T_ColdAisle_22
 + T_ColdAisle_23 = 38.0584002656
_C524: 58.4543103448 T_HotAisle_22 + T_HotAisle_23 - 59.4543103448 T_Rack_22
 = 0
_C525: - E_TES_22 + E_TES_23 - 0.000225 Q_Charge_TES_Watts_23
 + 0.000277777777778 Q_Discharge_TES_Watts_23 = 0
_C526: - Q_Discharge_TES_Watts_22 + Q_Discharge_TES_Watts_23 <= 1000000
_C527: - Q_Charge_TES_Watts_22 + Q_Charge_TES_Watts_23 <= 1000000
_C528: P_Chiller_HVAC_Watts_23 + P_Chiller_TES_Watts_23 <= 400000
_C529: - 1000 P_IT_Total_23 + Q_Cool_Watts_23 >= 0
_C53: P_UPS_Charge_3 - 270 Z_Charge_3 <= 0
_C530: - 0.001 P_Chiller_HVAC_Watts_24 - 0.001 P_Chiller_TES_Watts_24
 + P_Cooling_kW_24 = 0
_C531: - P_Grid_IT_24 + P_IT_Total_24 - P_UPS_Discharge_24 = 0
_C532: P_Grid_Other_24 - 0.1 P_IT_Total_24 = 0
_C533: - P_Cooling_kW_24 - P_Grid_IT_24 - P_Grid_Other_24 + P_Grid_Total_24
 - P_UPS_Charge_24 = 0
_C534: 10000 IsActive_24 + P_Grid_Total_24 <= 10924.33931
_C535: - E_UPS_23 + E_UPS_24 - 0.205 P_UPS_Charge_24
 + 0.271739130435 P_UPS_Discharge_24 = 0
_C536: P_UPS_Charge_24 - 270 Z_Charge_24 <= 0
_C537: P_UPS_Charge_24 - 40 Z_Charge_24 >= 0
_C538: P_UPS_Discharge_24 - 2700 Z_Discharge_24 <= 0
_C539: P_UPS_Discharge_24 - 100 Z_Discharge_24 >= 0
_C54: P_UPS_Charge_3 - 40 Z_Charge_3 >= 0
_C540: Z_Charge_24 + Z_Discharge_24 <= 1
_C541: - 6 P_Chiller_HVAC_Watts_24 + Q_Cool_Watts_24
 - Q_Discharge_TES_Watts_24 = 0
_C542: - 6 P_Chiller_TES_Watts_24 + Q_Charge_TES_Watts_24 = 0
_C543: 9.94579541499e-06 Q_Cool_Watts_24 + T_Aisle_In_24 - T_HotAisle_23 = 0
_C544: - 0.0503355704698 P_IT_Total_24 + 4.48616573826 T_IT_23 + T_IT_24
 - 5.48616573826 T_Rack_23 = 0
_C545: - 3.84820916835 T_ColdAisle_23 - 5.4436888589 T_IT_23
 + 8.29189802725 T_Rack_23 + T_Rack_24 = 0
_C546: - 29.7271551724 T_Aisle_In_24 + 30.4570824572 T_ColdAisle_23
 + T_ColdAisle_24 = 38.0584002656
_C547: 58.4543103448 T_HotAisle_23 + T_HotAisle_24 - 59.4543103448 T_Rack_23
 = 0
_C548: - E_TES_23 + E_TES_24 - 0.000225 Q_Charge_TES_Watts_24
 + 0.000277777777778 Q_Discharge_TES_Watts_24 = 0
_C549: - Q_Discharge_TES_Watts_23 + Q_Discharge_TES_Watts_24 <= 1000000
_C55: P_UPS_Discharge_3 - 2700 Z_Discharge_3 <= 0
_C550: - Q_Charge_TES_Watts_23 + Q_Charge_TES_Watts_24 <= 1000000
_C551: P_Chiller_HVAC_Watts_24 + P_Chiller_TES_Watts_24 <= 400000
_C552: - 1000 P_IT_Total_24 + Q_Cool_Watts_24 >= 0
_C553: - 0.001 P_Chiller_HVAC_Watts_25 - 0.001 P_Chiller_TES_Watts_25
 + P_Cooling_kW_25 = 0
_C554: - P_Grid_IT_25 + P_IT_Total_25 - P_UPS_Discharge_25 = 0
_C555: P_Grid_Other_25 - 0.1 P_IT_Total_25 = 0
_C556: - P_Cooling_kW_25 - P_Grid_IT_25 - P_Grid_Other_25 + P_Grid_Total_25
 - P_UPS_Charge_25 = 0
_C557: 10000 IsActive_25 + P_Grid_Total_25 <= 10924.33931
_C558: - E_UPS_24 + E_UPS_25 - 0.205 P_UPS_Charge_25
 + 0.271739130435 P_UPS_Discharge_25 = 0
_C559: P_UPS_Charge_25 - 270 Z_Charge_25 <= 0
_C56: P_UPS_Discharge_3 - 100 Z_Discharge_3 >= 0
_C560: P_UPS_Charge_25 - 40 Z_Charge_25 >= 0
_C561: P_UPS_Discharge_25 - 2700 Z_Discharge_25 <= 0
_C562: P_UPS_Discharge_25 - 100 Z_Discharge_25 >= 0
_C563: Z_Charge_25 + Z_Discharge_25 <= 1
_C564: - 6 P_Chiller_HVAC_Watts_25 + Q_Cool_Watts_25
 - Q_Discharge_TES_Watts_25 = 0
_C565: - 6 P_Chiller_TES_Watts_25 + Q_Charge_TES_Watts_25 = 0
_C566: 9.94579541499e-06 Q_Cool_Watts_25 + T_Aisle_In_25 - T_HotAisle_24 = 0
_C567: - 0.0503355704698 P_IT_Total_25 + 4.48616573826 T_IT_24 + T_IT_25
 - 5.48616573826 T_Rack_24 = 0
_C568: - 3.84820916835 T_ColdAisle_24 - 5.4436888589 T_IT_24
 + 8.29189802725 T_Rack_24 + T_Rack_25 = 0
_C569: - 29.7271551724 T_Aisle_In_25 + 30.4570824572 T_ColdAisle_24
 + T_ColdAisle_25 = 38.0584002656
_C57: Z_Charge_3 + Z_Discharge_3 <= 1
_C570: 58.4543103448 T_HotAisle_24 + T_HotAisle_25 - 59.4543103448 T_Rack_24
 = 0
_C571: - E_TES_24 + E_TES_25 - 0.000225 Q_Charge_TES_Watts_25
 + 0.000277777777778 Q_Discharge_TES_Watts_25 = 0
_C572: - Q_Discharge_TES_Watts_24 + Q_Discharge_TES_Watts_25 <= 1000000
_C573: - Q_Charge_TES_Watts_24 + Q_Charge_TES_Watts_25 <= 1000000
_C574: P_Chiller_HVAC_Watts_25 + P_Chiller_TES_Watts_25 <= 400000
_C575: - 1000 P_IT_Total_25 + Q_Cool_Watts_25 >= 0
_C576: - 0.001 P_Chiller_HVAC_Watts_26 - 0.001 P_Chiller_TES_Watts_26
 + P_Cooling_kW_26 = 0
_C577: - P_Grid_IT_26 + P_IT_Total_26 - P_UPS_Discharge_26 = 0
_C578: P_Grid_Other_26 - 0.1 P_IT_Total_26 = 0
_C579: - P_Cooling_kW_26 - P_Grid_IT_26 - P_Grid_Other_26 + P_Grid_Total_26
 - P_UPS_Charge_26 = 0
_C58: - 6 P_Chiller_HVAC_Watts_3 + Q_Cool_Watts_3 - Q_Discharge_TES_Watts_3
 = 0
_C580: 10000 IsActive_26 + P_Grid_Total_26 <= 10924.33931
_C581: - E_UPS_25 + E_UPS_26 - 0.205 P_UPS_Charge_26
 + 0.271739130435 P_UPS_Discharge_26 = 0
_C582: P_UPS_Charge_26 - 270 Z_Charge_26 <= 0
_C583: P_UPS_Charge_26 - 40 Z_Charge_26 >= 0
_C584: P_UPS_Discharge_26 - 2700 Z_Discharge_26 <= 0
_C585: P_UPS_Discharge_26 - 100 Z_Discharge_26 >= 0
_C586: Z_Charge_26 + Z_Discharge_26 <= 1
_C587: - 6 P_Chiller_HVAC_Watts_26 + Q_Cool_Watts_26
 - Q_Discharge_TES_Watts_26 = 0
_C588: - 6 P_Chiller_TES_Watts_26 + Q_Charge_TES_Watts_26 = 0
_C589: 9.94579541499e-06 Q_Cool_Watts_26 + T_Aisle_In_26 - T_HotAisle_25 = 0
_C59: - 6 P_Chiller_TES_Watts_3 + Q_Charge_TES_Watts_3 = 0
_C590: - 0.0503355704698 P_IT_Total_26 + 4.48616573826 T_IT_25 + T_IT_26
 - 5.48616573826 T_Rack_25 = 0
_C591: - 3.84820916835 T_ColdAisle_25 - 5.4436888589 T_IT_25
 + 8.29189802725 T_Rack_25 + T_Rack_26 = 0
_C592: - 29.7271551724 T_Aisle_In_26 + 30.4570824572 T_ColdAisle_25
 + T_ColdAisle_26 = 38.0584002656
_C593: 58.4543103448 T_HotAisle_25 + T_HotAisle_26 - 59.4543103448 T_Rack_25
 = 0
_C594: - E_TES_25 + E_TES_26 - 0.000225 Q_Charge_TES_Watts_26
 + 0.000277777777778 Q_Discharge_TES_Watts_26 = 0
_C595: - Q_Discharge_TES_Watts_25 + Q_Discharge_TES_Watts_26 <= 1000000
_C596: - Q_Charge_TES_Watts_25 + Q_Charge_TES_Watts_26 <= 1000000
_C597: P_Chiller_HVAC_Watts_26 + P_Chiller_TES_Watts_26 <= 400000
_C598: - 1000 P_IT_Total_26 + Q_Cool_Watts_26 >= 0
_C599: - 0.001 P_Chiller_HVAC_Watts_27 - 0.001 P_Chiller_TES_Watts_27
 + P_Cooling_kW_27 = 0
_C6: E_UPS_1 - 0.205 P_UPS_Charge_1 + 0.271739130435 P_UPS_Discharge_1 = 300
_C60: 9.94579541499e-06 Q_Cool_Watts_3 + T_Aisle_In_3 - T_HotAisle_2 = 0
_C600: - P_Grid_IT_27 + P_IT_Total_27 - P_UPS_Discharge_27 = 0
_C601: P_Grid_Other_27 - 0.1 P_IT_Total_27 = 0
_C602: - P_Cooling_kW_27 - P_Grid_IT_27 - P_Grid_Other_27 + P_Grid_Total_27
 - P_UPS_Charge_27 = 0
_C603: 10000 IsActive_27 + P_Grid_Total_27 <= 10924.33931
_C604: - E_UPS_26 + E_UPS_27 - 0.205 P_UPS_Charge_27
 + 0.271739130435 P_UPS_Discharge_27 = 0
_C605: P_UPS_Charge_27 - 270 Z_Charge_27 <= 0
_C606: P_UPS_Charge_27 - 40 Z_Charge_27 >= 0
_C607: P_UPS_Discharge_27 - 2700 Z_Discharge_27 <= 0
_C608: P_UPS_Discharge_27 - 100 Z_Discharge_27 >= 0
_C609: Z_Charge_27 + Z_Discharge_27 <= 1
_C61: - 0.0503355704698 P_IT_Total_3 + 4.48616573826 T_IT_2 + T_IT_3
 - 5.48616573826 T_Rack_2 = 0
_C610: - 6 P_Chiller_HVAC_Watts_27 + Q_Cool_Watts_27
 - Q_Discharge_TES_Watts_27 = 0
_C611: - 6 P_Chiller_TES_Watts_27 + Q_Charge_TES_Watts_27 = 0
_C612: 9.94579541499e-06 Q_Cool_Watts_27 + T_Aisle_In_27 - T_HotAisle_26 = 0
_C613: - 0.0503355704698 P_IT_Total_27 + 4.48616573826 T_IT_26 + T_IT_27
 - 5.48616573826 T_Rack_26 = 0
_C614: - 3.84820916835 T_ColdAisle_26 - 5.4436888589 T_IT_26
 + 8.29189802725 T_Rack_26 + T_Rack_27 = 0
_C615: - 29.7271551724 T_Aisle_In_27 + 30.4570824572 T_ColdAisle_26
 + T_ColdAisle_27 = 38.0584002656
_C616: 58.4543103448 T_HotAisle_26 + T_HotAisle_27 - 59.4543103448 T_Rack_26
 = 0
_C617: - E_TES_26 + E_TES_27 - 0.000225 Q_Charge_TES_Watts_27
 + 0.000277777777778 Q_Discharge_TES_Watts_27 = 0
_C618: - Q_Discharge_TES_Watts_26 + Q_Discharge_TES_Watts_27 <= 1000000
_C619: - Q_Charge_TES_Watts_26 + Q_Charge_TES_Watts_27 <= 1000000
_C62: - 3.84820916835 T_ColdAisle_2 - 5.4436888589 T_IT_2
 + 8.29189802725 T_Rack_2 + T_Rack_3 = 0
_C620: P_Chiller_HVAC_Watts_27 + P_Chiller_TES_Watts_27 <= 400000
_C621: - 1000 P_IT_Total_27 + Q_Cool_Watts_27 >= 0
_C622: - 0.001 P_Chiller_HVAC_Watts_28 - 0.001 P_Chiller_TES_Watts_28
 + P_Cooling_kW_28 = 0
_C623: - P_Grid_IT_28 + P_IT_Total_28 - P_UPS_Discharge_28 = 0
_C624: P_Grid_Other_28 - 0.1 P_IT_Total_28 = 0
_C625: - P_Cooling_kW_28 - P_Grid_IT_28 - P_Grid_Other_28 + P_Grid_Total_28
 - P_UPS_Charge_28 = 0
_C626: 10000 IsActive_28 + P_Grid_Total_28 <= 10924.33931
_C627: - E_UPS_27 + E_UPS_28 - 0.205 P_UPS_Charge_28
 + 0.271739130435 P_UPS_Discharge_28 = 0
_C628: P_UPS_Charge_28 - 270 Z_Charge_28 <= 0
_C629: P_UPS_Charge_28 - 40 Z_Charge_28 >= 0
_C63: - 29.7271551724 T_Aisle_In_3 + 30.4570824572 T_ColdAisle_2
 + T_ColdAisle_3 = 38.0584002656
_C630: P_UPS_Discharge_28 - 2700 Z_Discharge_28 <= 0
_C631: P_UPS_Discharge_28 - 100 Z_Discharge_28 >= 0
_C632: Z_Charge_28 + Z_Discharge_28 <= 1
_C633: - 6 P_Chiller_HVAC_Watts_28 + Q_Cool_Watts_28
 - Q_Discharge_TES_Watts_28 = 0
_C634: - 6 P_Chiller_TES_Watts_28 + Q_Charge_TES_Watts_28 = 0
_C635: 9.94579541499e-06 Q_Cool_Watts_28 + T_Aisle_In_28 - T_HotAisle_27 = 0
_C636: - 0.0503355704698 P_IT_Total_28 + 4.48616573826 T_IT_27 + T_IT_28
 - 5.48616573826 T_Rack_27 = 0
_C637: - 3.84820916835 T_ColdAisle_27 - 5.4436888589 T_IT_27
 + 8.29189802725 T_Rack_27 + T_Rack_28 = 0
_C638: - 29.7271551724 T_Aisle_In_28 + 30.4570824572 T_ColdAisle_27
 + T_ColdAisle_28 = 38.0584002656
_C639: 58.4543103448 T_HotAisle_27 + T_HotAisle_28 - 59.4543103448 T_Rack_27
 = 0
_C64: 58.4543103448 T_HotAisle_2 + T_HotAisle_3 - 59.4543103448 T_Rack_2 = 0
_C640: - E_TES_27 + E_TES_28 - 0.000225 Q_Charge_TES_Watts_28
 + 0.000277777777778 Q_Discharge_TES_Watts_28 = 0
_C641: - Q_Discharge_TES_Watts_27 + Q_Discharge_TES_Watts_28 <= 1000000
_C642: - Q_Charge_TES_Watts_27 + Q_Charge_TES_Watts_28 <= 1000000
_C643: P_Chiller_HVAC_Watts_28 + P_Chiller_TES_Watts_28 <= 400000
_C644: - 1000 P_IT_Total_28 + Q_Cool_Watts_28 >= 0
_C645: - 0.001 P_Chiller_HVAC_Watts_29 - 0.001 P_Chiller_TES_Watts_29
 + P_Cooling_kW_29 = 0
_C646: - P_Grid_IT_29 + P_IT_Total_29 - P_UPS_Discharge_29 = 0
_C647: P_Grid_Other_29 - 0.1 P_IT_Total_29 = 0
_C648: - P_Cooling_kW_29 - P_Grid_IT_29 - P_Grid_Other_29 + P_Grid_Total_29
 - P_UPS_Charge_29 = 0
_C649: 10000 IsActive_29 + P_Grid_Total_29 <= 10924.33931
_C65: - E_TES_2 + E_TES_3 - 0.000225 Q_Charge_TES_Watts_3
 + 0.000277777777778 Q_Discharge_TES_Watts_3 = 0
_C650: - E_UPS_28 + E_UPS_29 - 0.205 P_UPS_Charge_29
 + 0.271739130435 P_UPS_Discharge_29 = 0
_C651: P_UPS_Charge_29 - 270 Z_Charge_29 <= 0
_C652: P_UPS_Charge_29 - 40 Z_Charge_29 >= 0
_C653: P_UPS_Discharge_29 - 2700 Z_Discharge_29 <= 0
_C654: P_UPS_Discharge_29 - 100 Z_Discharge_29 >= 0
_C655: Z_Charge_29 + Z_Discharge_29 <= 1
_C656: - 6 P_Chiller_HVAC_Watts_29 + Q_Cool_Watts_29
 - Q_Discharge_TES_Watts_29 = 0
_C657: - 6 P_Chiller_TES_Watts_29 + Q_Charge_TES_Watts_29 = 0
_C658: 9.94579541499e-06 Q_Cool_Watts_29 + T_Aisle_In_29 - T_HotAisle_28 = 0
_C659: - 0.0503355704698 P_IT_Total_29 + 4.48616573826 T_IT_28 + T_IT_29
 - 5.48616573826 T_Rack_28 = 0
_C66: - Q_Discharge_TES_Watts_2 + Q_Discharge_TES_Watts_3 <= 1000000
_C660: - 3.84820916835 T_ColdAisle_28 - 5.4436888589 T_IT_28
 + 8.29189802725 T_Rack_28 + T_Rack_29 = 0
_C661: - 29.7271551724 T_Aisle_In_29 + 30.4570824572 T_ColdAisle_28
 + T_ColdAisle_29 = 38.0584002656
_C662: 58.4543103448 T_HotAisle_28 + T_HotAisle_29 - 59.4543103448 T_Rack_28
 = 0
_C663: - E_TES_28 + E_TES_29 - 0.000225 Q_Charge_TES_Watts_29
 + 0.000277777777778 Q_Discharge_TES_Watts_29 = 0
_C664: - Q_Discharge_TES_Watts_28 + Q_Discharge_TES_Watts_29 <= 1000000
_C665: - Q_Charge_TES_Watts_28 + Q_Charge_TES_Watts_29 <= 1000000
_C666: P_Chiller_HVAC_Watts_29 + P_Chiller_TES_Watts_29 <= 400000
_C667: - 1000 P_IT_Total_29 + Q_Cool_Watts_29 >= 0
_C668: - 0.001 P_Chiller_HVAC_Watts_30 - 0.001 P_Chiller_TES_Watts_30
 + P_Cooling_kW_30 = 0
_C669: - P_Grid_IT_30 + P_IT_Total_30 - P_UPS_Discharge_30 = 0
_C67: - Q_Charge_TES_Watts_2 + Q_Charge_TES_Watts_3 <= 1000000
_C670: P_Grid_Other_30 - 0.1 P_IT_Total_30 = 0
_C671: - P_Cooling_kW_30 - P_Grid_IT_30 - P_Grid_Other_30 + P_Grid_Total_30
 - P_UPS_Charge_30 = 0
_C672: 10000 IsActive_30 + P_Grid_Total_30 <= 10924.33931
_C673: - E_UPS_29 + E_UPS_30 - 0.205 P_UPS_Charge_30
 + 0.271739130435 P_UPS_Discharge_30 = 0
_C674: P_UPS_Charge_30 - 270 Z_Charge_30 <= 0
_C675: P_UPS_Charge_30 - 40 Z_Charge_30 >= 0
_C676: P_UPS_Discharge_30 - 2700 Z_Discharge_30 <= 0
_C677: P_UPS_Discharge_30 - 100 Z_Discharge_30 >= 0
_C678: Z_Charge_30 + Z_Discharge_30 <= 1
_C679: - 6 P_Chiller_HVAC_Watts_30 + Q_Cool_Watts_30
 - Q_Discharge_TES_Watts_30 = 0
_C68: P_Chiller_HVAC_Watts_3 + P_Chiller_TES_Watts_3 <= 400000
_C680: - 6 P_Chiller_TES_Watts_30 + Q_Charge_TES_Watts_30 = 0
_C681: 9.94579541499e-06 Q_Cool_Watts_30 + T_Aisle_In_30 - T_HotAisle_29 = 0
_C682: - 0.0503355704698 P_IT_Total_30 + 4.48616573826 T_IT_29 + T_IT_30
 - 5.48616573826 T_Rack_29 = 0
_C683: - 3.84820916835 T_ColdAisle_29 - 5.4436888589 T_IT_29
 + 8.29189802725 T_Rack_29 + T_Rack_30 = 0
_C684: - 29.7271551724 T_Aisle_In_30 + 30.4570824572 T_ColdAisle_29
 + T_ColdAisle_30 = 38.0584002656
_C685: 58.4543103448 T_HotAisle_29 + T_HotAisle_30 - 59.4543103448 T_Rack_29
 = 0
_C686: - E_TES_29 + E_TES_30 - 0.000225 Q_Charge_TES_Watts_30
 + 0.000277777777778 Q_Discharge_TES_Watts_30 = 0
_C687: - Q_Discharge_TES_Watts_29 + Q_Discharge_TES_Watts_30 <= 1000000
_C688: - Q_Charge_TES_Watts_29 + Q_Charge_TES_Watts_30 <= 1000000
_C689: P_Chiller_HVAC_Watts_30 + P_Chiller_TES_Watts_30 <= 400000
_C69: - 1000 P_IT_Total_3 + Q_Cool_Watts_3 >= 0
_C690: - 1000 P_IT_Total_30 + Q_Cool_Watts_30 >= 0
_C691: - 0.001 P_Chiller_HVAC_Watts_31 - 0.001 P_Chiller_TES_Watts_31
 + P_Cooling_kW_31 = 0
_C692: - P_Grid_IT_31 + P_IT_Total_31 - P_UPS_Discharge_31 = 0
_C693: P_Grid_Other_31 - 0.1 P_IT_Total_31 = 0
_C694: - P_Cooling_kW_31 - P_Grid_IT_31 - P_Grid_Other_31 + P_Grid_Total_31
 - P_UPS_Charge_31 = 0
_C695: 10000 IsActive_31 + P_Grid_Total_31 <= 10924.33931
_C696: - E_UPS_30 + E_UPS_31 - 0.205 P_UPS_Charge_31
 + 0.271739130435 P_UPS_Discharge_31 = 0
_C697: P_UPS_Charge_31 - 270 Z_Charge_31 <= 0
_C698: P_UPS_Charge_31 - 40 Z_Charge_31 >= 0
_C699: P_UPS_Discharge_31 - 2700 Z_Discharge_31 <= 0
_C7: P_UPS_Charge_1 - 270 Z_Charge_1 <= 0
_C70: - 0.001 P_Chiller_HVAC_Watts_4 - 0.001 P_Chiller_TES_Watts_4
 + P_Cooling_kW_4 = 0
_C700: P_UPS_Discharge_31 - 100 Z_Discharge_31 >= 0
_C701: Z_Charge_31 + Z_Discharge_31 <= 1
_C702: - 6 P_Chiller_HVAC_Watts_31 + Q_Cool_Watts_31
 - Q_Discharge_TES_Watts_31 = 0
_C703: - 6 P_Chiller_TES_Watts_31 + Q_Charge_TES_Watts_31 = 0
_C704: 9.94579541499e-06 Q_Cool_Watts_31 + T_Aisle_In_31 - T_HotAisle_30 = 0
_C705: - 0.0503355704698 P_IT_Total_31 + 4.48616573826 T_IT_30 + T_IT_31
 - 5.48616573826 T_Rack_30 = 0
_C706: - 3.84820916835 T_ColdAisle_30 - 5.4436888589 T_IT_30
 + 8.29189802725 T_Rack_30 + T_Rack_31 = 0
_C707: - 29.7271551724 T_Aisle_In_31 + 30.4570824572 T_ColdAisle_30
 + T_ColdAisle_31 = 38.0584002656
_C708: 58.4543103448 T_HotAisle_30 + T_HotAisle_31 - 59.4543103448 T_Rack_30
 = 0
_C709: - E_TES_30 + E_TES_31 - 0.000225 Q_Charge_TES_Watts_31
 + 0.000277777777778 Q_Discharge_TES_Watts_31 = 0
_C71: - P_Grid_IT_4 + P_IT_Total_4 - P_UPS_Discharge_4 = 0
_C710: - Q_Discharge_TES_Watts_30 + Q_Discharge_TES_Watts_31 <= 1000000
_C711: - Q_Charge_TES_Watts_30 + Q_Charge_TES_Watts_31 <= 1000000
_C712: P_Chiller_HVAC_Watts_31 + P_Chiller_TES_Watts_31 <= 400000
_C713: - 1000 P_IT_Total_31 + Q_Cool_Watts_31 >= 0
_C714: - 0.001 P_Chiller_HVAC_Watts_32 - 0.001 P_Chiller_TES_Watts_32
 + P_Cooling_kW_32 = 0
_C715: - P_Grid_IT_32 + P_IT_Total_32 - P_UPS_Discharge_32 = 0
_C716: P_Grid_Other_32 - 0.1 P_IT_Total_32 = 0
_C717: - P_Cooling_kW_32 - P_Grid_IT_32 - P_Grid_Other_32 + P_Grid_Total_32
 - P_UPS_Charge_32 = 0
_C718: 10000 IsActive_32 + P_Grid_Total_32 <= 10924.33931
_C719: - E_UPS_31 + E_UPS_32 - 0.205 P_UPS_Charge_32
 + 0.271739130435 P_UPS_Discharge_32 = 0
_C72: P_Grid_Other_4 - 0.1 P_IT_Total_4 = 0
_C720: P_UPS_Charge_32 - 270 Z_Charge_32 <= 0
_C721: P_UPS_Charge_32 - 40 Z_Charge_32 >= 0
_C722: P_UPS_Discharge_32 - 2700 Z_Discharge_32 <= 0
_C723: P_UPS_Discharge_32 - 100 Z_Discharge_32 >= 0
_C724: Z_Charge_32 + Z_Discharge_32 <= 1
_C725: - 6 P_Chiller_HVAC_Watts_32 + Q_Cool_Watts_32
 - Q_Discharge_TES_Watts_32 = 0
_C726: - 6 P_Chiller_TES_Watts_32 + Q_Charge_TES_Watts_32 = 0
_C727: 9.94579541499e-06 Q_Cool_Watts_32 + T_Aisle_In_32 - T_HotAisle_31 = 0
_C728: - 0.0503355704698 P_IT_Total_32 + 4.48616573826 T_IT_31 + T_IT_32
 - 5.48616573826 T_Rack_31 = 0
_C729: - 3.84820916835 T_ColdAisle_31 - 5.4436888589 T_IT_31
 + 8.29189802725 T_Rack_31 + T_Rack_32 = 0
_C73: - P_Cooling_kW_4 - P_Grid_IT_4 - P_Grid_Other_4 + P_Grid_Total_4
 - P_UPS_Charge_4 = 0
_C730: - 29.7271551724 T_Aisle_In_32 + 30.4570824572 T_ColdAisle_31
 + T_ColdAisle_32 = 38.0584002656
_C731: 58.4543103448 T_HotAisle_31 + T_HotAisle_32 - 59.4543103448 T_Rack_31
 = 0
_C732: - E_TES_31 + E_TES_32 - 0.000225 Q_Charge_TES_Watts_32
 + 0.000277777777778 Q_Discharge_TES_Watts_32 = 0
_C733: - Q_Discharge_TES_Watts_31 + Q_Discharge_TES_Watts_32 <= 1000000
_C734: - Q_Charge_TES_Watts_31 + Q_Charge_TES_Watts_32 <= 1000000
_C735: P_Chiller_HVAC_Watts_32 + P_Chiller_TES_Watts_32 <= 400000
_C736: - 1000 P_IT_Total_32 + Q_Cool_Watts_32 >= 0
_C737: - 0.001 P_Chiller_HVAC_Watts_33 - 0.001 P_Chiller_TES_Watts_33
 + P_Cooling_kW_33 = 0
_C738: - P_Grid_IT_33 + P_IT_Total_33 - P_UPS_Discharge_33 = 0
_C739: P_Grid_Other_33 - 0.1 P_IT_Total_33 = 0
_C74: 10000 IsActive_4 + P_Grid_Total_4 <= 10924.33931
_C740: - P_Cooling_kW_33 - P_Grid_IT_33 - P_Grid_Other_33 + P_Grid_Total_33
 - P_UPS_Charge_33 = 0
_C741: 10000 IsActive_33 + P_Grid_Total_33 <= 10924.33931
_C742: - E_UPS_32 + E_UPS_33 - 0.205 P_UPS_Charge_33
 + 0.271739130435 P_UPS_Discharge_33 = 0
_C743: P_UPS_Charge_33 - 270 Z_Charge_33 <= 0
_C744: P_UPS_Charge_33 - 40 Z_Charge_33 >= 0
_C745: P_UPS_Discharge_33 - 2700 Z_Discharge_33 <= 0
_C746: P_UPS_Discharge_33 - 100 Z_Discharge_33 >= 0
_C747: Z_Charge_33 + Z_Discharge_33 <= 1
_C748: - 6 P_Chiller_HVAC_Watts_33 + Q_Cool_Watts_33
 - Q_Discharge_TES_Watts_33 = 0
_C749: - 6 P_Chiller_TES_Watts_33 + Q_Charge_TES_Watts_33 = 0
_C75: - E_UPS_3 + E_UPS_4 - 0.205 P_UPS_Charge_4
 + 0.271739130435 P_UPS_Discharge_4 = 0
_C750: 9.94579541499e-06 Q_Cool_Watts_33 + T_Aisle_In_33 - T_HotAisle_32 = 0
_C751: - 0.0503355704698 P_IT_Total_33 + 4.48616573826 T_IT_32 + T_IT_33
 - 5.48616573826 T_Rack_32 = 0
_C752: - 3.84820916835 T_ColdAisle_32 - 5.4436888589 T_IT_32
 + 8.29189802725 T_Rack_32 + T_Rack_33 = 0
_C753: - 29.7271551724 T_Aisle_In_33 + 30.4570824572 T_ColdAisle_32
 + T_ColdAisle_33 = 38.0584002656
_C754: 58.4543103448 T_HotAisle_32 + T_HotAisle_33 - 59.4543103448 T_Rack_32
 = 0
_C755: - E_TES_32 + E_TES_33 - 0.000225 Q_Charge_TES_Watts_33
 + 0.000277777777778 Q_Discharge_TES_Watts_33 = 0
_C756: - Q_Discharge_TES_Watts_32 + Q_Discharge_TES_Watts_33 <= 1000000
_C757: - Q_Charge_TES_Watts_32 + Q_Charge_TES_Watts_33 <= 1000000
_C758: P_Chiller_HVAC_Watts_33 + P_Chiller_TES_Watts_33 <= 400000
_C759: - 1000 P_IT_Total_33 + Q_Cool_Watts_33 >= 0
_C76: P_UPS_Charge_4 - 270 Z_Charge_4 <= 0
_C760: - 0.001 P_Chiller_HVAC_Watts_34 - 0.001 P_Chiller_TES_Watts_34
 + P_Cooling_kW_34 = 0
_C761: - P_Grid_IT_34 + P_IT_Total_34 - P_UPS_Discharge_34 = 0
_C762: P_Grid_Other_34 - 0.1 P_IT_Total_34 = 0
_C763: - P_Cooling_kW_34 - P_Grid_IT_34 - P_Grid_Other_34 + P_Grid_Total_34
 - P_UPS_Charge_34 = 0
_C764: 10000 IsActive_34 + P_Grid_Total_34 <= 10924.33931
_C765: - E_UPS_33 + E_UPS_34 - 0.205 P_UPS_Charge_34
 + 0.271739130435 P_UPS_Discharge_34 = 0
_C766: P_UPS_Charge_34 - 270 Z_Charge_34 <= 0
_C767: P_UPS_Charge_34 - 40 Z_Charge_34 >= 0
_C768: P_UPS_Discharge_34 - 2700 Z_Discharge_34 <= 0
_C769: P_UPS_Discharge_34 - 100 Z_Discharge_34 >= 0
_C77: P_UPS_Charge_4 - 40 Z_Charge_4 >= 0
_C770: Z_Charge_34 + Z_Discharge_34 <= 1
_C771: - 6 P_Chiller_HVAC_Watts_34 + Q_Cool_Watts_34
 - Q_Discharge_TES_Watts_34 = 0
_C772: - 6 P_Chiller_TES_Watts_34 + Q_Charge_TES_Watts_34 = 0
_C773: 9.94579541499e-06 Q_Cool_Watts_34 + T_Aisle_In_34 - T_HotAisle_33 = 0
_C774: - 0.0503355704698 P_IT_Total_34 + 4.48616573826 T_IT_33 + T_IT_34
 - 5.48616573826 T_Rack_33 = 0
_C775: - 3.84820916835 T_ColdAisle_33 - 5.4436888589 T_IT_33
 + 8.29189802725 T_Rack_33 + T_Rack_34 = 0
_C776: - 29.7271551724 T_Aisle_In_34 + 30.4570824572 T_ColdAisle_33
 + T_ColdAisle_34 = 38.0584002656
_C777: 58.4543103448 T_HotAisle_33 + T_HotAisle_34 - 59.4543103448 T_Rack_33
 = 0
_C778: - E_TES_33 + E_TES_34 - 0.000225 Q_Charge_TES_Watts_34
 + 0.000277777777778 Q_Discharge_TES_Watts_34 = 0
_C779: - Q_Discharge_TES_Watts_33 + Q_Discharge_TES_Watts_34 <= 1000000
_C78: P_UPS_Discharge_4 - 2700 Z_Discharge_4 <= 0
_C780: - Q_Charge_TES_Watts_33 + Q_Charge_TES_Watts_34 <= 1000000
_C781: P_Chiller_HVAC_Watts_34 + P_Chiller_TES_Watts_34 <= 400000
_C782: - 1000 P_IT_Total_34 + Q_Cool_Watts_34 >= 0
_C783: - 0.001 P_Chiller_HVAC_Watts_35 - 0.001 P_Chiller_TES_Watts_35
 + P_Cooling_kW_35 = 0
_C784: - P_Grid_IT_35 + P_IT_Total_35 - P_UPS_Discharge_35 = 0
_C785: P_Grid_Other_35 - 0.1 P_IT_Total_35 = 0
_C786: - P_Cooling_kW_35 - P_Grid_IT_35 - P_Grid_Other_35 + P_Grid_Total_35
 - P_UPS_Charge_35 = 0
_C787: 10000 IsActive_35 + P_Grid_Total_35 <= 10924.33931
_C788: - E_UPS_34 + E_UPS_35 - 0.205 P_UPS_Charge_35
 + 0.271739130435 P_UPS_Discharge_35 = 0
_C789: P_UPS_Charge_35 - 270 Z_Charge_35 <= 0
_C79: P_UPS_Discharge_4 - 100 Z_Discharge_4 >= 0
_C790: P_UPS_Charge_35 - 40 Z_Charge_35 >= 0
_C791: P_UPS_Discharge_35 - 2700 Z_Discharge_35 <= 0
_C792: P_UPS_Discharge_35 - 100 Z_Discharge_35 >= 0
_C793: Z_Charge_35 + Z_Discharge_35 <= 1
_C794: - 6 P_Chiller_HVAC_Watts_35 + Q_Cool_Watts_35
 - Q_Discharge_TES_Watts_35 = 0
_C795: - 6 P_Chiller_TES_Watts_35 + Q_Charge_TES_Watts_35 = 0
_C796: 9.94579541499e-06 Q_Cool_Watts_35 + T_Aisle_In_35 - T_HotAisle_34 = 0
_C797: - 0.0503355704698 P_IT_Total_35 + 4.48616573826 T_IT_34 + T_IT_35
 - 5.48616573826 T_Rack_34 = 0
_C798: - 3.84820916835 T_ColdAisle_34 - 5.4436888589 T_IT_34
 + 8.29189802725 T_Rack_34 + T_Rack_35 = 0
_C799: - 29.7271551724 T_Aisle_In_35 + 30.4570824572 T_ColdAisle_34
 + T_ColdAisle_35 = 38.0584002656
_C8: P_UPS_Charge_1 - 40 Z_Charge_1 >= 0
_C80: Z_Charge_4 + Z_Discharge_4 <= 1
_C800: 58.4543103448 T_HotAisle_34 + T_HotAisle_35 - 59.4543103448 T_Rack_34
 = 0
_C801: - E_TES_34 + E_TES_35 - 0.000225 Q_Charge_TES_Watts_35
 + 0.000277777777778 Q_Discharge_TES_Watts_35 = 0
_C802: - Q_Discharge_TES_Watts_34 + Q_Discharge_TES_Watts_35 <= 1000000
_C803: - Q_Charge_TES_Watts_34 + Q_Charge_TES_Watts_35 <= 1000000
_C804: P_Chiller_HVAC_Watts_35 + P_Chiller_TES_Watts_35 <= 400000
_C805: - 1000 P_IT_Total_35 + Q_Cool_Watts_35 >= 0
_C806: - 0.001 P_Chiller_HVAC_Watts_36 - 0.001 P_Chiller_TES_Watts_36
 + P_Cooling_kW_36 = 0
_C807: - P_Grid_IT_36 + P_IT_Total_36 - P_UPS_Discharge_36 = 0
_C808: P_Grid_Other_36 - 0.1 P_IT_Total_36 = 0
_C809: - P_Cooling_kW_36 - P_Grid_IT_36 - P_Grid_Other_36 + P_Grid_Total_36
 - P_UPS_Charge_36 = 0
_C81: - 6 P_Chiller_HVAC_Watts_4 + Q_Cool_Watts_4 - Q_Discharge_TES_Watts_4
 = 0
_C810: 10000 IsActive_36 + P_Grid_Total_36 <= 10924.33931
_C811: - E_UPS_35 + E_UPS_36 - 0.205 P_UPS_Charge_36
 + 0.271739130435 P_UPS_Discharge_36 = 0
_C812: P_UPS_Charge_36 - 270 Z_Charge_36 <= 0
_C813: P_UPS_Charge_36 - 40 Z_Charge_36 >= 0
_C814: P_UPS_Discharge_36 - 2700 Z_Discharge_36 <= 0
_C815: P_UPS_Discharge_36 - 100 Z_Discharge_36 >= 0
_C816: Z_Charge_36 + Z_Discharge_36 <= 1
_C817: - 6 P_Chiller_HVAC_Watts_36 + Q_Cool_Watts_36
 - Q_Discharge_TES_Watts_36 = 0
_C818: - 6 P_Chiller_TES_Watts_36 + Q_Charge_TES_Watts_36 = 0
_C819: 9.94579541499e-06 Q_Cool_Watts_36 + T_Aisle_In_36 - T_HotAisle_35 = 0
_C82: - 6 P_Chiller_TES_Watts_4 + Q_Charge_TES_Watts_4 = 0
_C820: - 0.0503355704698 P_IT_Total_36 + 4.48616573826 T_IT_35 + T_IT_36
 - 5.48616573826 T_Rack_35 = 0
_C821: - 3.84820916835 T_ColdAisle_35 - 5.4436888589 T_IT_35
 + 8.29189802725 T_Rack_35 + T_Rack_36 = 0
_C822: - 29.7271551724 T_Aisle_In_36 + 30.4570824572 T_ColdAisle_35
 + T_ColdAisle_36 = 38.0584002656
_C823: 58.4543103448 T_HotAisle_35 + T_HotAisle_36 - 59.4543103448 T_Rack_35
 = 0
_C824: - E_TES_35 + E_TES_36 - 0.000225 Q_Charge_TES_Watts_36
 + 0.000277777777778 Q_Discharge_TES_Watts_36 = 0
_C825: - Q_Discharge_TES_Watts_35 + Q_Discharge_TES_Watts_36 <= 1000000
_C826: - Q_Charge_TES_Watts_35 + Q_Charge_TES_Watts_36 <= 1000000
_C827: P_Chiller_HVAC_Watts_36 + P_Chiller_TES_Watts_36 <= 400000
_C828: - 1000 P_IT_Total_36 + Q_Cool_Watts_36 >= 0
_C829: - 0.001 P_Chiller_HVAC_Watts_37 - 0.001 P_Chiller_TES_Watts_37
 + P_Cooling_kW_37 = 0
_C83: 9.94579541499e-06 Q_Cool_Watts_4 + T_Aisle_In_4 - T_HotAisle_3 = 0
_C830: - P_Grid_IT_37 + P_IT_Total_37 - P_UPS_Discharge_37 = 0
_C831: P_Grid_Other_37 - 0.1 P_IT_Total_37 = 0
_C832: - P_Cooling_kW_37 - P_Grid_IT_37 - P_Grid_Other_37 + P_Grid_Total_37
 - P_UPS_Charge_37 = 0
_C833: 10000 IsActive_37 + P_Grid_Total_37 <= 10924.33931
_C834: - E_UPS_36 + E_UPS_37 - 0.205 P_UPS_Charge_37
 + 0.271739130435 P_UPS_Discharge_37 = 0
_C835: P_UPS_Charge_37 - 270 Z_Charge_37 <= 0
_C836: P_UPS_Charge_37 - 40 Z_Charge_37 >= 0
_C837: P_UPS_Discharge_37 - 2700 Z_Discharge_37 <= 0
_C838: P_UPS_Discharge_37 - 100 Z_Discharge_37 >= 0
_C839: Z_Charge_37 + Z_Discharge_37 <= 1
_C84: - 0.0503355704698 P_IT_Total_4 + 4.48616573826 T_IT_3 + T_IT_4
 - 5.48616573826 T_Rack_3 = 0
_C840: - 6 P_Chiller_HVAC_Watts_37 + Q_Cool_Watts_37
 - Q_Discharge_TES_Watts_37 = 0
_C841: - 6 P_Chiller_TES_Watts_37 + Q_Charge_TES_Watts_37 = 0
_C842: 9.94579541499e-06 Q_Cool_Watts_37 + T_Aisle_In_37 - T_HotAisle_36 = 0
_C843: - 0.0503355704698 P_IT_Total_37 + 4.48616573826 T_IT_36 + T_IT_37
 - 5.48616573826 T_Rack_36 = 0
_C844: - 3.84820916835 T_ColdAisle_36 - 5.4436888589 T_IT_36
 + 8.29189802725 T_Rack_36 + T_Rack_37 = 0
_C845: - 29.7271551724 T_Aisle_In_37 + 30.4570824572 T_ColdAisle_36
 + T_ColdAisle_37 = 38.0584002656
_C846: 58.4543103448 T_HotAisle_36 + T_HotAisle_37 - 59.4543103448 T_Rack_36
 = 0
_C847: - E_TES_36 + E_TES_37 - 0.000225 Q_Charge_TES_Watts_37
 + 0.000277777777778 Q_Discharge_TES_Watts_37 = 0
_C848: - Q_Discharge_TES_Watts_36 + Q_Discharge_TES_Watts_37 <= 1000000
_C849: - Q_Charge_TES_Watts_36 + Q_Charge_TES_Watts_37 <= 1000000
_C85: - 3.84820916835 T_ColdAisle_3 - 5.4436888589 T_IT_3
 + 8.29189802725 T_Rack_3 + T_Rack_4 = 0
_C850: P_Chiller_HVAC_Watts_37 + P_Chiller_TES_Watts_37 <= 400000
_C851: - 1000 P_IT_Total_37 + Q_Cool_Watts_37 >= 0
_C852: - 0.001 P_Chiller_HVAC_Watts_38 - 0.001 P_Chiller_TES_Watts_38
 + P_Cooling_kW_38 = 0
_C853: - P_Grid_IT_38 + P_IT_Total_38 - P_UPS_Discharge_38 = 0
_C854: P_Grid_Other_38 - 0.1 P_IT_Total_38 = 0
_C855: - P_Cooling_kW_38 - P_Grid_IT_38 - P_Grid_Other_38 + P_Grid_Total_38
 - P_UPS_Charge_38 = 0
_C856: 10000 IsActive_38 + P_Grid_Total_38 <= 10924.33931
_C857: - E_UPS_37 + E_UPS_38 - 0.205 P_UPS_Charge_38
 + 0.271739130435 P_UPS_Discharge_38 = 0
_C858: P_UPS_Charge_38 - 270 Z_Charge_38 <= 0
_C859: P_UPS_Charge_38 - 40 Z_Charge_38 >= 0
_C86: - 29.7271551724 T_Aisle_In_4 + 30.4570824572 T_ColdAisle_3
 + T_ColdAisle_4 = 38.0584002656
_C860: P_UPS_Discharge_38 - 2700 Z_Discharge_38 <= 0
_C861: P_UPS_Discharge_38 - 100 Z_Discharge_38 >= 0
_C862: Z_Charge_38 + Z_Discharge_38 <= 1
_C863: - 6 P_Chiller_HVAC_Watts_38 + Q_Cool_Watts_38
 - Q_Discharge_TES_Watts_38 = 0
_C864: - 6 P_Chiller_TES_Watts_38 + Q_Charge_TES_Watts_38 = 0
_C865: 9.94579541499e-06 Q_Cool_Watts_38 + T_Aisle_In_38 - T_HotAisle_37 = 0
_C866: - 0.0503355704698 P_IT_Total_38 + 4.48616573826 T_IT_37 + T_IT_38
 - 5.48616573826 T_Rack_37 = 0
_C867: - 3.84820916835 T_ColdAisle_37 - 5.4436888589 T_IT_37
 + 8.29189802725 T_Rack_37 + T_Rack_38 = 0
_C868: - 29.7271551724 T_Aisle_In_38 + 30.4570824572 T_ColdAisle_37
 + T_ColdAisle_38 = 38.0584002656
_C869: 58.4543103448 T_HotAisle_37 + T_HotAisle_38 - 59.4543103448 T_Rack_37
 = 0
_C87: 58.4543103448 T_HotAisle_3 + T_HotAisle_4 - 59.4543103448 T_Rack_3 = 0
_C870: - E_TES_37 + E_TES_38 - 0.000225 Q_Charge_TES_Watts_38
 + 0.000277777777778 Q_Discharge_TES_Watts_38 = 0
_C871: - Q_Discharge_TES_Watts_37 + Q_Discharge_TES_Watts_38 <= 1000000
_C872: - Q_Charge_TES_Watts_37 + Q_Charge_TES_Watts_38 <= 1000000
_C873: P_Chiller_HVAC_Watts_38 + P_Chiller_TES_Watts_38 <= 400000
_C874: - 1000 P_IT_Total_38 + Q_Cool_Watts_38 >= 0
_C875: - 0.001 P_Chiller_HVAC_Watts_39 - 0.001 P_Chiller_TES_Watts_39
 + P_Cooling_kW_39 = 0
_C876: - P_Grid_IT_39 + P_IT_Total_39 - P_UPS_Discharge_39 = 0
_C877: P_Grid_Other_39 - 0.1 P_IT_Total_39 = 0
_C878: - P_Cooling_kW_39 - P_Grid_IT_39 - P_Grid_Other_39 + P_Grid_Total_39
 - P_UPS_Charge_39 = 0
_C879: 10000 IsActive_39 + P_Grid_Total_39 <= 10924.33931
_C88: - E_TES_3 + E_TES_4 - 0.000225 Q_Charge_TES_Watts_4
 + 0.000277777777778 Q_Discharge_TES_Watts_4 = 0
_C880: - E_UPS_38 + E_UPS_39 - 0.205 P_UPS_Charge_39
 + 0.271739130435 P_UPS_Discharge_39 = 0
_C881: P_UPS_Charge_39 - 270 Z_Charge_39 <= 0
_C882: P_UPS_Charge_39 - 40 Z_Charge_39 >= 0
_C883: P_UPS_Discharge_39 - 2700 Z_Discharge_39 <= 0
_C884: P_UPS_Discharge_39 - 100 Z_Discharge_39 >= 0
_C885: Z_Charge_39 + Z_Discharge_39 <= 1
_C886: - 6 P_Chiller_HVAC_Watts_39 + Q_Cool_Watts_39
 - Q_Discharge_TES_Watts_39 = 0
_C887: - 6 P_Chiller_TES_Watts_39 + Q_Charge_TES_Watts_39 = 0
_C888: 9.94579541499e-06 Q_Cool_Watts_39 + T_Aisle_In_39 - T_HotAisle_38 = 0
_C889: - 0.0503355704698 P_IT_Total_39 + 4.48616573826 T_IT_38 + T_IT_39
 - 5.48616573826 T_Rack_38 = 0
_C89: - Q_Discharge_TES_Watts_3 + Q_Discharge_TES_Watts_4 <= 1000000
_C890: - 3.84820916835 T_ColdAisle_38 - 5.4436888589 T_IT_38
 + 8.29189802725 T_Rack_38 + T_Rack_39 = 0
_C891: - 29.7271551724 T_Aisle_In_39 + 30.4570824572 T_ColdAisle_38
 + T_ColdAisle_39 = 38.0584002656
_C892: 58.4543103448 T_HotAisle_38 + T_HotAisle_39 - 59.4543103448 T_Rack_38
 = 0
_C893: - E_TES_38 + E_TES_39 - 0.000225 Q_Charge_TES_Watts_39
 + 0.000277777777778 Q_Discharge_TES_Watts_39 = 0
_C894: - Q_Discharge_TES_Watts_38 + Q_Discharge_TES_Watts_39 <= 1000000
_C895: - Q_Charge_TES_Watts_38 + Q_Charge_TES_Watts_39 <= 1000000
_C896: P_Chiller_HVAC_Watts_39 + P_Chiller_TES_Watts_39 <= 400000
_C897: - 1000 P_IT_Total_39 + Q_Cool_Watts_39 >= 0
_C898: - 0.001 P_Chiller_HVAC_Watts_40 - 0.001 P_Chiller_TES_Watts_40
 + P_Cooling_kW_40 = 0
_C899: - P_Grid_IT_40 + P_IT_Total_40 - P_UPS_Discharge_40 = 0
_C9: P_UPS_Discharge_1 - 2700 Z_Discharge_1 <= 0
_C90: - Q_Charge_TES_Watts_3 + Q_Charge_TES_Watts_4 <= 1000000
_C900: P_Grid_Other_40 - 0.1 P_IT_Total_40 = 0
_C901: - P_Cooling_kW_40 - P_Grid_IT_40 - P_Grid_Other_40 + P_Grid_Total_40
 - P_UPS_Charge_40 = 0
_C902: 10000 IsActive_40 + P_Grid_Total_40 <= 10924.33931
_C903: - E_UPS_39 + E_UPS_40 - 0.205 P_UPS_Charge_40
 + 0.271739130435 P_UPS_Discharge_40 = 0
_C904: P_UPS_Charge_40 - 270 Z_Charge_40 <= 0
_C905: P_UPS_Charge_40 - 40 Z_Charge_40 >= 0
_C906: P_UPS_Discharge_40 - 2700 Z_Discharge_40 <= 0
_C907: P_UPS_Discharge_40 - 100 Z_Discharge_40 >= 0
_C908: Z_Charge_40 + Z_Discharge_40 <= 1
_C909: - 6 P_Chiller_HVAC_Watts_40 + Q_Cool_Watts_40
 - Q_Discharge_TES_Watts_40 = 0
_C91: P_Chiller_HVAC_Watts_4 + P_Chiller_TES_Watts_4 <= 400000
_C910: - 6 P_Chiller_TES_Watts_40 + Q_Charge_TES_Watts_40 = 0
_C911: 9.94579541499e-06 Q_Cool_Watts_40 + T_Aisle_In_40 - T_HotAisle_39 = 0
_C912: - 0.0503355704698 P_IT_Total_40 + 4.48616573826 T_IT_39 + T_IT_40
 - 5.48616573826 T_Rack_39 = 0
_C913: - 3.84820916835 T_ColdAisle_39 - 5.4436888589 T_IT_39
 + 8.29189802725 T_Rack_39 + T_Rack_40 = 0
_C914: - 29.7271551724 T_Aisle_In_40 + 30.4570824572 T_ColdAisle_39
 + T_ColdAisle_40 = 38.0584002656
_C915: 58.4543103448 T_HotAisle_39 + T_HotAisle_40 - 59.4543103448 T_Rack_39
 = 0
_C916: - E_TES_39 + E_TES_40 - 0.000225 Q_Charge_TES_Watts_40
 + 0.000277777777778 Q_Discharge_TES_Watts_40 = 0
_C917: - Q_Discharge_TES_Watts_39 + Q_Discharge_TES_Watts_40 <= 1000000
_C918: - Q_Charge_TES_Watts_39 + Q_Charge_TES_Watts_40 <= 1000000
_C919: P_Chiller_HVAC_Watts_40 + P_Chiller_TES_Watts_40 <= 400000
_C92: - 1000 P_IT_Total_4 + Q_Cool_Watts_4 >= 0
_C920: - 1000 P_IT_Total_40 + Q_Cool_Watts_40 >= 0
_C921: - 0.001 P_Chiller_HVAC_Watts_41 - 0.001 P_Chiller_TES_Watts_41
 + P_Cooling_kW_41 = 0
_C922: - P_Grid_IT_41 + P_IT_Total_41 - P_UPS_Discharge_41 = 0
_C923: P_Grid_Other_41 - 0.1 P_IT_Total_41 = 0
_C924: - P_Cooling_kW_41 - P_Grid_IT_41 - P_Grid_Other_41 + P_Grid_Total_41
 - P_UPS_Charge_41 = 0
_C925: 10000 IsActive_41 + P_Grid_Total_41 <= 10924.33931
_C926: - E_UPS_40 + E_UPS_41 - 0.205 P_UPS_Charge_41
 + 0.271739130435 P_UPS_Discharge_41 = 0
_C927: P_UPS_Charge_41 - 270 Z_Charge_41 <= 0
_C928: P_UPS_Charge_41 - 40 Z_Charge_41 >= 0
_C929: P_UPS_Discharge_41 - 2700 Z_Discharge_41 <= 0
_C93: - 0.001 P_Chiller_HVAC_Watts_5 - 0.001 P_Chiller_TES_Watts_5
 + P_Cooling_kW_5 = 0
_C930: P_UPS_Discharge_41 - 100 Z_Discharge_41 >= 0
_C931: Z_Charge_41 + Z_Discharge_41 <= 1
_C932: - 6 P_Chiller_HVAC_Watts_41 + Q_Cool_Watts_41
 - Q_Discharge_TES_Watts_41 = 0
_C933: - 6 P_Chiller_TES_Watts_41 + Q_Charge_TES_Watts_41 = 0
_C934: 9.94579541499e-06 Q_Cool_Watts_41 + T_Aisle_In_41 - T_HotAisle_40 = 0
_C935: - 0.0503355704698 P_IT_Total_41 + 4.48616573826 T_IT_40 + T_IT_41
 - 5.48616573826 T_Rack_40 = 0
_C936: - 3.84820916835 T_ColdAisle_40 - 5.4436888589 T_IT_40
 + 8.29189802725 T_Rack_40 + T_Rack_41 = 0
_C937: - 29.7271551724 T_Aisle_In_41 + 30.4570824572 T_ColdAisle_40
 + T_ColdAisle_41 = 38.0584002656
_C938: 58.4543103448 T_HotAisle_40 + T_HotAisle_41 - 59.4543103448 T_Rack_40
 = 0
_C939: - E_TES_40 + E_TES_41 - 0.000225 Q_Charge_TES_Watts_41
 + 0.000277777777778 Q_Discharge_TES_Watts_41 = 0
_C94: - P_Grid_IT_5 + P_IT_Total_5 - P_UPS_Discharge_5 = 0
_C940: - Q_Discharge_TES_Watts_40 + Q_Discharge_TES_Watts_41 <= 1000000
_C941: - Q_Charge_TES_Watts_40 + Q_Charge_TES_Watts_41 <= 1000000
_C942: P_Chiller_HVAC_Watts_41 + P_Chiller_TES_Watts_41 <= 400000
_C943: - 1000 P_IT_Total_41 + Q_Cool_Watts_41 >= 0
_C944: - 0.001 P_Chiller_HVAC_Watts_42 - 0.001 P_Chiller_TES_Watts_42
 + P_Cooling_kW_42 = 0
_C945: - P_Grid_IT_42 + P_IT_Total_42 - P_UPS_Discharge_42 = 0
_C946: P_Grid_Other_42 - 0.1 P_IT_Total_42 = 0
_C947: - P_Cooling_kW_42 - P_Grid_IT_42 - P_Grid_Other_42 + P_Grid_Total_42
 - P_UPS_Charge_42 = 0
_C948: 10000 IsActive_42 + P_Grid_Total_42 <= 10924.33931
_C949: - E_UPS_41 + E_UPS_42 - 0.205 P_UPS_Charge_42
 + 0.271739130435 P_UPS_Discharge_42 = 0
_C95: P_Grid_Other_5 - 0.1 P_IT_Total_5 = 0
_C950: P_UPS_Charge_42 - 270 Z_Charge_42 <= 0
_C951: P_UPS_Charge_42 - 40 Z_Charge_42 >= 0
_C952: P_UPS_Discharge_42 - 2700 Z_Discharge_42 <= 0
_C953: P_UPS_Discharge_42 - 100 Z_Discharge_42 >= 0
_C954: Z_Charge_42 + Z_Discharge_42 <= 1
_C955: - 6 P_Chiller_HVAC_Watts_42 + Q_Cool_Watts_42
 - Q_Discharge_TES_Watts_42 = 0
_C956: - 6 P_Chiller_TES_Watts_42 + Q_Charge_TES_Watts_42 = 0
_C957: 9.94579541499e-06 Q_Cool_Watts_42 + T_Aisle_In_42 - T_HotAisle_41 = 0
_C958: - 0.0503355704698 P_IT_Total_42 + 4.48616573826 T_IT_41 + T_IT_42
 - 5.48616573826 T_Rack_41 = 0
_C959: - 3.84820916835 T_ColdAisle_41 - 5.4436888589 T_IT_41
 + 8.29189802725 T_Rack_41 + T_Rack_42 = 0
_C96: - P_Cooling_kW_5 - P_Grid_IT_5 - P_Grid_Other_5 + P_Grid_Total_5
 - P_UPS_Charge_5 = 0
_C960: - 29.7271551724 T_Aisle_In_42 + 30.4570824572 T_ColdAisle_41
 + T_ColdAisle_42 = 38.0584002656
_C961: 58.4543103448 T_HotAisle_41 + T_HotAisle_42 - 59.4543103448 T_Rack_41
 = 0
_C962: - E_TES_41 + E_TES_42 - 0.000225 Q_Charge_TES_Watts_42
 + 0.000277777777778 Q_Discharge_TES_Watts_42 = 0
_C963: - Q_Discharge_TES_Watts_41 + Q_Discharge_TES_Watts_42 <= 1000000
_C964: - Q_Charge_TES_Watts_41 + Q_Charge_TES_Watts_42 <= 1000000
_C965: P_Chiller_HVAC_Watts_42 + P_Chiller_TES_Watts_42 <= 400000
_C966: - 1000 P_IT_Total_42 + Q_Cool_Watts_42 >= 0
_C967: - 0.001 P_Chiller_HVAC_Watts_43 - 0.001 P_Chiller_TES_Watts_43
 + P_Cooling_kW_43 = 0
_C968: - P_Grid_IT_43 + P_IT_Total_43 - P_UPS_Discharge_43 = 0
_C969: P_Grid_Other_43 - 0.1 P_IT_Total_43 = 0
_C97: 10000 IsActive_5 + P_Grid_Total_5 <= 10924.33931
_C970: - P_Cooling_kW_43 - P_Grid_IT_43 - P_Grid_Other_43 + P_Grid_Total_43
 - P_UPS_Charge_43 = 0
_C971: 10000 IsActive_43 + P_Grid_Total_43 <= 10924.33931
_C972: - E_UPS_42 + E_UPS_43 - 0.205 P_UPS_Charge_43
 + 0.271739130435 P_UPS_Discharge_43 = 0
_C973: P_UPS_Charge_43 - 270 Z_Charge_43 <= 0
_C974: P_UPS_Charge_43 - 40 Z_Charge_43 >= 0
_C975: P_UPS_Discharge_43 - 2700 Z_Discharge_43 <= 0
_C976: P_UPS_Discharge_43 - 100 Z_Discharge_43 >= 0
_C977: Z_Charge_43 + Z_Discharge_43 <= 1
_C978: - 6 P_Chiller_HVAC_Watts_43 + Q_Cool_Watts_43
 - Q_Discharge_TES_Watts_43 = 0
_C979: - 6 P_Chiller_TES_Watts_43 + Q_Charge_TES_Watts_43 = 0
_C98: - E_UPS_4 + E_UPS_5 - 0.205 P_UPS_Charge_5
 + 0.271739130435 P_UPS_Discharge_5 = 0
_C980: 9.94579541499e-06 Q_Cool_Watts_43 + T_Aisle_In_43 - T_HotAisle_42 = 0
_C981: - 0.0503355704698 P_IT_Total_43 + 4.48616573826 T_IT_42 + T_IT_43
 - 5.48616573826 T_Rack_42 = 0
_C982: - 3.84820916835 T_ColdAisle_42 - 5.4436888589 T_IT_42
 + 8.29189802725 T_Rack_42 + T_Rack_43 = 0
_C983: - 29.7271551724 T_Aisle_In_43 + 30.4570824572 T_ColdAisle_42
 + T_ColdAisle_43 = 38.0584002656
_C984: 58.4543103448 T_HotAisle_42 + T_HotAisle_43 - 59.4543103448 T_Rack_42
 = 0
_C985: - E_TES_42 + E_TES_43 - 0.000225 Q_Charge_TES_Watts_43
 + 0.000277777777778 Q_Discharge_TES_Watts_43 = 0
_C986: - Q_Discharge_TES_Watts_42 + Q_Discharge_TES_Watts_43 <= 1000000
_C987: - Q_Charge_TES_Watts_42 + Q_Charge_TES_Watts_43 <= 1000000
_C988: P_Chiller_HVAC_Watts_43 + P_Chiller_TES_Watts_43 <= 400000
_C989: - 1000 P_IT_Total_43 + Q_Cool_Watts_43 >= 0
_C99: P_UPS_Charge_5 - 270 Z_Charge_5 <= 0
_C990: - 0.001 P_Chiller_HVAC_Watts_44 - 0.001 P_Chiller_TES_Watts_44
 + P_Cooling_kW_44 = 0
_C991: - P_Grid_IT_44 + P_IT_Total_44 - P_UPS_Discharge_44 = 0
_C992: P_Grid_Other_44 - 0.1 P_IT_Total_44 = 0
_C993: - P_Cooling_kW_44 - P_Grid_IT_44 - P_Grid_Other_44 + P_Grid_Total_44
 - P_UPS_Charge_44 = 0
_C994: 10000 IsActive_44 + P_Grid_Total_44 <= 10924.33931
_C995: - E_UPS_43 + E_UPS_44 - 0.205 P_UPS_Charge_44
 + 0.271739130435 P_UPS_Discharge_44 = 0
_C996: P_UPS_Charge_44 - 270 Z_Charge_44 <= 0
_C997: P_UPS_Charge_44 - 40 Z_Charge_44 >= 0
_C998: P_UPS_Discharge_44 - 2700 Z_Discharge_44 <= 0
_C999: P_UPS_Discharge_44 - 100 Z_Discharge_44 >= 0
Bounds
 E_TES_1 <= 1000
 E_TES_10 <= 1000
 E_TES_11 <= 1000
 E_TES_12 <= 1000
 E_TES_13 <= 1000
 E_TES_14 <= 1000
 E_TES_15 <= 1000
 E_TES_16 <= 1000
 E_TES_17 <= 1000
 E_TES_18 <= 1000
 E_TES_19 <= 1000
 E_TES_2 <= 1000
 E_TES_20 <= 1000
 E_TES_21 <= 1000
 E_TES_22 <= 1000
 E_TES_23 <= 1000
 E_TES_24 <= 1000
 E_TES_25 <= 1000
 E_TES_26 <= 1000
 E_TES_27 <= 1000
 E_TES_28 <= 1000
 E_TES_29 <= 1000
 E_TES_3 <= 1000
 E_TES_30 <= 1000
 E_TES_31 <= 1000
 E_TES_32 <= 1000
 E_TES_33 <= 1000
 E_TES_34 <= 1000
 E_TES_35 <= 1000
 E_TES_36 <= 1000
 E_TES_37 <= 1000
 E_TES_38 <= 1000
 E_TES_39 <= 1000
 E_TES_4 <= 1000
 E_TES_40 <= 1000
 E_TES_41 <= 1000
 E_TES_42 <= 1000
 E_TES_43 <= 1000
 E_TES_44 <= 1000
 E_TES_45 <= 1000
 E_TES_46 <= 1000
 E_TES_47 <= 1000
 E_TES_48 <= 1000
 E_TES_5 <= 1000
 E_TES_6 <= 1000
 E_TES_7 <= 1000
 E_TES_8 <= 1000
 E_TES_9 <= 1000
 300 <= E_UPS_1 <= 600
 300 <= E_UPS_10 <= 600
 300 <= E_UPS_11 <= 600
 300 <= E_UPS_12 <= 600
 300 <= E_UPS_13 <= 600
 300 <= E_UPS_14 <= 600
 300 <= E_UPS_15 <= 600
 300 <= E_UPS_16 <= 600
 300 <= E_UPS_17 <= 600
 300 <= E_UPS_18 <= 600
 300 <= E_UPS_19 <= 600
 300 <= E_UPS_2 <= 600
 300 <= E_UPS_20 <= 600
 300 <= E_UPS_21 <= 600
 300 <= E_UPS_22 <= 600
 300 <= E_UPS_23 <= 600
 300 <= E_UPS_24 <= 600
 300 <= E_UPS_25 <= 600
 300 <= E_UPS_26 <= 600
 300 <= E_UPS_27 <= 600
 300 <= E_UPS_28 <= 600
 300 <= E_UPS_29 <= 600
 300 <= E_UPS_3 <= 600
 300 <= E_UPS_30 <= 600
 300 <= E_UPS_31 <= 600
 300 <= E_UPS_32 <= 600
 300 <= E_UPS_33 <= 600
 300 <= E_UPS_34 <= 600
 300 <= E_UPS_35 <= 600
 300 <= E_UPS_36 <= 600
 300 <= E_UPS_37 <= 600
 300 <= E_UPS_38 <= 600
 300 <= E_UPS_39 <= 600
 300 <= E_UPS_4 <= 600
 300 <= E_UPS_40 <= 600
 300 <= E_UPS_41 <= 600
 300 <= E_UPS_42 <= 600
 300 <= E_UPS_43 <= 600
 300 <= E_UPS_44 <= 600
 300 <= E_UPS_45 <= 600
 300 <= E_UPS_46 <= 600
 300 <= E_UPS_47 <= 600
 300 <= E_UPS_48 <= 600
 300 <= E_UPS_5 <= 600
 300 <= E_UPS_6 <= 600
 300 <= E_UPS_7 <= 600
 300 <= E_UPS_8 <= 600
 300 <= E_UPS_9 <= 600
 166.7 <= P_IT_Total_1 <= 1000
 166.7 <= P_IT_Total_10 <= 1000
 166.7 <= P_IT_Total_11 <= 1000
 166.7 <= P_IT_Total_12 <= 1000
 166.7 <= P_IT_Total_13 <= 1000
 166.7 <= P_IT_Total_14 <= 1000
 166.7 <= P_IT_Total_15 <= 1000
 166.7 <= P_IT_Total_16 <= 1000
 166.7 <= P_IT_Total_17 <= 1000
 166.7 <= P_IT_Total_18 <= 1000
 166.7 <= P_IT_Total_19 <= 1000
 166.7 <= P_IT_Total_2 <= 1000
 166.7 <= P_IT_Total_20 <= 1000
 166.7 <= P_IT_Total_21 <= 1000
 166.7 <= P_IT_Total_22 <= 1000
 166.7 <= P_IT_Total_23 <= 1000
 166.7 <= P_IT_Total_24 <= 1000
 166.7 <= P_IT_Total_25 <= 1000
 166.7 <= P_IT_Total_26 <= 1000
 166.7 <= P_IT_Total_27 <= 1000
 166.7 <= P_IT_Total_28 <= 1000
 166.7 <= P_IT_Total_29 <= 1000
 166.7 <= P_IT_Total_3 <= 1000
 166.7 <= P_IT_Total_30 <= 1000
 166.7 <= P_IT_Total_31 <= 1000
 166.7 <= P_IT_Total_32 <= 1000
 166.7 <= P_IT_Total_33 <= 1000
 166.7 <= P_IT_Total_34 <= 1000
 166.7 <= P_IT_Total_35 <= 1000
 166.7 <= P_IT_Total_36 <= 1000
 166.7 <= P_IT_Total_37 <= 1000
 166.7 <= P_IT_Total_38 <= 1000
 166.7 <= P_IT_Total_39 <= 1000
 166.7 <= P_IT_Total_4 <= 1000
 166.7 <= P_IT_Total_40 <= 1000
 166.7 <= P_IT_Total_41 <= 1000
 166.7 <= P_IT_Total_42 <= 1000
 166.7 <= P_IT_Total_43 <= 1000
 166.7 <= P_IT_Total_44 <= 1000
 166.7 <= P_IT_Total_45 <= 1000
 166.7 <= P_IT_Total_46 <= 1000
 166.7 <= P_IT_Total_47 <= 1000
 166.7 <= P_IT_Total_48 <= 1000
 166.7 <= P_IT_Total_5 <= 1000
 166.7 <= P_IT_Total_6 <= 1000
 166.7 <= P_IT_Total_7 <= 1000
 166.7 <= P_IT_Total_8 <= 1000
 166.7 <= P_IT_Total_9 <= 1000
 Q_Charge_TES_Watts_1 <= 300000
 Q_Charge_TES_Watts_10 <= 300000
 Q_Charge_TES_Watts_11 <= 300000
 Q_Charge_TES_Watts_12 <= 300000
 Q_Charge_TES_Watts_13 <= 300000
 Q_Charge_TES_Watts_14 <= 300000
 Q_Charge_TES_Watts_15 <= 300000
 Q_Charge_TES_Watts_16 <= 300000
 Q_Charge_TES_Watts_17 <= 300000
 Q_Charge_TES_Watts_18 <= 300000
 Q_Charge_TES_Watts_19 <= 300000
 Q_Charge_TES_Watts_2 <= 300000
 Q_Charge_TES_Watts_20 <= 300000
 Q_Charge_TES_Watts_21 <= 300000
 Q_Charge_TES_Watts_22 <= 300000
 Q_Charge_TES_Watts_23 <= 300000
 Q_Charge_TES_Watts_24 <= 300000
 Q_Charge_TES_Watts_25 <= 300000
 Q_Charge_TES_Watts_26 <= 300000
 Q_Charge_TES_Watts_27 <= 300000
 Q_Charge_TES_Watts_28 <= 300000
 Q_Charge_TES_Watts_29 <= 300000
 Q_Charge_TES_Watts_3 <= 300000
 Q_Charge_TES_Watts_30 <= 300000
 Q_Charge_TES_Watts_31 <= 300000
 Q_Charge_TES_Watts_32 <= 300000
 Q_Charge_TES_Watts_33 <= 300000
 Q_Charge_TES_Watts_34 <= 300000
 Q_Charge_TES_Watts_35 <= 300000
 Q_Charge_TES_Watts_36 <= 300000
 Q_Charge_TES_Watts_37 <= 300000
 Q_Charge_TES_Watts_38 <= 300000
 Q_Charge_TES_Watts_39 <= 300000
 Q_Charge_TES_Watts_4 <= 300000
 Q_Charge_TES_Watts_40 <= 300000
 Q_Charge_TES_Watts_41 <= 300000
 Q_Charge_TES_Watts_42 <= 300000
 Q_Charge_TES_Watts_43 <= 300000
 Q_Charge_TES_Watts_44 <= 300000
 Q_Charge_TES_Watts_45 <= 300000
 Q_Charge_TES_Watts_46 <= 300000
 Q_Charge_TES_Watts_47 <= 300000
 Q_Charge_TES_Watts_48 <= 300000
 Q_Charge_TES_Watts_5 <= 300000
 Q_Charge_TES_Watts_6 <= 300000
 Q_Charge_TES_Watts_7 <= 300000
 Q_Charge_TES_Watts_8 <= 300000
 Q_Charge_TES_Watts_9 <= 300000
 Q_Discharge_TES_Watts_1 <= 300000
 Q_Discharge_TES_Watts_10 <= 300000
 Q_Discharge_TES_Watts_11 <= 300000
 Q_Discharge_TES_Watts_12 <= 300000
 Q_Discharge_TES_Watts_13 <= 300000
 Q_Discharge_TES_Watts_14 <= 300000
 Q_Discharge_TES_Watts_15 <= 300000
 Q_Discharge_TES_Watts_16 <= 300000
 Q_Discharge_TES_Watts_17 <= 300000
 Q_Discharge_TES_Watts_18 <= 300000
 Q_Discharge_TES_Watts_19 <= 300000
 Q_Discharge_TES_Watts_2 <= 300000
 Q_Discharge_TES_Watts_20 <= 300000
 Q_Discharge_TES_Watts_21 <= 300000
 Q_Discharge_TES_Watts_22 <= 300000
 Q_Discharge_TES_Watts_23 <= 300000
 Q_Discharge_TES_Watts_24 <= 300000
 Q_Discharge_TES_Watts_25 <= 300000
 Q_Discharge_TES_Watts_26 <= 300000
 Q_Discharge_TES_Watts_27 <= 300000
 Q_Discharge_TES_Watts_28 <= 300000
 Q_Discharge_TES_Watts_29 <= 300000
 Q_Discharge_TES_Watts_3 <= 300000
 Q_Discharge_TES_Watts_30 <= 300000
 Q_Discharge_TES_Watts_31 <= 300000
 Q_Discharge_TES_Watts_32 <= 300000
 Q_Discharge_TES_Watts_33 <= 300000
 Q_Discharge_TES_Watts_34 <= 300000
 Q_Discharge_TES_Watts_35 <= 300000
 Q_Discharge_TES_Watts_36 <= 300000
 Q_Discharge_TES_Watts_37 <= 300000
 Q_Discharge_TES_Watts_38 <= 300000
 Q_Discharge_TES_Watts_39 <= 300000
 Q_Discharge_TES_Watts_4 <= 300000
 Q_Discharge_TES_Watts_40 <= 300000
 Q_Discharge_TES_Watts_41 <= 300000
 Q_Discharge_TES_Watts_42 <= 300000
 Q_Discharge_TES_Watts_43 <= 300000
 Q_Discharge_TES_Watts_44 <= 300000
 Q_Discharge_TES_Watts_45 <= 300000
 Q_Discharge_TES_Watts_46 <= 300000
 Q_Discharge_TES_Watts_47 <= 300000
 Q_Discharge_TES_Watts_48 <= 300000
 Q_Discharge_TES_Watts_5 <= 300000
 Q_Discharge_TES_Watts_6 <= 300000
 Q_Discharge_TES_Watts_7 <= 300000
 Q_Discharge_TES_Watts_8 <= 300000
 Q_Discharge_TES_Watts_9 <= 300000
 14 <= T_Aisle_In_1 <= 30
 14 <= T_Aisle_In_10 <= 30
 14 <= T_Aisle_In_11 <= 30
 14 <= T_Aisle_In_12 <= 30
 14 <= T_Aisle_In_13 <= 30
 14 <= T_Aisle_In_14 <= 30
 14 <= T_Aisle_In_15 <= 30
 14 <= T_Aisle_In_16 <= 30
 14 <= T_Aisle_In_17 <= 30
 14 <= T_Aisle_In_18 <= 30
 14 <= T_Aisle_In_19 <= 30
 14 <= T_Aisle_In_2 <= 30
 14 <= T_Aisle_In_20 <= 30
 14 <= T_Aisle_In_21 <= 30
 14 <= T_Aisle_In_22 <= 30
 14 <= T_Aisle_In_23 <= 30
 14 <= T_Aisle_In_24 <= 30
 14 <= T_Aisle_In_25 <= 30
 14 <= T_Aisle_In_26 <= 30
 14 <= T_Aisle_In_27 <= 30
 14 <= T_Aisle_In_28 <= 30
 14 <= T_Aisle_In_29 <= 30
 14 <= T_Aisle_In_3 <= 30
 14 <= T_Aisle_In_30 <= 30
 14 <= T_Aisle_In_31 <= 30
 14 <= T_Aisle_In_32 <= 30
 14 <= T_Aisle_In_33 <= 30
 14 <= T_Aisle_In_34 <= 30
 14 <= T_Aisle_In_35 <= 30
 14 <= T_Aisle_In_36 <= 30
 14 <= T_Aisle_In_37 <= 30
 14 <= T_Aisle_In_38 <= 30
 14 <= T_Aisle_In_39 <= 30
 14 <= T_Aisle_In_4 <= 30
 14 <= T_Aisle_In_40 <= 30
 14 <= T_Aisle_In_41 <= 30
 14 <= T_Aisle_In_42 <= 30
 14 <= T_Aisle_In_43 <= 30
 14 <= T_Aisle_In_44 <= 30
 14 <= T_Aisle_In_45 <= 30
 14 <= T_Aisle_In_46 <= 30
 14 <= T_Aisle_In_47 <= 30
 14 <= T_Aisle_In_48 <= 30
 14 <= T_Aisle_In_5 <= 30
 14 <= T_Aisle_In_6 <= 30
 14 <= T_Aisle_In_7 <= 30
 14 <= T_Aisle_In_8 <= 30
 14 <= T_Aisle_In_9 <= 30
 18 <= T_ColdAisle_1 <= 22
 18 <= T_ColdAisle_10 <= 22
 18 <= T_ColdAisle_11 <= 22
 18 <= T_ColdAisle_12 <= 22
 18 <= T_ColdAisle_13 <= 22
 18 <= T_ColdAisle_14 <= 22
 18 <= T_ColdAisle_15 <= 22
 18 <= T_ColdAisle_16 <= 22
 18 <= T_ColdAisle_17 <= 22
 18 <= T_ColdAisle_18 <= 22
 18 <= T_ColdAisle_19 <= 22
 18 <= T_ColdAisle_2 <= 22
 18 <= T_ColdAisle_20 <= 22
 18 <= T_ColdAisle_21 <= 22
 18 <= T_ColdAisle_22 <= 22
 18 <= T_ColdAisle_23 <= 22
 18 <= T_ColdAisle_24 <= 22
 18 <= T_ColdAisle_25 <= 22
 18 <= T_ColdAisle_26 <= 22
 18 <= T_ColdAisle_27 <= 22
 18 <= T_ColdAisle_28 <= 22
 18 <= T_ColdAisle_29 <= 22
 18 <= T_ColdAisle_3 <= 22
 18 <= T_ColdAisle_30 <= 22
 18 <= T_ColdAisle_31 <= 22
 18 <= T_ColdAisle_32 <= 22
 18 <= T_ColdAisle_33 <= 22
 18 <= T_ColdAisle_34 <= 22
 18 <= T_ColdAisle_35 <= 22
 18 <= T_ColdAisle_36 <= 22
 18 <= T_ColdAisle_37 <= 22
 18 <= T_ColdAisle_38 <= 22
 18 <= T_ColdAisle_39 <= 22
 18 <= T_ColdAisle_4 <= 22
 18 <= T_ColdAisle_40 <= 22
 18 <= T_ColdAisle_41 <= 22
 18 <= T_ColdAisle_42 <= 22
 18 <= T_ColdAisle_43 <= 22
 18 <= T_ColdAisle_44 <= 22
 18 <= T_ColdAisle_45 <= 22
 18 <= T_ColdAisle_46 <= 22
 18 <= T_ColdAisle_47 <= 22
 18 <= T_ColdAisle_48 <= 22
 18 <= T_ColdAisle_5 <= 22
 18 <= T_ColdAisle_6 <= 22
 18 <= T_ColdAisle_7 <= 22
 18 <= T_ColdAisle_8 <= 22
 18 <= T_ColdAisle_9 <= 22
 14 <= T_HotAisle_1 <= 40
 14 <= T_HotAisle_10 <= 40
 14 <= T_HotAisle_11 <= 40
 14 <= T_HotAisle_12 <= 40
 14 <= T_HotAisle_13 <= 40
 14 <= T_HotAisle_14 <= 40
 14 <= T_HotAisle_15 <= 40
 14 <= T_HotAisle_16 <= 40
 14 <= T_HotAisle_17 <= 40
 14 <= T_HotAisle_18 <= 40
 14 <= T_HotAisle_19 <= 40
 14 <= T_HotAisle_2 <= 40
 14 <= T_HotAisle_20 <= 40
 14 <= T_HotAisle_21 <= 40
 14 <= T_HotAisle_22 <= 40
 14 <= T_HotAisle_23 <= 40
 14 <= T_HotAisle_24 <= 40
 14 <= T_HotAisle_25 <= 40
 14 <= T_HotAisle_26 <= 40
 14 <= T_HotAisle_27 <= 40
 14 <= T_HotAisle_28 <= 40
 14 <= T_HotAisle_29 <= 40
 14 <= T_HotAisle_3 <= 40
 14 <= T_HotAisle_30 <= 40
 14 <= T_HotAisle_31 <= 40
 14 <= T_HotAisle_32 <= 40
 14 <= T_HotAisle_33 <= 40
 14 <= T_HotAisle_34 <= 40
 14 <= T_HotAisle_35 <= 40
 14 <= T_HotAisle_36 <= 40
 14 <= T_HotAisle_37 <= 40
 14 <= T_HotAisle_38 <= 40
 14 <= T_HotAisle_39 <= 40
 14 <= T_HotAisle_4 <= 40
 14 <= T_HotAisle_40 <= 40
 14 <= T_HotAisle_41 <= 40
 14 <= T_HotAisle_42 <= 40
 14 <= T_HotAisle_43 <= 40
 14 <= T_HotAisle_44 <= 40
 14 <= T_HotAisle_45 <= 40
 14 <= T_HotAisle_46 <= 40
 14 <= T_HotAisle_47 <= 40
 14 <= T_HotAisle_48 <= 40
 14 <= T_HotAisle_5 <= 40
 14 <= T_HotAisle_6 <= 40
 14 <= T_HotAisle_7 <= 40
 14 <= T_HotAisle_8 <= 40
 14 <= T_HotAisle_9 <= 40
 14 <= T_IT_1 <= 60
 14 <= T_IT_10 <= 60
 14 <= T_IT_11 <= 60
 14 <= T_IT_12 <= 60
 14 <= T_IT_13 <= 60
 14 <= T_IT_14 <= 60
 14 <= T_IT_15 <= 60
 14 <= T_IT_16 <= 60
 14 <= T_IT_17 <= 60
 14 <= T_IT_18 <= 60
 14 <= T_IT_19 <= 60
 14 <= T_IT_2 <= 60
 14 <= T_IT_20 <= 60
 14 <= T_IT_21 <= 60
 14 <= T_IT_22 <= 60
 14 <= T_IT_23 <= 60
 14 <= T_IT_24 <= 60
 14 <= T_IT_25 <= 60
 14 <= T_IT_26 <= 60
 14 <= T_IT_27 <= 60
 14 <= T_IT_28 <= 60
 14 <= T_IT_29 <= 60
 14 <= T_IT_3 <= 60
 14 <= T_IT_30 <= 60
 14 <= T_IT_31 <= 60
 14 <= T_IT_32 <= 60
 14 <= T_IT_33 <= 60
 14 <= T_IT_34 <= 60
 14 <= T_IT_35 <= 60
 14 <= T_IT_36 <= 60
 14 <= T_IT_37 <= 60
 14 <= T_IT_38 <= 60
 14 <= T_IT_39 <= 60
 14 <= T_IT_4 <= 60
 14 <= T_IT_40 <= 60
 14 <= T_IT_41 <= 60
 14 <= T_IT_42 <= 60
 14 <= T_IT_43 <= 60
 14 <= T_IT_44 <= 60
 14 <= T_IT_45 <= 60
 14 <= T_IT_46 <= 60
 14 <= T_IT_47 <= 60
 14 <= T_IT_48 <= 60
 14 <= T_IT_5 <= 60
 14 <= T_IT_6 <= 60
 14 <= T_IT_7 <= 60
 14 <= T_IT_8 <= 60
 14 <= T_IT_9 <= 60
 14 <= T_Rack_1 <= 40
 14 <= T_Rack_10 <= 40
 14 <= T_Rack_11 <= 40
 14 <= T_Rack_12 <= 40
 14 <= T_Rack_13 <= 40
 14 <= T_Rack_14 <= 40
 14 <= T_Rack_15 <= 40
 14 <= T_Rack_16 <= 40
 14 <= T_Rack_17 <= 40
 14 <= T_Rack_18 <= 40
 14 <= T_Rack_19 <= 40
 14 <= T_Rack_2 <= 40
 14 <= T_Rack_20 <= 40
 14 <= T_Rack_21 <= 40
 14 <= T_Rack_22 <= 40
 14 <= T_Rack_23 <= 40
 14 <= T_Rack_24 <= 40
 14 <= T_Rack_25 <= 40
 14 <= T_Rack_26 <= 40
 14 <= T_Rack_27 <= 40
 14 <= T_Rack_28 <= 40
 14 <= T_Rack_29 <= 40
 14 <= T_Rack_3 <= 40
 14 <= T_Rack_30 <= 40
 14 <= T_Rack_31 <= 40
 14 <= T_Rack_32 <= 40
 14 <= T_Rack_33 <= 40
 14 <= T_Rack_34 <= 40
 14 <= T_Rack_35 <= 40
 14 <= T_Rack_36 <= 40
 14 <= T_Rack_37 <= 40
 14 <= T_Rack_38 <= 40
 14 <= T_Rack_39 <= 40
 14 <= T_Rack_4 <= 40
 14 <= T_Rack_40 <= 40
 14 <= T_Rack_41 <= 40
 14 <= T_Rack_42 <= 40
 14 <= T_Rack_43 <= 40
 14 <= T_Rack_44 <= 40
 14 <= T_Rack_45 <= 40
 14 <= T_Rack_46 <= 40
 14 <= T_Rack_47 <= 40
 14 <= T_Rack_48 <= 40
 14 <= T_Rack_5 <= 40
 14 <= T_Rack_6 <= 40
 14 <= T_Rack_7 <= 40
 14 <= T_Rack_8 <= 40
 14 <= T_Rack_9 <= 40
Binaries
IsActive_1
IsActive_10
IsActive_11
IsActive_12
IsActive_13
IsActive_14
IsActive_15
IsActive_16
IsActive_17
IsActive_18
IsActive_19
IsActive_2
IsActive_20
IsActive_21
IsActive_22
IsActive_23
IsActive_24
IsActive_25
IsActive_26
IsActive_27
IsActive_28
IsActive_29
IsActive_3
IsActive_30
IsActive_31
IsActive_32
IsActive_33
IsActive_34
IsActive_35
IsActive_36
IsActive_37
IsActive_38
IsActive_39
IsActive_4
IsActive_40
IsActive_41
IsActive_42
IsActive_43
IsActive_44
IsActive_45
IsActive_46
IsActive_47
IsActive_48
IsActive_5
IsActive_6
IsActive_7
IsActive_8
IsActive_9
Z_Charge_1
Z_Charge_10
Z_Charge_11
Z_Charge_12
Z_Charge_13
Z_Charge_14
Z_Charge_15
Z_Charge_16
Z_Charge_17
Z_Charge_18
Z_Charge_19
Z_Charge_2
Z_Charge_20
Z_Charge_21
Z_Charge_22
Z_Charge_23
Z_Charge_24
Z_Charge_25
Z_Charge_26
Z_Charge_27
Z_Charge_28
Z_Charge_29
Z_Charge_3
Z_Charge_30
Z_Charge_31
Z_Charge_32
Z_Charge_33
Z_Charge_34
Z_Charge_35
Z_Charge_36
Z_Charge_37
Z_Charge_38
Z_Charge_39
Z_Charge_4
Z_Charge_40
Z_Charge_41
Z_Charge_42
Z_Charge_43
Z_Charge_44
Z_Charge_45
Z_Charge_46
Z_Charge_47
Z_Charge_48
Z_Charge_5
Z_Charge_6
Z_Charge_7
Z_Charge_8
Z_Charge_9
Z_Discharge_1
Z_Discharge_10
Z_Discharge_11
Z_Discharge_12
Z_Discharge_13
Z_Discharge_14
Z_Discharge_15
Z_Discharge_16
Z_Discharge_17
Z_Discharge_18
Z_Discharge_19
Z_Discharge_2
Z_Discharge_20
Z_Discharge_21
Z_Discharge_22
Z_Discharge_23
Z_Discharge_24
Z_Discharge_25
Z_Discharge_26
Z_Discharge_27
Z_Discharge_28
Z_Discharge_29
Z_Discharge_3
Z_Discharge_30
Z_Discharge_31
Z_Discharge_32
Z_Discharge_33
Z_Discharge_34
Z_Discharge_35
Z_Discharge_36
Z_Discharge_37
Z_Discharge_38
Z_Discharge_39
Z_Discharge_4
Z_Discharge_40
Z_Discharge_41
Z_Discharge_42
Z_Discharge_43
Z_Discharge_44
Z_Discharge_45
Z_Discharge_46
Z_Discharge_47
Z_Discharge_48
Z_Discharge_5
Z_Discharge_6
Z_Discharge_7
Z_Discharge_8
Z_Discharge_9
End

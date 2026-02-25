#!/usr/bin/python

def domain_approach_mapping_dict():
    return {
        "STRUCTURE": {
            "domains": ["STRUCTURE"],
            "approaches": ["MRI"]
        }, 
        "BEH": {
            "domains": ["MOTOR", "MEMORY", "LANGUAGE"],
            "approaches": ["BEH"]
        },
        "FUNCTIONAL": {
            "domains": ["MOTOR", "MEMORY", "LANGUAGE"],
            "approaches": ["EEG", "MRI"]
        }, 
        "ST": {
            "domains": ["MOTOR", "MEMORY", "LANGUAGE"], 
            "approaches": ["ST_RAW"]
        }, 
        "ALL": {
            "domains": ["STRUCTURE", "MOTOR", "MEMORY", "LANGUAGE"], 
            "approaches": ["MRI", "BEH", "EEG"]
        }
    }

def ST_features(): # from standardized tests
    return [
        "LANGUAGE_ST_SCALED_SIMILARITY", 
        "LANGUAGE_ST_SCALED_VOCABULARY", 
        "LANGUAGE_ST_SCALED_INFORMATION", 
        "MEMORY_ST_SCALED_AudImm", 
        "MEMORY_ST_SCALED_VisImm", 
        "MEMORY_ST_SCALED_WorMem", 
        "MEMORY_ST_SCALED_LogMemI", 
        "MEMORY_ST_SCALED_LetNumSeq", 
        "MEMORY_ST_SCALED_FamPicI", 
        "MEMORY_ST_SCALED_VerPairI", 
        "MEMORY_ST_SCALED_FacI", 
        "MEMORY_ST_SCALED_SpaForward", 
        "MEMORY_ST_SCALED_SpaBackward", 
        "MOTOR_ST_SCALED_FineMotor", 
        "MOTOR_ST_SCALED_Balance", 
        "MOTOR_ST_SCALED_ProcessingSpeed"    
    ]

def basic_Q_features(): # from questionnaires
    return [
        "BASIC_Q_EHI_Sum", 
        "BASIC_Q_SF36_PhysicalFunct", 
        "BASIC_Q_SF36_PhysicalLimit", 
        "BASIC_Q_SF36_EmotionalWell", 
        "BASIC_Q_SF36_EmotionalLimit", 
        "BASIC_Q_SF36_Energy", 
        "BASIC_Q_SF36_SocialFunc", 
        "BASIC_Q_SF36_Pain", 
        "BASIC_Q_SF36_GeneralHealth", 
        "BASIC_Q_SF36_Physical", 
        "BASIC_Q_SF36_Mental", 
        "BASIC_Q_PSQI_SleepQuality", 
        "BASIC_Q_PSQI_SleepLatency", 
        "BASIC_Q_PSQI_SleepDuration", 
        "BASIC_Q_PSQI_SleepEfficiency", 
        "BASIC_Q_PSQI_SleepDisturbance", 
        "BASIC_Q_PSQI_SleepMedication", 
        "BASIC_Q_PSQI_DaytimeDysfunc", 
        "BASIC_Q_PSQI_Sum", 
        "BASIC_Q_IPAQ_MET", 
        "BASIC_Q_BFI_Extraversion", 
        "BASIC_Q_BFI_Agreeableness", 
        "BASIC_Q_BFI_Conscientiousness", 
        "BASIC_Q_BFI_EmotionalStability", 
        "BASIC_Q_BFI_Intellect", 
        "BASIC_Q_MSPSS_Sum", 
        "BASIC_Q_CogFailure_Sum", 
        "BASIC_Q_Beck_Anxiety", 
        "BASIC_Q_Beck_Depression", 
        "BASIC_Q_MOCA_SUM"
    ]

def Platform_features():
    return [
        "MOTOR_GOFITTS_BEH_ID1_LeaveTime", 
        "MOTOR_GOFITTS_BEH_ID2_LeaveTime", 
        "MOTOR_GOFITTS_BEH_ID3_LeaveTime", 
        "MOTOR_GOFITTS_BEH_ID4_LeaveTime", 
        "MOTOR_GOFITTS_BEH_ID5_LeaveTime", 
        "MOTOR_GOFITTS_BEH_ID6_LeaveTime", 
        "MOTOR_GOFITTS_BEH_ID1_PointTime", 
        "MOTOR_GOFITTS_BEH_ID2_PointTime", 
        "MOTOR_GOFITTS_BEH_ID3_PointTime", 
        "MOTOR_GOFITTS_BEH_ID4_PointTime", 
        "MOTOR_GOFITTS_BEH_ID5_PointTime", 
        "MOTOR_GOFITTS_BEH_ID6_PointTime", 
        "MOTOR_GOFITTS_BEH_SLOPE_LeaveTime", 
        "MOTOR_GOFITTS_BEH_SLOPE_PointTime", 
        "MEMORY_EXCLUSION_BEH_C1_FAMILIARITY", 
        "MEMORY_EXCLUSION_BEH_C2_FAMILIARITY", 
        "MEMORY_EXCLUSION_BEH_C3_FAMILIARITY", 
        "MEMORY_EXCLUSION_BEH_C1_RECOLLECTION", 
        "MEMORY_EXCLUSION_BEH_C2_RECOLLECTION", 
        "MEMORY_EXCLUSION_BEH_C3_RECOLLECTION", 
        "MEMORY_EXCLUSION_BEH_C1TarHit_PROPORTION", 
        "MEMORY_EXCLUSION_BEH_C1TarMiss_PROPORTION", 
        "MEMORY_EXCLUSION_BEH_C1NonTarFA_PROPORTION", 
        "MEMORY_EXCLUSION_BEH_C1NonTarCR_PROPORTION", 
        "MEMORY_EXCLUSION_BEH_C1NewFA_PROPORTION", 
        "MEMORY_EXCLUSION_BEH_C1NewCR_PROPORTION", 
        "MEMORY_EXCLUSION_BEH_C1TarHit_RT",
		"MEMORY_EXCLUSION_BEH_C1TarMiss_RT",
        "MEMORY_EXCLUSION_BEH_C1NonTarFA_RT", 
        "MEMORY_EXCLUSION_BEH_C1NonTarCR_RT", 
        "MEMORY_EXCLUSION_BEH_C1NewFA_RT", 
        "MEMORY_EXCLUSION_BEH_C1NewCR_RT", 
        "MEMORY_EXCLUSION_BEH_C2TarHit_PROPORTION", 
        "MEMORY_EXCLUSION_BEH_C2TarMiss_PROPORTION", 
        "MEMORY_EXCLUSION_BEH_C2NonTarFA_PROPORTION", 
        "MEMORY_EXCLUSION_BEH_C2NonTarCR_PROPORTION", 
        "MEMORY_EXCLUSION_BEH_C2NewFA_PROPORTION", 
        "MEMORY_EXCLUSION_BEH_C2NewCR_PROPORTION", 
        "MEMORY_EXCLUSION_BEH_C2TarHit_RT", 
        "MEMORY_EXCLUSION_BEH_C2TarMiss_RT", 
        "MEMORY_EXCLUSION_BEH_C2NonTarFA_RT", 
        "MEMORY_EXCLUSION_BEH_C2NonTarCR_RT", 
        "MEMORY_EXCLUSION_BEH_C2NewFA_RT", 
        "MEMORY_EXCLUSION_BEH_C2NewCR_RT", 
        "MEMORY_EXCLUSION_BEH_C3TarHit_PROPORTION", 
        "MEMORY_EXCLUSION_BEH_C3TarMiss_PROPORTION", 
        "MEMORY_EXCLUSION_BEH_C3NonTarFA_PROPORTION", 
        "MEMORY_EXCLUSION_BEH_C3NonTarCR_PROPORTION", 
        "MEMORY_EXCLUSION_BEH_C3NewFA_PROPORTION", 
        "MEMORY_EXCLUSION_BEH_C3NewCR_PROPORTION", 
        "MEMORY_EXCLUSION_BEH_C3TarHit_RT", 
        "MEMORY_EXCLUSION_BEH_C3TarMiss_RT", 
        "MEMORY_EXCLUSION_BEH_C3NonTarFA_RT", 
        "MEMORY_EXCLUSION_BEH_C3NonTarCR_RT", 
        "MEMORY_EXCLUSION_BEH_C3NewFA_RT", 
        "MEMORY_EXCLUSION_BEH_C3NewCR_RT", 
        "MEMORY_OSPAN_BEH_LETTER_ACCURACY", 
        "MEMORY_OSPAN_BEH_MATH_ACCURACY", 
        "LANGUAGE_SPEECHCOMP_BEH_PASSIVE_ACCURACY", 
        "LANGUAGE_SPEECHCOMP_BEH_PASSIVE_RT", 
        "LANGUAGE_READING_BEH_NULL_MeanSR"
    ]


import pandas as pd
import numpy as np
import re


# Loading the data

df = pd.read_csv('CSV_files/Dataset_maternal_mental_health_infant_sleep.csv', encoding='ISO-8859-1')

# Cleaning the data

def clean_columns(df):
    """Lowercase column names, trim whitespace (incl. non-breaking), replace spaces with underscores.
       Works for both single Index and MultiIndex columns."""
    def _clean(s: str) -> str:
        s = s.replace('\u00a0', ' ')
        s = s.strip().lower()
        s = re.sub(r'\s+', ' ', s)
        return s
    
    if isinstance(df.columns, pd.MultiIndex):
        flat = ['_'.join(_clean(str(p)) for p in tup if str(p) != 'None') for tup in df.columns]
        df.columns = flat
    else:
        df.columns = df.columns.map(lambda c: _clean(str(c)))
    return df

df = clean_columns(df)

df.drop(['birth_1mth_m_inclusion', 'birth_12mth_m_inclusion', 'child_survey_participation'], axis=1, inplace=True)

df.rename(columns={
    'type_parents': 'mother_or_partner',
    'marital_status_autre': 'marital_status_other',
    'gestationnal_age': 'infant_gestational_age',
    'type_pregnancy': 'pregnancy_type',
    'sex_baby1': 'infant_sex',
    'age_bb': 'infant_age_category',
    'sleep_night_duration_bb1': 'infant_nightly_sleep_duration',
    'night_awakening_number_bb1': 'infant_wakes_per_night',
    'how_falling_asleep_bb1': 'infant_sleeping_method'
}, inplace=True)

# Splitting the dataframe into two separate dataframes to make the analysis more manageable

participant_info = ['participant_number', 'mother_or_partner', 'age', 'marital_status', 'marital_status_other', 'marital_status_edit', 
                     'education', 'infant_gestational_age', 'pregnancy_type', 'infant_sex', 'infant_age_category', 'infant_nightly_sleep_duration',
                     'infant_wakes_per_night', 'infant_sleeping_method']
mental_health_info = ['participant_number', 'cbts_m_3', 'cbts_m_4', 'cbts_m_5', 'cbts_m_6', 'cbts_m_7', 'cbts_m_8', 'cbts_m_9', 'cbts_m_10', 
                      'cbts_m_11', 'cbts_m_12', 'cbts_13', 'cbts_14', 'cbts_15', 'cbts_16', 'cbts_17', 'cbts_18', 'cbts_19', 'cbts_20', 'cbts_21', 
                      'cbts_22', 'epds_1', 'epds_2', 'epds_3', 'epds_4', 'epds_5', 'epds_6', 'epds_7', 'epds_8', 'epds_9', 'epds_10', 'hads_1', 
                      'hads_3', 'hads_5', 'hads_7', 'hads_9', 'hads_11', 'hads_13', 'ibq_r_vsf_3_bb1', 'ibq_r_vsf_4_bb1', 'ibq_r_vsf_9_bb1', 
                      'ibq_r_vsf_10_bb1', 'ibq_r_vsf_16_bb1', 'ibq_r_vsf_17_bb1', 'ibq_r_vsf_28_bb1', 'ibq_r_vsf_29_bb1', 'ibq_r_vsf_32_bb1', 
                      'ibq_r_vsf_33_bb1']

participant_df = df[participant_info].copy()
mental_health_df = df[mental_health_info].copy()

# Cleaning the mental health evaluation columns

def clean_psych_columns(df):
    '''Clean specific patterns from psychological evaluation column names'''

    def clean_name(col):

        if col.startswith('cbts'):
            return re.sub(r'_m_', '_', col)
        
        elif col.startswith('ibq'):
            col = re.sub(r'_r_vsf', '', col)
            col = re.sub(r'_bb1', '', col)
            return col
        
        else:
            return col
    
    df.columns = [clean_name(c) for c in df.columns]
    return df

mental_health_df = clean_psych_columns(mental_health_df)

# Dropping the unnecessary marital status columns and creating a final marital status column

participant_df.drop(['mother_or_partner', 'marital_status', 'marital_status_other'], axis=1, inplace=True)
participant_df.rename(columns={
    'marital_status_edit': 'marital_status'
}, inplace=True)



# Decoding the participant dataframe

decode_marital_status = {
    1: 'Single',
    2: 'In a relationship',
    3: 'Separated, divorced or widowed'
}

participant_df['marital_status'] = participant_df['marital_status'].map(decode_marital_status)

#---------------------

decode_education = {
    1: 'No education',
    2: 'Compulsory education',
    3: 'Post-compulsory education (i.e. apprenticeship)',
    4: "Bachelor's degree or above in STEM field",
    5: "Bachelor's degree or above"
}

participant_df['education'] = participant_df['education'].map(decode_education)

#---------------------

decode_pregnancy_type = {
    1: 'Single pregnancy',
    2: 'Twin pregnancy'
}

participant_df['pregnancy_type'] = participant_df['pregnancy_type'].map(decode_pregnancy_type)

#---------------------

decode_infant_sex = {
    1: 'Female',
    2: 'Male'
}

participant_df['infant_sex'] = participant_df['infant_sex'].map(decode_infant_sex)

#---------------------

decode_infant_age_category = {
    1: '3-6 months',
    2: '6-9 months',
    3: '9-12 months'
}

participant_df['infant_age_category'] = participant_df['infant_age_category'].map(decode_infant_age_category)

#---------------------

decode_infant_sleeping_method = {
    1: 'While being fed',
    2: 'While being rocked',
    3: 'While being held',
    4: 'Alone in the crib',
    5: 'In the crib with parental presence'
}

participant_df['infant_sleeping_method'] = participant_df['infant_sleeping_method'].map(decode_infant_sleeping_method)

# Converting the hours column

def convert_to_hours(t):
    try:
        h, m = t.split(':')
        return int(h) + int(m)/60
    except:
        return np.nan

participant_df['infant_nightly_sleep_duration'] = participant_df['infant_nightly_sleep_duration'].apply(convert_to_hours)


participant_df = (
    participant_df
    .groupby('participant_number', as_index=False)
    .first()
)

mental_health_df = (
    mental_health_df
    .groupby('participant_number', as_index=False)
    .first()
)


participant_df.to_csv('CSV_files/participant.csv', index=False, encoding='utf-8')
mental_health_df.to_csv('CSV_files/mental_health.csv', index=False, encoding='utf-8')



'''Links for the evaluations:
CBTS: https://www.citybirthtraumascale.com/_files/ugd/70a10c_9ffc515c37eb489d90043a3201122b55.pdf?index=true
EPDS: https://www.mdcalc.com/calc/10466/edinburgh-postnatal-depression-scale-epds
HADS: https://epidose.ca/wp-content/uploads/2021/07/Hospital_Anxiety_and_Depression_Scale.pdf
'''

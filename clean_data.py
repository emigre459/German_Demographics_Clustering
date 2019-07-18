import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


def enhanced_clean(df, feature_summary, missing_breakpoint, 
    median_imputer):
    """
    Perform feature trimming, re-encoding, and engineering for demographics
    data. This version is enhanced relative to the version in the Jupyter
    notebook to include imputation of missing values.
    
    INPUTS
    ------
    df: Demographics DataFrame
    
    feature_summary: DataFrame that includes listing of the various missing value codes for each feature

    missing_breakpoint: int. Amount of missing values allowed in any given
        row in order to retain it

    median_imputer: sklearn SimpleImputer object with strategy = 'median'.
        This should be the imputer trained on the general population features
        that had more than 1% missing values
        
    OUTPUT: Trimmed and cleaned demographics DataFrame
    """
    
    # -------------------------------------------------------------------
    # convert missing value codes into NaNs, ...

    # Convert the strings for the missing values from the feature summary
    # To be proper lists of values to use for filling in NaNs

    # First, remove brackets
    # Then, split on comma separator
    feature_summary.loc[:, 'missing_or_unknown'] = \
        feature_summary.loc[:, 'missing_or_unknown'].str[1:-1].str.split(',')



    def fill_missing(df, missing_codes_mapping, inplace=False):
        '''
        Parses dataframe of missing values and their mapping to individual feature names
        and then fills any of those values found in a dataframe's matching feature columns
        with np.nan.

        Inputs
        ------
        df: pandas DataFrame. Table with features that match the ones for which we have
            missing mappings. Each sample is a person.

        missing_codes_mapping: pandas DataFrame. Contains columns 'attribute' and 
            'missing_or_unknown' that map codes used for missing/unknown values to 
            features/attributes. 'missing_or_unknown' is expected to have elements
            that are lists of str (usually ints, but sometimes chars or empty lists).

        Returns
        -------
        df with NaN values filled in according to missing_codes_mapping
        '''

        # Use deep copy if inplace = False, otherwise use actual inputs
        if inplace:
            data = df
            missing_codes = missing_codes_mapping
        else:            
            data = df.copy(deep=True)
            missing_codes = missing_codes_mapping.copy(deep=True)

        def parse_missing_codes(code_list):
            '''
            Goes through a list of str and converts the elements of the list according to the needs 
            of the dtypes in our demographic data such that the results can be used for 
            filling in NaN values.

            Inputs
            ------
            code_list: list of str. List is expected to contain the chars, floats, or ints 
                that are codes indicating a missing or unknown value.

            Returns
            -------
            list or np.nan. Each element of the list returned is typecast according to 
                the expected needs of the NaN-filling it will be doing. Empty lists
                (or lists with only an empty string in them) are returned as np.nan.
            '''

            # Make sure list isn't just empty string
            if '' not in code_list:
                # Check if list can be converted to int without issues - if so, do it
                try:
                    return [int(e) for e in code_list]

                # Not all are cast-able to int
                except ValueError:
                    return [float(e) if 'X' not in e else e for e in code_list]

            else:
                return np.nan

        # Typecast missing value codes appropriately
        missing_codes.loc[:, 'missing_or_unknown'] = \
            missing_codes.loc[:, 'missing_or_unknown'].apply(parse_missing_codes)

        # Create series that maps feature names (index) to missing codes (data)
        code_map = pd.Series(data=missing_codes['missing_or_unknown'].values,
                             index=missing_codes['attribute'].values)

        # When passing a Series into to_replace, index is key and data is value (like a dict)
        data.replace(to_replace=code_map,
                     value=np.nan,
                     inplace=True)

        return data
    

    df = fill_missing(df, feature_summary)

    # ------------------------------------------------------------------    
    # remove selected columns and rows, ...

    # Removing outlier features, except the one that provides a birth year
    df.drop(columns=['ALTER_HH', 'KBA05_BAUMAX', 'KK_KUNDENTYP',
        'AGER_TYP', 'TITEL_KZ'], inplace = True)

    # If I'm going to make sense out of these, I need more helpful names

    # Read in the names mapping CSV as a dict
    new_names = pd.read_csv('col_renaming.csv', header=None, index_col=0,
                            squeeze=True).to_dict()

    df.rename(columns=new_names, inplace=True)

    # Remove rows having more than the allowed threshold of missing values
    df = df.loc[df.isnull().sum(axis=1) <= missing_breakpoint,:]

    
    # -------------------------------------------------------------------
    # select, re-encode, and engineer column values.

    # Re-encode categorical variable(s) to be kept in the analysis.
    cat_cols = ['Bldg: Location Relative to E or W Germany',
               'Small or Home Office Owner', 'Insurance Type',
               'Nationality Based on Name', 'Shopper Type',
                'Socioeconomic Status - LowRes', 'Family Type - LowRes',
                'MoneyType__Primary', 'Energy Consumption Type',
                'Consumption Channel Type', 'Bldg: Building Type',
                'RR4: Life Stage Type - LowRes', 'Socioeconomic Status - HighRes',
                'Family Type - HighRes', 'Vacation Habits', 
                'RR4: Life Stage Type - HighRes']

    # Have to drop first dummy so avoid Dummy Variable Trap
    # Have to include NaN dummy so zero-vector can't be interpreted ambiguously
        # as either NaN or first dropped dummy (which are likely different)
    df = pd.get_dummies(
        df, prefix_sep='__', 
        drop_first=True, dummy_na=True,
        columns=cat_cols)



    def data_mapper(series, mapping_dict):
        '''
        Reads in a pandas Series object that represents the Generation Designation feature
        and returns a re-encoded series according to mapping_dict.

        Inputs
        ------
        series: pandas Series of integer codes (1 through 15) representing different
            Generation Designation values

        mapping_dict: dict of form {designation_code: new_code} used to determine what
            values to return


        Returns
        -------
        pandas Series with the new codes
        '''
        
        # Since NaN values aren't always propagated as expected, do a quick check
        print(f"There are {series.isnull().sum()} null values in the series \
        {series.name} prior to extraction")
        
        out = series.map(mapping_dict, na_action = 'ignore')
        
        print(f"There are {out.isnull().sum()} null values in the series \
        {series.name} after extraction")
        
        return out              


    # For extracting decade of birth info from Generation Designation feature
    decade_code_map = {
    1: 0,
    2: 0,
    3: 1,
    4: 1,
    5: 2,
    6: 2,
    7: 2,
    8: 3,
    9: 3,
    10: 4,
    11: 4,
    12: 4,
    13: 4,
    14: 5,
    15: 5
    }

    df['Generation Decade'] = \
    data_mapper(df['Generation Designation'],
        decade_code_map)

    # For extracting generational movement from Generation Designation feature
    movement_code_map = {
    1: 0,
    2: 1,
    3: 0,
    4: 1,
    5: 0,
    6: 1,
    7: 1,
    8: 0,
    9: 1,
    10: 0,
    11: 1,
    12: 0,
    13: 1,
    14: 0,
    15: 1
    }

    df['Generation Movement'] = \
    data_mapper(df['Generation Designation'],
        movement_code_map)

    # Drop Generation Designation feature now that we've captured
    # its information in the new features
    df.drop(columns='Generation Designation', inplace=True)

    # Drop the rows with missing int'l codes
    df.dropna(subset=["RR4: Life Stage Type - Int\'l Code Mapping"],
        inplace=True)

    # Cast as int since it seems to keep coming up as object type
    df["RR4: Life Stage Type - Int'l Code Mapping"] =\
    df["RR4: Life Stage Type - Int'l Code Mapping"].astype(int)

    # Extract the tens digit as an int - wealth code
    df["RR4: Life Stage Type - Int'l - Wealth"] = \
    (df["RR4: Life Stage Type - Int'l Code Mapping"] \
        / 10).astype(int)

    # Extract the ones digit - life stage code
    df["RR4: Life Stage Type - Int'l - Stage"] = \
    df["RR4: Life Stage Type - Int'l Code Mapping"] % 10

    # Drop lngering RR4: Life Stage Type - Int'l Code Mapping feature
    df.drop(
        columns="RR4: Life Stage Type - Int'l Code Mapping", inplace=True)

    # Make dummies for Life Stage - HighRes, and Life Stage - LowRes
    cat_cols = ['Life Stage - HighRes',
               'Life Stage - LowRes']

    df = pd.get_dummies(
        df, prefix_sep='__', 
        drop_first=True, dummy_na=True,
        columns=cat_cols)

    # For extracting rural neighborhood status from Neighborhood Quality
    rural_code_map = {
    0: np.nan,
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 0,
    7: 1,
    8: 2
    }

    df['Bldg: Rural Type'] = data_mapper(df['Bldg: Neighborhood Quality'],
        rural_code_map)

    # Make dummies out of the multi-level Rural Type categorical
    cat_cols = ['Bldg: Rural Type']

    df = pd.get_dummies(df,
        prefix_sep='__',
        drop_first=True,
        dummy_na=True,
        columns=cat_cols)

    # Exclude rural categories as though they weren't scored
    neighborhood_code_map = {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        7: 0,
        8: 0
        }

    df['Bldg: Neighborhood Quality'] = \
    data_mapper(df['Bldg: Neighborhood Quality'],
        neighborhood_code_map)

    # For extracting business building dominance and dropping info about rest
    biz_code_map = {
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 1
    }
    
    df['PLZ8: Primarily Business Bldgs'] = \
        data_mapper(df['PLZ8: Most Common Bldg Type'],
            biz_code_map)

    # Drop original feature, it's no longer useful
    df.drop(columns=['PLZ8: Most Common Bldg Type'],
                             inplace=True)


    # -------------------------------------------------------------------
    # Impute remaining missing values

    # Drop null rows features with 1% or less missing values
    cols = ['Bldg: Number of HHs',
    'PLZ8: Primarily Business Bldgs',
    'PLZ8: Bins of 1-2 Family Homes',
    'PLZ8: Bins of 6-10 Family Homes',
    'PLZ8: Bins of 10+ Family Homes',
    'PLZ8: Bin Counts of HHs',
    'PLZ8: Bin Counts of Bldgs',
    'PLZ8: Bins of 3-5 Family Homes',
    'Community: Share of Unemployment',
    'Community: Share of Unemployment Relative to Parent County',
    'Community: Size (bins)',
    'Bldg: Number of Academic Title Holders',
    'PLZ8: Number of Cars',
    'Age Bin',
    'PostCode: Distance to City Center (bins)',
    'PostCode: Density of HHs per km^2 (bins)',
    'PostCode: Distance to Nearest Urban Center (bins)',
    'Bldg: Distance to Point of Sale Category',
    'RR1: Residential-Commercial Activity Ratio (categories)']

    df.dropna(subset=cols, inplace=True)

    # Use Age Bin to impute Birth Year values, using midpoint of associated 
    # Age Bin to dictate year
    age_bin_map = {
        1: '< 30 years old',
        2: '30 - 45 years old',
        3: '46 - 60 years old',
        4: '> 60 years old'
    }

    # Assume data are all relative to 2017, as this is the maximum Birth Year value
    # Earliest Birth Year is 1900, making greatest age 117 
        # Thus we'll assume 20 year midpoint for code 4 (assuming anyone 100+ is outlier)
    age_bin_to_year = {
        1: 2002,
        2: 1979,
        3: 1963,
        4: 1937
    }

    # Make a series that is a mapping of Age Bin to birth years
    mapped_series = df['Age Bin'].map(age_bin_to_year)

    # Use this mapped series and fillna() on Birth Year 
        # to take midpoint Age Bin year as Birth Year
    df['Birth Year'].fillna(mapped_series, inplace=True)
    

    # Impute using the median for remaining features
    cols = ['RR1: Neighborhood Type',
    'RR1: Purchasing Power (bins)',
    'HH: Probability of Children in Residence',
    'Health Type',
    'RR3: Bins of 1-2 Family Homes',
    'RR3: Bins of 3-5 Family Homes',
    'RR3: Bins of 6-10 Family Homes',
    'RR3: Bins of 10+ Family Homes',
    'RR3: Bin Counts of Bldgs',
    'RR1: Movement Patterns',
    'Generation Movement',
    'Generation Decade']

    df[cols] = median_imputer.transform(df[cols])

    # -------------------------------------------------------------------
    # Return the cleaned dataframe.
    
    return df
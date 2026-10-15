#   PREPROCESSING PIPELINE

######################################### Group ID ######################################
# The first part of the passanger ID indicates that people belong to a group.
# It is not unreasonable that people from the same group would share the same Cabin.
# Extract the group ID from the passenger ID
df['Group'] = df['PassengerId'].apply(lambda x: x.split('_')[0])

# Forward fill or group-fill missing cabin data using group members
# If someone in group x has a cabin, give it to their traveling companions who are missing it.
df['Cabin'] = df.groupby('Group')['Cabin'].bfill().ffill()

########################################## Cabin
# List of categories to check and impute based on cabin
cabin_categories = ['CryoSleep', 'VIP', 'HomePlanet', 'Destination']

for col in cabin_categories:
    print (f'Processing column by Cabin: {col}')

    # making sure to target only cabins that exist
    valid_cabin = df['Cabin'].notna()

    # calculate how many unique non-null values exist in each cabin
    ## transform()calculates a summary value for each group and broadcasts it back to match the original df's shape
    cabin_nunique = df[valid_cabin].groupby('Cabin')[col].transform(lambda x: x.nunique(dropna=True))

    # find cabins where all members have ONE value for a given category and at least one member without that category
    uniform_cabin = valid_cabin & (cabin_nunique ==1) & df[col].isna()

    # grab the shared value in a given cabin
    shared_cabin_values = df[valid_cabin].groupby('Cabin')[col].transform(lambda x:x.iloc[0] if x.notna().any() else None)

    # fill mising values for the cabin members without one
    df.loc[uniform_cabin, col] = shared_cabin_values[uniform_cabin]

    print (f'Filled missing values for uniform cabins in {col}')

#################################### CryoSleep
# List of columns that represent charges due to different ameneties in the cruise
# (which one cannot use if sleeping)
awake_amenity_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

# Updating the missin values of these columns to 0 if the passanger is in cryosleep
df.loc[df['CryoSleep']==True, awake_amenity_cols] = df.loc[df['CryoSleep']==True, awake_amenity_cols].fillna(0)

###################################### Families
df['HadName']=df['Name'].notna().astype(int)
df.head()

df['Name']=df['Name'].fillna('Unknown')

# Let's first make a last name column to make our lives easier
df['LastName']=df['Name'].str.split().str[-1]

# for each cabin, let's see if they have at least one last name other than unknown
def get_cabin_lastname(group):
    # Filter out 'Unknown' and NaNs to find a real family name in the cabin
    valid_names = group[group != 'Unknown'].dropna()
    if not valid_names.empty:
        return valid_names.iloc[0] # Return the first real last name found
    return 'Unknown'

# Calculate the dominant last name per cabin
cabin_family_name = df[valid_cabin].groupby('Cabin')['LastName'].transform(get_cabin_lastname)

# Fill missing/Unknown names where the cabin shares a known family name
name_impute = valid_cabin & (df['LastName'] == 'Unknown') & (cabin_family_name != 'Unknown')

# Let's update the LastName column with the Cabin Name
df.loc[name_impute, 'LastName'] = cabin_family_name[name_impute]


features_to_check = ['HomePlanet', 'Destination', 'CryoSleep', 'VIP']

# Stablish a maximum family size as not to group all the unknowns into one family
MAX_FAMILY_SIZE = 20 # yes, the family of my syster-in-law goes ones a year in a cruise, they are 36!

for col in features_to_check:
    print(f"\n--- Processing column by Family: {col} ---")

    # Looking only for reasonably sized families with a valid last name
    valid_family_mask = (
        df['LastName'].notna() &
        (df['LastName'] != 'Unknown') &
        df['LastName'].map(lambda x: family_sizes.get(x, 0) <= MAX_FAMILY_SIZE)
    )

    # Check how many unique non-null values exist for each family group
    family_nunique = df[valid_family_mask].groupby('LastName')[col].transform(lambda x: x.nunique(dropna=True))

    # Impute: Family has uniform data (nunique == 1) AND the passenger is missing this value
    uniform_families_mask = valid_family_mask & (family_nunique == 1) & df[col].isna()

    # Grab the shared value for each family group safely
    shared_family_values = df[valid_family_mask].groupby('LastName')[col].transform(
        lambda x: x.iloc[0] if x.notna().any() else None
    )

    # Fill the missing values
    df.loc[uniform_families_mask, col] = shared_family_values[uniform_families_mask]
    print(f"Filled missing values for uniform families in {col}.")


# Define spending columns
spending_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

child_age_threshold = 18

# Apply 0 specifically to missing spending values for children
child_missing_spending_mask = (df['Age'] <= child_age_threshold) & df[spending_cols].isna().any(axis=1)

for col in spending_cols:
    # Fill missing spending with 0 where the passenger is a child
    child_mask = (df['Age'] <= child_age_threshold) & df[col].isna()
    df.loc[child_mask, col] = 0


########### Let's assume the 'missing' expenses incurred in the cruise are 'missing' because there were no expenses incurred
spending_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

# Fill missing spending values with 0
df[spending_cols] = df[spending_cols].fillna(0)

# Verify remaining missing values in spending columns
print("Remaining missing values in spending columns:")
print(df[spending_cols].isna().sum())


############### cabin distribution
# Parse the Cabin column WITHOUT filling missing strings first.
# Rows where 'Cabin' is NaN will result in NaN for Deck, Num, and Side
##### We want to be able to explode XGBoost's ability to handle NaNs
cabin_split = df['Cabin'].str.split('/', expand=True)
if cabin_split.shape[1] == 3:
    df['Deck'] = cabin_split[0]
    df['Num'] = pd.to_numeric(cabin_split[1], errors='coerce')
    df['Side'] = cabin_split[2]
else:
    # Fallback safety if any malformed rows exist
    df['Deck'] = pd.Series(dtype='object')
    df['Num'] = pd.Series(dtype='float')
    df['Side'] = pd.Series(dtype='object')



#################Preparing X for XG Boost Model
drop_cols= ['Cabin', 'PassengerId', 'Name', 'Transported']
X = df.drop(columns=[col for col in drop_cols if col in df.columns])


######## Casting Categorical FEatures
categorical_cols = ['HomePlanet', 'Destination', 'Deck', 'Side', 'LastName']
for col in categorical_cols:
    if col in X.columns:
        X[col] = X[col].astype('category')



############## Casting boolean features
bool_cols = ['CryoSleep', 'VIP']
for col in bool_cols:
    if col in X.columns:
        X[col] = X[col].astype('boolean')


############# Casting numerial features
df['Group'] = df['Group'].astype(int)
X['Group'] = X['Group'].astype(int)

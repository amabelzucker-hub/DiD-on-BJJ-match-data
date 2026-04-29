import numpy as np
import pandas as pd
import re
import warnings
import matplotlib.pyplot as plt
import causalpy as cp
from sklearn.linear_model import LinearRegression
import seaborn as sns
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore", category=FutureWarning)

def bin_event_time(k):
    if k <= -6:
        return "pre_-6+"
    elif k == -5:
        return "-5"
    elif k == -4:
        return "-4"
    elif k == -3:
        return "-3"
    elif k == -2:
        return "-2"
    elif k == -1:
        return "-1"
    elif k == 0:
        return "0"
    elif k >= 1 and k <= 2:
        return "1_2"
    elif k >= 3 and k<= 4:
        return "3_4"
    else:
        return "5+"

df = pd.read_csv("input/data.csv")
    #print(df.shape)
    #print(df.Method.value_counts())
    #print(df[(df.Fighter == "Quentin Rosenweig") & (df.Opponent == 'Aaron "Tex" Johnson')])

df_better = pd.concat([df.drop("Opponent", axis = 1), df.drop(["Fighter"], axis = 1).rename({"Opponent":"Fighter"}, axis = 1)])
var = df_better.Fighter.value_counts()

df_b2 = df_better[df_better.Fighter.isin(var[var>=6].index)]
pattern = r"^(\d{2,3}).*"

def clean_weight(weight_str):
    m = re.match(pattern, weight_str)
    if m:
        return int(m.group(1))
    else: 
        return 0

df_b2["Cleaned_Weight"] = df_b2.Weight.fillna(" ").apply(clean_weight)
    #print(df_b2.shape)
    #print(df_b2.Cleaned_Weight.value_counts().index.tolist())

df_b3 = df_b2[df_b2.Cleaned_Weight != 0].drop("Weight", axis = 1)
df_b3['Method'] = df_b3['Method'].fillna(0).astype(str)
    #print(df_b3.shape)

df_b3['Sub'] = np.where((~df_b3['Method'].str.contains('Pts', na = False)) & (~df_b3['Method'].isin(['Referee Decision', 'Points', 'Adv', 'Advantage', 'DQ', 'Injury', 'EBI/OT', '---', '0'])), 1, 0)

#filter for monotonicity

df_b4 = df_b3.sort_values("Year").groupby(["Fighter", "Year"]).agg({"Cleaned_Weight": [min, max]})
df_b4.columns = ["Min_Weight", "Max_Weight"]

increase_df = df_b4.sort_values(["Fighter", "Year"]).groupby("Fighter").filter(lambda g: g["Min_Weight"].is_monotonic_increasing)
increase_df = increase_df.sort_values(["Fighter", "Year"]).groupby("Fighter").filter(lambda g: g["Max_Weight"].is_monotonic_increasing)

decrease_df = df_b4.sort_values(["Fighter", "Year"]).groupby("Fighter").filter(lambda g: g["Min_Weight"].is_monotonic_decreasing)
decrease_df = decrease_df.sort_values(["Fighter", "Year"]).groupby("Fighter").filter(lambda g: g["Max_Weight"].is_monotonic_decreasing)

#reject the more sparse data in the beginning of the set
increase_df = increase_df[increase_df.index.get_level_values('Year') >= 1993]
decrease_df = decrease_df[decrease_df.index.get_level_values('Year') >= 1993]

#identify weight differences across years
increase_df = increase_df.sort_index()
decrease_df = decrease_df.sort_index()
weight_diffs = increase_df.groupby(level='Fighter')[['Min_Weight', 'Max_Weight']].diff().fillna(0)
weight_diffs_D = decrease_df.groupby(level='Fighter')[['Min_Weight', 'Max_Weight']].diff().fillna(0)
#print(weight_diffs)
increase_df['weight_increased'] = (weight_diffs > 0).any(axis=1)
decrease_df['weight_decreased'] = (weight_diffs_D < 0).any(axis=1)

#identify treatment groups
growth_groups = increase_df.groupby(level='Fighter')['weight_increased'].any()
growth_groups_D = decrease_df.groupby(level='Fighter')['weight_decreased'].any()
#It would be nice to weight heavier weight changes more heavily in the analysis
infty_control = growth_groups[growth_groups == False].index.tolist()
infty_control_D = growth_groups_D[growth_groups_D == False].index.tolist()
increase_counts_per_fighter = increase_df.groupby(level='Fighter')['weight_increased'].sum()
decrease_counts_per_fighter = decrease_df.groupby(level='Fighter')['weight_decreased'].sum()

#reject fighters that appear in more than one treatment group
multi_increase_fighters = increase_counts_per_fighter[increase_counts_per_fighter >= 2].reset_index()
multi_decrease_fighters = decrease_counts_per_fighter[decrease_counts_per_fighter >= 2].reset_index()

increase_df = increase_df.reset_index()
decrease_df = decrease_df.reset_index()

#dataframes of monotonic fighters with only one treatment year
increase_df = increase_df[~increase_df['Fighter'].isin(multi_increase_fighters['Fighter'])]
decrease_df = decrease_df[~decrease_df['Fighter'].isin(multi_decrease_fighters['Fighter'])]
#for visuallizing them...
year_groups = increase_df[increase_df['weight_increased']].reset_index().groupby('Year')['Fighter'].unique()
year_groups_D = decrease_df[decrease_df['weight_decreased']].reset_index().groupby('Year')['Fighter'].unique()


#INCREASE BATCH
year_groups_df = year_groups.reset_index(name='count').explode('count').rename(columns={'count': 'Fighter'})
year_groups_df = year_groups_df.groupby('Year').filter(lambda x: len(x) >= 4).reset_index(drop = 'True')
#print("Increase year group counts", year_groups_df["Year"].value_counts())

increase_treated_df = pd.merge(year_groups_df, df_b3, on='Fighter', how='left').rename(columns= {'Year_x': 'Treat_year', 'Year_y': 'Match_year'})
increase_treated_df['Match_year'] = increase_treated_df['Match_year'].astype(int)
increase_treated_df['Treated'] = 1
df_control_infty = df_b3[df_b3['Fighter'].isin(infty_control)].rename(columns = {'Year': 'Match_year'})
df_control_infty['Treated'] = 0
df_control_infty['Treat_year'] = np.inf
df_control_infty['Match_year'] = df_control_infty['Match_year'].astype(int)

increase_df = pd.concat([increase_treated_df, df_control_infty], ignore_index=True)

grouping_init = increase_df.groupby('Fighter')
has_post_init = grouping_init.apply(lambda g: (g['Match_year'] > g['Treat_year']).any())
has_pre_init  = grouping_init.apply(lambda g: (g['Match_year'] < g['Treat_year']).any())
is_treated_init = grouping_init['Treated'].max()
fighters_init = (is_treated_init == 0) | (has_post_init & has_pre_init)

#updated dataframe filtered to ensure pre and post treament observations and have all original columns
increase_df = increase_df[increase_df['Fighter'].isin(fighters_init.index)]

#make output variable columns
increase_df['Armbar'] = np.where((increase_df['Method'] == 'Armbar') & (increase_df['W/L'] == 'W'), 1, 0)
increase_df['Triangle'] = np.where((increase_df['Method'] == 'Triangle') & (increase_df['W/L'] == 'W'), 1, 0)
increase_df['RNC'] = np.where((increase_df['Method'] == 'RNC') & (increase_df['W/L'] == 'W'), 1, 0)
increase_df['Choke_from_back'] = np.where((increase_df['Method'] == 'Choke from back') & (increase_df['W/L'] == 'W'), 1, 0)
    #print(increase_df['Method'].value_counts().head(10))
increase_df['Sub_W'] = np.where((increase_df['Sub'] == 1) & (increase_df['W/L'] == 'W'), 1, 0)
increase_df['Sub_L'] = np.where((increase_df['Sub'] == 1) & (increase_df['W/L'] == 'L'), 1, 0)

#make columns for the models
increase_df["event_time"] = (increase_df["Match_year"] - increase_df["Treat_year"])
increase_df["event_bin"] = increase_df["event_time"].apply(bin_event_time)
increase_df["post"] = (increase_df["Match_year"] >= increase_df["Treat_year"]).astype(int)

#filter for only submission data but maintain pre and post treatment matches for each fighter
increase_sub_df = increase_df[increase_df['Sub'] == 1].copy()
grouping = increase_sub_df.groupby('Fighter')
has_post = grouping.apply(lambda g: (g['Match_year'] > g['Treat_year']).any())
has_pre  = grouping.apply(lambda g: (g['Match_year'] < g['Treat_year']).any())
is_treated = grouping['Treated'].max()
sub_fighters = (is_treated == 0) | (has_post & has_pre)
increase_sub_df = increase_sub_df[increase_sub_df['Fighter'].isin(sub_fighters.index)]

#RESTRICT TO ONLY ONE YEAR
#print(increase_sub_df['Treat_year'].value_counts())
increase_df_year = increase_df[(increase_df['Treat_year'] == 2022) | (increase_df['Treat_year'] == np.inf)] 
increase_df_year['post_treatment'] = (increase_df_year['Match_year'] > increase_df_year['Treat_year']).astype(int)
increase_df_year['unit'] = increase_df_year['Fighter']
increase_df_year['group'] = increase_df_year['Treated']

#debugging, checking post treament observations
dummy = increase_df_year.loc[increase_df_year['Treated'] == 1, 'Fighter']
dummy2 = dummy.nunique()
#print("Number of treated fighters in the year", dummy2)
#print("post treatment sum", increase_df_year['post_treatment'].sum())
#print(increase_df_year[increase_df_year["event_time"] > 0]["event_time"].value_counts().sort_index())


#exclude t = 0 as reference year becuase it's a mixture year, rather than an exact treatment time? nah, sticking with -1, think about it more later maybe

#DECREASE BATCH- doing all the same things I just did for the increase fighters for those that were monotonic decreasing

decrease_treated_df = pd.merge(year_groups_df, df_b3, on='Fighter', how='left').rename(columns= {'Year_x': 'Treat_year', 'Year_y': 'Match_year'})
decrease_treated_df['Match_year'] = decrease_treated_df['Match_year'].astype(int)
decrease_treated_df['Treated'] = 1
df_control_infty_D = df_b3[df_b3['Fighter'].isin(infty_control_D)].rename(columns = {'Year': 'Match_year'})
df_control_infty_D['Treated'] = 0
df_control_infty_D['Treat_year'] = np.inf
df_control_infty_D['Match_year'] = df_control_infty_D['Match_year'].astype(int)

decrease_df = pd.concat([decrease_treated_df, df_control_infty_D], ignore_index=True)

grouping_init_D = decrease_df.groupby('Fighter')
has_post_init_D = grouping_init_D.apply(lambda g: (g['Match_year'] > g['Treat_year']).any())
has_pre_init_D  = grouping_init_D.apply(lambda g: (g['Match_year'] < g['Treat_year']).any())
is_treated_init_D = grouping_init_D['Treated'].max()
fighters_init_D = (is_treated_init_D == 0) | (has_post_init_D & has_pre_init_D)
decrease_df = decrease_df[decrease_df['Fighter'].isin(fighters_init_D.index)]

decrease_df['Armbar'] = np.where((increase_df['Method'] == 'Armbar') & (decrease_df['W/L'] == 'W'), 1, 0)
decrease_df['Triangle'] = np.where((increase_df['Method'] == 'Triangle') & (decrease_df['W/L'] == 'W'), 1, 0)
decrease_df['RNC'] = np.where((decrease_df['Method'] == 'RNC') & (decrease_df['W/L'] == 'W'), 1, 0)
decrease_df['Choke_from_back'] = np.where((increase_df['Method'] == 'Choke from back') & (decrease_df['W/L'] == 'W'), 1, 0)
    #print(increase_df['Method'].value_counts().head(10))
decrease_df['Sub_W'] = np.where((decrease_df['Sub'] == 1) & (decrease_df['W/L'] == 'W'), 1, 0)
decrease_df['Sub_L'] = np.where((decrease_df['Sub'] == 1) & (decrease_df['W/L'] == 'L'), 1, 0)

#make columns for the models
decrease_df["event_time"] = (decrease_df["Match_year"] - decrease_df["Treat_year"])
decrease_df["event_bin"] = decrease_df["event_time"].apply(bin_event_time)
decrease_df["post"] = (decrease_df["Match_year"] >= decrease_df["Treat_year"]).astype(int)

#filter for only submission data but maintain pre and post treatment matches for each fighter
decrease_sub_df = decrease_df[decrease_df['Sub'] == 1].copy()
grouping_D = decrease_sub_df.groupby('Fighter')
has_post_D = grouping_D.apply(lambda g: (g['Match_year'] > g['Treat_year']).any())
has_pre_D  = grouping_D.apply(lambda g: (g['Match_year'] < g['Treat_year']).any())
is_treated_D = grouping['Treated'].max()
sub_fighters_D = (is_treated_D == 0) | (has_post_D & has_pre_D)
decrease_sub_df = decrease_sub_df[decrease_sub_df['Fighter'].isin(sub_fighters_D.index)]

#ONLY ONE YEAR
#print(increase_sub_df['Treat_year'].value_counts())
decrease_df_year = decrease_df[(decrease_df['Treat_year'] == 2022) | (decrease_df['Treat_year'] == np.inf)] 
decrease_df_year['post_treatment'] = (decrease_df_year['Match_year'] > decrease_df_year['Treat_year']).astype(int)
decrease_df_year['unit'] = decrease_df_year['Fighter']
decrease_df_year['group'] = decrease_df_year['Treated']


#METHODS APPLICATIONS

#DiD single year
result = cp.DifferenceInDifferences(
    decrease_df_year,
    formula="Sub_L ~ 1 + group*post_treatment",
    time_variable_name="Match_year",
    group_variable_name="group",
    model=LinearRegression(),
)

fig, ax = result.plot(round_to=3)

stats = result.effect_summary()
stats.table
table = stats.table.round(6)
print(table)



#paralell trends, looking at it as an event study
'''
event_df = decrease_df_year.copy()
event_df = event_df[event_df['event_time'] != -1]

model = smf.ols( "Sub_L ~ C(event_time) * group",data=event_df).fit()

coefs = model.params
conf = model.conf_int()

event_effects = []

for term in coefs.index:
    if "C(event_time)" in term and ":group" in term:
        time = int(float(term.split("[T.")[1].split("]")[0]))
        event_effects.append({
            "event_time": time,
            "effect": coefs[term],
            "lower": conf.loc[term, 0],
            "upper": conf.loc[term, 1],
        })

event_df_plot = pd.DataFrame(event_effects).sort_values("event_time")

plt.figure(figsize=(10,6))

plt.plot(event_df_plot["event_time"], event_df_plot["effect"], marker='o')
plt.fill_between(
    event_df_plot["event_time"],
    event_df_plot["lower"],
    event_df_plot["upper"],
    alpha=0.2
)

plt.axhline(0, color='black', linestyle='--')
plt.axvline(0, color='red', linestyle='--', label="Treatment")
plt.title("Event Study: Dynamic Treatment Effects")
plt.xlabel("Event Time (years relative to treatment)")
plt.ylabel("Effect on y")
plt.legend()
plt.show()
'''

#Staggered Difference in Differences for all treatment groups
'''
result = cp.StaggeredDifferenceInDifferences(
    increase_df,
    formula="Sub_W ~ 1 + C(Fighter) + (Treated * post) + C(Match_year)",
    unit_variable_name="Fighter",
    time_variable_name="Match_year",
    treated_variable_name="Treated",
    treatment_time_variable_name="Treat_year",
    model=cp.pymc_models.LinearRegression(
        sample_kwargs={
            "progressbar": True,
            "random_seed": 42,
            "tune": 200,    
            "draws": 200,   
            "chains": 2,    
            "cores": 2 
        }
    ),
)
print(result.summary())
'''
#+  + C(event_bin) for using the time group pooling
#fig, ax = result.plot() 


#FREQUENCY OF FINISHES AGAINST WEIGHT
'''
def method_freq(df, var, weight):
    count_t = (df['Cleaned_Weight'] == weight).sum()
    if count_t == 0: return 0
    count_var = (((df['Method']) == var) & (df['Cleaned_Weight'] == weight)).sum()
    return count_var / count_t
# 2. Setup variables
fixed_var = 'Armbar' # Replace with your specific method
unique_weights = sorted(df_b3['Cleaned_Weight'].unique())
# 3. Calculate frequencies for the y-axis
frequencies = [method_freq(df_b3, fixed_var, w) for w in unique_weights]
unique_weights = np.array(sorted(df_b3['Cleaned_Weight'].unique()))
frequencies = np.array([method_freq(df_b3, fixed_var, w) for w in unique_weights])
# 4. Create the plot
plt.figure(figsize=(10, 6))
plt.scatter(unique_weights, frequencies, color='blue', label='Actual Data')
m, b = np.polyfit(unique_weights, frequencies, 1)
plt.plot(unique_weights, m * unique_weights + b, color='red', 
         label=f'Linear Fit (y={m:.4f}x + {b:.4f})')
plt.title(f'Frequency of {fixed_var} by Weight Class')
plt.xlabel('Weight')
plt.ylabel('ArmbarFrequency')
plt.grid(True)
plt.show()

#print(method_freq(df_b3, 'Armbar', ))
'''

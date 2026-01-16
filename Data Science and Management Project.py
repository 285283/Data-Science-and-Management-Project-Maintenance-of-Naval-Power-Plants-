import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pymysql
import cryptography
from sqlalchemy import create_engine
from sqlalchemy import text
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. ------ LOADING THE DATASET --------

try:
    df = pd.read_csv("navalplantmaintenance.csv", header=None, sep=r"\s+")
except FileNotFoundError:
    print("ERROR: File 'navalplantmaintenance.csv' not found.")
    exit()

df.columns = [
    'lp', 'v', 'GTT', 'GTn', 'GGn', 'Ts', 'Tp', 'T48', 'T1', 'T2',
    'P48', 'P1', 'P2', 'Pexh', 'TIC', 'mf', 'GT1', 'GT2'
]

# -------------  POINT 1 - DATASET EXPLORATION ---------------
if __name__ == '__main__':

    ### Structural Analysis
    # Check 1: Printing dataset dimensions
    print("\nDataset Shape (Rows, Columns):")
    print(df.shape)
    # Check 2: Data Types
    print("\nData Types:")
    print(df.dtypes)
    # Check 3: Missing Values
    print("\nMissing Values Count:")
    print(df.isnull().sum())

    ### Statistics
    statistica = (df.describe())
    pd.set_option('display.max_columns', None)
    print("\nDescriptive Statistics:")
    print(statistica)

    # --- GRAPHICAL EXPLORATION ---

    # GRAPH 1: Distribution Analysis (Fuel Flow)
    # Useful to understand ship operating regimes (low vs high load)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(df['mf'], bins=40, color='purple', alpha=0.6, edgecolor='black')
    ax.set_title("Fuel Flow Distribution")
    ax.set_xlabel("Fuel Flow (mf) [kg/s]")
    ax.set_ylabel("Frequency (Number of observations)")
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.show()

    # GRAPH 2: Control Analysis (Scatterplot)
    # Shows the logical relationship between command (Lever) and response (Speed)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df['lp'], df['v'], color='teal', alpha=0.5, s=15)
    ax.set_title("Control Verification: Lever Position vs Speed")
    ax.set_xlabel("Lever Position (lp)")
    ax.set_ylabel("Ship Speed (v) [knots]")
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.show()

    # GRAPH 3: Efficiency Analysis (Scatterplot)
    # Relationship energy input (mf) vs mechanical output (GTT)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df['mf'], df['GTT'], color='blue', alpha=0.3, s=15)
    ax.set_title("Efficiency Analysis: Fuel vs Torque")
    ax.set_xlabel("Fuel Flow (mf) [kg/s]")
    ax.set_ylabel("GT Shaft Torque (GTT) [kN m]")
    ax.grid(True)
    plt.show()

# --------------- POINT 2 - CONSISTENCY & OUTLIERS-------------------

print(" \n --- POINT 2 ANALYSIS: CONSISTENCY & OUTLIERS ---")

# 3. NORMALIZED BOXPLOT
# Normalize data (0-1) to compare sensors with different units
# Formula: (x - min) / (max - min)
df_norm = (df - df.min()) / (df.max() - df.min())

plt.figure(figsize=(14, 8))
plt.boxplot(df_norm.values, tick_labels=df_norm.columns)
plt.title("Outlier Analysis (Normalized Boxplot)")
plt.ylabel("Normalized Scale (0-1)")
plt.xlabel("Variables")
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# 4. STATISTICAL OUTLIERS (IQR)
print("\n3. Outlier Count (IQR Method):")

for col in df.columns:
    # Calculate quartiles as per theory
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    # Define limits (boxplot whiskers)
    low = Q1 - 1.5 * IQR
    high = Q3 + 1.5 * IQR

    # Filter values outside limits
    outliers = df[(df[col] < low) | (df[col] > high)]
    num = len(outliers)

    if num > 0:
        print(f" - {col}: {num} outliers")


# --- PHYSICAL RANGE CHECKS  ---
print("\n--- PHYSICAL RANGE CHECKS (Domain Knowledge) ---")

# 1. Lever position (lp)
physical_outliers = df[(df['lp'] < 0) | (df['lp'] > 10)]
print(len(physical_outliers), "Outliers in Lever position")

# 2. Ship speed (v) [knots]
physical_outliers = df[(df['v'] < 0) | (df['v'] > 35)]
print(len(physical_outliers), "Outliers in Ship speed")

# 3. Gas Turbine (GT) shaft torque (GTT) [kN m]
physical_outliers = df[(df['GTT'] < 0) | (df['GTT'] > 3000)]
print(len(physical_outliers), "Outliers in Gas Turbine")

# 4. GT rate of revolutions (GTn) [rpm]
physical_outliers = df[(df['GTn'] < 0) | (df['GTn'] > 4500)]
print(len(physical_outliers), "Outliers in GT rate of revolutions")

# 5. Gas Generator rate of revolutions (GGn) [rpm]
physical_outliers = df[(df['GGn'] < 5000) | (df['GGn'] > 12000)]
print(len(physical_outliers), "Outliers in Gas Generator rate of revolutions")

# 6. Starboard Propeller Torque (Ts) [kN]
physical_outliers = df[(df['Ts'] < 0) | (df['Ts'] > 1000)]
print(len(physical_outliers), "Outliers in Starboard Propeller Torque")

# 7. Port Propeller Torque (Tp) [kN]
physical_outliers = df[(df['Tp'] < 0) | (df['Tp'] > 1000)]
print(len(physical_outliers), "Outliers in Port Propeller Torque")

# 8. HP Turbine exit temperature (T48)
physical_outliers = df[(df['T48'] < 400) | (df['T48'] > 1500)]
print(len(physical_outliers), "Outliers in Hight Pressure (HP) Turbine exit temp (Kelvin)")

# 9. GT Compressor inlet air temperature (T1)
physical_outliers = df[(df['T1'] < 200) | (df['T1'] > 350)]
print(len(physical_outliers), "Outliers in GT Compressor inlet air temp (Kelvin)")

# 10. GT Compressor outlet air temperature (T2)
physical_outliers = df[(df['T2'] < 200) | (df['T2'] > 900)]
print(len(physical_outliers), "Outliers in GT Compressor outlet air temp (Kelvin)")

# 11. HP Turbine exit pressure (P48) [bar]
physical_outliers = df[(df['P48'] < 1) | (df['P48'] > 10)]
print(len(physical_outliers), "Outliers in HP Turbine exit pressure (P48)")

# 12. GT Compressor inlet air pressure (P1) [bar]
physical_outliers = df[(df['P1'] < 0.8) | (df['P1'] > 1.1)]
print(len(physical_outliers), "Outliers in GT Compressor inlet air pressure (P1)")

# 13. GT Compressor outlet air pressure (P2) [bar]
physical_outliers = df[(df['P2'] < 8) | (df['P2'] > 25)]
print(len(physical_outliers), "Outliers in GT Compressor outlet air pressure (P2)")

# 14. GT exhaust gas pressure (Pexh) [bar]
physical_outliers = df[(df['Pexh'] < 1.0) | (df['Pexh'] > 1.1)]
print(len(physical_outliers), "Outliers in GT exhaust gas pressure (Pexh)")

# 15. Turbine Injection Control (TIC) [%]
physical_outliers = df[(df['TIC'] < 0) | (df['TIC'] > 100)]
print(len(physical_outliers), "Outliers in Turbine Injection Control (TIC)")

# 16. Fuel flow (mf) [kg/s]
physical_outliers = df[(df['mf'] < 0) | (df['mf'] > 4.0)]
print(len(physical_outliers), "Outliers in Fuel flow (mf)")

# 17. GT Compressor decay state coefficient
physical_outliers = df[(df['GT1'] < 0.9) | (df['GT1'] > 1.0)]
print(len(physical_outliers), "Outliers in GT Compressor decay state coefficient")

# 18. GT Turbine decay state coefficient
physical_outliers = df[(df['GT2'] < 0.9) | (df['GT2'] > 1.0)]
print(len(physical_outliers), "Outliers in GT Turbine decay state coefficient")

# --------------- POINT 3 - DATABASE CREATION -------------------
print("\n--- POINT 3: SQL DATABASE POPULATION (Relational Schema) ---")

# 1. Reaload Dataset with full names (For DB Metadata)
try:
    df_sql = pd.read_csv("navalplantmaintenance.csv", sep='\s+', header=None,
                         names=['Lever position (lp) [ ]', 'Ship speed (v) [knots]',
                                'Gas Turbine (GT) shaft torque (GTT) [kN m]', 'GT rate of revolutions (GTn) [rpm]',
                                'Gas Generator rate of revolutions (GGn) [rpm]', 'Starboard Propeller Torque (Ts) [kN]',
                                'Port Propeller Torque (Tp) [kN]',
                                'Hight Pressure (HP) Turbine exit temperature (T48) [C]',
                                'GT Compressor inlet air temperature (T1) [C]',
                                'GT Compressor outlet air temperature (T2) [C]', 'HP Turbine exit pressure (P48) [bar]',
                                'GT Compressor inlet air pressure (P1) [bar]',
                                'GT Compressor outlet air pressure (P2) [bar]', 'GT exhaust gas pressure (Pexh) [bar]',
                                'Turbine Injection Control (TIC) [%]', 'Fuel flow (mf) [kg/s]',
                                'GT Compressor decay state coefficient', 'GT Turbine decay state coefficient'])

    # 2. MySQL Connection Parameters
    USER = 'root'
    PASSWORD = '1234321'
    HOST = 'localhost'
    DATABASE = 'naval_vessel'

    # Connection String
    engine = create_engine(f'mysql+pymysql://{USER}:{PASSWORD}@{HOST}/{DATABASE}')
    print("Connecting to MySQL")

    # 3. Master table load
    df_sql.to_sql('dati', engine, if_exists='replace', index=False)
    print("Master table 'dati' loaded successfully!")

    # --- DATA WAREHOUSING: SPLITTING INTO LOGICAL TABLES ---

    # TABLE 1: GAS TURBINE (GT)
    mapping_gt = {
        'Gas Turbine (GT) shaft torque (GTT) [kN m]': 'Gas_Turbine_GT_shaft_torque',
        'GT rate of revolutions (GTn) [rpm]': 'GT_rate_of_revolutions',
        'GT exhaust gas pressure (Pexh) [bar]': 'GT_exhaust_gas_pressure',
        'GT Turbine decay state coefficient': 'GT_Turbine_decay_state_coefficient'
    }
    df_gt = df_sql[list(mapping_gt.keys())].rename(columns=mapping_gt)
    if 'id' not in df_gt.columns: df_gt.insert(0, 'id', range(1, 1 + len(df_gt)))

    # TABLE 2: COMPRESSOR
    mapping_gt_compressor = {
        'GT Compressor inlet air temperature (T1) [C]': 'GT_Compressor_inlet_air_temperature',
        'GT Compressor outlet air pressure (P2) [bar]': 'GT_Compressor_outlet_air_pressure',
        'GT Compressor inlet air pressure (P1) [bar]': 'GT_Compressor_inlet_air_pressure',
        'GT Compressor outlet air temperature (T2) [C]': 'GT_Compressor_outlet_air_temperature',
        'GT Compressor decay state coefficient': 'GT_Compressor_decay_state_coefficient'
    }
    df_gt_compressor = df_sql[list(mapping_gt_compressor.keys())].rename(columns=mapping_gt_compressor)
    if 'id' not in df_gt_compressor.columns: df_gt_compressor.insert(0, 'id', range(1, 1 + len(df_gt_compressor)))

    # TABLE 3: HIGH PRESSURE TURBINE
    mapping_hight_pressure = {
        'Hight Pressure (HP) Turbine exit temperature (T48) [C]': 'Hight_Pressure_HP_Turbine_exit_temperature',
        'HP Turbine exit pressure (P48) [bar]': 'HP_Turbine_exit_pressure'
    }
    df_hight_pressure = df_sql[list(mapping_hight_pressure.keys())].rename(columns=mapping_hight_pressure)
    if 'id' not in df_hight_pressure.columns: df_hight_pressure.insert(0, 'id', range(1, 1 + len(df_hight_pressure)))

    # TABLE 4: PROPELLERS
    mapping_propeller = {
        'Starboard Propeller Torque (Ts) [kN]': 'Starboard_Propeller_Torque',
        'Port Propeller Torque (Tp) [kN]': 'Port_Propeller_Torque'
    }
    df_propeller = df_sql[list(mapping_propeller.keys())].rename(columns=mapping_propeller)
    if 'id' not in df_propeller.columns: df_propeller.insert(0, 'id', range(1, 1 + len(df_propeller)))

    # TABLE 5: SHIP DYNAMICS
    mapping_ship = {
        'Lever position (lp) [ ]': 'Lever_position',
        'Fuel flow (mf) [kg/s]': 'Fuel_flow',
        'Ship speed (v) [knots]': 'Ship_speed',
        'Turbine Injection Control (TIC) [%]': 'Turbine_Injection_Control',
        'Gas Generator rate of revolutions (GGn) [rpm]': 'Gas_Generator_rate_of_revolutions',
    }
    df_ship = df_sql[list(mapping_ship.keys())].rename(columns=mapping_ship)
    if 'id' not in df_ship.columns: df_ship.insert(0, 'id', range(1, 1 + len(df_ship)))

    # --- SAFE LOADING WITH TRANSACTION ---
    print("Populating Normalized Tables...")
    with engine.begin() as conn:
        conn.execute(text("SET FOREIGN_KEY_CHECKS = 0"))

        # Load all tables
        df_gt.to_sql('gt', conn, if_exists='replace', index=False)
        df_gt_compressor.to_sql('gt_compressor', conn, if_exists='replace', index=False)
        df_hight_pressure.to_sql('hight_pressure', conn, if_exists='replace', index=False)
        df_propeller.to_sql('propeller', conn, if_exists='replace', index=False)
        df_ship.to_sql('ship', conn, if_exists='replace', index=False)
        conn.execute(text("SET FOREIGN_KEY_CHECKS = 1"))
    print("SUCCESS: Data Warehouse Schema created and populated in MySQL.")

except Exception as e:
    print(f"\n[ERROR] MySQL Connection failed: {e}")
    print("Check: 1. Is MySQL running? 2. Did you create 'naval_vessel' DB? 3. Is password correct?")

#---- POINT 4: TEMPORAL DECAY ANALYSIS -----
print("\n--- POINT 4 EXECUTION: DECAY ANALYSIS ---")

# Time axis creation (Index acts as simulation time step)
time_axis = df.index
# Plot Generation
# GT1 = Compressor Decay | GT2 = Turbine Decay
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Plot 1: Compressor Decay (GT1)
ax1.plot(time_axis, df['GT1'], color='#1f77b4', linewidth=0.5, alpha=0.8)
ax1.set_title("Target 1: Compressor Decay State (GT1)", fontsize=12, fontweight='bold')
ax1.set_ylabel("Coefficient (1.0 = Optimal)", fontsize=10)
ax1.grid(True, linestyle='--', alpha=0.6)

# Plot 2: Turbine Decay (GT2)
ax2.plot(time_axis, df['GT2'], color='#d62728', linewidth=0.5, alpha=0.8)
ax2.set_title("Target 2: Turbine Decay State (GT2)", fontsize=12, fontweight='bold')
ax2.set_xlabel("Simulation Time (Instance Index)", fontsize=10)
ax2.set_ylabel("Coefficient (1.0 = Optimal)", fontsize=10)
ax2.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()

# --------------- POINT 5: PREDICTIVE MODELING ---------------------------------
print("\n--- POINT 5: PREDICTIVE MODELING ---")

# 1. DATA PREPARATION
X = df.drop(columns=['GT1', 'GT2']) #Features
y = df[['GT1', 'GT2']] #Target

# 2. SPLIT DATI
# We use 70/30 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- Data summary section ---
print("\n              DATASET COMPOSITION INFO")
print(f" -> Total Dataset Rows:      {len(df)}")
print(f" -> Training Set (70%):      {len(X_train)} samples")
print(f" -> Test Set (30%):          {len(X_test)} samples")
print(f" -> Input Columns (X):       {X.shape[1]} sensors")
print(f" -> Target Columns (y):      {y.shape[1]} (GT1 Compressor, GT2 Turbine)")

# 3. LINEAR REGRESSION
print("\n[A] Training Linear Regression ")

# Setup variables
X3_train = X_train
X3_test = X_test
model3 = LinearRegression()
model3.fit(X3_train, y_train)
y3_pred = model3.predict(X3_test)

# Metrics
print("Linear Regression MAE:", mean_absolute_error(y_test, y3_pred))
print("Linear Regression MSE:", mean_squared_error(y_test, y3_pred))

# Plot Linear Regression
plt.figure(figsize=(12, 5))
# GT1
plt.subplot(1, 2, 1)
plt.scatter(y_test['GT1'], y3_pred[:, 0], alpha=0.5, color='blue')
plt.plot([0.95, 1.0], [0.95, 1.0], 'r--')
plt.title(f"Linear Regression - GT1")
plt.xlabel("Actual GT1")
plt.ylabel("Predicted GT1")
plt.grid(True)

# GT2
plt.subplot(1,2,2)
plt.scatter(y_test['GT2'], y3_pred[:,1], alpha=0.5, color='green')
plt.plot([0.975,1.0],[0.975,1.0], 'r--')
plt.title(f"Linear Regression - GT2")
plt.xlabel("Actual GT2")
plt.ylabel("Predicted GT2")
plt.xlim(0.975, 1.000)
plt.ylim(0.975, 1.000)
plt.grid(True)
plt.tight_layout()
plt.show()

# 4. RANDOM FOREST
print("\n[B] Training Random Forest")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X3_train, y_train)
y_pred_rf = rf_model.predict(X3_test)

# Metrics
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"Random Forest MAE:     {mae_rf:.6f}")
print(f"Random Forest R2:      {r2_rf:.4f}")

# Plot Random Forest
plt.figure(figsize=(12, 5))

# GT1 Random Forest
plt.subplot(1, 2, 1)
plt.scatter(y_test['GT1'], y_pred_rf[:, 0], alpha=0.5, color='purple')
plt.plot([0.95, 1.00], [0.95, 1.00], 'r--', linewidth=2)
plt.title(f"Random Forest - GT1 (R2: {r2_score(y_test['GT1'], y_pred_rf[:,0]):.3f})")
plt.xlabel("Actual GT1")
plt.ylabel("Predicted GT1")
plt.xlim(0.95, 1.00)
plt.ylim(0.95, 1.00)
plt.grid(True)

# GT2 Random Forest
plt.subplot(1, 2, 2)
plt.scatter(y_test['GT2'], y_pred_rf[:, 1], alpha=0.5, color='orange')
plt.plot([0.975, 1.00], [0.975, 1.00], 'r--', linewidth=2)
plt.title(f"Random Forest - GT2 (R2: {r2_score(y_test['GT2'], y_pred_rf[:,1]):.3f})")
plt.xlabel("Actual GT2")
plt.ylabel("Predicted GT2")
plt.xlim(0.975, 1.00)
plt.ylim(0.975, 1.00)
plt.grid(True)

plt.tight_layout()
plt.show()
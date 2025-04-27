# --- Imports ---
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="🌬️ Wind Farm Predictor", layout="wide")
st.title("🌬️ Wind Farm Production Predictor")

# --- Function Definitions ---

def generate_weighted_forecasts(df_input, alpha=0.9):
    def parse_forecast_column(col_name):
        parts = col_name.split('_')
        model = parts[0]
        hour = int(parts[1].replace('h', ''))
        day_part = parts[2]
        if day_part == 'D':
            day_offset = 0
        elif day_part.startswith('D-'):
            day_offset = -int(day_part.replace('D-', ''))
        elif day_part.startswith('D+'):
            day_offset = int(day_part.replace('D+', ''))
        else:
            raise ValueError(f"Unexpected day part: {day_part}")
        var = parts[3]
        offset = pd.Timedelta(days=day_offset, hours=hour)
        predictor = f"{model}_{var}"
        return predictor, offset

    def reshape_forecast_block(block):
        col_name = [c for c in block.columns if c.startswith('NWP')][0]
        predictor, offset = parse_forecast_column(col_name)
        b = block.copy()
        b['predictor'] = predictor
        b['Time'] = pd.to_datetime(b['Time'], format='%d/%m/%Y %H:%M')
        issued_at = b['Time'].dt.normalize() + offset
        b['delay_hours'] = (b['Time'] - issued_at).dt.total_seconds() / 3600.0
        b = b.rename(columns={col_name: 'forecast_value'})
        b = b.drop(columns=['Time'])
        return b.dropna(subset=['forecast_value'])

    long_df = pd.concat(
        [reshape_forecast_block(df_input[['Time', 'ID', col]])
         for col in df_input.columns if col.startswith('NWP')],
        ignore_index=True
    )

    long_df['weight'] = alpha ** long_df['delay_hours']
    long_df['w_value'] = long_df['weight'] * long_df['forecast_value']

    best = (
        long_df.groupby(['ID', 'predictor'])[['w_value', 'weight']]
        .sum()
        .eval('w_value / weight')
        .unstack(level='predictor')
    )

    non_nwp = [c for c in df_input.columns if not c.startswith('NWP')]
    non_nwp = pd.Index(non_nwp).drop_duplicates().tolist()

    result = pd.concat(
        [df_input[non_nwp].reset_index(drop=True),
         best.reset_index(drop=True)],
        axis=1
    )

    return result

def interpolate_nans(df):
    nwp_cols = [c for c in df.columns if c.startswith('NWP')]
    df = df.sort_values(['WF', 'Time'])
    df[nwp_cols] = df.groupby('WF')[nwp_cols].transform(lambda grp: grp.interpolate(method='linear', limit_direction='both'))
    return df

def append_features(df):
    nwp_bases = sorted({col.split('_')[0] for col in df.columns if col.startswith('NWP')})
    for base in nwp_bases:
        u_col = f"{base}_U"
        v_col = f"{base}_V"
        if u_col in df.columns and v_col in df.columns:
            df[f"{base}_WS"] = np.sqrt(df[u_col]**2 + df[v_col]**2)
            df[f"{base}_WD"] = (np.degrees(np.arctan2(df[v_col], df[u_col])) + 360) % 360
    return df

# --- Load Model and Scaler ---
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load('model.pkl')   # This matches your model file
    scaler = joblib.load('scalers.pkl')  # This matches your scaler file
    return model, scaler

model, scaler = load_model_and_scaler()

# --- Streamlit Interface ---



uploaded_file = st.file_uploader("Upload New Test Data (CSV)", type="csv")

if uploaded_file:
    df_test = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data Preview")
    st.dataframe(df_test.head())

    # --- Preprocessing ---

    # 1. Weighted forecasts
    df_work = df_test.copy().reset_index()
    df_weighted = generate_weighted_forecasts(df_work, alpha=0.9)

    cols = ['ID'] + [c for c in df_weighted.columns if c != 'ID']
    df_weighted = df_weighted[cols]
    df_weighted['WF'] = df_weighted['WF'].str.replace(r'^WF', '', regex=True).astype(int)
    df_weighted.set_index('ID', drop=True, inplace=True)

    # 2. Interpolation
    df_interpolated = interpolate_nans(df_weighted.copy())

    # 3. Add WS and WD
    df_interpolated = append_features(df_interpolated)

    # 4. Select feature columns
    feature_cols = [c for c in df_interpolated.columns if any(c.endswith(suffix) for suffix in ['_U', '_V', '_T', '_WS', '_WD'])]

    X_new = df_interpolated[feature_cols]

    # --- Scaling ---
    X_new_scaled = []

    for wf, sub in df_interpolated.groupby('WF'):
        sub = sub.copy()
        feature_cols = [c for c in sub.columns if any(c.endswith(suffix) for suffix in ['_U', '_V', '_T', '_WS', '_WD'])]
        
        if wf in scaler:
            scaler_wf = scaler[wf]  # Get the right scaler for this farm
            # Only use the features that the scaler was trained on
            expected_cols = scaler_wf.feature_names_in_
            available_cols = [col for col in expected_cols if col in sub.columns]

            # Optional: Warn if some expected columns are missing
            missing_cols = set(expected_cols) - set(available_cols)
            if missing_cols:
                st.warning(f"Missing columns for WF {wf}: {missing_cols}")

            # Transform only available features
            sub_scaled = scaler_wf.transform(sub[available_cols])
            X_new_scaled.append((wf, sub, sub_scaled))

            X_new_scaled.append((wf, sub, sub_scaled))
        else:
            st.warning(f"No scaler found for WF {wf}!")


    # --- Prediction ---
    X_all_scaled = np.vstack([scaled for wf, sub, scaled in X_new_scaled])
    predictions = model.predict(X_all_scaled)


    # Now update the df_interpolated with predictions
    all_subs = pd.concat([sub.reset_index(drop=True) for wf, sub, scaled in X_new_scaled], ignore_index=True)
    all_subs['Predicted_Production'] = predictions

    # --- Show Results ---
    st.subheader("Predicted Results")
    st.dataframe(all_subs[['WF', 'Time', 'Predicted_Production']])

    # --- Download Results ---
    csv = all_subs[['WF', 'Time', 'Predicted_Production']].to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions as CSV", data=csv, file_name='predicted_output.csv', mime='text/csv')



else:
    st.info("Please upload a CSV file to continue.")

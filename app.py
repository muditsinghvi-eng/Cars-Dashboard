import streamlit as st
import pandas as pd
import plotly.express as px

# 1. Page Config
st.set_page_config(layout="wide", page_title="Car Performance Dashboard")

# 2. Load Data (Local Files)
@st.cache_data(ttl=600)
def load_all_data():
    file1 = "Cars Dashboard Mobile.xlsx"
    file2 = "State wise DBH.xlsx"
    sov_sheets = {
        "search": "Search", "pv": "pv", "uu": "uu", 
        "mofu": "mofu", "leads": "leads", "gender": "gender", 
        "age": "Age", "reg_traffic": "Region", "reg_leads": "Lead Region",
        "te_rte": "te_rte", "le_rle": "le_rle"
    }
    sov_data = {}
    try:
        for key, sheet in sov_sheets.items():
            df = pd.read_excel(file1, sheet_name=sheet)
            df = df.dropna(how='all', axis=0).dropna(how='all', axis=1)
            df.columns = [str(c).strip() for c in df.columns]
            name_col = 'model name' if 'model name' in df.columns else 'Model Name'
            if name_col in df.columns:
                df[name_col] = df[name_col].astype(str).str.strip()
                df = df.rename(columns={name_col: 'model name'})
            sov_data[key] = df
    except Exception as e:
        st.error(f"Error loading {file1}: {e}")

    bench_data = None
    try:
        bench_df = pd.read_excel(file2, sheet_name="Dump")
        bench_df = bench_df.dropna(how='all', axis=0).dropna(how='all', axis=1)
        
        # Mapping Column H (Index 7) as Leads as per user instruction
        if len(bench_df.columns) >= 8:
            new_cols = list(bench_df.columns)
            new_cols[7] = "Leads"
            bench_df.columns = new_cols

        bench_df.columns = [str(c).strip() for c in bench_df.columns]
        
        def robust_date_parse(series):
            res = pd.to_datetime(series, format='%b-%y', errors='coerce')
            mask = res.isna()
            if mask.any():
                res[mask] = pd.to_datetime(series[mask], errors='coerce')
            return res

        bench_df['MMM-YY_DT'] = robust_date_parse(bench_df['MMM-YY'])
        bench_df = bench_df.dropna(subset=['MMM-YY_DT'])
        bench_data = bench_df
    except Exception as e:
        st.error(f"Error loading {file2}: {e}")

    return sov_data, bench_data

sov_data, bench_data = load_all_data()

# 3. Sidebar Filters
if sov_data:
    all_models = sorted(sov_data['search']['model name'].dropna().unique().tolist())
    st.sidebar.header("Global Controls")
    main_model = st.sidebar.selectbox("Main Model", all_models, index=0)
    competitors = st.sidebar.multiselect("Select Competitors", [m for m in all_models if m != main_model])
    
    possible_months = ["Jan'25", "Feb'25", "Mar'25", "Apr'25", "May'25", "Jun'25", 
                       "Jul'25", "Aug'25", "Sep'25", "Oct'25", "Nov'25", "Dec'25"]
    existing_cols = sov_data['search'].columns.tolist()
    months_list = [m for m in possible_months if m in existing_cols]
    
    st.sidebar.subheader("Select Timeframe")
    start_m = st.sidebar.selectbox("Start Month", months_list, index=0)
    end_m = st.sidebar.selectbox("End Month", months_list, index=len(months_list)-1)

    s_idx, e_idx = months_list.index(start_m), months_list.index(end_m)
    if s_idx > e_idx:
        st.sidebar.error("Error: Start Month must be before End Month.")
        st.stop()
        
    selected_range = months_list[s_idx : e_idx + 1]
    all_selected_global = [main_model] + competitors
    mobile_config = {'staticPlot': True}

# Updated Helper for Consistent Bold Heatmaps & Dimensions
def style_heatmap(fig, range_len):
    fs = 14 if range_len <= 7 else (11 if range_len <= 10 else 9)
    fig.update_layout(
        xaxis_title=None, yaxis_title=None,
        coloraxis_showscale=False,
        margin=dict(t=40, l=5, r=5, b=5),
        height=280 # Matching Tab 1 PV/UU best dimension
    )
    fig.update_traces(textfont=dict(size=fs, weight='bold')) # Bold Text
    fig.update_xaxes(tickangle=45)
    return fig

tab_sov, tab_bench = st.tabs(["Market SOV Overview", "State Benchmarking"])

# ==========================================
# TAB 1: MARKET SOV OVERVIEW
# ==========================================
with tab_sov:
    st.title(f"Market Performance: {main_model}")
    
    # 1. SOV Trend
    sov_trend = []
    for m in selected_range:
        row = {"Month": m}
        for metric in ["search", "pv", "uu", "mofu", "leads"]:
            df = sov_data[metric]
            if m in df.columns:
                grp = df[df['model name'].isin(all_selected_global)]
                total = grp[m].sum()
                main_v = grp[grp['model name'] == main_model][m].sum()
                row[metric.upper()] = (main_v/total*100) if total > 0 else 0
            else: row[metric.upper()] = 0
        sov_trend.append(row)
    
    line_df = pd.DataFrame(sov_trend).melt(id_vars="Month")
    line_df['Label'] = line_df.apply(lambda x: f"<b>{x['value']:.1f}%</b>" if x['Month'] in [selected_range[0], selected_range[-1]] else "", axis=1)
    fig_line = px.line(line_df, x="Month", y="value", color="variable", markers=True, text="Label")
    fig_line.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, title=None),
        xaxis_title=None, yaxis_title=None, height=450, margin=dict(l=10, r=10, t=60, b=10)
    )
    fig_line.update_traces(textposition="top center")
    st.plotly_chart(fig_line, use_container_width=True, config=mobile_config)

    # 2. Performance Heatmaps
    st.divider()
    st.header("Performance Heatmaps")
    h_cols = st.columns(2)
    for i, m in enumerate(["search", "pv", "uu", "mofu", "leads", "pv/uu", "t2l"]):
        u_df = sov_data["uu"][sov_data["uu"]['model name'].isin(all_selected_global)].set_index('model name')[selected_range]
        if m in ["pv/uu", "t2l"]:
            if m == "pv/uu":
                p_df = sov_data["pv"][sov_data["pv"]['model name'].isin(all_selected_global)].set_index('model name')[selected_range]
                disp = (p_df / u_df).fillna(0); title, fmt = "PV/UU", ".2f"
            else:
                l_df = sov_data["leads"][sov_data["leads"]['model name'].isin(all_selected_global)].set_index('model name')[selected_range]
                disp = ((l_df/u_df)*100).fillna(0); title, fmt = "T2L %", ".2f"
        else:
            raw = sov_data[m][sov_data[m]['model name'].isin(all_selected_global)].set_index('model name')[selected_range]
            disp = (raw.div(raw.sum(), axis=1)*100).fillna(0); title, fmt = f"SOV: {m.upper()}", ".1f"
        
        with h_cols[i % 2]:
            fig = px.imshow(disp.reindex(all_selected_global), text_auto=fmt, color_continuous_scale="Reds", aspect="auto")
            fig = style_heatmap(fig, len(selected_range))
            fig.update_layout(title=title)
            st.plotly_chart(fig, use_container_width=True, config=mobile_config)

    # DISCLAIMER
    st.warning(f"**Note:** The data below (Erosion and State Contribution) is just for the month of {end_m}.")

    # 3. Competitive Erosion Analysis
    st.header("Competitive Erosion Analysis")
    def display_erosion(sheet_key, title):
        if sheet_key in sov_data:
            df = sov_data[sheet_key].copy()
            idx = next((i for i, col in enumerate(df.columns) if df[col].astype(str).str.contains(main_model).any()), None)
            if idx is not None:
                p_col = df.columns[idx]
                subset = df[df[p_col].astype(str).str.strip() == main_model].copy()
                cols = [c for i, c in enumerate(df.columns) if i != idx and "Unnamed" not in c]
                rank_col = df.columns[idx + 2] if idx + 2 < len(df.columns) else cols[0]
                st.subheader(title)
                st.dataframe(subset.sort_values(by=rank_col).head(10)[cols], use_container_width=True, hide_index=True)

    e1, e2 = st.columns(2)
    with e1: display_erosion("te_rte", "Traffic Erosion (UU)")
    with e2: display_erosion("le_rle", "Lead Erosion")

    # 4. States Contribution (Axis Bolded)
    st.divider()
    st.header("States Contribution")
    def get_geo_fig(key, main_m, comps, title, base_states=None):
        df = sov_data[key]
        cols = [c for c in df.columns if all(ex not in c.lower() for ex in ['model name', 'total', 'unnamed'])]
        m_row = df[df['model name'] == main_m]
        if base_states is None:
            if m_row.empty: return None, []
            states = m_row[cols].iloc[0][m_row[cols].iloc[0] > 0].sort_values(ascending=True).tail(10).index.tolist()
        else: states = [s for s in base_states if s in df.columns][::-1]
        p_data = []
        if not m_row.empty and states:
            t = m_row[cols].sum(axis=1).values[0]
            for s in states: p_data.append({"State": s, "Entity": main_m, "V": (m_row[s].values[0]/t*100 if t>0 else 0)})
        c_rows = df[df['model name'].isin(comps)]
        if not c_rows.empty and states:
            avgs = c_rows[cols].mean(); t = avgs.sum()
            for s in states: p_data.append({"State": s, "Entity": "Comp Avg", "V": (avgs[s]/t*100 if t>0 else 0)})
        if not p_data: return None, []
        fig = px.bar(pd.DataFrame(p_data), y="State", x="V", color="Entity", barmode="group", text_auto='.1f', orientation='h', title=title)
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, title=None), xaxis_title=None, yaxis_title=None, height=500, margin=dict(l=10, r=10, t=60, b=10))
        fig.update_traces(textfont=dict(weight='bold')) # Bold Bar Labels
        return fig, states[::-1]

    f_t, t_st = get_geo_fig('reg_traffic', main_model, competitors, "Traffic Contribution %")
    if f_t: st.plotly_chart(f_t, use_container_width=True, config=mobile_config)
    
    f_l, _ = get_geo_fig('reg_leads', main_model, competitors, "Leads Contribution %", base_states=t_st)
    if f_l: st.plotly_chart(f_l, use_container_width=True, config=mobile_config)

    # 5. State-wise T2L Heatmap
    st.subheader("State-wise Conversion (T2L %)")
    if t_st:
        v_s = [s for s in t_st if s in sov_data['reg_traffic'].columns and s in sov_data['reg_leads'].columns]
        if v_s:
            t_sb = sov_data['reg_traffic'][sov_data['reg_traffic']['model name'].isin(all_selected_global)].set_index('model name')[v_s]
            l_sb = sov_data['reg_leads'][sov_data['reg_leads']['model name'].isin(all_selected_global)].set_index('model name')[v_s]
            t2l_g = (l_sb / t_sb * 100).fillna(0).T 
            f_z = px.imshow(t2l_g, text_auto=".2f", color_continuous_scale="RdYlGn", aspect="auto")
            f_z.update_layout(coloraxis_showscale=False, xaxis_title=None, yaxis_title=None, height=550, margin=dict(l=5, r=5, t=10, b=5))
            f_z.update_traces(textfont=dict(size=12, color="black", weight='bold')) # Bold Heatmap Labels
            st.plotly_chart(f_z, use_container_width=True, config=mobile_config)

# ==========================================
# TAB 2: STATE DEEP-DIVE BENCHMARKING
# ==========================================
with tab_bench:
    st.title("State-Wise Benchmarking")
    if bench_data is not None:
        c1, c2 = st.columns(2)
        with c1:
            sel_models = st.multiselect("Select Models (Max 15)", options=sorted(bench_data['ParentModelName'].unique()), default=[main_model] if main_model in bench_data['ParentModelName'].unique() else None, max_selections=15)
        with c2:
            sel_regions = st.multiselect("Select States (Max 5)", options=sorted(bench_data['Region'].unique()), default=["Pan India"] if "Pan India" in bench_data['Region'].unique() else None, max_selections=5)
        
        start_p, end_p = pd.to_datetime(start_m.replace("'", "-"), format="%b-%y").to_period('M'), pd.to_datetime(end_m.replace("'", "-"), format="%b-%y").to_period('M')
        mask = (bench_data['ParentModelName'].isin(sel_models)) & (bench_data['Region'].isin(sel_regions)) & (bench_data['MMM-YY_DT'].dt.to_period('M') >= start_p) & (bench_data['MMM-YY_DT'].dt.to_period('M') <= end_p)
        f_df = bench_data[mask].copy()
        f_df['Month_Disp'] = f_df['MMM-YY_DT'].dt.strftime('%b-%y')

        if not f_df.empty:
            st.divider()
            st.subheader("Model Share & Engagement Benchmarking")
            hm_cols = st.columns(3) # Top Row (UU, PV, Depth)
            hm_cols2 = st.columns(2) # Bottom Row (Leads, T2L)
            
            metrics = ["UU", "PV", "PV/UU", "Leads", "T2L"]
            
            for i, m in enumerate(metrics):
                if m == "T2L":
                    agg = f_df.groupby(['ParentModelName', 'Month_Disp', 'MMM-YY_DT']).agg({'Leads':'sum', 'UU':'sum'}).reset_index()
                    agg['Val'] = (agg['Leads'] / agg['UU'] * 100).fillna(0)
                    title, fmt = "T2L %", ".2f"
                elif m == "PV/UU":
                    agg = f_df.groupby(['ParentModelName', 'Month_Disp', 'MMM-YY_DT']).agg({'PV':'sum', 'UU':'sum'}).reset_index()
                    agg['Val'] = (agg['PV'] / agg['UU']).fillna(0)
                    title, fmt = "Depth (PV/UU)", ".2f"
                elif m in ["UU", "PV", "Leads"]:
                    agg = f_df.groupby(['ParentModelName', 'Month_Disp', 'MMM-YY_DT'])[m].sum().reset_index()
                    agg['Total'] = agg.groupby('Month_Disp')[m].transform('sum')
                    agg['Val'] = (agg[m] / agg['Total'] * 100).fillna(0)
                    title, fmt = f"{m} SOV %", ".1f"
                
                hm = agg.pivot(index='ParentModelName', columns='Month_Disp', values='Val').reindex(sel_models)
                cols_ord = [d.strftime('%b-%y') for d in sorted(agg['MMM-YY_DT'].unique())]
                hm = hm[[c for c in cols_ord if c in hm.columns]]
                
                target_col = hm_cols[i] if i < 3 else hm_cols2[i-3]
                
                with target_col:
                    fig = px.imshow(hm, text_auto=fmt, color_continuous_scale="Reds", aspect="auto")
                    fig = style_heatmap(fig, len(selected_range)) # Applies Bold & 280 Height
                    fig.update_layout(title=title)
                    st.plotly_chart(fig, use_container_width=True, config=mobile_config)
        else: st.warning(f"No benchmarking data found for the window {start_m} to {end_m}.")
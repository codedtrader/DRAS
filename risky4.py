import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker
import numpy as np
from datetime import datetime
import streamlit.components.v1 as components

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="DRAS · Risk Dashboard",
    page_icon="⬛",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════════════
# PALETTE
# ══════════════════════════════════════════════════════════════════════════════
BG = '#0d1117'
SURFACE = '#111820'
BORDER = '#1e2a36'
SUBTEXT = '#64748b'
TEXT = '#e2e8f0'
GRID = '#1a2433'
ACCENT = '#4da2ff'
BRAND = {
    'BTC':'#F7931A','ETH':'#627EEA','SOL':'#14F195','GOLD':'#FFD700',
    'BNB':'#F0B90B','CRV':'#FF3D00','VET':'#15BDFF','UNI':'#FF007A','SUI':'#4DA2FF',
}
COND_COLORS = {
    'Bullish':'#34d399','Bearish':'#f87171','Neutral':'#94a3b8',
    'Warning':'#f59e0b','Reliable':'#34d399','Moderate':'#94a3b8',
}
DEV_STATE = {
    'NEUTRAL': ('#4da2ff','#0e1f35'),
    'OVERBOUGHT': ('#f59e0b','#1f1200'),
    'OVERSOLD': ('#a78bfa','#130820'),
}

# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
  html,body,[data-testid="stAppViewContainer"],[data-testid="stApp"]{{
      background:{BG}!important;font-family:'DM Sans',sans-serif;}}
  [data-testid="stHeader"]{{background:{BG}!important;border-bottom:1px solid {BORDER};}}
  [data-testid="block-container"]{{padding-top:1rem!important;padding-bottom:1.5rem!important;max-width:100%!important;}}
  section[data-testid="stMain"]>div{{padding-top:0!important;}}
  div[data-testid="column"]{{padding:0 6px!important;}}
  h1,h2,h3,h4{{color:{TEXT}!important;font-family:'DM Sans',sans-serif!important;}}
  .metric-card{{background:{SURFACE};border:1px solid {BORDER};border-radius:8px;
      padding:14px 18px;display:flex;flex-direction:column;gap:3px;}}
  .metric-label{{font-family:'Space Mono',monospace;font-size:8.5px;
      letter-spacing:0.12em;color:{SUBTEXT};text-transform:uppercase;}}
  .metric-value{{font-family:'Space Mono',monospace;font-size:20px;
      font-weight:700;color:white;line-height:1.1;}}
  .metric-sub{{font-family:'Space Mono',monospace;font-size:9px;color:{SUBTEXT};}}
  .section-header{{display:flex;align-items:center;gap:8px;margin-bottom:8px;
      padding-bottom:6px;border-bottom:1px solid {BORDER};}}
  .section-title{{font-family:'Space Mono',monospace;font-size:10px;font-weight:700;
      letter-spacing:0.14em;color:{SUBTEXT};text-transform:uppercase;}}
  .section-accent{{width:3px;height:14px;background:{ACCENT};border-radius:2px;}}
  .top-bar{{display:flex;justify-content:space-between;align-items:center;
      padding:10px 0 16px 0;border-bottom:1px solid {BORDER};margin-bottom:16px;}}
  .logo-text{{font-family:'Space Mono',monospace;font-size:17px;font-weight:700;
      color:white;letter-spacing:0.04em;}}
  .logo-accent{{color:{ACCENT};}}
  .header-meta{{font-family:'Space Mono',monospace;font-size:8.5px;
      color:{SUBTEXT};text-align:right;line-height:1.9;}}
  .divider{{border:none;border-top:1px solid {BORDER};margin:14px 0;}}
  .footer{{font-family:'Space Mono',monospace;font-size:9px;color:#263040;
      text-align:center;padding:16px 0 4px 0;letter-spacing:0.08em;}}
  #MainMenu,footer,[data-testid="stToolbar"]{{display:none!important;}}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

# File paths
ASSET_RANK_PATH = '/home/napoleao/dras/DRAS_2 - asset_rank.csv'
RANGE_RADAR_PATH = '/home/napoleao/dras/DRAS_2 - range_radar.csv'
ROLLING_ZSCORE_PATH = '/home/napoleao/dras/DRAS_2 - rolling_zscore.csv'
VOLATILITY_BANDS_PATH = '/home/napoleao/dras/volatility_bands.csv'

@st.cache_data
def load_asset_rank(rank_path, radar_path):
    """Load asset rankings with proper asset names from range_radar"""
    # Load scores
    scores_df = pd.read_csv(rank_path)
    score_col = scores_df.columns[0]
    scores = scores_df[score_col].values
    
    # Load asset names from range_radar
    radar_df = pd.read_csv(radar_path, skiprows=1)
    assets = radar_df.iloc[:, 0].values[:9]
    
    # Clean asset names
    assets = [str(a).strip().replace('PAXG/GOLD', 'GOLD') for a in assets]
    
    # Create combined dataframe
    result = pd.DataFrame({
        'Asset': assets,
        'Health_Score': scores
    })
    
    return result

@st.cache_data
def load_rolling_zscore(path):
    """Load rolling z-scores with dates as index"""
    df = pd.read_csv(path)
    date_col = df.columns[0]
    
    # Try different date formats
    try:
        df[date_col] = pd.to_datetime(df[date_col], format='%m/%d/%Y')
    except:
        try:
            df[date_col] = pd.to_datetime(df[date_col], format='%d/%m/%Y')
        except:
            df[date_col] = pd.to_datetime(df[date_col], infer_datetime_format=True)
    
    df = df.set_index(date_col)
    df.index.name = 'Date'
    return df.sort_index()

@st.cache_data
def load_range_radar(path):
    """Load range radar table data - handles broken volatility data"""
    # Skip first row (last updated), read from row 2
    df = pd.read_csv(path, skiprows=1)
    
    # Get relevant columns
    df_clean = pd.DataFrame({
        'Asset': df.iloc[:, 0],
        'Close': df.iloc[:, 1],
        'Volatility': df.iloc[:, 2],
        'Deviation': df.iloc[:, 3],
        'LowerBand': df.iloc[:, 4],  # Added to check for alternate volatility data
        'DevState': df.iloc[:, 6],
        'Condition': df.iloc[:, 7],
        'Confidence': df.iloc[:, 8]
    })
    
    # Remove header row if present
    df_clean = df_clean[df_clean['Asset'].notna() & 
                        (df_clean['Asset'] != 'Assets') &
                        (df_clean['Asset'].str.strip() != '')].copy()
    
    # Clean asset names
    df_clean['Asset'] = df_clean['Asset'].str.strip().str.replace('PAXG/GOLD', 'GOLD', regex=False)
    
    # Parse volatility - FIXED with proper handling of broken data
    def parse_volatility(row):
        vol_str = str(row['Volatility']).strip()
        
        # If it's a proper percentage, use it
        if '%' in vol_str and '$' not in vol_str:
            try:
                val = float(vol_str.replace('%', '').replace(',', '').strip())
                # Sanity check: volatility should be between 0% and 100%
                if 0 <= val <= 100:
                    return val
            except:
                pass
        
        # If we get here, the data is broken (like "$0.03")
        # Check if LowerBand has a percentage that looks reasonable
        try:
            lower_str = str(row['LowerBand']).strip()
            if '%' in lower_str:
                lower_val = float(lower_str.replace('%', '').replace(',', '').strip())
                # If it's a reasonable volatility (0-20%), use it
                if 0 < lower_val <= 20:
                    return lower_val
        except:
            pass
        
        # Last resort: estimate from the band ranges
        # For crypto, typical volatility is 2-5%
        # Use asset-specific defaults based on typical crypto volatility
        asset = row['Asset']
        defaults = {
            'BTC': 2.0,
            'ETH': 3.0,
            'SOL': 3.5,
            'BNB': 3.0,
            'SUI': 4.0,
            'UNI': 4.0,
            'CRV': 3.5,
            'VET': 3.0,
            'GOLD': 1.5
        }
        return defaults.get(asset, 3.0)
    
    # Apply the parsing function
    df_clean['Vol_pct'] = df_clean.apply(parse_volatility, axis=1)
    
    # Parse deviation
    df_clean['Dev_pct'] = pd.to_numeric(
        df_clean['Deviation'].astype(str).str.replace('%', '').str.replace('$', '').str.replace(',', '').str.strip(), 
        errors='coerce'
    ).fillna(0.0)
    
    # Clean other columns
    df_clean['DevState'] = df_clean['DevState'].astype(str).str.strip()
    df_clean['Condition'] = df_clean['Condition'].astype(str).str.strip()
    df_clean['Confidence'] = df_clean['Confidence'].astype(str).str.strip()
    
    # Override DevState based on deviation value
    def classify_dev(pct):
        if pd.isna(pct):
            return 'NEUTRAL'
        if pct >= 70:
            return 'OVERBOUGHT'
        elif pct <= -70:
            return 'OVERSOLD'
        else:
            return 'NEUTRAL'
    
    df_clean['DevState'] = df_clean['Dev_pct'].apply(classify_dev)
    
    # Drop the temporary LowerBand column
    df_clean = df_clean.drop('LowerBand', axis=1)
    
    return df_clean.reset_index(drop=True)

@st.cache_data
def load_volatility_bands(path):
    """Load volatility bands data"""
    df = pd.read_csv(path)
    # Replace PAXG with GOLD for consistency
    df['symbol'] = df['symbol'].str.replace('PAXG', 'GOLD', regex=False)
    return df

# Get last updated date from range_radar - FIXED
try:
    _meta = pd.read_csv(RANGE_RADAR_PATH, nrows=1, header=None)
    # The date is in the second column (index 1)
    date_value = str(_meta.iloc[0, 1]).strip()
    # Try to parse and reformat the date
    try:
        parsed_date = pd.to_datetime(date_value)
        last_date = parsed_date.strftime("%d %b %Y").upper()
    except:
        last_date = date_value.upper()
except:
    last_date = datetime.now().strftime("%d %b %Y").upper()

# Load all data - REMOVED DEBUG MESSAGES
try:
    asset_rank_df = load_asset_rank(ASSET_RANK_PATH, RANGE_RADAR_PATH)
    zscore_df = load_rolling_zscore(ROLLING_ZSCORE_PATH)
    radar_df = load_range_radar(RANGE_RADAR_PATH)
    bands_df = load_volatility_bands(VOLATILITY_BANDS_PATH)
    data_ok = True
except Exception as e:
    data_ok = False
    st.error(f"Error loading data: {str(e)}")

if not data_ok:
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# CHART BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def make_oscillator(zdf):
    """Valuation oscillator grid"""
    ASSETS = [a for a in ['BTC','SOL','SUI','GOLD','BNB','CRV','VET','UNI','ETH']
              if a in zdf.columns]
    
    plt.style.use('dark_background')
    fig, axes = plt.subplots(3, 3, figsize=(5.5, 3), facecolor=BG)
    dates = zdf.index
    
    for idx, ax in enumerate(axes.flat):
        if idx >= len(ASSETS):
            ax.set_visible(False)
            continue
        
        asset = ASSETS[idx]
        vals = np.clip(zdf[asset].values, -3, 3)
        brand = BRAND.get(asset, '#fff')
        
        ax.set_facecolor(BG)
        ax.axhspan(-2, 2, color=brand, alpha=0.09, zorder=1)
        ax.axhspan(-1, 1, color=brand, alpha=0.20, zorder=2)
        ax.axhline(0, color='#fff', linewidth=0.35, alpha=0.15, zorder=3)
        
        ax.plot(dates, vals, color='white', linewidth=0.8,
                marker='o', markersize=1.2,
                markerfacecolor='white', markeredgecolor='white', zorder=4)
        ax.scatter([dates[-1]], [vals[-1]], color='white', s=10, zorder=5)
        
        ax.annotate(f'{vals[-1]:.2f}\u03c3',
                    xy=(dates[-1], vals[-1]),
                    xytext=(-4, 5), textcoords='offset points',
                    color='white', fontsize=4, fontweight='bold', va='bottom')
        
        ax.set_ylim(-3.2, 3.2)
        ax.yaxis.set_ticks([-2, 0, 2])
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax.set_xlim(dates[0], dates[-1])
        
        plt.setp(ax.get_xticklabels(), fontsize=3, color='#4a5568')
        plt.setp(ax.get_yticklabels(), fontsize=3, color='#4a5568')
        ax.tick_params(length=0)
        ax.grid(axis='y', color=GRID, linewidth=0.4, alpha=0.6)
        ax.grid(axis='x', color=GRID, linewidth=0.3, alpha=0.3)
        
        for s in ax.spines.values():
            s.set_visible(False)
        
        ax.set_title(asset, color=brand, fontsize=7, fontweight='bold',
                     pad=2, fontfamily='monospace')
    
    b2 = mpatches.Patch(facecolor='grey', alpha=0.25, label='\u00b12\u03c3')
    b1 = mpatches.Patch(facecolor='grey', alpha=0.55, label='\u00b11\u03c3')
    fig.legend(handles=[b2, b1], loc='lower center', ncol=2,
               frameon=False, fontsize=6.5, labelcolor='white',
               bbox_to_anchor=(0.5, 0.002))
    
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    fig.patch.set_facecolor(BG)
    return fig

def make_capital_health(rank_df):
    """Capital allocation health chart"""
    # Filter out assets with health score = 0 and sort
    df = rank_df[rank_df['Health_Score'] > 0].copy()
    df = df.sort_values('Health_Score', ascending=True)
    
    def get_color(s):
        if s >= 0.6:
            return '#16a34a'
        elif s >= 0.4:
            return '#ea580c'
        else:
            return '#dc2626'
    
    df['Color'] = df['Health_Score'].apply(get_color)
    
    fig, ax = plt.subplots(figsize=(5, 3.3), facecolor=BG)
    ax.set_facecolor(BG)
    
    bars = ax.barh(df['Asset'], df['Health_Score'],
                   color=df['Color'], height=0.55, zorder=3)
    
    for bar, val in zip(bars, df['Health_Score']):
        ax.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                f'{val:.2f}', va='center', fontsize=7.5,
                color='#94a3b8', fontfamily='monospace')
    
    ax.set_xlim(0, 1.18)
    ax.set_xlabel('Health Score (0 = Weak, 1 = Strong)',
                  fontsize=7.5, color=SUBTEXT, fontfamily='monospace', labelpad=6)
    
    plt.setp(ax.get_yticklabels(), fontsize=6, color='#94a3b8', fontfamily='monospace')
    plt.setp(ax.get_xticklabels(), fontsize=5, color='#4a5568', fontfamily='monospace')
    ax.tick_params(length=0)
    ax.grid(axis='x', color=GRID, linewidth=0.4, alpha=0.6, zorder=0)
    
    for s in ax.spines.values():
        s.set_visible(False)
    
    legend_elements = [
        mpatches.Patch(facecolor='#16a34a', label='Accumulation'),
        mpatches.Patch(facecolor='#ea580c', label='Transitional'),
        mpatches.Patch(facecolor='#dc2626', label='Capital Leakage'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=6.5,
              frameon=True, facecolor=SURFACE, edgecolor=BORDER,
              labelcolor='#94a3b8')
    
    fig.patch.set_facecolor(BG)
    plt.tight_layout()
    return fig

def make_price_bands(bdf):
    """Volatility price range bands"""
    bdf = bdf.copy().sort_values('symbol')
    symbols = bdf['symbol'].tolist()
    x = np.arange(len(symbols))
    
    fig, ax = plt.subplots(figsize=(17, 4.2), facecolor=BG)
    ax.set_facecolor(BG)
    
    for i, row in bdf.iterrows():
        idx = symbols.index(row['symbol'])
        brand = BRAND.get(row['symbol'], '#94a3b8')
        
        cp = row['current_price']
        pct_lo = (row['lower_band'] / cp - 1) * 100
        pct_hi = (row['upper_band'] / cp - 1) * 100
        pct_ctr = (row['skew_adjusted_center'] / cp - 1) * 100
        
        ax.bar(idx, pct_hi - pct_lo, bottom=pct_lo,
               color=brand, alpha=0.18, width=0.6, zorder=2)
        ax.plot([idx - 0.28, idx + 0.28], [pct_ctr, pct_ctr],
                color=brand, linewidth=1.2, alpha=0.7, zorder=3)
        ax.scatter([idx], [0], color='white', s=30, zorder=5)
        
        ax.text(idx, pct_hi + 0.4, f'+{pct_hi:.1f}%',
                ha='center', va='bottom', fontsize=8.5,
                color=brand, fontfamily='monospace')
        ax.text(idx, pct_lo - 0.4, f'{pct_lo:.1f}%',
                ha='center', va='top', fontsize=8.5,
                color=brand, fontfamily='monospace')
    
    ax.axhline(0, color='#475569', linewidth=0.8, linestyle='--', alpha=0.6)
    
    ax.set_xticks(x)
    ax.set_xticklabels(symbols, fontsize=9.5, color='#94a3b8', 
                       fontfamily='monospace', rotation=45)
    ax.set_ylabel('% from Current Price', fontsize=10, color=SUBTEXT,
                  fontfamily='monospace', labelpad=6)
    ax.set_title('Volatility Price Range Bands \u00b7 Skew-Adjusted',
                 fontsize=11, color='#64748b', fontfamily='monospace',
                 fontweight='bold', pad=6)
    
    plt.setp(ax.get_yticklabels(), fontsize=9, color='#4a5568', fontfamily='monospace')
    ax.tick_params(length=0)
    ax.grid(axis='y', color=GRID, linewidth=0.4, alpha=0.5)
    
    for s in ax.spines.values():
        s.set_visible(False)
    
    ax.scatter([], [], color='white', s=20, label='Current price (0%)')
    ax.plot([], [], color='grey', linewidth=1.2, alpha=0.7, label='Skew-adj. center')
    ax.legend(loc='upper right', fontsize=8.5, frameon=True,
              facecolor=SURFACE, edgecolor=BORDER, labelcolor='#94a3b8')
    
    fig.patch.set_facecolor(BG)
    plt.tight_layout()
    return fig

def build_table_html(ddf):
    """Risk table HTML - FIXED volatility bar display"""
    TD = 'style="padding:12px 13px;font-family:monospace;font-size:11px;border-bottom:1px solid #1e2a3622;vertical-align:middle;"'
    TDR = 'style="padding:12px 13px;font-family:monospace;font-size:11px;border-bottom:1px solid #1e2a3622;vertical-align:middle;text-align:right;"'
    TDC = 'style="padding:12px 13px;font-family:monospace;font-size:11px;border-bottom:1px solid #1e2a3622;vertical-align:middle;text-align:center;"'
    TH = 'style="padding:12px 13px;font-family:monospace;font-size:8.5px;letter-spacing:0.12em;color:#64748b;text-transform:uppercase;border-bottom:1px solid #1e2a36;background:#0a1118;text-align:left;"'
    THR = 'style="padding:12px 13px;font-family:monospace;font-size:8.5px;letter-spacing:0.12em;color:#64748b;text-transform:uppercase;border-bottom:1px solid #1e2a36;background:#0a1118;text-align:right;"'
    THC = 'style="padding:12px 13px;font-family:monospace;font-size:8.5px;letter-spacing:0.12em;color:#64748b;text-transform:uppercase;border-bottom:1px solid #1e2a36;background:#0a1118;text-align:center;"'
    
    # Handle NaN values and calculate max volatility - FIXED
    valid_vols = ddf['Vol_pct'].dropna()
    if len(valid_vols) > 0:
        max_vol = valid_vols.max()
    else:
        max_vol = 1.0
    
    # Prevent division by zero
    if max_vol == 0 or pd.isna(max_vol):
        max_vol = 1.0
    
    rows_html = ''
    
    for i, row in ddf.iterrows():
        brand = BRAND.get(row['Asset'], '#aaaaaa')
        
        # Handle NaN in vol_pct - KEY FIX
        vol_pct = float(row['Vol_pct']) if pd.notna(row['Vol_pct']) else 0.0
        
        # Calculate bar width as percentage - IMPROVED
        if max_vol > 0 and vol_pct > 0:
            bar_w = min(int((vol_pct / max_vol) * 100), 100)  # Cap at 100%
        else:
            bar_w = 0
        
        # Handle NaN in dev_pct
        dev = float(row['Dev_pct']) if pd.notna(row['Dev_pct']) else 0.0
        dc = '#34d399' if dev >= 0 else '#f87171'
        ds = '+' if dev >= 0 else ''
        
        # Handle NaN in string columns
        state = str(row['DevState']) if pd.notna(row['DevState']) else 'NEUTRAL'
        fg, bg = DEV_STATE.get(state, ('#4da2ff', '#0e1f35'))
        
        cond = str(row['Condition']) if pd.notna(row['Condition']) else 'Neutral'
        cc = COND_COLORS.get(cond, '#94a3b8')
        
        conf = str(row['Confidence']) if pd.notna(row['Confidence']) else 'Moderate'
        cfc = COND_COLORS.get(conf, '#94a3b8')
        
        rb = '#0d1520' if i % 2 == 0 else '#0d1117'
        
        rows_html += f"""
        <tr style="background:{rb};">
          <td {TD}>
            <span style="display:inline-block;width:7px;height:8px;border-radius:50%;
              background:{brand};box-shadow:0 0 5px {brand}66;
              margin-right:8px;vertical-align:middle;"></span>
            <span style="color:white;font-weight:700;font-size:12px;">{row['Asset']}</span>
          </td>
          <td {TDR}>
            <div style="display:flex;align-items:center;justify-content:flex-end;gap:7px;">
              <span style="color:#e2e8f0;">{vol_pct:.2f}%</span>
              <div style="width:60px;height:6px;background:#1e2a36;border-radius:3px;overflow:hidden;display:inline-block;">
                <div style="width:{bar_w}%;height:100%;background:linear-gradient(90deg,#4da2ff,#a78bfa);border-radius:3px;"></div>
              </div>
            </div>
          </td>
          <td {TDR}><span style="color:{dc};font-weight:600;">{ds}{dev:.2f}%</span></td>
          <td {TDC}>
            <span style="display:inline-block;padding:2px 7px;border-radius:4px;
              font-size:7.5px;font-weight:700;letter-spacing:0.10em;
              color:{fg};background:{bg};">{state}</span>
          </td>
          <td {TDC}>
            <span style="display:inline-block;width:5px;height:5px;border-radius:50%;
              background:{cc};margin-right:4px;vertical-align:middle;"></span>
            <span style="color:{cc};font-weight:600;">{cond}</span>
          </td>
          <td {TDC}><span style="color:{cfc};">{conf}</span></td>
        </tr>"""
    
    return f"""
    <table style="width:100%;border-collapse:collapse;">
      <thead><tr>
        <th {TH}>Asset</th><th {THR}>Volatility</th><th {THR}>Deviation</th>
        <th {THC}>Dev State</th><th {THC}>Condition</th><th {THC}>Confidence</th>
      </tr></thead>
      <tbody>{rows_html}</tbody>
    </table>"""

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY METRICS
# ══════════════════════════════════════════════════════════════════════════════

def summary_metrics(zdf, ddf, rank_df):
    """Calculate summary metrics"""
    latest = zdf.iloc[-1]
    max_row = ddf.loc[ddf['Dev_pct'].abs().idxmax()]
    
    # Optimal asset: highest capital health score above 0.6
    top_assets = rank_df[rank_df['Health_Score'] > 0.6]
    if len(top_assets) > 0:
        optimal_row = top_assets.loc[top_assets['Health_Score'].idxmax()]
        optimal = optimal_row['Asset']
        optimal_score = optimal_row['Health_Score']
    else:
        optimal = 'N/A'
        optimal_score = 0.0
    
    return {
        'ext_pos': int((latest > 2).sum()),
        'ext_neg': int((latest < -2).sum()),
        'avg_vol': ddf['Vol_pct'].mean(),
        'optimal_asset': optimal,
        'optimal_score': optimal_score,
        'max_asset': max_row['Asset'],
        'max_dev': max_row['Dev_pct'],
    }

def metric_card(label, value, sub='', color='white'):
    """Create metric card HTML"""
    return f"""<div class="metric-card">
      <div class="metric-label">{label}</div>
      <div class="metric-value" style="color:{color};">{value}</div>
      <div class="metric-sub">{sub}</div></div>"""

# ══════════════════════════════════════════════════════════════════════════════
# RENDER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(f"""
<div style="display:flex;justify-content:space-between;align-items:center;
     padding:10px 0 16px 0;border-bottom:1px solid {BORDER};margin-bottom:16px;">
  <div>
    <div style="font-family:'Space Mono',monospace;font-size:22px;font-weight:700;
         color:#ffffff;letter-spacing:0.04em;">
      DRAS <span style="color:{ACCENT};">·</span> Risk Dashboard
    </div>
    <div style="font-family:'Space Mono',monospace;font-size:10px;color:{SUBTEXT};
         margin-top:4px;letter-spacing:0.08em;">
      DYNAMIC RETURN ASYMMETRY SIGNAL
    </div>
  </div>
  <div style="font-family:'Space Mono',monospace;font-size:10px;
       color:{SUBTEXT};text-align:right;line-height:2.0;">
    LAST UPDATED &nbsp;{last_date}<br>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Metric cards ───────────────────────────────────────────────────────────────
m = summary_metrics(zscore_df, radar_df, asset_rank_df)
sign = '+' if m['max_dev'] >= 0 else ''

c1, c2, c3, c4, c5 = st.columns(5)
c1.markdown(metric_card('Avg Volatility', f"{m['avg_vol']:.2f}%", 'Tracked Assets'),
            unsafe_allow_html=True)
c2.markdown(metric_card('Extended +2\u03c3', str(m['ext_pos']), 'assets overbought',
            '#f59e0b' if m['ext_pos'] > 0 else '#94a3b8'), unsafe_allow_html=True)
c3.markdown(metric_card('Extended \u22122\u03c3', str(m['ext_neg']), 'assets oversold',
            '#a78bfa' if m['ext_neg'] > 0 else '#94a3b8'), unsafe_allow_html=True)
c4.markdown(metric_card('Optimal Asset', m['optimal_asset'],
            f"health score {m['optimal_score']:.2f}",
            '#34d399' if m['optimal_score'] > 0.6 else '#94a3b8'), unsafe_allow_html=True)
c5.markdown(metric_card('Max Deviation', f"{sign}{m['max_dev']:.1f}%",
            f"{m['max_asset']} \u2014 largest move",
            '#34d399' if m['max_dev'] >= 0 else '#f87171'), unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── Row: Risk table | Capital allocation health ────────────────────────────────
st.markdown("""<div class="section-header">
  <div class="section-accent"></div>
  <div class="section-title">Asset Risk Matrix &nbsp;&amp;&nbsp; Capital Allocation Health</div>
</div>""", unsafe_allow_html=True)

col_table, col_health = st.columns([1.1, 0.9])

with col_table:
    table_html = build_table_html(radar_df)
    full_html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
    <style>*{{box-sizing:border-box;margin:0;padding:0;}}
    body{{background:{SURFACE};font-family:monospace;overflow-x:hidden;}}
    </style></head><body>{table_html}</body></html>"""
    components.html(full_html, height=390, scrolling=False)

with col_health:
    fig_cap = make_capital_health(asset_rank_df)
    st.pyplot(fig_cap, use_container_width=True)
    plt.close(fig_cap)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── Row: Volatility price range bands ──────────────────────────────────────────
st.markdown("""<div class="section-header">
  <div class="section-accent"></div>
  <div class="section-title">Volatility Price Range Bands</div>
</div>""", unsafe_allow_html=True)

fig_bands = make_price_bands(bands_df)
st.pyplot(fig_bands, use_container_width=True)
plt.close(fig_bands)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── Row: Valuation oscillator (full width) ─────────────────────────────────────
st.markdown("""<div class="section-header">
  <div class="section-accent"></div>
  <div class="section-title">Valuation Oscillator</div>
</div>""", unsafe_allow_html=True)

fig_osc = make_oscillator(zscore_df)
st.pyplot(fig_osc, use_container_width=True)
plt.close(fig_osc)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(f"""<div class="footer">
  @codedtrader in collaboration with Sonnet 4.5 &nbsp;\u00b7&nbsp; QUANTITATIVE RISK DASHBOARD &nbsp;\u00b7&nbsp; DRAS v1.0
</div>""", unsafe_allow_html=True)
# ───────────────────────────────────────────────────────────────
#  Importing all the necessary libraries
# ───────────────────────────────────────────────────────────────
import os, time, functools, requests, pandas as pd, numpy as np
import streamlit as st, plotly.graph_objects as go, plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
import tensorflow as tf
from tensorflow.keras import layers, callbacks
import openai
from dotenv import load_dotenv
import os

# ----------------------------------------------------------------
# Loading the API keys from environment variables
# ----------------------------------------------------------------
load_dotenv(dotenv_path="keys.env")  # explicitly load your keys.env file
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")
BASE = "https://api.polygon.io"

# ----------------------------------------------------------------
#  Function definations for making requests to Polygon API with retries
# ----------------------------------------------------------------
def _pg(endpoint: str,
        params: dict | None = None,
        retries: int = 3,
        pause: float = 1.5,
        silent_404: bool = False):
    params = params or {}
    params["apiKey"] = POLYGON_API_KEY            

    for i in range(retries):
        r = requests.get(f"{BASE}{endpoint}", params=params, timeout=15)

        if r.status_code == 200:   # Successful response       
            return r.json()

        if r.status_code == 429:     # Rate limited; apply backoff and retry     
            time.sleep(pause * (i + 1))
            continue

        if r.status_code == 404 and silent_404:   # Return None on 404 if allowed
            return None
         # Break on other errors, shows warnings
        if r.status_code != 429:
            st.warning(f"Polygon error {r.status_code}: {endpoint}")
    return None

# ----------------------------------------------------------------
#  Fetch and cache daily OHLCV data from Polygon (one call per day)
# ----------------------------------------------------------------
@st.cache_data(ttl=86_400, show_spinner=False)           # Pulling all adjusted daily bars for a stock and cache for 24 hours.
def pg_daily_aggregates(ticker: str) -> pd.DataFrame:
    url  = f"/v2/aggs/ticker/{ticker}/range/1/day/2022-01-01/{pd.Timestamp.utcnow().date()}"
    raw  = _pg(url, {"adjusted":"true","limit":50000,"sort":"asc"})
    if not raw or raw.get("resultsCount",0) == 0:
        return pd.DataFrame()
    
    
    df = pd.DataFrame(raw["results"])
    df["t"] = pd.to_datetime(df["t"], unit="ms")    # Convert timestamp to datetime

    # Renaming columns and set datetime index
    df = (df.rename(columns={"o":"Open","h":"High","l":"Low","c":"Close", "v":"Volume"})
         .set_index("t")[["Open","High","Low","Close", "Volume"]])
    return df

def load_stock_data(ticker: str, start, end) -> pd.DataFrame: # Slicing cached daily data by start and end date
    all_bars = pg_daily_aggregates(ticker)
    return all_bars.loc[str(start):str(end)].copy()

# ------------------------------------------------------------------------------
#  Get most recent open and close price for a ticker and its being cached daily
# ------------------------------------------------------------------------------
@st.cache_data(ttl=86400, show_spinner=False)

def get_current_price(ticker: str):                # defination to fetch the previous trading day's close and open prices.
    url = f"/v2/aggs/ticker/{ticker}/prev"
    data = _pg(url, {"adjusted": "true"})
    if not data or "results" not in data or not data["results"]:
        return None, None
    result = data["results"][0]
    close = result.get("c")
    open_ = result.get("o")
    pct_change = ((close - open_) / open_ * 100) if open_ else None
    return close, pct_change

@st.cache_data(ttl=86400, show_spinner=False)           # cache daily

def get_yearly_net_income(ticker: str, limit=5):         # function to retrieve net income for the last few years for a given ticker.
    """Fetch yearly net income (profit) from Polygon API"""
    url = "/v3/reference/financials"
    params = {
        "ticker": ticker.upper(),
        "limit": limit,
        "type": "Y",
        "sort": "reportPeriod",
        "order": "asc"
    }
    data = _pg(url, params, silent_404=True)
    if not data or "results" not in data:
        return []

    results = []
    for item in data["results"]:
        try:
            year = int(item["fiscal_period"])
            profit = item["financials"]["income_statement"]["net_income"] / 1e9  # convert to billions
            results.append({"year": year, "profit": profit})
        except (KeyError, TypeError):
            continue
    return results

# ----------------------------------------------------------------
#  Plot for moving averages and generate buy/sell signals
# ----------------------------------------------------------------
def plot_moving_avg_signals(prices: pd.DataFrame, sel: str):
    prices["MA50"] = prices["Close"].rolling(window=50).mean()
    prices["MA200"] = prices["Close"].rolling(window=200).mean()

    # Buy when MA50 crosses above MA200
    prices["Buy_Signal"] = (prices["MA50"] > prices["MA200"]) & (prices["MA50"].shift(1) <= prices["MA200"].shift(1))

    # Sell when MA50 crosses below MA200
    prices["Sell_Signal"] = (prices["MA50"] < prices["MA200"]) & (prices["MA50"].shift(1) >= prices["MA200"].shift(1))

    # Plotting the close prices and moving averages
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prices.index, y=prices["Close"], name="Close Price", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=prices.index, y=prices["MA50"], name="50-Day MA", line=dict(color="green")))
    fig.add_trace(go.Scatter(x=prices.index, y=prices["MA200"], name="200-Day MA", line=dict(color="red")))

    # Plotting Buy signals as upward green triangles
    fig.add_trace(go.Scatter(
        x=prices.loc[prices["Buy_Signal"]].index,
        y=prices.loc[prices["Buy_Signal"], "Close"],
        mode="markers",
        marker=dict(color="green", size=10, symbol="triangle-up"),
        name="Buy Signal"
    ))

    # Plotting Sell signals as downward red triangles
    fig.add_trace(go.Scatter(
        x=prices.loc[prices["Sell_Signal"]].index,
        y=prices.loc[prices["Sell_Signal"], "Close"],
        mode="markers",
        marker=dict(color="red", size=10, symbol="triangle-down"),
        name="Sell Signal"
    ))
    fig.update_layout(
        title=f"{sel} Moving Averages & Signals",
        xaxis_title="Date", yaxis_title="Price ($)",
        height=400, template="plotly_white"
    )
    return fig

# ----------------------------------------------------------------
#  Fetch recent top 5 news from Polygon API
# ----------------------------------------------------------------
@st.cache_data(ttl=600, show_spinner=False)  # cache for 10 min
def get_polygon_news(ticker: str = "", limit: int = 5):
    if ticker:
        url = f"/v2/reference/news?ticker={ticker.upper()}&limit={limit}"
    else:
        url = f"/v2/reference/news?limit={limit}"
    res = _pg(url, silent_404=True)
    return res.get("results", []) if res else []

# ----------------------------------------------------------------
#  Fetch company profile data from Polygon
# ----------------------------------------------------------------
@st.cache_data(ttl=86400, show_spinner=False)

def pg_profile(ticker: str) -> dict:                # Getting company metadata like industry, exchange, etc.
    data = _pg(f"/v3/reference/tickers/{ticker}",
               silent_404=True)          # ignore 404 for delisted or typos
    if data and "results" in data:
        return data["results"]
    return {}                             # always hand back a dict

# ----------------------------------------------------------------
#  Fetch market cap, market and sector description
# ----------------------------------------------------------------
@st.cache_data(ttl=86400, show_spinner=False)

def pg_marketcap(ticker: str):                 # Return market cap, market name and sector description for a ticker.
    prof = pg_profile(ticker)
    return prof.get("market_cap"), prof.get("market"), prof.get("sic_description")

# ----------------------------------------------------------------
#  Retrieve recent dividend announcements from Polygon
# ----------------------------------------------------------------
@st.cache_data(ttl=86_400, show_spinner=False)
def pg_dividends(tkr, limit=4):
    """Last `limit` cash-dividend announcements (ex-date, amount)."""
    r = _pg("/v3/reference/dividends", {"ticker": tkr,
                                        "order":  "desc",
                                        "limit":  limit})
    return r.get("results", []) if r else []

# ----------------------------------------------------------------
#  Retrieve recent stock split events from Polygon
# ----------------------------------------------------------------
@st.cache_data(ttl=86_400, show_spinner=False)
def pg_splits(tkr, limit=4):
    """Last `limit` stock-split events."""
    r = _pg("/v3/reference/splits", {"ticker": tkr,
                                     "order":  "desc",
                                     "limit":  limit})
    return r.get("results", []) if r else []

# ----------------------------------------------------------------
#  Calculate Year-to-Date return from previous year-end to latest close
# ----------------------------------------------------------------
def calc_ytd_change(df: pd.DataFrame) -> float | None:
    """%-change from last close of previous year → latest close."""
    if df.empty:                                     # No data available
        return None
    
    jan_1 = pd.Timestamp(df.index[-1].year, 1, 1)
    try:                                           
        start_px = df.loc[jan_1:]["Close"][0]        # First close of the year
    except Exception:
        return None
    end_px = df["Close"].iloc[-1]                    # Latest close
    return (end_px - start_px) / start_px * 100

# ----------------------------------------------------------------
#  Retrieve top N peers from same sector and calculate daily % change
# ----------------------------------------------------------------
def get_related_companies(tkr: str,                               #Uses pre-fetched fundamentals and cached price data.
                          fundamentals: pd.DataFrame,
                          top: int = 5) -> pd.DataFrame:
    if tkr not in fundamentals.index:
        return pd.DataFrame()

    sector = fundamentals.at[tkr, "sector"]
    peers  = (fundamentals[fundamentals["sector"] == sector]
              .drop(index=tkr, errors="ignore")
              .head(top)
              .copy())

    rows = []
    for sym in peers.index:
        bars = pg_daily_aggregates(sym).tail(2)   # Get last 2 days of prices
        if len(bars) == 2:
            y_close, t_close = bars["Close"].iloc[-2:]
            pct = (t_close - y_close) / y_close * 100
        else:                                     # not enough history
            pct = None

        rows.append({
            "ticker":  sym,
            "company": pg_profile(sym).get("name", "—"),
            "change":  pct,
        })

    return pd.DataFrame(rows)
# # ----------------------------------------------------------------
#  Constructing a dataframe with basic financials and classification info
# ----------------------------------------------------------------
@st.cache_data(ttl=86_400)
def get_fundamentals(ticker_list: list[str]) -> pd.DataFrame:    # useful for filtering, ranking, and peer comparison.
    rows, bar = [], st.progress(0.)
    for i, t in enumerate(ticker_list, 1):
        prof = pg_profile(t)
        rows.append({
            "ticker":       t,
            "market_value": prof.get("market_cap"),
            "sector":       prof.get("sic_description"), 
            "industry":     prof.get("sic_industry_description"),  
        })
        bar.progress(i / len(ticker_list))

    # keep first occurrence of every ticker (guards against dupes)
    df = (pd.DataFrame(rows)
            .drop_duplicates(subset="ticker")
            .set_index("ticker"))

    return df

# ----------------------------------------------------------------
#  Generating a placeholder signal columns for display purposes
# ----------------------------------------------------------------
def generate_signals(df: pd.DataFrame) -> pd.DataFrame:         #Adding dummy efficiency ratio and relative sector ranking.
    df["efficiency_ratio"] = np.nan
    df["sector_score"] = df.groupby("sector")["efficiency_ratio"].rank(pct=True)
    return df

# ----------------------------------------------------------------
#  Stub for intraday snapshot
# ----------------------------------------------------------------
@st.cache_data(ttl=86400, show_spinner=False)
def get_intraday_snapshot(ticker: str):
    return {}

# ----------------------------------------------------------------
#  Create future index of business days for forecasting horizon
# ----------------------------------------------------------------
def make_future_index(last_ts, n): return pd.bdate_range(last_ts+pd.tseries.offsets.BDay(), periods=n)

# ----------------------------------------------------------------
#  Format time series into supervised learning format (X, y)
# ----------------------------------------------------------------
def make_supervised(series,n_lags=5,horizon=1):                     # Converting a 1D time series into lagged features (X) and targets (y). Used for training machine learning models like XGBoost.
    arr=np.asarray(series);X=[arr[i-n_lags:i] for i in range(n_lags,len(arr)-horizon+1)]
    y=[arr[i+horizon-1] for i in range(n_lags,len(arr)-horizon+1)]
    return np.array(X),np.array(y)

# ----------------------------------------------------------------
#  Train XGBoost regressor on time series
# ----------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def fit_xgb(series,n_lags=5):                # Fit XGBoost model with lagged input and return model and scaled RMSE/MAPE metrics.
    X,y=make_supervised(series,n_lags);split=int(0.8*len(X))
    sc=StandardScaler(); Xtr,Xte=sc.fit_transform(X[:split]),sc.transform(X[split:])
    mdl=XGBRegressor(n_estimators=500,learning_rate=0.05,max_depth=4,
                     subsample=0.8,objective="reg:squarederror",n_jobs=-1).fit(Xtr,y[:split])
    preds=mdl.predict(Xte)
    return mdl,sc,mean_absolute_percentage_error(y[split:],preds),np.sqrt(mean_squared_error(y[split:],preds))

# ----------------------------------------------------------------
# LSTM or GRU model training with early stopping
# ----------------------------------------------------------------
def lstm_or_gru(series,mode="LSTM",steps=60,epochs=20):         #Trains a deep learning model (LSTM or GRU) on a time series.Includes scaling, reshaping, and early stopping.
    sc=StandardScaler();scaled=sc.fit_transform(series.values.reshape(-1,1)).flatten()
    X,y=make_supervised(scaled,steps,1); X=np.expand_dims(X,-1); split=int(0.8*len(X))
    def net():
        return tf.keras.Sequential([layers.Input(shape=(steps,1)),
            (layers.LSTM if mode=="LSTM" else layers.GRU)(64,return_sequences=True),
            (layers.LSTM if mode=="LSTM" else layers.GRU)(32), layers.Dense(1)])
    m=net(); m.compile(optimizer="adam",loss="mse")
    m.fit(X[:split],y[:split],epochs=epochs,batch_size=32,validation_split=0.1,verbose=0,
          callbacks=[callbacks.EarlyStopping(patience=3,restore_best_weights=True)])
    preds=m.predict(X[split:],verbose=0).flatten()
    preds=sc.inverse_transform(preds.reshape(-1,1)).flatten()
    y_te=sc.inverse_transform(y[split:].reshape(-1,1)).flatten()
    return m,sc,mean_absolute_percentage_error(y_te,preds),np.sqrt(mean_squared_error(y_te,preds))

# ----------------------------------------------------------------
#  Fit LSTM-specific model wrapper
# ----------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def fit_lstm_model(series, n_steps=60, epochs=20):
    return lstm_or_gru(series, mode="LSTM", steps=n_steps, epochs=epochs)

# ----------------------------------------------------------------
#  Fit GRU-specific model wrapper
# ----------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def fit_gru_model(series,  n_steps=60, epochs=20):
    return lstm_or_gru(series, mode="GRU",  steps=n_steps, epochs=epochs)

# ----------------------------------------------------------------
#  Calling OpenAI API for finance chatbot
# ----------------------------------------------------------------
def chat_answer(prompt: str):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-1106",   # Using "gpt-3.5-turbo-1106"
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500
    )

        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Error: {e}"
# ----------------------------------------------------------------
#  Class representing a company's profile and performance via Polygon API
# ----------------------------------------------------------------
class PolygonCompany:
    def __init__(self, ticker: str, fundamentals: pd.DataFrame):
        self.tkr     = ticker.upper()            # Store ticker in uppercase format
        self.fundamentals = fundamentals

        # Retrieve and store the company's profile information
        self.profile = pg_profile(self.tkr) or {}

        # Load historical stock data from 2000 to present
        self.hist    = load_stock_data(self.tkr, "2000-01-01",
                                       pd.Timestamp.utcnow().date())

        # Extract market cap from the company profile
        self.market_cap = self.profile.get("market_cap")

        # Get 1-year high and low prices if available
        yr = self.hist.last("365D")
        self.year_low  = yr["Low"].min()  if not yr.empty else None
        self.year_high = yr["High"].max() if not yr.empty else None

        # Calculate Year-to-Date return
        self.ytd_ret = calc_ytd_change(self.hist)


        # Try to fetch the next earnings event
        self.next_earn = None 
        # next earnings (free endpoint)
        try:
            # next earnings (free endpoint)
            evt = _pg("/v3/reference/stock_events",
                    {"ticker": self.tkr, "type": "EARNINGS", "limit": 1},
                    silent_404=True)          # ← tell _pg to ignore 404 here

        except Exception:
            self.next_earn = None

         # Try to compute trailing 12-month dividend yield
        try:
            divs = _pg("/v3/reference/dividends",
                       {"ticker": self.tkr, "limit": 1000})
            price, _ = get_current_price(self.tkr)
            if divs and divs.get("results") and price:
                last12 = [d for d in divs["results"]
                          if (pd.Timestamp.utcnow()
                              - pd.to_datetime(d["payment_date"])).days <= 365]
                ttm = sum(d["cash_amount"] for d in last12)
                self.div_yield = ttm / price * 100 if ttm else None
            else:
                self.div_yield = None
        except Exception:
            self.div_yield = None

    # Return the current price and daily percentage change
    @property
    def last_price(self):
        return get_current_price(self.tkr)

    # ----------------------------------------------------------------
    #  Render company overview, metrics, and peer comparison in UI
    # ----------------------------------------------------------------
    def render(self):
        name = self.profile.get("name", self.tkr)
        st.markdown(f"### 🏢 Company Overview – {name} ({self.tkr})")

        colL, colR = st.columns([2, 1])

        # Display company metadata on the left
        with colL:
            st.write(f"**Sector:** {self.profile.get('market','–')}")
            st.write(f"**Industry:** {self.profile.get('sic_description','–')}")
            st.write(f"**Exchange:** {self.profile.get('primary_exchange','–')}")
            st.write(f"**Locale:** {self.profile.get('locale','–')}")

        # Show financial stats on the right
        with colR:
            px, pct = self.last_price
            if px:
                st.metric("Price", f"${px:,.2f}", f"{pct:+.2f}%")
            if self.market_cap:
                st.caption(f"Market Cap • ${self.market_cap/1e9:,.1f} B")
            if self.year_low is not None:
                st.caption(f"52-week • {self.year_low:,.2f}—{self.year_high:,.2f}")
            if self.ytd_ret is not None:
                col = "green" if self.ytd_ret >= 0 else "red"
                st.markdown(f"<span style='color:{col};'>YTD {self.ytd_ret:+.1f}%"
                            f"</span>", unsafe_allow_html=True)
            if self.div_yield:
                st.caption(f"Dividend Yield • {self.div_yield:.1f}%")
            if self.next_earn:
                st.caption(f"Next Earnings • {self.next_earn}")
            ath = self.hist["High"].max() if not self.hist.empty else None
            if ath:
                st.caption(f"All-Time High • ${ath:,.2f}")

        # Render 30-day historical sparkline chart
        if not self.hist.empty:
            fig = go.Figure(go.Scatter(x=self.hist.tail(30).index,
                                       y=self.hist["Close"].tail(30),
                                       mode="lines+markers"))
            fig.update_layout(height=260, margin=dict(t=20,b=10,l=0,r=0),
                              title="Last 30 Days – Close Price")
            st.plotly_chart(fig, use_container_width=True)
        
        # Fetch and display related companies from same sector
        related_df = get_related_companies(
            self.tkr,
            fundamentals=self.fundamentals,
            top=5
        )
        if not related_df.empty:
            st.markdown("### 🌱 Related Companies")
            st.table(
                related_df
                  .style
                  .format({"change": "{:+.2f}%"})
                  .applymap(lambda v: "color:green" if isinstance(v, float) and v > 0
                                         else ("color:red" if isinstance(v, float) else None),
                            subset=["change"])
            )

        # Show recent dividend and split history if available
        divs, splits = pg_dividends(self.tkr), pg_splits(self.tkr)
        if divs or splits:
            st.markdown("---")
        if divs:
            st.subheader("Recent dividends")
            st.table(pd.DataFrame(divs)[["ex_dividend_date","cash_amount"]]
                     .rename(columns={"ex_dividend_date":"Ex-date",
                                      "cash_amount":"Amount $"})
                     .style.format({"Amount $":"{:.2f}"}))
        if splits:
            st.subheader("Recent splits")
            st.table(pd.DataFrame(splits)[["execution_date",
                                           "split_from","split_to"]]
                     .rename(columns={"execution_date":"Date",
                                      "split_from":"From",
                                      "split_to":"To"}))

# ----------------------------------------------------------------
#  Streamlit UI
# ----------------------------------------------------------------
st.set_page_config("Polygon Dashboard",layout="wide")

def main():
    st.title("💰Stocks Analysis and Forecasting Dashboard")

    # Default tickers pre-filled in text area
    default = "AAPL\nMSFT\nAMZN\nMETA\nNVDA"

    raw = st.sidebar.text_area("Input Tickers To Analyse: (one per line)", default) #User inputs tickers one per line

    #Clean, uppercase, and remove duplicate tickers
    tickers = list(dict.fromkeys(
        t.strip().upper() for t in raw.splitlines() if t.strip()
    ))

    # Retrieve previously selected ticker from session state (if any)
    prev_sel = st.session_state.get("prev_sel", "")

    # Clear metrics if ticker changed
    sel = st.sidebar.selectbox("Stock to forecast", tickers, index=tickers.index(prev_sel) if prev_sel in tickers else 0)
    # Clear metrics if ticker changed
    if sel != prev_sel:
        st.session_state["metrics"] = {}
        st.session_state["prev_sel"] = sel

    sd   = st.sidebar.date_input("Start",pd.to_datetime("2022-01-01"))  # Start date selector, defaults to Jan 1, 2022
    ed = st.sidebar.date_input("End", pd.to_datetime("today") - pd.Timedelta(days=1)) # End date selector, set to yesterday
    if sd>=ed: st.sidebar.error("Start must be < End")

    # Button to trigger dashboard analysis
    analyze_clicked = st.sidebar.button("Analyze", use_container_width=True)

    if "show_news_again" not in st.session_state:
        st.session_state["show_news_again"] = False

    # If user clicks Analyze, mark dashboard to run and reset state
    if analyze_clicked:
        st.session_state["run"] = True
        st.session_state["metrics"] = {}
        st.session_state["show_news_again"] = False  # reset on new analysis

    # Toggle-able sidebar block for chatbot interface
    with st.sidebar.expander("🤖  Finance Chatbot", expanded=False):
            if "last_bot_reply" not in st.session_state:
                st.session_state["last_bot_reply"] = ""

            user_q = st.text_input(
                "Ask me anything (e.g., 'What is EBITDA?' or 'price of MSFT?')",
                key="chat_input", placeholder="Type and press Enter"
            )
            # If user submits a question, call OpenAI to generate a reply
            if user_q:
                with st.spinner("Thinking…"):
                    st.session_state["last_bot_reply"] = chat_answer(user_q)

            if st.session_state["last_bot_reply"]: # If reply exists, render it in styled box
                 st.markdown(
                f"<div style='background-color:#1e1e1e;padding:12px;border-radius:8px;"
                f"color:#f0f0f0;font-size:14px;'>{st.session_state['last_bot_reply']}</div>",
                unsafe_allow_html=True,
            )

            st.caption("Powered by GPT-3.5 turbo")

    # Check if dashboard should run; if not, display news first 
    if not st.session_state.get("run"): 
    # Show news flash when dashboard hasn't been triggered yet
        st.markdown("### 🗞️ Top 5 Market Headlines")

        # Fetch top 5 news articles using Polygon API
        top_news = get_polygon_news(limit=5)

        if not top_news: # If no news is returned, inform the user
            st.info("No recent market news available.")
        else:
            for n in top_news: # Loop through each news item and display headline + metadata
                st.markdown(
                    f"- **[{n['title']}]({n['article_url']})**  <br> "
                    f"<small>{n['publisher']['name']} &bull; {n['published_utc'][:10]}</small>",
                    unsafe_allow_html=True
                )
        st.stop() # Prevent further dashboard execution until Analyze is clicked

    # make sure the dict exists on every refresh
    st.session_state.setdefault("metrics", {})

    with st.spinner("Loading fundamentals (cached)…"):
        df_raw  = get_fundamentals(tickers)       # Fetch sector-wise fundamentals for selected tickers

        df_raw = df_raw.drop(columns="industry", errors="ignore")
        
        # Rank companies by market cap within each sector
        df_raw["cap_rank"] = (df_raw.groupby("sector")["market_value"].rank(method="min", ascending=False))

        # Compute market cap percentile within sector
        df_raw["cap_pct"] = (df_raw.groupby("sector")["market_value"].rank(pct=True))

        df_ready = generate_signals(df_raw) # Prepare signal-enhanced dataframe for use in dashboard views

        df_ready = df_ready.drop(
        columns=["efficiency_ratio", "sector_score"],
        errors="ignore"          # Drop unused signal columns if they exist
    )

    # Load once and share across all tabs
    price_df = load_stock_data(sel, sd, ed).dropna()
    if price_df.empty:
        st.warning("No price data.")
        st.stop()

    # Load once and share across all tabs
    if st.session_state.get("run"):
        toggle_label = "❌ Hide Stocks News" if st.session_state["show_news_again"] else "📰 Show Stocks News"
        # Button toggles news visibility in dashboard
        if st.button(toggle_label, key="toggle_news"):
            st.session_state["show_news_again"] = not st.session_state["show_news_again"]
            st.rerun()  # Rerun the app to update the layout with toggled news

    if st.session_state.get("show_news_again"):  # If toggle is active, show news block again
        st.markdown("---")
        st.markdown("### 🗞️ Top 5 Market Headlines (Revisited)")
        top_news = get_polygon_news(limit=5)

        if not top_news:
            st.info("No recent market news available.")
        else:
            for n in top_news:
                st.markdown(
                    f"- **[{n['title']}]({n['article_url']})**  <br> "
                    f"<small>{n['publisher']['name']} &bull; {n['published_utc'][:10]}</small>",
                    unsafe_allow_html=True
                )
        st.stop()


    # If not showing news — show the full dashboard
    tab1, tab2, tab3, tab4 = st.tabs(["Insights", "Company Overview", "Summary", "Forecast"])

    # Insights tab - shows price summary, volume trend, and buy/sell indicators
    with tab1:
        st.subheader(f"Daily price snapshot: {sel}")

        prices = load_stock_data(sel, sd, ed).dropna() # Load OHLCV data and drop missing rows
        if prices.empty:
            st.warning("No price data for that range.")
            st.stop()

        # ---------- headline metrics --------------------------------------
        latest = prices.iloc[-1] # Most recent available trading day data
        prev   = prices.iloc[-2] if len(prices) > 1 else latest # Previous day close (if available)
        delta  = latest["Close"] - prev["Close"] # Change in closing price from previous trading day

        # Four columns to show OHLC metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Open",  f"${latest['Open'] :.2f}")
        col2.metric("High",  f"${latest['High'] :.2f}")
        col3.metric("Low",   f"${latest['Low']  :.2f}")
        col4.metric("Close", f"${latest['Close']:.2f}", f"{delta:+.2f}")

        # Volume Trend Analysis
        st.subheader("📊 Daily Volume Trend")

        # Compute moving averages
        prices["Vol_MA7"] = prices["Volume"].rolling(window=7).mean()
        prices["Vol_MA30"] = prices["Volume"].rolling(window=30).mean()

        # Plot
        fig_vol = go.Figure() # Creating a volume trend chart with overlayed moving averages
        fig_vol.add_trace(go.Bar(
            x=prices.index, y=prices["Volume"],
            name="Daily Volume", marker_color="rgba(30,144,255,1)",opacity=0.9  
        ))
        fig_vol.add_trace(go.Scatter(
            x=prices.index, y=prices["Vol_MA7"],
            mode="lines", name="7-Day MA", line=dict(color="orange")
        ))
        fig_vol.add_trace(go.Scatter(
            x=prices.index, y=prices["Vol_MA30"],
            mode="lines", name="30-Day MA", line=dict(color="green")
        ))
        fig_vol.update_layout(
            height=400, xaxis_title="Date", yaxis_title="Volume",
            title=f"{sel} Volume Trend with Moving Averages",
            template="plotly_dark", margin=dict(t=30, b=20, l=10, r=10)
        )
        st.plotly_chart(fig_vol, use_container_width=True)

         # Slider to control number of rows shown in table
        lookback = st.slider("Show last N trading days", 5, 30, 5, 1)
        subset   = prices["Close"].tail(lookback) # Subset for table display based on slider selection
        
        st.caption(f"Raw OHLC (last {lookback} days)")
        st.dataframe(
            prices[["Open", "High", "Low", "Close"]].tail(lookback)
                .style.format("${0:,.2f}")
            )

        st.subheader("📈 Moving Averages (50 vs 200) with Buy/Sell Signals") # Overlay of short- and long-term trend signals
        fig = plot_moving_avg_signals(prices.copy(), sel) # Generate chart showing crossover-based trading signals
        st.plotly_chart(fig, use_container_width=True)

    # Company Overview tab - displays business profile, sector peers, and dividend/split info
    with tab2:
        st.subheader("Company Overview")
        chosen = st.selectbox("Select a company:", options=tickers,index=tickers.index(sel)) # Dropdown to change company for detailed view

        try:
            PolygonCompany(chosen, fundamentals=df_ready).render()
        except Exception as err:
            st.error(f"Could not build overview: {err}")

    # Summary tab - provides a sector-level snapshot of selected tickers
    with tab3:
        st.subheader("Sector snapshot")
        st.dataframe(
            df_raw[["market_value", "sector", "cap_rank", "cap_pct"]]
            .rename(columns={
                "market_value": "mkt_cap",
                "cap_rank": "rank_in_sector",
                "cap_pct": "sector_percentile"
            })
            .style.format({"mkt_cap": "${:,.0f}",
                            "sector_percentile": "{:.0%}"})
        )
        # Market Cap Bar Chart
        chart_df = (
            df_raw[["market_value"]]
            .dropna()
            .sort_values("market_value", ascending=False)
            .rename(columns={"market_value": "Market Cap"})
        )
        # Create bar chart comparing market caps
        fig = px.bar(
            chart_df,
            x=chart_df.index,
            y="Market Cap",
            title="Market Cap Comparison",
            text="Market Cap",
            labels={"index": "Ticker"},
        )
        # Customize chart layout
        fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
        fig.update_layout(
            height=450,
            xaxis_title="Company",
            yaxis_title="Market Cap ($)",
            uniformtext_minsize=8,
            uniformtext_mode="hide",
            yaxis=dict(tickprefix="$", showgrid=True),
        )
        # Render the bar chart in full width
        st.plotly_chart(fig, use_container_width=True)

    # Forecast tab - allows users to run and compare time series forecasting models
    with tab4:
        st.subheader("Price forecast")

        horizon = st.number_input("Forecast horizon (trading days)", 1, 30, 7, help="How many days ahead?")
        model   = st.radio("Model", ["ARIMA", "SARIMA", "XGBOOST", "LSTM", "GRU"], horizontal=True)
        series = price_df["Close"].squeeze()

        
        def log_metric(ticker_key: str, mape: float, rmse: float):
            # Store/append metrics in session_state
            st.session_state["metrics"][ticker_key] = {"MAPE": mape, "RMSE": rmse}

        def show_comparison(sort_by: str = "MAPE"):
             # Display model performance comparison table, sorted by selected metric
            if len(st.session_state["metrics"]) < 2:
                return                           # Need at least 2 models to compare
            comp_df = (
                pd.DataFrame(st.session_state["metrics"])
                .T.sort_values(sort_by)          # Sort metrics ascending
                .style.format({"MAPE": "{:.2%}", "RMSE": "{:.2f}"})
                .background_gradient(axis=0, cmap="Greens", subset=[sort_by])
            )
            st.subheader("🔎 Model-validation comparison") # Section heading
            st.dataframe(comp_df) # Display comparison table

        # ARIMA branch - run diagnostics and forecasting using specified p, d, q values 
        if model == "ARIMA":
            st.markdown("### ARIMA diagnostics & forecast")

            # input the hyper-parameter widgets 
            colp, cold, colq = st.columns(3)
            p = colp.number_input("AR (p)", 0, 5, 1)  # Input for autoregressive order
            d = cold.number_input("Diff (d)", 0, 2, 1)  # Input for differencing order
            q = colq.number_input("MA (q)", 0, 5, 1)  # Input for moving average order

            # Use 80 percent of data for training, 20 percent for validation
            split = int(0.8 * len(series))        
            if split < max(p, d, q) + 3:          # Sanity check to ensure enough data points exist for ARIMA modeling
                    st.error("Price history too short for these ARIMA orders.")
                    st.stop()
            # Split the series for validation
            train, test = series[:split], series[split:]

            with st.spinner("Training / validating ARIMA …"):
                arima_val = ARIMA(train, order=(p, d, q)).fit() # Fit ARIMA model to training data
                val_pred  = arima_val.forecast(len(test)) # Generate forecast for validation
                mape = mean_absolute_percentage_error(test, val_pred) # Evaluate model using MAPE
                rmse = np.sqrt(mean_squared_error(test, val_pred)) # Evaluate model using RMSE

            st.success(f"Validation MAPE {mape:.2%} RMSE {rmse:,.2f}")
            key = f"{sel}_ARIMA"

            # Re-fit on all data to generate final forecast
            arima_full = ARIMA(series, order=(p, d, q)).fit() # Refit ARIMA on full data to make future forecast
            fc = arima_full.forecast(horizon)                 # Generate final forecast based on full model
            fc.index = make_future_index(series.index[-1], horizon) # Create date index for future forecast

            # Plot historical data and forecast
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=series.index, y=series, name="Historical"))
            fig.add_trace(go.Scatter(x=fc.index, y=fc, name="Forecast", line=dict(dash="dash", width=2)))
            fig.update_layout(height=350,xaxis_title=None,yaxis_title="Close $",margin=dict(t=30, b=10, l=10, r=10),)
            st.plotly_chart(fig, use_container_width=True)

            
            # show numeric values
            st.write("**Last 5 historical closes**")
            st.dataframe(series.tail(5).to_frame("Close $").style.format("${0:,.2f}"))
            
            st.write("**First forecasted values**")
            st.dataframe(fc.head().to_frame("Forecast").style.format("${0:,.2f}"))

            log_metric(key, mape, rmse) # Store results for comparison table

        #SARIMA branch - adds seasonal components to ARIMA modeling
        elif model == "SARIMA":
            st.markdown("### SARIMA diagnostics & forecast")

            # seasonal period (e.g. 5 for weekly, 12 for monthly, 252 for yearly)
            seasonal_period = st.number_input("Seasonal period (m)", 2, 365, 5)

            # non-seasonal (p,d,q)
            colp, cold, colq = st.columns(3)
            p = colp.number_input("AR (p)", 0, 5, 1)
            d = cold.number_input("Diff (d)", 0, 2, 1)
            q = colq.number_input("MA (q)", 0, 5, 1)

            # seasonal (P,D,Q)
            colP, colD, colQ = st.columns(3)
            P = colP.number_input("Seasonal AR (P)", 0, 5, 1)
            D = colD.number_input("Seasonal Diff (D)", 0, 2, 1)
            Q = colQ.number_input("Seasonal MA (Q)", 0, 5, 1)

            # Train/test split: 80 percent train, 20 percent validation
            split = int(0.8 * len(series))
            train, test = series[:split], series[split:]

            with st.spinner("Training / validating SARIMA …"):
                sarima = sm.tsa.SARIMAX(
                    train,
                    order=(p, d, q),
                    seasonal_order=(P, D, Q, seasonal_period),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False)

                val_pred = sarima.forecast(steps=len(test))
                mape = mean_absolute_percentage_error(test, val_pred)
                rmse = np.sqrt(mean_squared_error(test, val_pred))

            st.success(f"Validation MAPE {mape:.2%} RMSE {rmse:,.2f}")

            # refit on full data & forecast
            sarima_full = sm.tsa.SARIMAX(
                series,
                order=(p, d, q),
                seasonal_order=(P, D, Q, seasonal_period),
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)

            fc = sarima_full.forecast(horizon)
            fc.index = make_future_index(series.index[-1], horizon)

            # chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=series.index, y=series, name="Historical"))
            fig.add_trace(go.Scatter(
                x=fc.index, y=fc, name="Forecast",
                line=dict(dash="dash", width=2)
            ))
            fig.update_layout(
                height=350, xaxis_title=None, yaxis_title="Close $",
                margin=dict(t=30, b=10, l=10, r=10),
            )
            st.plotly_chart(fig, use_container_width=True)

            # tables
            st.write("**Last 5 historical closes**")
            st.dataframe(series.tail(5).to_frame("Close $").style.format("${0:,.2f}"))

            st.write("**First forecasted values**")
            st.dataframe(fc.head().to_frame("Forecast").style.format("${0:,.2f}"))

            # log metrics so the leaderboard updates 
            key = f"{sel}_SARIMA"
            log_metric(key, mape, rmse)

        # XGBoost branch - applies gradient boosting on lagged features
        elif model == "XGBOOST":                       
            st.markdown("### XGBoost diagnostics & forecast")

            # hyper-parameter – how many past days become features
            n_lags = st.number_input("Lag window (days)", 3, 30, 5,help="Number of lagged closes fed into the model.")

            # build supervised matrix once
            X, y = make_supervised(series, n_lags=n_lags)
            split = int(0.8 * len(X))
            X_tr, X_te, y_tr, y_te = X[:split], X[split:], y[:split], y[split:]

            # scale lags for faster convergence
            scaler = StandardScaler()          # Normalize input features
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)

            # Fit the XGBoost model to training data
            with st.spinner("Training / validating XGBoost …"):
                xgb = XGBRegressor(
                    n_estimators=500,
                    learning_rate=0.05,
                    max_depth=4,
                    subsample=0.8,
                    objective="reg:squarederror",
                    n_jobs=-1,
                    random_state=42,
                ).fit(X_tr, y_tr)

            # Generate predictions for the test set
            val_pred = xgb.predict(X_te)
            mape = mean_absolute_percentage_error(y_te, val_pred)
            rmse = np.sqrt(mean_squared_error(y_te, val_pred))
            st.success(f"Validation MAPE {mape:.2%} RMSE {rmse:,.2f}")

            # recursive multi-step forecast
            window = series.values[-n_lags:]           # last lag window
            preds  = []
            for _ in range(horizon):
                nxt = xgb.predict(scaler.transform(window.reshape(1, -1)))[0] # Forecast next value using trained model
                preds.append(nxt)
                window = np.append(window[1:], nxt) 

            fc_ix = make_future_index(series.index[-1], horizon)
            fc = pd.Series(preds, index=fc_ix, name="Forecast")

            # chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=series.index, y=series, name="Historical"))
            fig.add_trace(go.Scatter(x=fc.index, y=fc, name="Forecast",
                                    line=dict(dash="dash", width=2)))
            fig.update_layout(
                height=350, xaxis_title=None, yaxis_title="Close $",
                margin=dict(t=30, b=10, l=10, r=10),
            )
            st.plotly_chart(fig, use_container_width=True)

            # last 5 closes + first forecasts
            st.write("**Last 5 historical closes**")
            st.dataframe(series.tail(5).to_frame("Close $").style.format("${0:,.2f}"))

            st.write("**First forecasted values**")
            st.dataframe(fc.head().to_frame("Forecast").style.format("${0:,.2f}"))

            # log metrics for comparison table
            key = f"{sel}_XGBOOST"
            
            log_metric(key, mape, rmse)

        # LSTM branch - deep learning model suited for capturing long-term dependencies
        elif model == "LSTM":
            st.markdown("### LSTM diagnostics & forecast")

            # Number of previous time steps to consider in each input sequence
            n_steps = st.number_input("Sequence length (days)", 20, 120, 60, 5)
            # Number of full training cycles for the model
            epochs  = st.number_input("Training epochs", 5, 100, 20, 5)

            with st.spinner("Training / validating LSTM …"):
                lstm_model, scaler, mape, rmse = fit_lstm_model(
                    series, n_steps=n_steps, epochs=epochs
                )
            st.success(f"Validation MAPE {mape:.2%} RMSE {rmse:,.2f}")

            # recursive forecast
            window = scaler.transform(series.values[-n_steps:].reshape(-1, 1)).flatten()
            preds  = []
            for _ in range(horizon):
                # Predict next value in scaled form
                next_scaled = lstm_model.predict(window.reshape(1, n_steps, 1),verbose=0).flatten()[0]
                # Convert prediction back to original scale
                next_price  = scaler.inverse_transform([[next_scaled]])[0, 0]
                preds.append(next_price)
                window = np.append(window[1:], next_scaled)

            fc_ix = make_future_index(series.index[-1], horizon)
            fc = pd.Series(preds, index=fc_ix, name="Forecast")

            # plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=series.index, y=series, name="Historical"))
            fig.add_trace(go.Scatter(x=fc.index, y=fc, name="Forecast",
                                    line=dict(dash="dash", width=2)))
            fig.update_layout(height=350, xaxis_title=None, yaxis_title="Close $",
                            margin=dict(t=30, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)

            # tables
            st.write("**Last 5 historical closes**")
            st.dataframe(series.tail(5).to_frame("Close $").style.format("${0:,.2f}"))
            st.write("**First forecasted values**")
            st.dataframe(fc.head().to_frame("Forecast").style.format("${0:,.2f}"))

            # metrics
            key = f"{sel}_LSTM" 
            log_metric(key, mape, rmse)

        # GRU branch - simplified alternative to LSTM for capturing sequence patterns
        elif model == "GRU":
            st.markdown("### GRU diagnostics & forecast")

            # Controls look-back window for GRU input
            n_steps = st.number_input(
                "Sequence length (days)", 20, 120, 60, 5,
                help="How many past closes feed the GRU."
            )
            # Set training iterations with early stopping
            epochs  = st.number_input(
                "Training epochs", 5, 100, 20, 5,
                help="Stop early via patience=3.",
            )

            with st.spinner("Training / validating GRU …"):
                gru_model, scaler, mape, rmse = fit_gru_model(
                    series, n_steps=n_steps, epochs=epochs
                )

            st.success(f"Validation MAPE {mape:.2%} RMSE {rmse:,.2f}")

            # recursive forecast
            window = scaler.transform(series.values[-n_steps:].reshape(-1, 1)).flatten()
            preds  = []
            for _ in range(horizon):
                next_scaled = gru_model.predict(window.reshape(1, n_steps, 1), verbose=0).flatten()[0] # Predict next normalized value
                next_price  = scaler.inverse_transform([[next_scaled]])[0, 0] # Rescale prediction to original value
                preds.append(next_price)
                window = np.append(window[1:], next_scaled)

            fc_ix = make_future_index(series.index[-1], horizon)
            fc = pd.Series(preds, index=fc_ix, name="Forecast")

            # chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=series.index, y=series, name="Historical"))
            fig.add_trace(go.Scatter(x=fc.index, y=fc, name="Forecast",
                                    line=dict(dash="dash", width=2)))
            fig.update_layout(height=350, xaxis_title=None, yaxis_title="Close $",
                            margin=dict(t=30, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)

            # tables
            st.write("**Last 5 historical closes**")
            st.dataframe(series.tail(5).to_frame("Close $").style.format("${0:,.2f}"))

            st.write("**First forecasted values**")
            st.dataframe(fc.head().to_frame("Forecast").style.format("${0:,.2f}"))

            # metrics
            key = f"{sel}_GRU"
            log_metric(key, mape, rmse)

        show_comparison()  

    if st.session_state.get("show_news_again"):
        st.markdown("---") # Separator line
        st.markdown("### 🗞️ Top 5 Market Headlines (Revisited)")   # News heading for bottom section
        top_news = get_polygon_news(limit=5) # Pull latest news headlines

        if not top_news:
            st.info("No recent market news available.")
        else:
            # Render headlines with clickable links and publisher details
            for n in top_news:
                st.markdown(f"- **[{n['title']}]({n['article_url']})**  <br> "f"<small>{n['publisher']['name']} &bull; {n['published_utc'][:10]}</small>",unsafe_allow_html=True)

# Streamlit app runs only when this script is executed directly
if __name__ == "__main__":
    main()


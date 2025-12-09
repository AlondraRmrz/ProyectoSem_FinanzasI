import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize
from datetime import datetime


st.set_page_config(page_title="Gestión de Portafolios - Seminario Finanzas", layout="wide")

ESTRATEGIAS = {
    "Regiones": {
        "tickers": ["SPLG", "EWC", "IEUR", "EEM", "EWJ"],
        "benchmark_weights": [0.7062, 0.0323, 0.1176, 0.0902, 0.0537]
    },
    "Sectores (EE.UU.)": {
        "tickers": ["XLC", "XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLB", "XLRE", "XLK", "XLU"],
        "benchmark_weights": [0.0999, 0.1025, 0.0482, 0.0295, 0.1307, 0.0958, 0.0809, 0.0166, 0.0187, 0.3535, 0.0237]
    }
}

RF_RATE = 0.04 # tasa libre de riesto anual





# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Metrics and data
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

@st.cache_data
def descargar_datos(tickers, start_date, end_date):

    df = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)
    
    if 'Adj Close' in df.columns:
        data = df['Adj Close']
    elif 'Close' in df.columns:
        st.warning("Warning: Se utilizó la col 'Close' porque 'Adj Close' no se encontró.")
        data = df['Close']
    else:
        st.error("Error: No se encontraron datos de precios.")
        return pd.DataFrame()

    data = data.ffill().dropna() # missing data
    return data


def calcular_rendimientos(precios):
    return precios.pct_change().dropna()


def calcular_metricas(series_retornos, series_benchmark, rf=RF_RATE):

    # Mean
    media_anual = series_retornos.mean() * 252

    # Volatility
    volatilidad_anual = series_retornos.std() * np.sqrt(252)
    
    # Sharpe Ratio
    sharpe = (media_anual - rf) / volatilidad_anual if volatilidad_anual != 0 else 0

    # Sortino Ratio
    target = 0
    downside_returns = series_retornos[series_retornos < target]
    downside_dev = np.sqrt((downside_returns**2).mean()) * np.sqrt(252)
    sortino = (media_anual - rf) / downside_dev if downside_dev != 0 else 0

    # Beta
    df_join = pd.concat([series_retornos, series_benchmark], axis=1).dropna()
    cov_matrix = np.cov(df_join.iloc[:,0], df_join.iloc[:,1])
    cov_ab = cov_matrix[0, 1]
    var_b = cov_matrix[1, 1]
    beta = cov_ab / var_b if var_b != 0 else 0

    # Max Drawdown
    cumulative = (1 + series_retornos).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_drawdown = drawdown.min()

    # VaR (95%)
    var_95 = np.percentile(series_retornos, 5)

    # CVaR (95%)
    cvar_95 = series_retornos[series_retornos <= var_95].mean()

    # Curtosis
    curtosis = series_retornos.kurtosis()

    # Sesgo
    sesgo = series_retornos.skew()

    # Treynor Ratio
    treynor = (media_anual - rf) / beta if beta != 0 else 0

    # Information Ratio
    active_return = series_retornos - series_benchmark
    mean_active_return = active_return.mean() * 252
    tracking_error = active_return.std() * np.sqrt(252)
    info_ratio = mean_active_return / tracking_error if tracking_error != 0 else 0

    # Calmar Ratio
    total_return = (1 + series_retornos).prod()
    n_years = len(series_retornos) / 252
    cagr = (total_return**(1/n_years)) - 1
    calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else 0

    return {
        "Retorno Anual Esperado": f"{media_anual:.2%}",
        "Volatilidad (Anual)": f"{volatilidad_anual:.2%}",
        "Sharpe Ratio": f"{sharpe:.4f}",
        "Sortino Ratio": f"{sortino:.4f}",
        "Treynor Ratio": f"{treynor:.4f}",
        "Information Ratio": f"{info_ratio:.4f}",
        "Calmar Ratio": f"{calmar:.4f}",
        "Beta (vs Benchmark)": f"{beta:.4f}",
        "Max Drawdown": f"{max_drawdown:.2%}",
        "VaR (95%)": f"{var_95:.2%}",
        "CVaR (95%)": f"{cvar_95:.2%}",
        "Curtosis": f"{curtosis:.4f}",
        "Sesgo": f"{sesgo:.4f}"
    }

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Streamlit Interace
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

st.title("Gestor de Portafolios")
st.markdown("Proyecto Final - Seminario de Finanzas | Facultad de Ciencias UNAM")

# ------ SIDEBAR
st.sidebar.header("Configuración de la Estrategia")

# Strategies
tipo_estrategia = st.sidebar.selectbox("Selecciona el Universo de Inversión:", list(ESTRATEGIAS.keys()))
activos = ESTRATEGIAS[tipo_estrategia]["tickers"]
pesos_benchmark = ESTRATEGIAS[tipo_estrategia]["benchmark_weights"]

st.sidebar.markdown("---")
st.sidebar.header("Rango de Fechas")
start_val = datetime(2020, 1, 1)
end_val = datetime.today()
fecha_inicio = st.sidebar.date_input("Fecha Inicio", start_val)
fecha_fin = st.sidebar.date_input("Fecha Fin", end_val)

if fecha_inicio >= fecha_fin:
    st.error("La fecha de inicio debe ser anterior a la fecha final")
    st.stop()





# ------ DATA LOAD
with st.spinner('Descargando datos de mercado...'):
    precios = descargar_datos(activos, fecha_inicio, fecha_fin)
    retornos = calcular_rendimientos(precios)

if precios.empty:
    st.error("No se encontraron datos para el rango seleccionado.")
    st.stop()

# benchmark built
retorno_benchmark = retornos.dot(pesos_benchmark)
acumulado_benchmark = (1 + retorno_benchmark).cumprod()





# ------ PRIMARY VIEW
tab1, tab2, tab3 = st.tabs(["Análisis de Mercado", "Portafolio Arbitrario", "Portafolio Optimizado"])

# --- MARKET ANALISYS
with tab1:
    st.header(f"Desempeño del Universo: {tipo_estrategia}")
    st.write("Visualización de los precios.")
    
    precios_norm = precios / precios.iloc[0] * 100 # normalize to base 100 for comparison
    fig = px.line(precios_norm, title="Evolución de Precios")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Ver Matriz de Correlación", expanded=True):
        corr = retornos.corr()
        fig_corr = px.imshow(corr, text_auto='.4f', aspect="auto", color_continuous_scale='RdBu_r', zmin=-1, zmax=1, title="Matriz de Correlación de Activos")
        st.plotly_chart(fig_corr, use_container_width=True)

# --- ARBITRARY PORTFOLIO
with tab2:
    st.header("Portafolio Arbitrario (Definido por Usuario)")
    st.markdown("Define los pesos para cada activo. **La suma debe ser 100%**.")
    
    col_inputs, col_graph = st.columns([1, 2])
    
    pesos_usuario = []
    suma_pesos = 0
    
    with col_inputs:
        with st.form("pesos_form"):
            for activo in activos:
                val = st.number_input(f"Peso para {activo} (%)", min_value=0.0, max_value=100.0, value=100.0/len(activos), step=1.0)
                pesos_usuario.append(val/100) # convert to decimal
            
            submitted = st.form_submit_button("Calcular Portafolio")
            
    suma_pesos = sum(pesos_usuario)
    
    if abs(suma_pesos - 1.0) > 0.01:
        st.warning(f"La suma de los pesos es {suma_pesos*100:.2f}%. Ajustar para que sea 100%.")
    else:
        # portfolio calculation
        retorno_usuario = retornos.dot(pesos_usuario)
        acumulado_usuario = (1 + retorno_usuario).cumprod()
        
        # metrics
        metrics_user = calcular_metricas(retorno_usuario, retorno_benchmark)
        metrics_bench = calcular_metricas(retorno_benchmark, retorno_benchmark) # beta es 1
        
        # Dataframe
        df_metrics = pd.DataFrame([metrics_user, metrics_bench], index=["Tu Portafolio", "Benchmark"]).T
        
        with col_graph:
            # comparative plot
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Scatter(x=acumulado_usuario.index, y=acumulado_usuario, mode='lines', name='Tu Portafolio'))
            fig_comp.add_trace(go.Scatter(x=acumulado_benchmark.index, y=acumulado_benchmark, mode='lines', name='Benchmark', line=dict(dash='dash')))
            fig_comp.update_layout(title="Rendimiento Acumulado vs Benchmark")
            st.plotly_chart(fig_comp, use_container_width=True)
        
        st.subheader("Métricas de Rendimiento y Riesgo")
        st.dataframe(df_metrics)

# --- OPTIMIZED PORTFOLIO
with tab3:
    st.header("Optimización de Portafolio (Markowitz)")
    st.markdown("Introduce un **Rendimiento Objetivo Anual** y el modelo buscará la combinación de activos que minimice la varianza para ese retorno.")
    
    col_opt_input, col_opt_res = st.columns([1, 2])
    
    with col_opt_input:
        target_return = st.number_input("Rendimiento Objetivo Anual (%)", min_value=0.0, max_value=50.0, value=10.0, step=0.5) / 100
        btn_optimizar = st.button("Optimizar Portafolio")
        
    if btn_optimizar:
        # optimization logic
        mu = retornos.mean() * 252
        cov = retornos.cov() * 252
        num_assets = len(activos)
        
        # variance minimization
        def portfolio_variance(weights, cov_matrix):
            return weights.T @ cov_matrix @ weights
        
        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: x.T @ mu - target_return}
        )
        bounds = tuple((0, 1) for _ in range(num_assets))
        init_guess = num_assets * [1. / num_assets,]
        
        try:
            opt_result = minimize(portfolio_variance, init_guess, args=(cov,), method='SLSQP', bounds=bounds, constraints=constraints)
            
            if opt_result.success:
                pesos_optimos = opt_result.x
                
                # results calculation
                retorno_opt = retornos.dot(pesos_optimos)
                acumulado_opt = (1 + retorno_opt).cumprod()
                metrics_opt = calcular_metricas(retorno_opt, retorno_benchmark)
                
                with col_opt_res:
                    st.success("Optimización Exitosa")
                    
                    # optimal weights
                    df_pesos = pd.DataFrame({"Activo": activos, "Peso Óptimo": pesos_optimos})
                    fig_pie = px.pie(df_pesos, values='Peso Óptimo', names='Activo', title="Distribución Óptima de Activos")
                    st.plotly_chart(fig_pie)
                
                st.subheader("Desempeño del Portafolio Optimizado")
                col_g1, col_g2 = st.columns(2)
                
                with col_g1:
                    # plot
                    fig_opt = go.Figure()
                    fig_opt.add_trace(go.Scatter(x=acumulado_opt.index, y=acumulado_opt, mode='lines', name='Portafolio Optimizado'))
                    fig_opt.add_trace(go.Scatter(x=acumulado_benchmark.index, y=acumulado_benchmark, mode='lines', name='Benchmark', line=dict(dash='dash')))
                    st.plotly_chart(fig_opt, use_container_width=True)
                
                with col_g2:
                    # metrics
                    st.dataframe(pd.DataFrame(metrics_opt, index=["Optimizado"]).T)
                    
            else:
                st.error("No se pudo encontrar una solución. El rendimiento objetivo puede ser demasiado alto para los activos seleccionados.")
                
        except Exception as e:
            st.error(f"Error en la optimización: {e}")
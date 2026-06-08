import os
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from model import Kronos, KronosTokenizer, KronosPredictor

def main():
    ticker = "BTC-EUR"
    print(f"1. Estrazione dati storici reali per {ticker}...")
    df_btc = yf.download(ticker, period="1y", interval="1d")
    
    # Formattazione per il modello Kronos
    df_btc = df_btc.reset_index()
    df_btc.columns = [col[0] if isinstance(col, tuple) else col for col in df_btc.columns]
    df_btc = df_btc.rename(columns={
        'Date': 'timestamps', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'
    })
    df_btc['volume'] = df_btc['volume'].astype(float)
    df_btc['amount'] = df_btc['close'] * df_btc['volume']

    # Impostazioni finestra temporale: mostriamo gli ultimi 45 giorni nel grafico
    lookback = 45  
    steps_da_prevedere = 3   
    
    x_df = df_btc.iloc[-lookback:].copy()
    df_input = x_df[['open', 'high', 'low', 'close', 'volume', 'amount']].copy()
    x_timestamp = pd.Series(pd.to_datetime(x_df['timestamps']).dt.tz_localize(None))
    
    y_timestamp = pd.Series(pd.date_range(
        start=x_timestamp.iloc[-1] + pd.Timedelta(days=1), 
        periods=steps_da_prevedere, 
        freq='D' # Frequenza giornaliera continua per le crypto
    ))

    print("2. Calcolo delle proiezioni tramite Kronos...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
    predictor = KronosPredictor(tokenizer=tokenizer, model=model)

    previsioni = predictor.predict(
        df=df_input, x_timestamp=x_timestamp, y_timestamp=y_timestamp, pred_len=steps_da_prevedere
    )

    print("3. Generazione del report minimale ed elegante...")
    
    # Costruiamo il grafico a candele con Plotly
    fig = go.Figure()

    # Serie Storica Reale (Colori istituzionali opachi)
    fig.add_trace(go.Candlestick(
        x=x_timestamp,
        open=df_input['open'], high=df_input['high'],
        low=df_input['low'], close=df_input['close'],
        name="Storico di Mercato",
        increasing_line_color='#2ebd85', decreasing_line_color='#e0294a',
        increasing_fillcolor='#2ebd85', decreasing_fillcolor='#e0294a'
    ))

    # Proiezione Modello (Evidenziata in grigio chiaro/sfondo neutro per distacco analitico)
    fig.add_trace(go.Candlestick(
        x=y_timestamp,
        open=previsioni['open'], high=previsioni['high'],
        low=previsioni['low'], close=previsioni['close'],
        name="Proiezione Statistica AI",
        increasing_line_color='#4776e6', decreasing_line_color='#8e54e9',
        increasing_fillcolor='rgba(71, 118, 230, 0.4)', decreasing_fillcolor='rgba(142, 84, 233, 0.4)'
    ))

    # Layout Minimal ed Editoriale (Fondo antracite opaco, linee di griglia finissime)
    fig.update_layout(
        template="plotly_dark",
        title=None, # Togliamo il titolo interno al grafico per lasciarlo all'HTML pulito
        yaxis_title="Valore in USD ($)",
        xaxis_title=None,
        xaxis_rangeslider_visible=False,
        paper_bgcolor="#161a25",
        plot_bgcolor="#161a25",
        margin=dict(l=10, r=10, t=20, b=20),
        font=dict(color="#848e9c", family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"),
        xaxis=dict(gridcolor="#212633", zeroline=False),
        yaxis=dict(gridcolor="#212633", zeroline=False)
    )

    graph_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

    # Costruzione righe della tabella dati con formattazione minimalista
    table_rows = ""
    for idx, row in previsioni.iterrows():
        date_str = str(idx.date()) if hasattr(idx, 'date') else str(idx)[:10]
        # Calcolo al volo del delta ipotetico (Close vs Open della previsione)
        delta = row['close'] - row['open']
        delta_color = "#2ebd85" if delta >= 0 else "#e0294a"
        delta_sign = "+" if delta >= 0 else ""

        table_rows += f"""
        <tr>
            <td style="padding: 14px 16px; border-bottom: 1px solid #212633; color: #ffffff; font-weight: 500;">{date_str}</td>
            <td style="padding: 14px 16px; border-bottom: 1px solid #212633;">${row['open']:,.2f}</td>
            <td style="padding: 14px 16px; border-bottom: 1px solid #212633; color: #2ebd85;">${row['high']:,.2f}</td>
            <td style="padding: 14px 16px; border-bottom: 1px solid #212633; color: #e0294a;">${row['low']:,.2f}</td>
            <td style="padding: 14px 16px; border-bottom: 1px solid #212633; color: #ffffff; font-weight: 600;">${row['close']:,.2f}</td>
            <td style="padding: 14px 16px; border-bottom: 1px solid #212633; color: {delta_color}; font-weight: 500;">{delta_sign}${delta:,.2f}</td>
        </tr>
        """

    # Template HTML stile Terminale di Ricerca / Economico Minimalista
    html_content = f"""
    <!DOCTYPE html>
    <html lang="it">
    <head>
        <meta charset="UTF-8">
        <title>Analisi Quantitativa - {ticker}</title>
        <style>
            body {{ background-color: #0b0e11; color: #848e9c; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; margin: 0; padding: 40px 20px; -webkit-font-smoothing: antialiased; }}
            .container {{ max-width: 1100px; margin: 0 auto; }}
            .header {{ display: flex; justify-content: space-between; align-items: flex-end; border-bottom: 1px solid #212633; padding-bottom: 20px; margin-bottom: 30px; }}
            h1 {{ color: #ffffff; font-size: 24px; font-weight: 600; margin: 0; letter-spacing: -0.5px; }}
            .meta-info {{ font-size: 13px; color: #474d57; text-transform: uppercase; letter-spacing: 0.5px; }}
            .card {{ background: #161a25; border-radius: 4px; border: 1px solid #212633; padding: 24px; margin-bottom: 24px; }}
            h2 {{ color: #eaecef; font-size: 16px; font-weight: 500; margin-top: 0; margin-bottom: 20px; }}
            table {{ width: 100%; border-collapse: collapse; text-align: left; font-size: 14px; }}
            th {{ color: #474d57; padding: 12px 16px; font-weight: 500; border-bottom: 2px solid #212633; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; }}
            .footer {{ text-align: left; margin-top: 30px; font-size: 12px; color: #474d57; border-top: 1px solid #212633; padding-top: 20px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div>
                    <h1>Asset Analytics: {ticker}</h1>
                    <div style="font-size: 14px; margin-top: 4px; color: #b7bdc6;">Modellazione predittiva basata su serie temporali storiche</div>
                </div>
                <div class="meta-info">Data Generazione: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</div>
            </div>
            
            <div class="card">
                <h2>Orizzonte di Coda & Avanzamento Autoregressivo (Prossimi {steps_da_prevedere} Giorni)</h2>
                {graph_html}
            </div>
            
            <div class="card">
                <h2>Dati Tabellari di Proiezione</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Data Target</th>
                            <th>Apertura Prevista</th>
                            <th>Massimo Massimo</th>
                            <th>Minimo Minimo</th>
                            <th>Chiusura Prevista</th>
                            <th>Delta Interno (C-O)</th>
                        </tr>
                    </thead>
                    <tbody>
                        {table_rows}
                    </tbody>
                </table>
            </div>
            
            <div class="footer">
                Documentazione ad uso puramente informativo. I calcoli sono generati tramite inferenza neurale autoregressiva dal modello Kronos-base (Zero-Shot Forecasting framework). Non costituisce consulenza finanziaria.
            </div>
        </div>
    </body>
    </html>
    """

    # Salvataggio del report raffinato
    output_filename = "Report_Inference_BTC.html"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"\n[SUCCESSO] Dashboard professionale pronta! File: {os.path.abspath(output_filename)}")
    print("Aprilo nel browser per analizzare i risultati senza distrazioni grafiche inutili.")

if __name__ == "__main__":
    main()
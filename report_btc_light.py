import os
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from model import Kronos, KronosTokenizer, KronosPredictor

def main():
    ticker = "BTC-EUR"  
    print(f"1. Estrazione dati storici reali per {ticker} (Stile Editoriale)...")
    df_btc = yf.download(ticker, period="1y", interval="1d")
    
    # Formattazione per il modello Kronos
    df_btc = df_btc.reset_index()
    df_btc.columns = [col[0] if isinstance(col, tuple) else col for col in df_btc.columns]
    df_btc = df_btc.rename(columns={
        'Date': 'timestamps', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'
    })
    df_btc['volume'] = df_btc['volume'].astype(float)
    df_btc['amount'] = df_btc['close'] * df_btc['volume']

    # Impostazioni: 90 giorni di contesto per 7 giorni di previsione
    lookback = 90  
    steps_da_prevedere = 7   
    
    x_df = df_btc.iloc[-lookback:].copy()
    df_input = x_df[['open', 'high', 'low', 'close', 'volume', 'amount']].copy()
    x_timestamp = pd.Series(pd.to_datetime(x_df['timestamps']).dt.tz_localize(None))
    
    y_timestamp = pd.Series(pd.date_range(
        start=x_timestamp.iloc[-1] + pd.Timedelta(days=1), 
        periods=steps_da_prevedere, 
        freq='D'
    ))

    print("2. Elaborazione stime matematiche con Kronos...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
    predictor = KronosPredictor(tokenizer=tokenizer, model=model)

    previsioni = predictor.predict(
        df=df_input, x_timestamp=x_timestamp, y_timestamp=y_timestamp, pred_len=steps_da_prevedere
    )

    print("3. Generazione della GUI Light Minimal...")
    
    # Costruzione Grafico a Candele con Plotly (Light Theme)
    fig = go.Figure()

    # Storico di mercato (Verde foresta e Rosso mattone opachi)
    fig.add_trace(go.Candlestick(
        x=x_timestamp,
        open=df_input['open'], high=df_input['high'],
        low=df_input['low'], close=df_input['close'],
        name="Dati Storici Rilevati",
        increasing_line_color='#22c55e', decreasing_line_color='#ef4444',
        increasing_fillcolor='#22c55e', decreasing_fillcolor='#ef4444'
    ))

    # Previsione Settimanale (Evidenziata in un Blu Aviazione istituzionale raffinato)
    fig.add_trace(go.Candlestick(
        x=y_timestamp,
        open=previsioni['open'], high=previsioni['high'],
        low=previsioni['low'], close=previsioni['close'],
        name="Proiezione Statistica AI",
        increasing_line_color='#2563eb', decreasing_line_color='#3b82f6',
        increasing_fillcolor='rgba(37, 99, 235, 0.2)', decreasing_fillcolor='rgba(59, 130, 246, 0.2)'
    ))

    # Layout Light pulitissimo
    fig.update_layout(
        template="plotly_white",
        title=None,
        yaxis_title="Valore in EUR (€)",
        xaxis_rangeslider_visible=False,
        paper_bgcolor="#fcfbf9", # Sfondo avorio chiarissimo stile carta stampata
        plot_bgcolor="#fcfbf9",
        margin=dict(l=10, r=10, t=20, b=20),
        font=dict(color="#1e293b", family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto"),
        xaxis=dict(gridcolor="#e2e8f0", zeroline=False, linecolor="#cbd5e1"),
        yaxis=dict(gridcolor="#e2e8f0", zeroline=False, linecolor="#cbd5e1")
    )

    graph_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

    # Generazione righe della tabella in stile minimal-light
    table_rows = ""
    for idx, row in previsioni.iterrows():
        date_str = str(idx.date()) if hasattr(idx, 'date') else str(idx)[:10]
        delta = row['close'] - row['open']
        delta_color = "#15803d" if delta >= 0 else "#b91c1c" # Colori scuri per il testo su sfondo bianco
        delta_sign = "+" if delta >= 0 else ""

        table_rows += f"""
        <tr style="background-color: #fcfbf9;">
            <td style="padding: 16px; border-bottom: 1px solid #e2e8f0; color: #0f172a; font-weight: 500;">{date_str}</td>
            <td style="padding: 16px; border-bottom: 1px solid #e2e8f0; color: #334155;">€{row['open']:,.2f}</td>
            <td style="padding: 16px; border-bottom: 1px solid #e2e8f0; color: #16a34a; font-weight: 500;">€{row['high']:,.2f}</td>
            <td style="padding: 16px; border-bottom: 1px solid #e2e8f0; color: #dc2626; font-weight: 500;">€{row['low']:,.2f}</td>
            <td style="padding: 16px; border-bottom: 1px solid #e2e8f0; color: #0f172a; font-weight: 600;">€{row['close']:,.2f}</td>
            <td style="padding: 16px; border-bottom: 1px solid #e2e8f0; color: {delta_color}; font-weight: 600;">{delta_sign}€{delta:,.2f}</td>
        </tr>
        """

    html_content = f"""
    <!DOCTYPE html>
    <html lang="it">
    <head>
        <meta charset="UTF-8">
        <title>Quantitative Research Report - {ticker}</title>
        <style>
            body {{ background-color: #f4f2ee; color: #334155; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 50px 20px; -webkit-font-smoothing: antialiased; }}
            .container {{ max-width: 1100px; margin: 0 auto; background: #fcfbf9; padding: 40px; border-radius: 2px; box-shadow: 0 4px 20px rgba(0,0,0,0.04); border: 1px solid #e2e8f0; }}
            .header {{ border-bottom: 2px solid #0f172a; padding-bottom: 24px; margin-bottom: 35px; display: flex; justify-content: space-between; align-items: flex-end; }}
            h1 {{ color: #0f172a; font-size: 28px; font-weight: 700; margin: 0; font-family: 'Georgia', serif; letter-spacing: -0.5px; }}
            .meta-info {{ font-size: 12px; color: #64748b; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; }}
            .card {{ background: #fcfbf9; margin-bottom: 35px; }}
            h2 {{ color: #0f172a; font-size: 16px; font-weight: 600; margin-top: 0; margin-bottom: 20px; text-transform: uppercase; letter-spacing: 0.5px; border-bottom: 1px solid #cbd5e1; padding-bottom: 8px; }}
            table {{ width: 100%; border-collapse: collapse; text-align: left; font-size: 14px; }}
            th {{ color: #64748b; padding: 12px 16px; font-weight: 600; border-bottom: 2px solid #0f172a; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; }}
            .footer {{ text-align: left; margin-top: 40px; font-size: 11px; color: #94a3b8; border-top: 1px solid #e2e8f0; padding-top: 20px; line-height: 1.5; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div>
                    <h1>Quantitative Research: {ticker}</h1>
                    <div style="font-size: 14px; margin-top: 6px; color: #64748b; font-style: italic; font-family: 'Georgia', serif;">Analisi stocastica e proiezioni su serie temporali storiche</div>
                </div>
                <div class="meta-info">Report Data: {pd.Timestamp.now().strftime('%d %B %Y • %H:%M')}</div>
            </div>
            
            <div class="card">
                <h2>Modellazione Grafica delle Tendenze (Orizzonte 7 Giorni)</h2>
                <div style="background-color: #fcfbf9; padding: 10px 0;">
                    {graph_html}
                </div>
            </div>
            
            <div class="card">
                <h2>Matrice Analitica delle Previsioni</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Data Target</th>
                            <th>Apertura Stimata</th>
                            <th>Massimo Teorico</th>
                            <th>Minimo Teorico</th>
                            <th>Chiusura Stimata</th>
                            <th>Delta Atteso (C-O)</th>
                        </tr>
                    </thead>
                    <tbody>
                        {table_rows}
                    </tbody>
                </table>
            </div>
            
            <div class="footer">
                <strong>INFORMATIVA METODOLOGICA:</strong> Questo documento è stato redatto mediante l'utilizzo di algoritmi di intelligenza artificiale specializzati nell'interpolazione di serie storiche (Framework Kronos-base). I dati di input storici sono estratti in tempo reale da Yahoo Finance. Le stime fornite hanno una valenza puramente matematica e accademica basata sulla geometria statistica passata del prezzo e non considerano eventi macroeconomici esogeni, notizie o variazioni di liquidità del mercato. Non costituisce sollecitazione al pubblico risparmio né consulenza finanziaria personalizzata.
            </div>
        </div>
    </body>
    </html>
    """

    # Salvataggio del report raffinato in stile light
    output_filename = "Report_Inference_BTC_Light.html"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"\n[SUCCESSO] Documento editoriale pronto! File: {os.path.abspath(output_filename)}")
    print("Aprilo con Chrome o Edge per visualizzare l'impaginazione in stile istituzionale.")

if __name__ == "__main__":
    main()
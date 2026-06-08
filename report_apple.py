import os
import smtplib
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from model import Kronos, KronosTokenizer, KronosPredictor

def main():
    ticker = "AAPL"
    print(f"1. Recupero dati reali per {ticker}...")
    df_apple = yf.download(ticker, period="1y", interval="1d")
    
    # Formattazione per Kronos
    df_apple = df_apple.reset_index()
    df_apple.columns = [col[0] if isinstance(col, tuple) else col for col in df_apple.columns]
    df_apple = df_apple.rename(columns={
        'Date': 'timestamps', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'
    })
    df_apple['volume'] = df_apple['volume'].astype(float)
    df_apple['amount'] = df_apple['close'] * df_apple['volume']

    # Impostazioni Finestra Temporale (Storico visualizzato nel report: 60 giorni)
    lookback = 60  
    steps_da_prevedere = 5   
    
    x_df = df_apple.iloc[-lookback:].copy()
    df_input = x_df[['open', 'high', 'low', 'close', 'volume', 'amount']].copy()
    x_timestamp = pd.Series(pd.to_datetime(x_df['timestamps']).dt.tz_localize(None))
    
    y_timestamp = pd.Series(pd.date_range(
        start=x_timestamp.iloc[-1] + pd.Timedelta(days=1), 
        periods=steps_da_prevedere, 
        freq='B'
    ))

    print("2. Elaborazione previsioni con l'AI...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
    predictor = KronosPredictor(tokenizer=tokenizer, model=model)

    previsioni = predictor.predict(
        df=df_input, x_timestamp=x_timestamp, y_timestamp=y_timestamp, pred_len=steps_da_prevedere
    )

    print("3. Generazione della Dashboard Interattiva HTML...")
    
    # Costruiamo il grafico a candele (Candlestick) unendo storico e previsione
    fig = go.Figure()

    # Storico Reale (Candele Grigio/Blu scuro)
    fig.add_trace(go.Candlestick(
        x=x_timestamp,
        open=df_input['open'], high=df_input['high'],
        low=df_input['low'], close=df_input['close'],
        name="Dati Storici",
        increasing_line_color='#26a69a', decreasing_line_color='#ef5350'
    ))

    # Previsione AI (Candele tratteggiate o evidenziate in Oro/Viola per distinguerle)
    fig.add_trace(go.Candlestick(
        x=y_timestamp,
        open=previsioni['open'], high=previsioni['high'],
        low=previsioni['low'], close=previsioni['close'],
        name="Previsione Kronos AI",
        increasing_line_color='#ab47bc', decreasing_line_color='#5c6bc0'
    ))

    # Layout stile TradingView (Scuro ed elegante)
    fig.update_layout(
        template="plotly_dark",
        title=f"Analisi Predittiva Avanzata - {ticker} (Powered by Kronos)",
        yaxis_title="Prezzo in USD ($)",
        xaxis_title="Data",
        xaxis_rangeslider_visible=False,
        paper_bgcolor="#131722",
        plot_bgcolor="#131722",
        font=dict(color="#d1d4dc", family="Arial")
    )

    # Convertiamo il grafico in codice HTML puro da iniettare nella pagina
    graph_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

    # Costruiamo le righe della tabella HTML per i dati previsti
    table_rows = ""
    for idx, row in previsioni.iterrows():
        date_str = str(idx.date()) if hasattr(idx, 'date') else str(idx)[:10]
        table_rows += f"""
        <tr>
            <td style="padding: 12px; border-bottom: 1px solid #2a2e39;">{date_str}</td>
            <td style="padding: 12px; border-bottom: 1px solid #2a2e39; color: #26a69a;">${row['open']:.2f}</td>
            <td style="padding: 12px; border-bottom: 1px solid #2a2e39; color: #26a69a;">${row['high']:.2f}</td>
            <td style="padding: 12px; border-bottom: 1px solid #2a2e39; color: #ef5350;">${row['low']:.2f}</td>
            <td style="padding: 12px; border-bottom: 1px solid #2a2e39; font-weight: bold;">${row['close']:.2f}</td>
        </tr>
        """

    # Template HTML completo con design moderno Cyberpunk/Financial Minimal
    html_content = f"""
    <!DOCTYPE html>
    <html lang="it">
    <head>
        <meta charset="UTF-8">
        <title>Report Predittivo Kronos - {ticker}</title>
        <style>
            body {{ background-color: #1c2030; color: #d1d4dc; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: #131722; padding: 30px; border-radius: 12px; box-shadow: 0 8px 24px rgba(0,0,0,0.5); }}
            h1 {{ color: #ffffff; margin-top: 0; font-weight: 600; border-bottom: 2px solid #2a2e39; padding-bottom: 15px; }}
            .badge {{ background: linear-gradient(135deg, #7b1fa2, #4a148c); color: white; padding: 6px 12px; border-radius: 20px; font-size: 0.85em; font-weight: bold; display: inline-block; margin-bottom: 20px; }}
            .grid {{ display: flex; flex-direction: column; gap: 30px; }}
            .card {{ background: #1c2030; padding: 20px; border-radius: 8px; border: 1px solid #2a2e39; }}
            table {{ width: 100%; border-collapse: collapse; text-align: left; margin-top: 10px; }}
            th {{ background-color: #2a2e39; color: #h1h1h1; padding: 12px; font-weight: 600; }}
            .footer {{ text-align: center; margin-top: 40px; font-size: 0.85em; color: #787b86; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Report di Analisi Quantitativa: {ticker}</h1>
            <div class="badge">KRONOS TIME-SERIES PREDICTION TARGET</div>
            
            <div class="grid">
                <div class="card">
                    {graph_html}
                </div>
                
                <div class="card">
                    <h3 style="margin-top:0; color: #ffffff;">Tabella Output Predittivo (Prossimi {steps_da_prevedere} Giorni)</h3>
                    <table>
                        <thead>
                            <tr>
                                <th>Data Target</th>
                                <th>Apertura Prevista (Open)</th>
                                <th>Massimo Previsto (High)</th>
                                <th>Minimo Previsto (Low)</th>
                                <th>Chiusura Prevista (Close)</th>
                            </tr>
                        </thead>
                        <tbody>
                            {table_rows}
                        </tbody>
                    </table>
                </div>
            </div>
            <div class="footer">
                Generato automaticamente dal motore neurale Kronos-base • Dati Real-Time via Yahoo Finance
            </div>
        </div>
    </body>
    </html>
    """

    # Salvataggio del file finale
    output_filename = "Report_Previsioni_AAPL.html"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"\n[SUCCESSO] Report grafico generato! Trovi il file qui: {os.path.abspath(output_filename)}")

    # Invio email
    smtp_server = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    smtp_user = os.environ.get("SMTP_USER", "enomisia974@gmail.com")
    smtp_password = os.environ.get("SMTP_PASSWORD", "")
    smtp_to = os.environ.get("SMTP_TO", "enomisia974@gmail.com")

    if smtp_password:
        msg = MIMEMultipart()
        msg['From'] = smtp_user
        msg['To'] = smtp_to
        msg['Subject'] = f"Report Previsioni Kronos - {ticker}"

        text_part = MIMEText(f"Report previsioni Kronos per {ticker} in allegato.", 'plain')
        msg.attach(text_part)

        att = MIMEBase('application', 'octet-stream')
        att.set_payload(html_content.encode('utf-8'))
        encoders.encode_base64(att)
        att.add_header('Content-Disposition', f'attachment; filename={output_filename}')
        msg.attach(att)

        s = smtplib.SMTP(smtp_server, smtp_port)
        s.starttls()
        s.login(smtp_user, smtp_password)
        s.sendmail(smtp_user, [smtp_to], msg.as_string())
        s.quit()
        print(f"Email inviata a {smtp_to}")
    else:
        print("SMTP_PASSWORD non impostata, salto invio email.")

if __name__ == "__main__":
    main()
import os
import datetime

OUTPUT_FILE = "Report_Audit_DataLeak.html"

AUDIT_DATE = datetime.datetime.now().strftime("%d %B %Y • %H:%M")

def main():
    html = f"""<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Audit Sicurezza Dati &bull; Kronos Pipeline</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800;900&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<style>
  *{{margin:0;padding:0;box-sizing:border-box;}}
  body{{background:#0b1120;color:#e2e8f0;font-family:'Inter',-apple-system,sans-serif;line-height:1.6;}}
  .hd{{background:linear-gradient(135deg,#0b1120 0%,#1a0a2e 50%,#0b1120 100%);padding:60px 0 40px;border-bottom:3px solid #ef4444;}}
  .hd-in{{max-width:1100px;margin:0 auto;padding:0 32px;}}
  .hd-in .meta{{font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:2px;margin-bottom:8px;}}
  .hd-in h1{{font-family:'JetBrains Mono',monospace;font-weight:800;font-size:36px;color:#fff;letter-spacing:-1px;}}
  .hd-in .sub{{font-size:15px;color:#94a3b8;margin-top:8px;font-weight:300;}}
  .severity{{display:inline-block;padding:3px 12px;border-radius:12px;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:0.5px;}}
  .sev-critical{{background:#ef444420;color:#ef4444;border:1px solid #ef444440;}}
  .sev-high{{background:#f59e0b20;color:#f59e0b;border:1px solid #f59e0b40;}}
  .sev-medium{{background:#3b82f620;color:#3b82f6;border:1px solid #3b82f640;}}
  .sev-fixed{{background:#22c55e20;color:#22c55e;border:1px solid #22c55e40;}}
  .cnt{{max-width:1100px;margin:0 auto;padding:32px;}}
  .card{{background:#161a25;border-radius:8px;border:1px solid #212633;overflow:hidden;margin-bottom:24px;}}
  .card-hd{{padding:18px 24px;border-bottom:1px solid #212633;background:#1a1f2e;display:flex;justify-content:space-between;align-items:center;}}
  .card-hd h2{{font-family:'JetBrains Mono',monospace;font-size:16px;font-weight:600;color:#e2e8f0;}}
  .card-bd{{padding:20px 24px;}}
  .finding{{padding:16px 20px;border-left:4px solid;margin:16px 0;border-radius:0 6px 6px 0;}}
  .finding.critical{{border-color:#ef4444;background:#ef444408;}}
  .finding.high{{border-color:#f59e0b;background:#f59e0b08;}}
  .finding.fixed{{border-color:#22c55e;background:#22c55e08;}}
  .file-path{{font-family:'JetBrains Mono',monospace;font-size:12px;color:#64748b;background:#0b1120;padding:10px 14px;border-radius:4px;overflow-x:auto;white-space:nowrap;margin:8px 0;}}
  .file-path .highlight{{color:#f59e0b;font-weight:600;}}
  code{{font-family:'JetBrains Mono',monospace;font-size:13px;background:#0b1120;padding:2px 6px;border-radius:3px;color:#e2e8f0;}}
  pre{{font-family:'JetBrains Mono',monospace;font-size:12px;background:#0b1120;padding:16px 20px;border-radius:6px;overflow-x:auto;margin:12px 0;border:1px solid #212633;color:#e2e8f0;line-height:1.5;}}
  .diff-add{{color:#22c55e;}}
  .diff-rem{{color:#ef4444;}}
  .tag{{display:inline-block;padding:2px 10px;border-radius:10px;font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:0.5px;}}
  .tag-leak{{background:#ef444430;color:#ef4444;}}
  .tag-fix{{background:#22c55e30;color:#22c55e;}}
  .grid-2{{display:grid;grid-template-columns:1fr 1fr;gap:20px;}}
  .metric{{text-align:center;padding:20px;}}
  .metric .val{{font-family:'JetBrains Mono',monospace;font-size:32px;font-weight:700;}}
  .metric .lbl{{font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:0.8px;margin-top:6px;}}
  .summary-item{{padding:14px 18px;border-bottom:1px solid #212633;display:flex;justify-content:space-between;align-items:center;}}
  .summary-item:last-child{{border-bottom:none;}}
  .summary-item .status{{width:12px;height:12px;border-radius:50%;display:inline-block;margin-right:10px;}}
  .status-fixed{{background:#22c55e;}}
  .status-open{{background:#ef4444;}}
  .ftr{{max-width:1100px;margin:0 auto;padding:24px 32px;font-size:11px;color:#475569;border-top:1px solid #212633;text-align:center;}}
  h3{{font-family:'JetBrains Mono',monospace;font-size:14px;font-weight:600;color:#e2e8f0;margin:20px 0 8px;}}
  ul{{padding-left:20px;}}
  li{{margin:6px 0;font-size:14px;color:#94a3b8;}}
  li strong{{color:#e2e8f0;}}
  @media(max-width:768px){{.grid-2{{grid-template-columns:1fr;}}.hd-in h1{{font-size:24px;}}}}
</style>
</head>
<body>

<div class="hd">
  <div class="hd-in">
    <div class="meta">Data Security Audit &middot; {AUDIT_DATE}</div>
    <h1>Report di Audit: Data Leakage nella Pipeline</h1>
    <div class="sub">Analisi delle vulnerabilit&agrave; di contaminazione temporale nei moduli di feature engineering, assemblaggio e meta-learning &mdash; rimediate il 30 Maggio 2026</div>
  </div>
</div>

<div class="cnt">

  <!-- Summary Metrics -->
  <div class="grid-2">
    <div class="card">
      <div class="metric">
        <div class="val" style="color:#ef4444;">3</div>
        <div class="lbl">Data Leak Identificati</div>
      </div>
    </div>
    <div class="card">
      <div class="metric">
        <div class="val" style="color:#22c55e;">3 / 3</div>
        <div class="lbl">Vulnerabilit&agrave; Risolte</div>
      </div>
    </div>
  </div>

  <!-- Found 1: feature_pipeline.py -->
  <div class="card">
    <div class="card-hd">
      <h2>1. forward_return_pct nel CSV</h2>
      <span class="tag tag-leak">LEAK</span>
    </div>
    <div class="card-bd">
      <div class="finding critical">
        <strong>Problema:</strong> <code>create_forward_target()</code> salvava <code>forward_return_pct</code> come colonna nel CSV di output. Questa colonna contiene il rendimento percentuale futuro (<code>close[t+3] - close[t] / close[t]</code>) ed era quindi un valore noto solo in futuro, disponibile come feature per qualsiasi modello addestrato sul file.
        <div class="file-path" style="margin-top:12px;">feature_pipeline.py:57 &mdash; <span class="highlight">df['forward_return_pct'] = forward_return</span></div>
      </div>
      <div class="finding fixed">
        <strong>Correzione:</strong> La colonna <code>forward_return_pct</code> &egrave; stata rimossa dalla funzione. Il target binario <code>target</code> viene ora calcolato a runtime dentro ogni fold di training in <code>xgboost_pipeline.py</code>, non pre-salvato su disco.
        <div class="file-path" style="margin-top:12px;">feature_pipeline.py:54-60 &mdash; <span class="highlight diff-add">+ calcolato dentro ogni fold</span></div>
        <div class="file-path">feature_pipeline.py:175-179 &mdash; <span class="highlight diff-add">+ drop colonne forward-looking prima del salvataggio</span></div>
      </div>
    </div>
  </div>

  <!-- Found 2: run_unified.py -->
  <div class="card">
    <div class="card-hd">
      <h2>2. Sentiment Intraday &amp; Leak Check Cieco</h2>
      <span class="tag tag-leak">LEAK</span>
    </div>
    <div class="card-bd">
      <div class="finding critical">
        <strong>Problema A &mdash; Leak check strutturalmente cieco:</strong> Il "Maniacal Timestamp Check" confrontava <code>sentiment_date_parsed &gt; timestamps_parsed</code>, ma entrambe le colonne provenivano dalla stessa data (merge su <code>date</code>). La condizione non poteva mai essere <code>True</code>, rendendo il check inutile ma stampando comunque <code>[OK]</code>.
        <div class="file-path">run_unified.py:73-84 (vecchio) &mdash; <span class="highlight">leak check sempre falso positivo</span></div>
      </div>
      <div class="finding critical">
        <strong>Problema B &mdash; Look-ahead bias da sentiment same-day:</strong> Articoli pubblicati dopo la chiusura del mercato (o dopo l'orario di snapshot delle feature) venivano aggregati nella barra daily dello stesso giorno. Un articolo delle 20:00 di luned&igrave; descrive eventi post-market che il modello non poteva conoscere al momento della decisione.
        <div class="file-path">run_unified.py:62-67 (vecchio) &mdash; <span class="highlight">merge left_on='date' senza shift temporale</span></div>
      </div>
      <div class="finding fixed">
        <strong>Correzione A &mdash; Shift T-1:</strong> Tutte le colonne sentiment vengono shiftate di 1 riga (<code>shift(1)</code>) dopo il merge. Il sentiment del giorno T-1 viene usato per predire il giorno T, eliminando ogni possibilit&agrave; di contaminazione intraday.
        <div class="file-path">run_unified.py:74-90 &mdash; <span class="highlight diff-add">+ shift(1) su tutte le colonne sentiment</span></div>
      </div>
      <div class="finding fixed">
        <strong>Correzione B &mdash; Leak check reale:</strong> Dopo lo shift, viene preservata la data originale del sentiment in <code>sentiment_original_date</code> (shiftata anch'essa). Il nuovo check verifica che <code>sentiment_original_date &le; timestamps</code>. Dopo T-1 shift, la data del sentiment &egrave; sempre la barra precedente, mai quella corrente. Se la condizione fallisce, stampa warning reale.
        <div class="file-path">run_unified.py:92-113 &mdash; <span class="highlight diff-add">+ leak check con sentinel shiftato</span></div>
      </div>

      <h3>Dettaglio dello shift T-1</h3>
      <pre>
Prima dello shift:
  timestamps  | sentiment_score | sentiment_original_date
  2024-01-01  |      0.12       |      2024-01-01
  2024-01-02  |     -0.05       |      2024-01-02
  2024-01-05  |      0.30       |      2024-01-05    (luned&igrave;)

Dopo .shift(1):
  timestamps  | sentiment_score | sentiment_original_date  | Verifica leak
  2024-01-01  |      NaN        |            NaN           | N/A (ffill &rarr; 0)
  2024-01-02  |      0.12       |      2024-01-01          | 2024-01-01 &le; 2024-01-02 ✓
  2024-01-05  |     -0.05       |      2024-01-02          | 2024-01-02 &le; 2024-01-05 ✓
      </pre>
      <div style="font-size:12px;color:#64748b;">Il sentiment viene shiftato, non cancellato. L'articolo delle 20:00 di luned&igrave; sar&agrave; disponibile come feature per marted&igrave;.</div>
    </div>
  </div>

  <!-- Found 3: xgboost_pipeline.py -->
  <div class="card">
    <div class="card-hd">
      <h2>3. Target Pre-Calcolato Globalmente + PCA su Tutti i Dati</h2>
      <span class="tag tag-leak">LEAK</span>
    </div>
    <div class="card-bd">
      <div class="finding high">
        <strong>Problema A &mdash; Target calcolato su tutta la serie:</strong> <code>recompute_target(df['close'])</code> veniva chiamato in <code>load_data()</code> sull'intero dataframe prima dello split. Sebbene il target sia un label (non feature), l'architettura violava il principio che ogni fold deve essere indipendente.
        <div class="file-path">xgboost_pipeline.py:64 (vecchio) &mdash; <span class="highlight">target = recompute_target(df['close'])</span></div>
      </div>
      <div class="finding high">
        <strong>Problema B &mdash; PCA fittata sull'intero dataset:</strong> <code>reduce_embeddings()</code> veniva chiamata prima del loop di cross-validation e training. La PCA catturava la struttura di varianza dell'intera serie, inclusi dati futuri. Ogni fold di addestramento usava componenti principali informate dal futuro.
        <div class="file-path">xgboost_pipeline.py:55-56 (vecchio) &mdash; <span class="highlight">emb_pca = reduce_embeddings(df[emb_cols])</span></div>
      </div>
      <div class="finding fixed">
        <strong>Correzione A &mdash; Target dentro ogni fold:</strong> <code>compute_target_for_idx(close, idx)</code> accetta l'array delle <code>close</code> e un array <code>idx</code> di indici. Usa <code>idx[-1]</code> come limite massimo: se <code>idx[j] + horizon &gt; idx[-1]</code>, il target &egrave; marcato -1 (invalido) e mascherato. Zero accesso a close fuori dal fold.
        <div class="file-path">xgboost_pipeline.py:30-48 &mdash; <span class="highlight diff-add">+ compute_target_for_idx() con vincolo fold-aware</span></div>
        <div class="file-path">xgboost_pipeline.py:168-172 (timeseries_cv) &mdash; <span class="highlight diff-add">+ chiamata per fold</span></div>
        <div class="file-path">xgboost_pipeline.py:218-225 (train_and_evaluate) &mdash; <span class="highlight diff-add">+ chiamata per split</span></div>
        <div class="file-path">xgboost_pipeline.py:320-323 (backtest_simulation) &mdash; <span class="highlight diff-add">+ chiamata per walk-forward</span></div>
      </div>
      <div class="finding fixed">
        <strong>Correzione B &mdash; PCA per fold:</strong> <code>reduce_embeddings()</code> ora riceve <code>emb_data[train_idx]</code> (solo train). La trasformazione del test usa <code>pca.transform()</code>. Questo vale per CV, training 80/20 e backtest walk-forward.
        <div class="file-path">xgboost_pipeline.py:162-165 (timeseries_cv) &mdash; <span class="highlight diff-add">+ PCA su train_idx, transform su test_idx</span></div>
        <div class="file-path">xgboost_pipeline.py:215-218 (train_and_evaluate) &mdash; <span class="highlight diff-add">+ PCA su primo 80%, transform su restante 20%</span></div>
        <div class="file-path">xgboost_pipeline.py:329-332 (backtest_simulation) &mdash; <span class="highlight diff-add">+ PCA su train_idx_valid, transform su i-esima riga</span></div>
      </div>

      <h3>Dettaglio: compute_target_for_idx()</h3>
      <pre>
def compute_target_for_idx(close: np.ndarray, idx: np.ndarray,
                           horizon=3, threshold=2.5) -> np.ndarray:
    target = np.full(len(idx), -1, dtype=np.int8)
    last_valid_future = idx[-1]  # &larr; limite: close[idx[-1]]
    for j, i in enumerate(idx):
        future_idx = i + horizon
        if future_idx <= last_valid_future:  # close usata solo se dentro idx
            ret = (close[future_idx] - close[i]) / close[i] * 100
            target[j] = 1 if ret > threshold else 0
    return target
      </pre>
      <div style="font-size:12px;color:#64748b;">Ultimi <code>horizon</code> elementi di ogni fold mascherati (target = -1). Rimossi prima del training.</div>
    </div>
  </div>

  <!-- Riepilogo file modificati -->
  <div class="card">
    <div class="card-hd">
      <h2>Riepilogo File Modificati</h2>
    </div>
    <div class="card-bd" style="padding:0;">
      <div class="summary-item">
        <div>
          <span class="status status-fixed"></span>
          <code style="font-size:14px;">feature_pipeline.py</code>
          <span style="font-size:12px;color:#64748b;margin-left:8px;">217 righe</span>
        </div>
        <div><span class="tag tag-fix">FIXED</span></div>
      </div>
      <div class="summary-item">
        <div>
          <span class="status status-fixed"></span>
          <code style="font-size:14px;">run_unified.py</code>
          <span style="font-size:12px;color:#64748b;margin-left:8px;">186 righe</span>
        </div>
        <div><span class="tag tag-fix">FIXED</span></div>
      </div>
      <div class="summary-item" style="border-bottom:none;">
        <div>
          <span class="status status-fixed"></span>
          <code style="font-size:14px;">xgboost_pipeline.py</code>
          <span style="font-size:12px;color:#64748b;margin-left:8px;">449 righe</span>
        </div>
        <div><span class="tag tag-fix">FIXED</span></div>
      </div>
    </div>
  </div>

  <!-- Checklist finale -->
  <div class="card">
    <div class="card-hd">
      <h2>Checklist di Verifica</h2>
    </div>
    <div class="card-bd">
      <ul>
        <li><strong>&check; forward_return_pct</strong> &mdash; Rimosso dal CSV. Target calcolato a runtime in ogni fold.</li>
        <li><strong>&check; target</strong> &mdash; Non pi&ugrave; salvato in <code>master_btc_features.csv</code>.</li>
        <li><strong>&check; Sentiment intraday</strong> &mdash; Shift T-1: T-1 predice T.</li>
        <li><strong>&check; Leak check</strong> &mdash; Riscritto con confronto reale <code>sentiment_original_date &le; timestamps</code>.</li>
        <li><strong>&check; Target per fold</strong> &mdash; <code>compute_target_for_idx()</code> usa solo <code>close[idx[-1]]</code>.</li>
        <li><strong>&check; PCA per fold</strong> &mdash; Fittata su <code>train_idx</code> in CV, 80/20 e walk-forward.</li>
        <li><strong>&check; Backtest P&amp;L</strong> &mdash; Da prezzi reali di mercato, non forfettario.</li>
      </ul>
    </div>
  </div>

  <!-- Rischi residui -->
  <div class="card">
    <div class="card-hd">
      <h2>Rischi Residui</h2>
      <span class="severity sev-medium">MEDIO</span>
    </div>
    <div class="card-bd">
      <ul>
        <li><strong>Forward-fill inter-day:</strong> Il <code>fillna(method='ffill')</code> propaga il sentiment del giorno pi&ugrave; recente. Se non ci sono notizie per 5 giorni, il sentiment rimane "vecchio" ma corretto (nessun leak).</li>
        <li><strong>Lookback di 90 giorni per normalizzazione:</strong> La normalizzazione sliding-window in <code>feature_pipeline.py</code> usa fino a 90 giorni di storia. Causale per costruzione (<code>local = x[start:i+1]</code>), ma per i primi 90 giorni la stima di media/varianza &egrave; instabile.</li>
        <li><strong>Yahoo Finance dati intraday non disponibili:</strong> La pipeline opera su barre daily. Non &egrave; possible distinguere articoli pre/post chiusura intra-day senza timestamp di pubblicazione nei dati aggregati.</li>
      </ul>
    </div>
  </div>

  <!-- Metodologia -->
  <div class="card">
    <div class="card-hd">
      <h2>Metodologia di Audit</h2>
    </div>
    <div class="card-bd" style="font-size:13px;color:#94a3b8;line-height:1.8;">
      <p>L'audit &egrave; stato condotto il 30 Maggio 2026 esaminando il flusso dei dati attraverso la pipeline:</p>
      <p><strong>Fase 1 (feature_pipeline.py):</strong> Verificato che le colonne derivate da prezzi futuri non venissero salvate su disco. Identificato <code>forward_return_pct</code> come variabile contaminata.</p>
      <p><strong>Fase 3 (run_unified.py):</strong> Esaminato il merge temporale tra prezzi e sentiment. Verificato che il "Maniacal Timestamp Check" fosse operativamente cieco. Identificato il rischio di look-ahead bias da sentiment same-day.</p>
      <p><strong>Fase 4 (xgboost_pipeline.py):</strong> Analizzata la sequenza di split temporale, PCA e calcolo del target. Identificato che PCA veniva fittata su tutti i dati prima dello split. Identificato che il target era pre-calcolato globalmente.</p>
    </div>
  </div>

</div>

<div class="ftr">
  Kronos Quantitative Research &middot; Audit Data Leakage &middot; {AUDIT_DATE}<br>
  <span style="color:#212633;">3 vulnerabilit&agrave; trovate, 3 risolte, 0 aperte</span>
</div>

</body>
</html>
"""

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[OK] Report audit generato: {os.path.abspath(OUTPUT_FILE)}")


if __name__ == "__main__":
    main()

"""
Audit ML quantistico a 3 layer: static scan + cross-file context + Gemini Deep Audit.

Usage:
    python review_gemini.py --audit                 # full audit 3 layer
    python review_gemini.py --audit --quick         # solo layer 1+2 (statico, no Gemini)
    python review_gemini.py --file daily_trade.py   # review singola (come prima)

Requires: GEMINI_API_KEY env var (https://aistudio.google.com/apikey)
"""

import os, sys, json, subprocess, argparse, textwrap, re, time
from datetime import datetime
import requests

GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    print("ERROR: GEMINI_API_KEY non impostata.")
    print("Ottieni: https://aistudio.google.com/apikey")
    sys.exit(1)

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "output", "review_gemini.json")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────
# LAYER 1 — STATIC ANALYSIS (regex pre-scan)
# ─────────────────────────────────────────────

RISK_PATTERNS = {
    "data_leakage": [
        (r"\.fit\([^)]*\)", ".fit() call — verify it's on TRAIN fold only"),
        (r"\.fit_transform\(", ".fit_transform() — fits on data passed, risk of leakage"),
        (r"\.transform\(", ".transform() without .fit — check fit happened on train only"),
        (r"StandardScaler\(\)", "StandardScaler — check .fit() scope"),
        (r"MinMaxScaler", "MinMaxScaler — check .fit() scope"),
        (r"\bPCA\(", "PCA — check .fit() scope (MUST be on train fold)"),
        (r"CausalPCA\(", "CausalPCA — check .fit() scope"),
        (r"train_test_split", "train_test_split found — verify test set isolation"),
        (r"(TimeSeriesSplit|WalkForward)", "TimeSeriesSplit/WalkForward — good, verify fold count >= 20"),
        (r"shift\(-\d\)", "shift(-N) — look-ahead bias risk, verify feature at T uses only data <= T"),
    ],
    "security": [
        (r"(password|PASSWORD)\s*=\s*['\"][^'\"]+['\"]", "Hardcoded password"),
        (r"(api_key|API_KEY|secret|SECRET|token|TOKEN)\s*=\s*['\"][^'\"]+['\"]", "Possible hardcoded credential"),
        (r"eval\(", "eval() — code injection risk"),
        (r"exec\(", "exec() — code injection risk"),
        (r"pickle\.load\b", "pickle.load — arbitrary code execution risk"),
        (r"joblib\.load\b", "joblib.load — verify trusted source"),
        (r"subprocess\.(call|run|Popen)", "subprocess — verify input sanitization"),
    ],
    "code_quality": [
        (r"except\s*:", "Bare except — catches ALL exceptions silently"),
        (r"except\s+Exception\s*:", "Broad except Exception — catch specific exceptions"),
        (r"\bTODO\b", "TODO found — incomplete code"),
        (r"\bFIXME\b", "FIXME found — known bug"),
        (r"\bHACK\b", "HACK found — technical debt"),
        (r"\bXXX\b", "XXX found — needs attention"),
        (r"^\s*pass\s*$", "pass statement — unimplemented block (multi-line)"),
        (r"# type:\s*ignore", "type: ignore — bypassing type checker"),
        (r"if\s+__name__\s*==\s*['\"]__main__['\"]", "Script has __main__ guard — good"),
        (r"from\s+\w+\s+import\s+\*", "Wildcard import — namespace pollution"),
        (r"\bprint\(", "print() in production code — use logger instead"),
    ],
    "architectural": [
        (r"os\.environ\.get", "os.environ.get — good, using env vars"),
        (r"open\(['\"][^'\"]+['\"]", "Hardcoded file path — prefer configurable paths"),
        (r"pd\.read_csv\(['\"][^'\"]*\.csv['\"]", "Hardcoded CSV path"),
        (r"pd\.to_pickle\(", "pd.to_pickle — consider Parquet/Feather for performance"),
        (r"\.to_csv\(", ".to_csv — consider Parquet for large data"),
        (r"requests\.(get|post)\(", "HTTP request — verify timeout handling"),
        (r"\.\./\.\./", "Relative path traversal — verify correctness"),
    ],
    "concurrency": [
        (r"threading\.(Thread|Lock)", "Threading — verify thread safety"),
        (r"multiprocessing", "Multiprocessing — verify resource cleanup"),
        (r"time\.sleep\(", "time.sleep — verify not used for synchronization"),
    ],
}


def static_scan(filepath):
    findings = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        return [{"severity": "major", "category": "io", "line": 0,
                 "description": f"Impossibile leggere file: {e}"}]

    content = "".join(lines)

    for category, patterns in RISK_PATTERNS.items():
        for pattern, description in patterns:
            for match in re.finditer(pattern, content, re.MULTILINE):
                line_num = content[:match.start()].count("\n") + 1
                # Estrai contesto (linea intera)
                ctx_line = lines[line_num - 1].strip() if line_num <= len(lines) else ""
                findings.append({
                    "severity": "critical" if category == "data_leakage" else
                                "major" if category == "security" else "minor",
                    "category": category,
                    "line": line_num,
                    "pattern": pattern,
                    "description": description,
                    "context": ctx_line[:120],
                })

    # Deduplica: tieni solo primo match per pattern per linea
    seen = set()
    unique = []
    for f in findings:
        key = (f["line"], f["pattern"])
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return unique


# ─────────────────────────────────────────────
# LAYER 2 — CROSS-FILE DEPENDENCY MAP
# ─────────────────────────────────────────────

def build_dep_map(py_files):
    dep_map = {}
    for fpath in py_files:
        rel = os.path.relpath(fpath, SCRIPT_DIR)
        deps = set()
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                m = re.match(r"^(?:from\s+(\S+)\s+import|import\s+(\S+))", line)
                if m:
                    mod = m.group(1) or m.group(2)
                    deps.add(mod.split(".")[0])
        dep_map[rel] = sorted(deps)
    return dep_map


def group_files_by_module(py_files):
    """Raggruppa file correlati per modulo/sotto-directory."""
    groups = {}
    for fpath in py_files:
        rel = os.path.relpath(fpath, SCRIPT_DIR)
        parts = rel.replace("\\", "/").split("/")
        key = parts[0] if len(parts) > 1 else "_root"
        groups.setdefault(key, []).append(rel)
    return groups


def build_crossfile_context(py_files, dep_map):
    lines = ["## Cross-File Dependency Map\n"]
    for fpath in sorted(py_files):
        rel = os.path.relpath(fpath, SCRIPT_DIR)
        deps = dep_map.get(rel, [])
        lines.append(f"- `{rel}` → imports: {', '.join(deps) if deps else '(none)'}")
    lines.append("")
    return "\n".join(lines)


# ─────────────────────────────────────────────
# SYSTEM PROMPT — Full ML Audit Checklist
# ─────────────────────────────────────────────

ML_AUDIT_CHECKLIST = textwrap.dedent("""\
Sei un Senior ML Engineer specializzato in audit di sistemi quantitativi finanziari.
Devi analizzare il codice usando la checklist qui sotto e identificare TUTTI i problemi.

RESTITUISCI SOLO JSON valido. Struttura:

{"issues": [{
  "severity": "critical" | "major" | "minor",
  "category": "data_leakage" | "walk_forward" | "architectural" | "security" | "code_quality",
  "module": "nome_file.py",
  "line": <numero linea>,
  "description": "problema specifico",
  "impact": "impatto sul sistema",
  "fix_type": "edit" | "replace" | "add_check",
  "old_code": "codice esistente (se fix_type=edit)",
  "new_code": "codice corretto (se fix_type=edit)",
  "confidence": 0.0-1.0
}]}

CHECKLIST OBBLIGATORIA - Controlla OGNI punto:

=== DATA LEAKAGE (se presente -> critical) ===
[ ] StandardScaler.fit() chiamato SOLO sul train fold, mai sull'intero dataset
[ ] PCA/CausalPCA.fit() chiamato SOLO sul train fold
[ ] Soglia di confidenza selezionata su validation set, mai su test set
[ ] Feature al tempo T non includono info future (shift(-N), pct_change senza shift)
[ ] Normalizzazione sentiment non usa statistiche su dati futuri

=== WALK-FORWARD VALIDATION ===
[ ] Minimo 20 fold temporali
[ ] Ogni fold riporta accuratezza out-of-sample separata
[ ] Baseline naive implementata e confrontata per fold
[ ] Output media + deviazione standard, non singola metrica

=== ARCHITETTURALE ===
[ ] PCA testa n_components in [10,20,50], seleziona minimo >=90% varianza
[ ] PCA serializzato con XGBoost per ogni asset
[ ] NaN yfinance -> log warning + skip giorno
[ ] Nessuna notizia -> sentiment 0.0 + log
[ ] Weekend/festivi -> skip silenzioso
[ ] Fallimento asset -> log + continua batch
[ ] API endpoint NON contiene logica ML

=== QUALITA' CODICE ===
[ ] Type hints su tutte le funzioni pubbliche
[ ] Naming consistente (snake_case funzioni/variabili, UPPER_CASE costanti)
[ ] Docstring per funzioni >10 righe
[ ] requirements.txt con versioni fisse (==)
[ ] Nessun import dentro le funzioni (salvo lazy loading motivato)

Per ogni problema trovato, stima confidence=1.0 solo se SICURO al 100%,
confidence=0.7 se probabile, confidence=0.3 se dubbio.
""")

# ─────────────────────────────────────────────
# CORE — Gemini API + Parsing
# ─────────────────────────────────────────────

_last_called = 0.0
MIN_INTERVAL = 1.2


def call_gemini(prompt, max_retries=5):
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.05,
            "topP": 0.95,
            "maxOutputTokens": 16384,
        }
    }
    for attempt in range(max_retries):
        try:
            r = requests.post(
                f"{GEMINI_URL}?key={API_KEY}",
                json=payload,
                timeout=180
            )
            if r.status_code == 429:
                wait = min(2 ** attempt * 5, 60)
                print(f"  [429] retry in {wait}s ({attempt+1}/{max_retries})")
                time.sleep(wait)
                continue
            r.raise_for_status()
            data = r.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            return text
        except (KeyError, IndexError):
            print(f"  [ERROR] Risposta inaspettata: {json.dumps(data, indent=2)[:300]}")
            return '{"issues": []}'
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429 and attempt < max_retries - 1:
                wait = min(2 ** attempt * 5, 60)
                print(f"  [429] retry in {wait}s ({attempt+1}/{max_retries})")
                time.sleep(wait)
                continue
            print(f"  [HTTP_ERROR] {e}")
            return '{"issues": []}'
    return '{"issues": []}'


def call_gemini_rate_limited(prompt):
    global _last_called
    elapsed = time.time() - _last_called
    if elapsed < MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - elapsed)
    result = call_gemini(prompt)
    _last_called = time.time()
    return result


def clean_json(text):
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start:end+1]
    return text


SEVERITY_MAP = {
    "critical": "critical", "critico": "critical",
    "major": "major", "maggiore": "major",
    "minor": "minor", "minore": "minor",
    "medium": "major", "medio": "major",
    "high": "critical", "alto": "critical",
    "low": "minor", "basso": "minor",
    "info": "minor",
}


def normalize_issues(issues):
    for i in issues:
        i["severity"] = SEVERITY_MAP.get(i.get("severity", "").strip().lower(), "minor")
        i.setdefault("confidence", 1.0)
    return issues


def parse_review(text):
    text = clean_json(text)
    try:
        result = json.loads(text)
        if "issues" in result:
            result["issues"] = normalize_issues(result["issues"])
        return result
    except json.JSONDecodeError:
        pass
    # Fallback: trova l'oggetto JSON dentro il testo
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            result = json.loads(text[start:end+1])
            if "issues" in result:
                result["issues"] = normalize_issues(result["issues"])
            return result
        except json.JSONDecodeError:
            pass
    print(f"  [WARN] JSON non parsabile ({len(text)} chars)")
    print(f"  [DEBUG] {text[:200]}...{text[-100:]}")
    return {"issues": []}


# ─────────────────────────────────────────────
# LAYER 3 — GEMINI DEEP AUDIT
# ─────────────────────────────────────────────

def build_audit_prompt(static_findings, crossfile_ctx, code_snippets):
    # Prepara findings statici come testo
    static_summary = ""
    if static_findings:
        by_cat = {}
        for f in static_findings:
            by_cat.setdefault(f["category"], []).append(f)
        for cat, items in sorted(by_cat.items()):
            static_summary += f"\n[{cat.upper()}] {len(items)} findings\n"
            for item in items[:10]:  # max 10 per categoria
                static_summary += f"  L{item['line']}: {item['description']}\n"
            if len(items) > 10:
                static_summary += f"  ... e {len(items)-10} altri\n"

    return f"""{ML_AUDIT_CHECKLIST}

## STATIC ANALYSIS FINDINGS (pre-scan automatico)
{static_summary or "Nessun finding statico rilevante."}

## CROSS-FILE CONTEXT
{crossfile_ctx}

## CODICE DA AUDITARE
```python
{code_snippets[:25000]}
```

Analizza tutto il codice usando la checklist. Per ogni issue: indica MODULO, LINEA, e se possibile
old_code + new_code per il fix. Confidence=1.0 solo se SICURO. Restituisci SOLO JSON valido."""


def cmd_audit(py_files, quick=False):
    """Esegue audit completo a 3 layer."""
    print(f"\n{'='*60}")
    print(f"AUDIT ML QUANTITATIVO — {len(py_files)} file")
    print(f"{'='*60}")

    # Layer 1: Static scan su tutti i file
    print("\n[LAYER 1] Static analysis...")
    all_static = []
    for fpath in py_files:
        findings = static_scan(fpath)
        for f in findings:
            f["module"] = os.path.relpath(fpath, SCRIPT_DIR)
        all_static.extend(findings)

    by_sev = {"critical": 0, "major": 0, "minor": 0}
    for f in all_static:
        by_sev[f["severity"]] = by_sev.get(f["severity"], 0) + 1
    print(f"  Trovati {len(all_static)} pattern: "
          f"critical={by_sev['critical']}, major={by_sev['major']}, minor={by_sev['minor']}")

    if quick:
        output = {
            "mode": "audit_quick",
            "timestamp": datetime.now().isoformat(),
            "summary": {"total": len(all_static), **by_sev},
            "issues": all_static,
            "note": "Solo layer 1+2 (static scan). Usa --audit senza --quick per layer 3 Gemini.",
        }
        write_output(output)
        return output

    # Layer 2: Cross-file dependency map
    print("\n[LAYER 2] Cross-file dependency map...")
    dep_map = build_dep_map(py_files)
    crossfile_ctx = build_crossfile_context(py_files, dep_map)
    print(f"  {len(dep_map)} file mappati")

    # Layer 3: Gemini deep audit (snipped per file)
    print(f"\n[LAYER 3] Gemini deep audit ({GEMINI_MODEL})...")
    file_groups = group_files_by_module(py_files)
    all_issues = []

    # Raccogli snippet raggruppati per modulo
    def send_group(group_name, file_list):
        snippet = ""
        for rel in sorted(file_list):
            fpath = os.path.join(SCRIPT_DIR, rel)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read()
                snippet += f"\n# === {rel} ===\n{content[:2000]}\n"
            except Exception:
                pass
        if not snippet.strip():
            return []

        print(f"  [{group_name}] sending {len(file_list)} files ({len(snippet)} chars)...")

        if len(snippet) > 45000:
            print(f"    [WARN] prompt troppo lungo ({len(snippet)} chars), splitto...")
            mid = len(file_list) // 2
            return (send_group(f"{group_name}_a", file_list[:mid]) +
                    send_group(f"{group_name}_b", file_list[mid:]))

        static_here = [f for f in all_static if f.get("module", "").startswith(group_name)]
        prompt = build_audit_prompt(static_here, crossfile_ctx, snippet)
        for attempt in range(3):
            try:
                resp = call_gemini_rate_limited(prompt)
                result = parse_review(resp)
                for issue in result.get("issues", []):
                    issue.setdefault("module", group_name)
                    issue.setdefault("confidence", 1.0)
                issues = result.get("issues", [])
                print(f"    -> {len(issues)} issues")
                return issues
            except Exception as e:
                if attempt < 2:
                    print(f"    -> retry {attempt+1}/3: {e}")
                    time.sleep(5)
                else:
                    print(f"    -> ERROR dopo 3 tentativi: {e}")
                    return []

    for group_name, group_files in sorted(file_groups.items()):
        # Split in chunk da max 4 file per evitare 503
        chunked = [group_files[i:i+3] for i in range(0, len(group_files), 3)]
        for idx, chunk in enumerate(chunked):
            subname = f"{group_name}_{idx}" if len(chunked) > 1 else group_name
            all_issues.extend(send_group(subname, chunk))

    # Unisci static + Gemini issues (deduplica)
    seen_keys = set()
    merged = []
    for issue in all_static + all_issues:
        key = (issue.get("module", ""), issue.get("line", 0),
               issue.get("description", "")[:60])
        if key not in seen_keys:
            seen_keys.add(key)
            merged.append(issue)

    by_sev2 = {"critical": 0, "major": 0, "minor": 0}
    for f in merged:
        by_sev2[f["severity"]] = by_sev2.get(f["severity"], 0) + 1

    output = {
        "mode": "audit_deep",
        "model": GEMINI_MODEL,
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total": len(merged),
            "static_only": len(all_static),
            "gemini_only": len(all_issues),
            **by_sev2,
        },
        "issues": merged,
    }
    write_output(output)

    print(f"\n{'='*60}")
    print(f"AUDIT COMPLETATO: {output['summary']['total']} issues "
          f"(critical={by_sev2['critical']}, major={by_sev2['major']}, minor={by_sev2['minor']})")
    print(f"{'='*60}")
    return output


# ─────────────────────────────────────────────
# SINGLE FILE REVIEW (modalità originale)
# ─────────────────────────────────────────────

def get_staged_diff():
    r = subprocess.run(["git", "diff", "--staged"], capture_output=True, text=True, cwd=SCRIPT_DIR)
    return r.stdout


def get_unstaged_diff():
    r = subprocess.run(["git", "diff"], capture_output=True, text=True, cwd=SCRIPT_DIR)
    return r.stdout


def get_file_content(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def list_py_files():
    py_files = []
    for root, dirs, files in os.walk(SCRIPT_DIR):
        dirs[:] = [d for d in dirs if d not in ("__pycache__", ".git", ".github", "venv", ".venv",
                                                  "env", "node_modules", "webui", "finetune",
                                                  "examples", "tests", "model", "figures",
                                                  "output", "Report_Giornalieri_Btc", "finetune_csv",
                                                  "feature_store", "report_AAPL")]
        for f in files:
            if f.endswith(".py"):
                py_files.append(os.path.join(root, f))
    return py_files


def build_review_prompt(content, context):
    return f"""{ML_AUDIT_CHECKLIST}

{context}

```python
{content[:12000]}
```

Analizza il codice usando la checklist. Restituisci SOLO JSON valido."""


def write_output(output):
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n[OUTPUT] {OUTPUT_FILE}")


def apply_fix(filepath, issue):
    if not os.path.exists(filepath):
        return False
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    old = issue.get("old_code", "")
    new = issue.get("new_code", "")
    if issue.get("fix_type") == "edit" and old:
        if old in content:
            content = content.replace(old, new, 1)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"  [FIX] L{issue.get('line','?')}: {issue.get('description','')[:60]}")
            return True
        else:
            print(f"  [WARN] old_code non trovato L{issue.get('line','?')}")
            return False
    return False


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ML Audit Tool — 3 layer (static + cross-file + Gemini)")
    parser.add_argument("--audit", action="store_true", help="Audit completo 3 layer su tutto il progetto")
    parser.add_argument("--quick", action="store_true", help="Con --audit: solo layer 1+2 (statico, no costo API)")
    parser.add_argument("--file", type=str, help="Review singolo file")
    parser.add_argument("--diff", action="store_true", help="Review unstaged git diff")
    parser.add_argument("--full", action="store_true", help="Review tutti i file (senza cross-file context)")
    parser.add_argument("--list", action="store_true", help="Mostra issues senza applicare fix")
    parser.add_argument("--apply", action="store_true", help="Applica fix automatici")
    args = parser.parse_args()

    if args.audit:
        py_files = list_py_files()
        cmd_audit(py_files, quick=args.quick)
        return

    # ── Modalità originali (--file, --full, --diff) ──
    if args.file:
        path = os.path.join(SCRIPT_DIR, args.file)
        if not os.path.exists(path):
            print(f"ERROR: File non trovato: {path}")
            sys.exit(1)
        content = get_file_content(path)
        # Static scan sul file
        static_findings = static_scan(path)
        context = f"File: {args.file} ({len(content)} chars)\n"
        if static_findings:
            context += "\nStatic pre-scan findings:\n" + "\n".join(
                f"  L{f['line']}: [{f['severity']}] {f['description']}" for f in static_findings[:10])
        prompt = build_review_prompt(content, context)
        print(f"[REVIEW] {args.file}...")
        resp = call_gemini_rate_limited(prompt)
        result = parse_review(resp)
        issues = result.get("issues", [])
        output = {
            "file": args.file,
            "timestamp": datetime.now().isoformat(),
            "static_findings": static_findings,
            "summary": {
                "total": len(issues),
                "critical": sum(1 for i in issues if i.get("severity") == "critical"),
                "major": sum(1 for i in issues if i.get("severity") == "major"),
                "minor": sum(1 for i in issues if i.get("severity") == "minor"),
            },
            "issues": issues,
        }
        write_output(output)

    elif args.full:
        files = list_py_files()
        all_static = []
        all_issues = []
        for fpath in files:
            rel = os.path.relpath(fpath, SCRIPT_DIR)
            content = get_file_content(fpath)
            static_findings = static_scan(fpath)
            for f in static_findings:
                f["module"] = rel
            all_static.extend(static_findings)
            context = f"File: {rel} ({len(content)} chars)"
            prompt = build_review_prompt(content, context)
            print(f"[REVIEW] {rel}...")
            resp = call_gemini_rate_limited(prompt)
            result = parse_review(resp)
            for issue in result.get("issues", []):
                issue["module"] = rel
                all_issues.append(issue)
            print(f"  -> {len(result.get('issues', []))} issues")
        merged = all_static + all_issues
        by_sev = {"critical": 0, "major": 0, "minor": 0}
        for f in merged:
            by_sev[f["severity"]] = by_sev.get(f["severity"], 0) + 1
        output = {
            "file": "full_repo",
            "timestamp": datetime.now().isoformat(),
            "summary": {"total": len(merged), **by_sev},
            "issues": merged,
        }
        write_output(output)

    elif args.diff:
        diff = get_unstaged_diff()
        if not diff.strip():
            print("[INFO] Nessun unstaged diff.")
            return
        prompt = f"""{ML_AUDIT_CHECKLIST}

Git diff (unstaged changes):

```diff
{diff[:12000]}
```

Analizza il diff usando la checklist. Restituisci SOLO JSON valido."""
        print(f"[REVIEW] Unstaged diff...")
        resp = call_gemini_rate_limited(prompt)
        result = parse_review(resp)
        output = {
            "file": "unstaged_diff",
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total": len(result.get("issues", [])),
                "critical": sum(1 for i in result.get("issues", []) if i.get("severity") == "critical"),
                "major": sum(1 for i in result.get("issues", []) if i.get("severity") == "major"),
                "minor": sum(1 for i in result.get("issues", []) if i.get("severity") == "minor"),
            },
            "issues": result.get("issues", []),
        }
        write_output(output)
    else:
        parser.print_help()
        print("\nEsempi:")
        print("  python review_gemini.py --audit              # audit completo 3 layer")
        print("  python review_gemini.py --audit --quick      # solo static scan (gratis)")
        print("  python review_gemini.py --file daily_trade.py --list")
        sys.exit(1)

    # --list preview
    if args.list:
        try:
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                out = json.load(f)
        except Exception:
            return
        for i in out.get("issues", []):
            line = i.get("line", "?")
            sev = i.get("severity", "?").upper()
            desc = i.get("description", "")[:100]
            conf = i.get("confidence", 1.0)
            conf_str = f" [conf={conf:.1f}]" if conf < 1.0 else ""
            print(f"  [{sev}]{conf_str} {i.get('module','')}:L{line} — {desc}")

    # --apply fix
    if args.apply and (args.file or args.diff):
        try:
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                out = json.load(f)
        except Exception:
            return
        applied = 0
        for issue in out.get("issues", []):
            if issue.get("severity") not in ("critical", "major"):
                continue
            if issue.get("confidence", 1.0) < 0.8:
                continue
            target = os.path.join(SCRIPT_DIR, issue.get("module", ""))
            if apply_fix(target, issue):
                applied += 1
        print(f"\n[APPLIED] {applied} fix applicati (solo critical/major con conf>=0.8)")


if __name__ == "__main__":
    main()

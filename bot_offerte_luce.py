import os, asyncio, logging
from datetime import datetime
from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    ConversationHandler, filters, ContextTypes
)

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)

UTENTI_AUTORIZZATI = {
    365726063,  # enomisia974
    6282141653, # rikk68
}

CAP, RESIDENTE, USO, POTENZA, CONSUMO, PREZZO, TARIFFA, RINNOVABILI, PV = range(9)

OFFERTE_FISSO = [
    {"fornitore": "IREN", "offerta": "36 Fisso Luce Summer Edition", "prezzo": 0.1012, "q_fissa": 12.25, "durata": "36 mesi", "green": False, "sconto": 12, "note": "prezzo kWh piu' basso in assoluto"},
    {"fornitore": "OCTOPUS ENERGY", "offerta": "Fissa 12M", "prezzo": 0.1080, "q_fissa": 6.00, "durata": "12 mesi", "green": False, "sconto": 0, "note": "q.fissa minima del mercato (6\u20ac/mese)"},
    {"fornitore": "NeN", "offerta": "Special 36 Luce", "prezzo": 0.1090, "q_fissa": 9.00, "durata": "36 mesi", "green": False, "sconto": 0, "note": "prezzo bloccato 36 mesi, gruppo A2A"},
    {"fornitore": "ALPERIA", "offerta": "Direct Energy", "prezzo": 0.1094, "q_fissa": 9.08, "durata": "60 mesi", "green": False, "sconto": 0, "note": "prezzo bloccato 5 anni, q.fissa contenuta"},
    {"fornitore": "PULSEE", "offerta": "Luce RELAX Fix 36", "prezzo": 0.1179, "q_fissa": 12.00, "durata": "36 mesi", "green": False, "sconto": 60, "note": "60\u20ac sconto primo anno, 36 mesi bloccato"},
    {"fornitore": "ILLUMIA", "offerta": "Energia Lunghissima", "prezzo": 0.1190, "q_fissa": 13.00, "durata": "36 mesi", "green": False, "sconto": 75, "note": "75\u20ac sconto primo anno, fornitore affidabile"},
    {"fornitore": "EDISON", "offerta": "Web Luce", "prezzo": 0.1240, "q_fissa": 7.50, "durata": "12 mesi", "green": False, "sconto": 30, "note": "q.fissa bassa, 30\u20ac sconto, fornitore top"},
    {"fornitore": "ACEA", "offerta": "Fix Luce", "prezzo": 0.1250, "q_fissa": 7.50, "durata": "12 mesi", "green": False, "sconto": 0, "note": "prezzo bloccato 12 mesi, q.fissa contenuta"},
    {"fornitore": "LENE", "offerta": "Leggera Luce 24", "prezzo": 0.1260, "q_fissa": 7.00, "durata": "24 mesi", "green": False, "sconto": 0, "note": "q.fissa minima (7\u20ac/mese), bloccato 2 anni"},
    {"fornitore": "E.ON", "offerta": "LuceClick Verde", "prezzo": 0.1290, "q_fissa": 9.00, "durata": "12 mesi", "green": True, "sconto": 0, "note": "energia 100% green certificata"},
]

OFFERTE_VARIABILE = [
    {"fornitore": "NeN", "offerta": "Surf Luce", "prezzo": "PUN + 0.000", "q_fissa": 11.00, "green": False, "sconto": 0, "note": "spread ZERO, q.fissa 11\u20ac/mese"},
    {"fornitore": "EDISON", "offerta": "World Luce", "prezzo": "PUN + 0.000", "q_fissa": 10.00, "green": False, "sconto": 0, "note": "spread ZERO, q.fissa 10\u20ac/mese"},
    {"fornitore": "E.ON", "offerta": "Luce Drive Smarty", "prezzo": "PUN + 0.002", "q_fissa": 10.00, "green": True, "sconto": 0, "note": "spread 0.002, energia green"},
    {"fornitore": "ACEA", "offerta": "Sprint Web Luce", "prezzo": "PUN + 0.004", "q_fissa": 8.00, "green": False, "sconto": 0, "note": "spread bassissimo (0.004), q.fissa contenuta"},
    {"fornitore": "E.ON", "offerta": "Flex Click Luce", "prezzo": "PUN + 0.007", "q_fissa": 9.00, "green": True, "sconto": 0, "note": "spread 0.007, energia 100% green"},
    {"fornitore": "OCTOPUS ENERGY", "offerta": "Flex Luce", "prezzo": "PUN + 0.009", "q_fissa": 6.00, "green": False, "sconto": 0, "note": "q.fissa minima del mercato (6\u20ac/mese)"},
    {"fornitore": "HERA", "offerta": "Piu' Controllo Active Easy", "prezzo": "PUN + 0.009", "q_fissa": 10.00, "green": False, "sconto": 0, "note": "spread 0.009, fornitore rinomato"},
    {"fornitore": "SORGENIA", "offerta": "Next Energy Sunlight", "prezzo": "PUN + 0.010", "q_fissa": 6.70, "green": True, "sconto": 0, "note": "spread 0.010, 100% green, q.fissa bassa"},
    {"fornitore": "EDISON", "offerta": "Dynamic Luce", "prezzo": "PUN + 0.014", "q_fissa": 8.25, "green": False, "sconto": 30, "note": "30\u20ac sconto primo anno, spread 0.014"},
    {"fornitore": "ENEL", "offerta": "Flex Box", "prezzo": "PUN + 0.041", "q_fissa": 5.00, "green": False, "sconto": 0, "note": "q.fissa minima (5\u20ac/mese), spread alto"},
]

PUN_MEDIO = 0.13170
PUN_ANNO_PREC = 0.116  # media 2025

CONTESTO_MERCATO = {
    "titolo": "CONTESTO DI MERCATO",
    "trend": "PUN in crescita nel {anno} (media {pun_medio:.4f} EUR/kWh vs {pun_prec:.3f} nel {anno_prec})",
    "fattori": [
        "Tensioni geopolitiche in Medio Oriente e Iran che spingono i prezzi",
        "Previsioni di stabilizzazione solo su base cautelativa",
    ],
    "outlook_fisso": "bloccare un prezzo ora protegge da futuri shock energetici",
    "outlook_variabile": "Se invece vuoi scommettere su un calo del PUN (scenario meno probabile oggi)",
}

def calcola_spesa(prezzo_kwh, q_fissa_mese, consumo, potenza, ha_pv=False, residente=True, sconto=0):
    prelievo = consumo * 0.5 if ha_pv else consumo
    energia = prezzo_kwh * prelievo
    q_fissa_annua = q_fissa_mese * 12
    trasporto = 120 + (8.50 * potenza)
    regolati = trasporto + (0.0227 * prelievo)
    materia = energia + q_fissa_annua - sconto
    iva_aliquota = 0.10 if residente else 0.22
    iva = (materia * iva_aliquota) + (regolati * 0.22)
    totale = materia + regolati + iva
    return round(totale / 12, 2), round(totale, 2)

SEP = "\n" + "\u2500" * 44

def genera_report(consumo, tipo_prezzo, tipo_tariffa, ha_pv, cap, potenza, residente, uso, solo_verde):
    lines = []
    lines.append("REPORT OFFERTE LUCE \u00b7 " + datetime.now().strftime("%d/%m/%Y"))
    res_label = "Residente" if residente else "Non residente"
    p_str = f"{potenza:.1f}".rstrip("0").rstrip(".")
    lines.append(f"CAP {cap} \u00b7 {consumo} kWh/anno \u00b7 {p_str} kW \u00b7 {res_label} \u00b7 {uso}")
    lines.append(f"Prezzo: {tipo_prezzo} \u00b7 {tipo_tariffa} \u00b7 Verde: {'Si' if solo_verde else 'No'} \u00b7 PV: {'Si' if ha_pv else 'No'}")
    lines.append(SEP)

    if ha_pv:
        lines.append("\u26a1 CON FOTOVOLTAICO: priorit\u00e0 alla quota fissa pi\u00f9 bassa")
        lines.append("   (autoconsumo riduce il prelievo dalla rete)")
        lines.append(SEP)

    fisso = [o for o in OFFERTE_FISSO if not solo_verde or o["green"]]
    variabile = [o for o in OFFERTE_VARIABILE if not solo_verde or o["green"]]

    if tipo_prezzo in ("Fisso", "Indifferente") and fisso:
        lines.append(SEP)
        lines.append("OFFERTE A PREZZO FISSO" + (" \u2192 solo rinnovabili" if solo_verde else ""))
        lines.append("")
        def costo_ord(o):
            _, a = calcola_spesa(o["prezzo"], o["q_fissa"], consumo, potenza, ha_pv, residente, o["sconto"])
            return a
        ordinate = sorted(fisso, key=costo_ord)
        for i, o in enumerate(ordinate, 1):
            mese, anno = calcola_spesa(o["prezzo"], o["q_fissa"], consumo, potenza, ha_pv, residente, o["sconto"])
            qf_annua = round(o["q_fissa"] * 12, 2)
            lines.append(f"#{i}")
            lines.append(f"\u2022 Fornitore: {o['fornitore']} \u2013 {o['offerta']}")
            if o["green"]:
                lines.append("  \u2606 Energia 100% rinnovabile")
            lines.append(f"\u2022 SPESA TOTALE STIMATA: {anno}\u20ac")
            lines.append(f"\u2022 Impatto Mensile: {mese}\u20ac")
            lines.append(f"\u2022 Quota Fissa Inclusa: {qf_annua}\u20ac")
            if o["sconto"]:
                lines.append(f"\u2022 Sconto Applicato: {o['sconto']}\u20ac (1\u00b0 anno)")
            lines.append(f"\u2022 Tariffa al kWh: {o['prezzo']:.4f}\u20ac/kWh ({o['durata']})")
            lines.append("")

    if tipo_prezzo in ("Variabile", "Indifferente") and variabile:
        lines.append(SEP)
        lines.append("OFFERTE A PREZZO VARIABILE (PUN medio: {:.4f}\u20ac/kWh)".format(PUN_MEDIO) + (" \u2192 solo rinnovabili" if solo_verde else ""))
        lines.append("")
        def costo_ord_var(o):
            s = o["prezzo"].split("+")[1].strip() if "+" in o["prezzo"] else "0"
            _, a = calcola_spesa(PUN_MEDIO + float(s), o["q_fissa"], consumo, potenza, ha_pv, residente, o["sconto"])
            return a
        ordinate = sorted(variabile, key=costo_ord_var)
        for i, o in enumerate(ordinate, 1):
            s = o["prezzo"].split("+")[1].strip() if "+" in o["prezzo"] else "0"
            p_kwh = PUN_MEDIO + float(s)
            mese, anno = calcola_spesa(p_kwh, o["q_fissa"], consumo, potenza, ha_pv, residente, o["sconto"])
            qf_annua = round(o["q_fissa"] * 12, 2)
            lines.append(f"#{i}")
            lines.append(f"\u2022 Fornitore: {o['fornitore']} \u2013 {o['offerta']}")
            if o["green"]:
                lines.append("  \u2606 Energia 100% rinnovabile")
            lines.append(f"\u2022 SPESA TOTALE STIMATA: {anno}\u20ac*")
            lines.append(f"\u2022 Impatto Mensile: {mese}\u20ac")
            lines.append(f"\u2022 Quota Fissa Inclusa: {qf_annua}\u20ac")
            if o["sconto"]:
                lines.append(f"\u2022 Sconto Applicato: {o['sconto']}\u20ac (1\u00b0 anno)")
            lines.append(f"\u2022 Tariffa al kWh: {p_kwh:.4f}\u20ac/kWh ({o['prezzo']})")
            lines.append("")
        lines.append("* Stima su PUN medio, la spesa varia mensilmente")
        lines.append("")

    ctx = CONTESTO_MERCATO
    anno = datetime.now().year
    anno_prec = anno - 1
    lines.append(SEP)
    lines.append(ctx["titolo"])
    lines.append("")
    lines.append("- " + ctx["trend"].format(anno=anno, pun_medio=PUN_MEDIO, pun_prec=PUN_ANNO_PREC, anno_prec=anno_prec))
    for f in ctx["fattori"]:
        lines.append("- " + f)
    lines.append("")
    lines.append(SEP)
    lines.append("RACCOMANDAZIONE STRATEGICA")
    lines.append("")
    if ha_pv:
        tutte = fisso + variabile
        if tutte:
            best = min(tutte, key=lambda x: x["q_fissa"])
            lines.append("SCELTA CONSIGLIATA: MINIMIZZARE QUOTA FISSA")
            lines.append("")
            lines.append("Motivazione: Con impianto fotovoltaico il prelievo dalla rete e' ridotto")
            lines.append("significativamente. La spesa variabile incide meno della componente fissa.")
            lines.append("L'offerta migliore e' quella con la quota fissa piu' bassa sul mercato.")
            lines.append("")
            lines.append(f"  {best['fornitore']} \u2013 {best['offerta']}")
            lines.append(f"  Quota fissa: {best['q_fissa']:.2f}\u20ac/mese \u2013 {best['durata']}")
            lines.append(f"  Prezzo energia: {best['prezzo'] if isinstance(best['prezzo'], float) else best['prezzo']} EUR/kWh")
            lines.append("")
            lines.append("Con PV il risparmio non viene dal prezzo/kWh ma dalla quota fissa,")
            lines.append("perche' l'autoconsumo riduce i kWh prelevati dalla rete.")
    else:
        lines.append("SCELTA CONSIGLIATA: PREZZO FISSO")
        lines.append("")
        lines.append("Motivazione: Con {0} kWh/anno la differenza tra fisso e variabile".format(consumo))
        lines.append("e' contenuta (~1-2 EUR/mese oggi), ma il rischio di rialzo del PUN e' concreto,")
        lines.append(ctx["outlook_fisso"] + ".")
        if consumo <= 1500:
            lines.append("Bloccare ora a 0.119-0.122 EUR/kWh garantisce certezza di spesa per 12-36 mesi,")
            lines.append(ctx["outlook_fisso"] + ".")
        elif consumo <= 2700:
            lines.append("Bloccare ora intorno a 0.122 EUR/kWh garantisce un risparmio significativo")
            lines.append("a fronte di un rialzo anche contenuto del PUN (+0.01 EUR/kWh).")
        else:
            lines.append("Con consumi elevati, il fisso e' ancora piu' vantaggioso: anche un piccolo")
            lines.append("rialzo del PUN si tradurrebbe in decine di euro extra all'anno.")
        lines.append("")
        lines.append("Offerta consigliata:")
        lines.append("  ALPERIA \u2013 Direct Energy (0.1094 EUR/kWh, 9.08 EUR/mese, 60 mesi)")
        lines.append("  per chi cerca la massima stabilita' a lungo termine, oppure")
        lines.append("  EDISON \u2013 Web Luce (0.124 EUR/kWh, 7.50 EUR/mese, 12 mesi, -30 EUR sconto)")
        lines.append("  se preferisci flessibilita' con un fornitore top-rated.")
        lines.append("  ILLUMIA \u2013 Energia Lunghissima (0.119 EUR/kWh, 75\u20ac sconto, 36 mesi)")
        lines.append("  e' un ottimo compromesso qualita'/prezzo.")
        lines.append("")
        lines.append(ctx["outlook_variabile"] + ",")
        lines.append("Acea Sprint Web Luce offre lo spread piu' basso (PUN + 0.004 EUR/kWh)")
        lines.append("con quota fissa contenuta (8 EUR/mese), oppure Sorgenia Next Energy Sunlight")
        lines.append("se preferisci energia 100% green (PUN + 0.010, 6.70 EUR/mese).")
    lines.append("")
    lines.append(SEP)
    lines.append("* Nota metodologica: Le stime includono tutte le componenti di spesa")
    lines.append("  (materia energia, trasporto, oneri di sistema, imposte, IVA)")
    lines.append("  secondo la metodologia ARERA. I prezzi sono aggiornati al 09/06/2026")
    lines.append("  e potrebbero variare. Dati aggiornati al 09/06/2026 da Facile.it,")
    lines.append("  SOStariffe, Selectra e luce-gas.it. Verifica sempre sul Portale Offerte")
    lines.append("  prima di sottoscrivere: https://www.ilportaleofferte.it")
    return "\n".join(lines)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    uid = update.effective_user.id
    if uid not in UTENTI_AUTORIZZATI:
        logging.warning(f"Accesso negato a user {uid} (@{update.effective_user.username})")
        await update.message.reply_text("Accesso negato. Bot riservato a utenti autorizzati.")
        return ConversationHandler.END
    await update.message.reply_text("Benvenuto! Ti guider\u00f2 nella scelta dell'offerta luce migliore.\n\nInserisci il tuo CAP (es. 00100):")
    return CAP

async def myid(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = update.effective_user.id
    uname = update.effective_user.username or "nessuno"
    await update.message.reply_text(f"Il tuo ID Telegram: {uid}\nUsername: @{uname}\n\nAggiungi questo numero a UTENTI_AUTORIZZATI nel file bot_offerte_luce.py per ottenere accesso.")



async def cap_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data["cap"] = update.message.text.strip()
    reply = ReplyKeyboardMarkup([["Casa", "Altri usi"]], one_time_keyboard=True, resize_keyboard=True)
    await update.message.reply_text("Uso dell'immobile?", reply_markup=reply)
    return USO

async def uso_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data["uso"] = update.message.text.strip()
    reply = ReplyKeyboardMarkup([["Si", "No"]], one_time_keyboard=True, resize_keyboard=True)
    await update.message.reply_text("Sei residente nell'immobile?", reply_markup=reply)
    return RESIDENTE

async def residente_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data["residente"] = update.message.text.strip().lower() == "si"
    reply = ReplyKeyboardMarkup([["3 kW", "6 kW", "9 kW"]], one_time_keyboard=True, resize_keyboard=True)
    await update.message.reply_text("Potenza impegnata?\n(Scegli o digita un valore, es. 4.5)", reply_markup=reply)
    return POTENZA

async def potenza_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    val = update.message.text.strip().lower().replace("kw", "").strip()
    try:
        p = float(val)
        if p < 1 or p > 20:
            raise ValueError
        context.user_data["potenza"] = p
    except ValueError:
        await update.message.reply_text("Valore non valido. Scegli 3, 6, 9 kW o digita un numero (1-20):")
        return POTENZA
    reply = ReplyKeyboardMarkup([["800", "1000", "1500"], ["2000", "2700", "3500"]], one_time_keyboard=True, resize_keyboard=True)
    await update.message.reply_text("Consumo annuo stimato in kWh?\n(Scegli o digita un valore)", reply_markup=reply)
    return CONSUMO

async def consumo_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        context.user_data["consumo"] = int(update.message.text.strip())
    except ValueError:
        await update.message.reply_text("Inserisci un numero valido (es. 1000):")
        return CONSUMO
    reply = ReplyKeyboardMarkup([["Fisso", "Variabile", "Indifferente"]], one_time_keyboard=True, resize_keyboard=True)
    await update.message.reply_text("Tipo di prezzo preferito?", reply_markup=reply)
    return PREZZO

async def prezzo_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data["tipo_prezzo"] = update.message.text.strip()
    reply = ReplyKeyboardMarkup([["Monoraria", "Fasce orarie", "Indifferente"]], one_time_keyboard=True, resize_keyboard=True)
    await update.message.reply_text("Tipo di tariffa?", reply_markup=reply)
    return TARIFFA

async def tariffa_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data["tipo_tariffa"] = update.message.text.strip()
    reply = ReplyKeyboardMarkup([["Si", "No"]], one_time_keyboard=True, resize_keyboard=True)
    await update.message.reply_text("Solo energia da rinnovabili?", reply_markup=reply)
    return RINNOVABILI

async def rinnovabili_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data["solo_verde"] = update.message.text.strip().lower() == "si"
    reply = ReplyKeyboardMarkup([["Si", "No"]], one_time_keyboard=True, resize_keyboard=True)
    await update.message.reply_text("Hai un impianto fotovoltaico?", reply_markup=reply)
    return PV

async def pv_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    ha_pv = update.message.text.strip().lower() == "si"
    context.user_data["pv"] = ha_pv
    await update.message.reply_text("Elaborazione in corso...", reply_markup=ReplyKeyboardRemove())
    report = genera_report(
        consumo=context.user_data["consumo"],
        tipo_prezzo=context.user_data["tipo_prezzo"],
        tipo_tariffa=context.user_data["tipo_tariffa"],
        ha_pv=ha_pv,
        cap=context.user_data["cap"],
        potenza=context.user_data["potenza"],
        residente=context.user_data["residente"],
        uso=context.user_data["uso"],
        solo_verde=context.user_data["solo_verde"]
    )
    for chunk in [report[i:i+4096] for i in range(0, len(report), 4096)]:
        await update.message.reply_text(chunk)
    await update.message.reply_text("Per un nuovo confronto, invia /start")
    return ConversationHandler.END

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("Arrivederci!", reply_markup=ReplyKeyboardRemove())
    return ConversationHandler.END

def main():
    TOKEN = os.environ.get("BOT_TOKEN")
    app = Application.builder().token(TOKEN).build()
    conv = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            CAP: [MessageHandler(filters.TEXT & ~filters.COMMAND, cap_handler)],
            RESIDENTE: [MessageHandler(filters.TEXT & ~filters.COMMAND, residente_handler)],
            USO: [MessageHandler(filters.TEXT & ~filters.COMMAND, uso_handler)],
            POTENZA: [MessageHandler(filters.TEXT & ~filters.COMMAND, potenza_handler)],
            CONSUMO: [MessageHandler(filters.TEXT & ~filters.COMMAND, consumo_handler)],
            PREZZO: [MessageHandler(filters.TEXT & ~filters.COMMAND, prezzo_handler)],
            TARIFFA: [MessageHandler(filters.TEXT & ~filters.COMMAND, tariffa_handler)],
            RINNOVABILI: [MessageHandler(filters.TEXT & ~filters.COMMAND, rinnovabili_handler)],
            PV: [MessageHandler(filters.TEXT & ~filters.COMMAND, pv_handler)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )
    app.add_handler(CommandHandler("myid", myid))
    app.add_handler(conv)
    print("Bot avviato! Premi Ctrl+C per fermarlo.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()

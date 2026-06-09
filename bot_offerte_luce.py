import os, asyncio, logging
from datetime import datetime
from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    ConversationHandler, filters, ContextTypes
)

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)

CAP, RESIDENTE, USO, POTENZA, CONSUMO, PREZZO, TARIFFA, RINNOVABILI, PV = range(9)

OFFERTE_FISSO = [
    {"fornitore": "ILLUMIA", "offerta": "Lunga Luce Easy", "prezzo": 0.108, "q_fissa": 5.50, "durata": "36 mesi", "green": False, "sconto": 0, "note": "quota fissa piu' bassa in assoluto"},
    {"fornitore": "ILLUMIA", "offerta": "Energia Lunghissima", "prezzo": 0.119, "q_fissa": 7.00, "durata": "36 mesi", "green": False, "sconto": 0, "note": "prezzo bloccato 3 anni"},
    {"fornitore": "EDISON", "offerta": "Web Luce", "prezzo": 0.122, "q_fissa": 7.50, "durata": "12 mesi", "green": False, "sconto": 30, "note": "fornitore top"},
    {"fornitore": "E.ON", "offerta": "LuceClick Verde", "prezzo": 0.129, "q_fissa": 9.00, "durata": "12 mesi", "green": True, "sconto": 0, "note": "energia 100% green"},
    {"fornitore": "OCTOPUS ENERGY", "offerta": "Fissa 12M", "prezzo": 0.138, "q_fissa": 6.00, "durata": "12 mesi", "green": False, "sconto": 0, "note": "quota fissa molto bassa"},
]

OFFERTE_VARIABILE = [
    {"fornitore": "SORGENIA", "offerta": "Next Energy Sunlight", "prezzo": "PUN + 0.008", "q_fissa": 6.70, "green": True, "sconto": 0, "note": "spread bassissimo, 100% green"},
    {"fornitore": "OCTOPUS ENERGY", "offerta": "Flex Luce", "prezzo": "PUN + 0.009", "q_fissa": 6.00, "green": False, "sconto": 0, "note": "quota fissa minima del mercato"},
    {"fornitore": "HERA", "offerta": "Piu' Controllo Active Easy", "prezzo": "PUN + 0.009", "q_fissa": 10.00, "green": False, "sconto": 0, "note": "spread contenuto"},
    {"fornitore": "EDISON", "offerta": "Dynamic Luce", "prezzo": "PUN + 0.014", "q_fissa": 8.25, "green": False, "sconto": 30, "note": "sconto primo anno"},
]

PUN_MEDIO = 0.13170

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

SEP = "\n" + "\u2500" * 40 + "\n"

def genera_report(consumo, tipo_prezzo, tipo_tariffa, ha_pv, cap, potenza, residente, uso, solo_verde):
    lines = []
    lines.append("REPORT OFFERTE LUCE \u00b7 " + datetime.now().strftime("%d/%m/%Y"))
    res_label = "Residente" if residente else "Non residente"
    lines.append(f"CAP {cap} \u00b7 {consumo} kWh/anno \u00b7 {potenza} kW \u00b7 {res_label} \u00b7 {uso}")
    lines.append(f"Prezzo: {tipo_prezzo} \u00b7 {tipo_tariffa} \u00b7 Verde: {'Si' if solo_verde else 'No'} \u00b7 PV: {'Si' if ha_pv else 'No'}")
    lines.append(SEP)

    if ha_pv:
        lines.append("\u26a1 CON FOTOVOLTAICO: priorit\u00e0 alla quota fissa pi\u00f9 bassa")
        lines.append("   (autoconsumo riduce il prelievo dalla rete)")
        lines.append(SEP)

    fisso = [o for o in OFFERTE_FISSO if not solo_verde or o["green"]]
    variabile = [o for o in OFFERTE_VARIABILE if not solo_verde or o["green"]]

    if tipo_prezzo in ("Fisso", "Indifferente") and fisso:
        lines.append("OFFERTE PREZZO FISSO")
        if solo_verde:
            lines.append("   (solo energia da rinnovabili)")
        lines.append("")
        ordinate = sorted(fisso, key=lambda x: x["q_fissa"] if ha_pv else x["prezzo"])
        for i, o in enumerate(ordinate, 1):
            mese, anno = calcola_spesa(o["prezzo"], o["q_fissa"], consumo, potenza, ha_pv, residente, o["sconto"])
            lines.append(f"#{i} {o['fornitore']} \u2013 {o['offerta']}")
            if o["green"]:
                lines.append("   \u2606 VERDE")
            lines.append(f"   Prezzo: {o['prezzo']:.3f}  Quota: {o['q_fissa']:.2f}\u20ac/mese  {o['durata']}")
            if o["sconto"]:
                lines.append(f"   Spesa 1\u00b0 anno ~{mese}\u20ac/mese (~{anno}\u20ac/anno, \u2212{o['sconto']}\u20ac sconto)")
            else:
                lines.append(f"   Spesa ~{mese}\u20ac/mese (~{anno}\u20ac/anno)")
            lines.append("   " + o["note"])
            lines.append("")

    if tipo_prezzo in ("Variabile", "Indifferente") and variabile:
        lines.append("OFFERTE PREZZO VARIABILE (PUN medio: {:.4f}\u20ac/kWh)".format(PUN_MEDIO))
        if solo_verde:
            lines.append("   (solo energia da rinnovabili)")
        lines.append("")
        ordinate = sorted(variabile, key=lambda x: x["q_fissa"] if ha_pv else 0)
        for i, o in enumerate(ordinate, 1):
            s = o["prezzo"].split("+")[1].strip() if "+" in o["prezzo"] else "0"
            p_kwh = PUN_MEDIO + float(s)
            mese, anno = calcola_spesa(p_kwh, o["q_fissa"], consumo, potenza, ha_pv, residente, o["sconto"])
            lines.append(f"#{i} {o['fornitore']} \u2013 {o['offerta']}")
            if o["green"]:
                lines.append("   \u2606 VERDE")
            lines.append(f"   Indice: {o['prezzo']}  Quota: {o['q_fissa']:.2f}\u20ac/mese")
            if o["sconto"]:
                lines.append(f"   Spesa 1\u00b0 anno ~{mese}\u20ac/mese (~{anno}\u20ac/anno, \u2212{o['sconto']}\u20ac sconto)*")
            else:
                lines.append(f"   Spesa ~{mese}\u20ac/mese (~{anno}\u20ac/anno)*")
            lines.append("   " + o["note"])
            lines.append("")
        lines.append("* Stima su PUN medio, la spesa varia mensilmente")
        lines.append("")

    lines.append(SEP)
    lines.append("RACCOMANDAZIONE")
    lines.append("")
    if ha_pv:
        tutte = fisso + variabile
        if tutte:
            best = min(tutte, key=lambda x: x["q_fissa"])
            lines.append(f"  {best['fornitore']} \u2013 {best['offerta']}")
            lines.append(f"  Quota fissa: {best['q_fissa']:.2f}\u20ac/mese (la pi\u00f9 bassa)")
            lines.append("  Con PV minimizza la quota fissa, non il prezzo/kWh")
    else:
        lines.append("  Mercato in rialzo (PUN 0,132\u20ac/kWh). Meglio un fisso.")
        lines.append("  EDISON Web Luce (0,122\u20ac/kWh, 7,50\u20ac/mese, \u221230\u20ac sconto)")
    lines.append("")
    lines.append(SEP)
    lines.append("Fonte: ilportaleofferte.it \u00b7 Open Data ARERA (09/06/2026)")
    return "\n".join(lines)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("Benvenuto! Ti guider\u00f2 nella scelta dell'offerta luce migliore.\n\nInserisci il tuo CAP (es. 00100):")
    return CAP

async def cap_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data["cap"] = update.message.text.strip()
    reply = ReplyKeyboardMarkup([["Si", "No"]], one_time_keyboard=True, resize_keyboard=True)
    await update.message.reply_text("Sei residente nell'immobile?", reply_markup=reply)
    return RESIDENTE

async def residente_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data["residente"] = update.message.text.strip().lower() == "si"
    reply = ReplyKeyboardMarkup([["Casa", "Altri usi"]], one_time_keyboard=True, resize_keyboard=True)
    await update.message.reply_text("Uso dell'immobile?", reply_markup=reply)
    return USO

async def uso_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data["uso"] = update.message.text.strip()
    reply = ReplyKeyboardMarkup([["3 kW", "6 kW", "9 kW"]], one_time_keyboard=True, resize_keyboard=True)
    await update.message.reply_text("Potenza impegnata?", reply_markup=reply)
    return POTENZA

async def potenza_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    val = update.message.text.strip().lower().replace("kw", "").strip()
    try:
        context.user_data["potenza"] = int(val)
    except ValueError:
        await update.message.reply_text("Scegli tra 3, 6 o 9 kW:")
        return POTENZA
    reply = ReplyKeyboardMarkup([["800", "1000", "1500"], ["2000", "2700", "3500"]], one_time_keyboard=True, resize_keyboard=True)
    await update.message.reply_text("Consumo annuo stimato in kWh?", reply_markup=reply)
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
    app.add_handler(conv)
    print("Bot avviato! Premi Ctrl+C per fermarlo.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()

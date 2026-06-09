import os, asyncio, logging
from datetime import datetime
from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    ConversationHandler, filters, ContextTypes
)

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)

CAP, POTENZA, CONSUMO, PREZZO, TARIFFA, PV = range(6)

OFFERTE_FISSO = [
    {"fornitore": "ILLUMIA", "offerta": "Lunga Luce Easy", "prezzo": 0.108, "q_fissa": 5.50, "durata": "36 mesi", "note": "quota fissa piu' bassa in assoluto"},
    {"fornitore": "ILLUMIA", "offerta": "Energia Lunghissima", "prezzo": 0.119, "q_fissa": 7.00, "durata": "36 mesi", "note": "prezzo bloccato 3 anni"},
    {"fornitore": "EDISON", "offerta": "Web Luce", "prezzo": 0.122, "q_fissa": 7.50, "durata": "12 mesi", "note": "sconto -30 EUR/anno, fornitore top"},
    {"fornitore": "E.ON", "offerta": "LuceClick Verde", "prezzo": 0.129, "q_fissa": 9.00, "durata": "12 mesi", "note": "energia 100% green"},
    {"fornitore": "OCTOPUS ENERGY", "offerta": "Fissa 12M", "prezzo": 0.138, "q_fissa": 6.00, "durata": "12 mesi", "note": "quota fissa molto bassa"},
]

OFFERTE_VARIABILE = [
    {"fornitore": "SORGENIA", "offerta": "Next Energy Sunlight", "prezzo": "PUN + 0.008", "q_fissa": 6.70, "note": "spread bassissimo, 100% green"},
    {"fornitore": "OCTOPUS ENERGY", "offerta": "Flex Luce", "prezzo": "PUN + 0.009", "q_fissa": 6.00, "note": "quota fissa minima del mercato"},
    {"fornitore": "HERA", "offerta": "Piu' Controllo Active Easy", "prezzo": "PUN + 0.009", "q_fissa": 10.00, "note": "spread contenuto"},
    {"fornitore": "EDISON", "offerta": "Dynamic Luce", "prezzo": "PUN + 0.014", "q_fissa": 8.25, "note": "sconto -30 EUR/anno"},
]

PUN_MEDIO = 0.13170

def calcola_spesa(prezzo_kwh, q_fissa_mese, consumo, potenza, ha_pv=False):
    prelievo = consumo * 0.5 if ha_pv else consumo
    energia = prezzo_kwh * prelievo
    q_fissa_annua = q_fissa_mese * 12
    trasporto = 120 + (8.50 * potenza)
    regolati = trasporto + (0.0227 * prelievo)
    materia = energia + q_fissa_annua
    iva = (materia * 0.10) + (regolati * 0.22)
    totale = materia + regolati + iva
    return round(totale / 12, 2), round(totale, 2)

SEP = "\n" + "\u2500" * 40 + "\n"

def genera_report(consumo, tipo_prezzo, tipo_tariffa, ha_pv, cap, potenza):
    lines = []
    lines.append(f"REPORT OFFERTE LUCE · {datetime.now().strftime('%d/%m/%Y')}")
    lines.append(f"CAP {cap} · {consumo} kWh/anno · {potenza} kW · Domestico Residente")
    lines.append(f"Preferenze: {tipo_prezzo} · {tipo_tariffa} · Fotovoltaico: {'Si' if ha_pv else 'No'}")
    lines.append(SEP)

    if ha_pv:
        lines.append("\u26a1 CON FOTOVOLTAICO: quota fissa > prezzo/kWh")
        lines.append("   (autoconsumo riduce il prelievo dalla rete)")
        lines.append(SEP)

    if tipo_prezzo in ("Fisso", "Indifferente"):
        lines.append("OFFERTE PREZZO FISSO")
        lines.append("")
        ordinate = sorted(OFFERTE_FISSO, key=lambda x: x["q_fissa"] if ha_pv else x["prezzo"])
        for i, o in enumerate(ordinate, 1):
            mese, anno = calcola_spesa(o["prezzo"], o["q_fissa"], consumo, potenza, ha_pv)
            lines.append(f"#{i} {o['fornitore']} \u2013 {o['offerta']}")
            lines.append(f"   Prezzo: {o['prezzo']:.3f}  Quota: {o['q_fissa']:.2f}\u20ac/mese  {o['durata']}")
            lines.append(f"   Spesa ~{mese}\u20ac/mese (~{anno}\u20ac/anno)")
            lines.append(f"   {o['note']}")
            lines.append("")

    if tipo_prezzo in ("Variabile", "Indifferente"):
        lines.append("OFFERTE PREZZO VARIABILE (PUN medio: {:.4f}\u20ac/kWh)".format(PUN_MEDIO))
        lines.append("")
        ordinate = sorted(OFFERTE_VARIABILE, key=lambda x: x["q_fissa"] if ha_pv else 0)
        for i, o in enumerate(ordinate, 1):
            s = o["prezzo"].split("+")[1].strip() if "+" in o["prezzo"] else "0"
            p_kwh = PUN_MEDIO + float(s)
            mese, anno = calcola_spesa(p_kwh, o["q_fissa"], consumo, potenza, ha_pv)
            lines.append(f"#{i} {o['fornitore']} \u2013 {o['offerta']}")
            lines.append(f"   Indice: {o['prezzo']}  Quota: {o['q_fissa']:.2f}\u20ac/mese")
            lines.append(f"   Spesa ~{mese}\u20ac/mese (~{anno}\u20ac/anno)*")
            lines.append(f"   {o['note']}")
            lines.append("")
        lines.append("* Stima su PUN medio, la spesa varia mensilmente")
        lines.append("")

    lines.append(SEP)
    lines.append("RACCOMANDAZIONE")
    lines.append("")
    if ha_pv:
        best = min(OFFERTE_FISSO + OFFERTE_VARIABILE, key=lambda x: x["q_fissa"])
        lines.append(f"  {best['fornitore']} \u2013 {best['offerta']}")
        lines.append(f"  Quota fissa: {best['q_fissa']:.2f}\u20ac/mese (la piu' bassa)")
        lines.append("  Con PV minimizza la quota fissa, non il prezzo/kWh")
    else:
        lines.append("  Mercato in rialzo (PUN 0,132\u20ac/kWh). Meglio un fisso.")
        lines.append("  EDISON Web Luce (0,122\u20ac/kWh, 7,50\u20ac/mese, \u221230\u20ac sconto)")
    lines.append("")
    lines.append(SEP)
    lines.append("Fonte: ilportaleofferte.it \u00b7 Open Data ARERA (09/06/2026)")
    return "\n".join(lines)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text(
        "Benvenuto! Ti guidero' nella scelta dell'offerta luce migliore.\n\nInserisci il tuo CAP (es. 00100):"
    )
    return CAP

async def cap_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data["cap"] = update.message.text.strip()
    reply = ReplyKeyboardMarkup([["3 kW", "6 kW", "9 kW"]], one_time_keyboard=True, resize_keyboard=True)
    await update.message.reply_text("Qual e' la potenza impegnata?", reply_markup=reply)
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
    reply = ReplyKeyboardMarkup([["Monoraria", "Multioraria", "Indifferente"]], one_time_keyboard=True, resize_keyboard=True)
    await update.message.reply_text("Tipo di tariffa oraria?", reply_markup=reply)
    return TARIFFA

async def tariffa_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data["tipo_tariffa"] = update.message.text.strip()
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
        potenza=context.user_data["potenza"]
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
            POTENZA: [MessageHandler(filters.TEXT & ~filters.COMMAND, potenza_handler)],
            CONSUMO: [MessageHandler(filters.TEXT & ~filters.COMMAND, consumo_handler)],
            PREZZO: [MessageHandler(filters.TEXT & ~filters.COMMAND, prezzo_handler)],
            TARIFFA: [MessageHandler(filters.TEXT & ~filters.COMMAND, tariffa_handler)],
            PV: [MessageHandler(filters.TEXT & ~filters.COMMAND, pv_handler)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )
    app.add_handler(conv)
    print("Bot avviato! Premi Ctrl+C per fermarlo.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()

import logging
from datetime import datetime
from kronos_system.config import ASSETS
from kronos_system.ml.trainer import run_full_pipeline
from kronos_system.data.database import init_db

logger = logging.getLogger(__name__)


def run_daily_batch(asset_ids: list[str] | None = None):
    """Run full pipeline for all (or specified) assets.
    
    Each asset is independent: if one fails, others continue.
    Logs structured results for monitoring.
    """
    if asset_ids is None:
        asset_ids = ASSETS

    logger.info("=" * 60)
    logger.info("DAILY BATCH START — %s", datetime.utcnow().isoformat())
    logger.info("Assets: %s", ", ".join(asset_ids))
    logger.info("=" * 60)

    results = {}
    for aid in asset_ids:
        try:
            logger.info("--- Processing %s ---", aid)
            result = run_full_pipeline(aid)
            results[aid] = result["status"]
            logger.info("%s → status=%s | signal=%s | prob=%s",
                        aid, result["status"],
                        result.get("prediction", {}).get("signal", "N/A"),
                        result.get("prediction", {}).get("probability", "N/A"))
        except Exception as e:
            logger.error("Unhandled exception for %s: %s", aid, e, exc_info=True)
            results[aid] = f"crash: {e}"

    ok = sum(1 for s in results.values() if s == "ok")
    fail = sum(1 for s in results.values() if s != "ok")
    logger.info("=" * 60)
    logger.info("DAILY BATCH END — OK=%d FAIL=%d", ok, fail)
    logger.info("=" * 60)
    return results

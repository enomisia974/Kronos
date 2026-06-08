import logging
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from kronos_system.config import CRON_HOUR, CRON_MINUTE, CRON_TIMEZONE
from kronos_system.inference import run_daily_batch

logger = logging.getLogger(__name__)

_scheduler = None


def start_scheduler():
    """Start APScheduler with daily cron at configured time."""
    global _scheduler
    if _scheduler is not None and _scheduler.running:
        logger.warning("Scheduler already running")
        return _scheduler

    _scheduler = BackgroundScheduler(daemon=True)
    trigger = CronTrigger(hour=CRON_HOUR, minute=CRON_MINUTE, timezone=CRON_TIMEZONE)
    _scheduler.add_job(
        run_daily_batch,
        trigger=trigger,
        id="kronos_daily_batch",
        name="Daily Kronos pipeline for all assets",
        replace_existing=True,
        misfire_grace_time=3600,
    )
    _scheduler.start()
    logger.info("Scheduler started: daily at %02d:%02d %s", CRON_HOUR, CRON_MINUTE, CRON_TIMEZONE)
    return _scheduler


def stop_scheduler():
    global _scheduler
    if _scheduler:
        _scheduler.shutdown(wait=False)
        logger.info("Scheduler stopped")

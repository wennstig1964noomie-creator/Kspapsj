"""Flask web app. Users paste their Alpaca keys, start/stop the bot, monitor status."""
from __future__ import annotations

import logging
import threading

from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask, flash, jsonify, redirect, render_template, request, session, url_for
from flask_session import Session

import config
from alpaca_client import AlpacaCredentials
from trading_bot import TradingBot

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = Flask(__name__)
app.config["SECRET_KEY"] = config.SECRET_KEY
app.config["SESSION_TYPE"] = config.SESSION_TYPE
Session(app)

_bots: dict[str, TradingBot] = {}
_bot_lock = threading.Lock()
_scheduler = BackgroundScheduler(daemon=True)
_scheduler.start()


def _bot_for_session() -> TradingBot | None:
    return _bots.get(session.get("sid", ""))


def _job_id(sid: str) -> str:
    return f"bot-{sid}"


@app.route("/")
def index():
    bot = _bot_for_session()
    if bot is None:
        return render_template("login.html")
    try:
        status = bot.status()
    except Exception as e:
        flash(f"Could not load account: {e}", "error")
        return redirect(url_for("logout"))
    running = _scheduler.get_job(_job_id(session["sid"])) is not None
    return render_template("dashboard.html", status=status, running=running)


@app.route("/connect", methods=["POST"])
def connect():
    key = request.form.get("api_key", "").strip()
    secret = request.form.get("api_secret", "").strip()
    paper = request.form.get("paper", "on") == "on"
    if not key or not secret:
        flash("API key and secret are required.", "error")
        return redirect(url_for("index"))

    creds = AlpacaCredentials(api_key=key, api_secret=secret, paper=paper)
    try:
        bot = TradingBot(creds)
        bot.client.account()
    except Exception as e:
        flash(f"Could not authenticate with Alpaca: {e}", "error")
        return redirect(url_for("index"))

    sid = f"{key[:6]}-{id(bot)}"
    session["sid"] = sid
    with _bot_lock:
        _bots[sid] = bot
    flash("Connected to Alpaca.", "ok")
    return redirect(url_for("index"))


@app.route("/start", methods=["POST"])
def start():
    bot = _bot_for_session()
    if bot is None:
        return redirect(url_for("index"))
    job_id = _job_id(session["sid"])
    if _scheduler.get_job(job_id) is None:
        _scheduler.add_job(
            bot.run_once,
            "interval",
            minutes=config.TRADE_INTERVAL_MINUTES,
            id=job_id,
        )
        threading.Thread(target=bot.run_once, daemon=True).start()
        flash(f"Bot started. Evaluating every {config.TRADE_INTERVAL_MINUTES} min.", "ok")
    return redirect(url_for("index"))


@app.route("/stop", methods=["POST"])
def stop():
    sid = session.get("sid")
    if sid:
        job = _scheduler.get_job(_job_id(sid))
        if job:
            job.remove()
            flash("Bot stopped.", "ok")
    return redirect(url_for("index"))


@app.route("/close-all", methods=["POST"])
def close_all():
    bot = _bot_for_session()
    if bot is None:
        return redirect(url_for("index"))
    try:
        bot.client.close_all()
        flash("Submitted close-all order.", "ok")
    except Exception as e:
        flash(f"Close-all failed: {e}", "error")
    return redirect(url_for("index"))


@app.route("/status.json")
def status_json():
    bot = _bot_for_session()
    if bot is None:
        return jsonify({"error": "not connected"}), 401
    return jsonify(bot.status())


@app.route("/logout")
def logout():
    sid = session.pop("sid", None)
    if sid:
        with _bot_lock:
            _bots.pop(sid, None)
        job = _scheduler.get_job(_job_id(sid))
        if job:
            job.remove()
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

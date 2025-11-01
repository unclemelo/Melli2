# bot.py — Melli v2.0 (Training Edition, Robust Feedback)
# ──────────────────────────────────────────────

import discord
import os
import asyncio
from discord.ext import commands, tasks
from dotenv import load_dotenv
from colorama import Fore, Style, init
from datetime import datetime, timedelta
from ai_module import AIManager
import random
import re

init(autoreset=True)

# ──────────────────────────────────────────────
# Load environment
load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")

# ──────────────────────────────────────────────
# Bot setup
intents = discord.Intents.all()
client = commands.AutoShardedBot(command_prefix="!", shard_count=1, intents=intents)
client.remove_command("help")

# ──────────────────────────────────────────────
# Terminal helpers
def terminal_banner():
    print(f"""{Fore.MAGENTA}{Style.BRIGHT}
╔════════════════════════════════════════════════╗
║               MELLI SYSTEM v2.0                ║
╚════════════════════════════════════════════════╝
    """)

def log(msg: str, level: str = "info"):
    time = datetime.now().strftime("%H:%M:%S")
    levels = {
        "info": Fore.CYAN + "[INFO]",
        "success": Fore.GREEN + "[SUCCESS]",
        "warn": Fore.YELLOW + "[WARN]",
        "error": Fore.RED + "[ERROR]",
        "critical": Fore.MAGENTA + "[CRITICAL]",
    }
    tag = levels.get(level, Fore.WHITE + "[LOG]")
    print(f"{Fore.BLACK}[{time}]{Style.RESET_ALL} {tag} {Fore.WHITE}{msg}{Style.RESET_ALL}")

# ──────────────────────────────────────────────
# AI Setup
ai_manager = AIManager("models/model.pt", "models/vectorizer.pkl")
last_channel_creation = datetime.min
CHANNEL_COOLDOWN = timedelta(minutes=1)
TRUSTED_DEVS = [954135885392252940, 667032667732312115]

# ──────────────────────────────────────────────
# Events
@client.event
async def on_ready():
    terminal_banner()
    log(f"System online as {client.user} ({client.user.id})", "success")
    log(f"Connected to {len(client.guilds)} guilds.", "info")

@client.event
async def on_message(message):
    if message.author.bot:
        return

    # Log all messages from trusted devs for training
    ai_manager.log_user_message(message.author.id, message.content)

    global last_channel_creation
    now = datetime.now()

    if ai_manager.should_create_channel(message.content):
        if now - last_channel_creation >= CHANNEL_COOLDOWN:
            guild = message.guild
            channel_name = ai_manager.suggest_channel_name(message.content)
            try:
                await guild.create_text_channel(channel_name)
                await message.channel.send(f"Created channel: {channel_name}")
                last_channel_creation = now
                log(f"Channel '{channel_name}' created in {guild.name}", "success")
            except Exception as e:
                log(f"Failed to create channel: {e}", "error")
        else:
            log("Channel creation skipped due to cooldown.", "warn")

    await client.process_commands(message)

@client.event
async def on_reaction_add(reaction, user):
    
    if reaction.message.author.id != client.user.id:
        return

    # Safely extract suggested channel name
    suggested_name = None
    match = re.search(r"Created channel:\s*(.+)", reaction.message.content)
    if match:
        suggested_name = match.group(1).strip()
    else:
        suggested_name = "unknown-channel"

    if reaction.emoji == "👍":
        ai_manager.adjust_mood(success=True)
        ai_manager.log_feedback(
            message=reaction.message.content,
            suggested_name=suggested_name,
            reaction="up"
        )
        log(f"Received positive reinforcement from {user}", "success")
    elif reaction.emoji == "👎":
        ai_manager.adjust_mood(success=False)
        ai_manager.log_feedback(
            message=reaction.message.content,
            suggested_name=suggested_name,
            reaction="down"
        )
        log(f"Received negative reinforcement from {user}", "warn")

@client.event
async def on_member_remove(member):
    ai_manager.delete_user_data(member.id)
    log(f"Deleted data for user {member}", "info")

# ──────────────────────────────────────────────
# Main entry
async def main():
    try:
        log("Starting Melli client...", "info")
        await client.start(TOKEN)
    except KeyboardInterrupt:
        log("Manual shutdown requested (Ctrl+C)", "warn")
        await client.close()
    except Exception as e:
        log(f"Failed to start bot: {e}", "critical")

if __name__ == "__main__":
    asyncio.run(main())

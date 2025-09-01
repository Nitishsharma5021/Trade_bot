import os
import logging
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from binance.client import Client
from binance.exceptions import BinanceAPIException
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackContext, CallbackQueryHandler
from telegram.constants import ParseMode
from dotenv import load_dotenv

load_dotenv()  # This loads variables from .env file

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Configuration - Set your API keys from environment variables
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHANNEL_ID = os.getenv('TELEGRAM_CHANNEL_ID')

# Check if all required environment variables are set
required_vars = {
    'BINANCE_API_KEY': BINANCE_API_KEY,
    'BINANCE_SECRET_KEY': BINANCE_SECRET_KEY,
    'TELEGRAM_BOT_TOKEN': TELEGRAM_BOT_TOKEN,
    'TELEGRAM_CHANNEL_ID': TELEGRAM_CHANNEL_ID
}

missing_vars = [var for var, value in required_vars.items() if value is None]
if missing_vars:
    logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
    logger.error("Please set these variables in your .env file")
    exit(1)

# Initialize Binance client
try:
    client = Client(BINANCE_API_KEY, BINANCE_SECRET_KEY, testnet=False)
    logger.info("Binance client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Binance client: {str(e)}")
    client = None

# Global variables - Top 20 cryptocurrencies for analysis
symbols = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT',
    'SOLUSDT', 'DOGEUSDT', 'MATICUSDT', 'DOTUSDT', 'LTCUSDT',
    'AVAXUSDT', 'LINKUSDT', 'ATOMUSDT', 'UNIUSDT', 'XLMUSDT',
    'ALGOUSDT', 'XTZUSDT', 'BCHUSDT', 'VETUSDT', 'FILUSDT'
]

# Store last sent signals to avoid duplicates
last_signals_sent = {}

# Technical Analysis Functions
def calculate_rsi(prices, period=14):
    """Calculate RSI indicator"""
    if len(prices) < period + 1:
        return np.zeros_like(prices)
    
    deltas = np.diff(prices)
    seed = deltas[:period+1]
    up = seed[seed >= 0].sum()/period
    down = -seed[seed < 0].sum()/period
    rs = up/down if down != 0 else 1
    rsi = np.zeros_like(prices)
    rsi[:period] = 100. - 100./(1. + rs)

    for i in range(period, len(prices)):
        delta = deltas[i-1]
        if delta > 0:
            up_val = delta
            down_val = 0.
        else:
            up_val = 0.
            down_val = -delta

        up = (up*(period-1) + up_val)/period
        down = (down*(period-1) + down_val)/period
        rs = up/down if down != 0 else 1
        rsi[i] = 100. - 100./(1. + rs)

    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD indicator"""
    if len(prices) < slow:
        return np.zeros_like(prices), np.zeros_like(prices), np.zeros_like(prices)
    
    exp1 = pd.Series(prices).ewm(span=fast, min_periods=fast).mean()
    exp2 = pd.Series(prices).ewm(span=slow, min_periods=slow).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, min_periods=signal).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_bollinger_bands(prices, window=20, num_std=2):
    """Calculate Bollinger Bands"""
    if len(prices) < window:
        return np.zeros_like(prices), np.zeros_like(prices), np.zeros_like(prices)
    
    rolling_mean = pd.Series(prices).rolling(window=window, min_periods=1).mean()
    rolling_std = pd.Series(prices).rolling(window=window, min_periods=1).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, rolling_mean, lower_band

async def analyze_symbol(symbol, interval=Client.KLINE_INTERVAL_1HOUR):
    """Analyze a cryptocurrency symbol using multiple indicators"""
    try:
        if client is None:
            logger.error("Binance client not initialized")
            return None
            
        klines = client.get_klines(symbol=symbol, interval=interval, limit=100)
        if not klines or len(klines) < 50:
            logger.warning(f"Insufficient data for {symbol}")
            return None

        closes = [float(k[4]) for k in klines]
        highs = [float(k[2]) for k in klines]
        lows = [float(k[3]) for k in klines]
        volumes = [float(k[5]) for k in klines]
        
        current_price = closes[-1]
        
        rsi = calculate_rsi(closes)
        if len(rsi) == 0:
            return None
        rsi_value = rsi[-1]
        
        macd, signal_line, histogram = calculate_macd(closes)
        if len(macd) == 0:
            return None
            
        macd_value = macd.iloc[-1] if hasattr(macd, 'iloc') else macd[-1]
        macd_signal = signal_line.iloc[-1] if hasattr(signal_line, 'iloc') else signal_line[-1]
        
        upper_bb, middle_bb, lower_bb = calculate_bollinger_bands(closes)
        if len(upper_bb) == 0:
            return None
            
        upper_bb_value = upper_bb.iloc[-1] if hasattr(upper_bb, 'iloc') else upper_bb[-1]
        lower_bb_value = lower_bb.iloc[-1] if hasattr(lower_bb, 'iloc') else lower_bb[-1]
        
        if len(volumes) >= 20:
            avg_volume = sum(volumes[-20:]) / 20
            current_volume = volumes[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        else:
            volume_ratio = 1
        
        signal_strength = 0
        signal_direction = "NEUTRAL"
        
        if rsi_value > 70:
            signal_strength -= 2
            signal_direction = "BEARISH"
        elif rsi_value < 30:
            signal_strength += 2
            signal_direction = "BULLISH"
        
        if macd_value > macd_signal:
            signal_strength += 1
            signal_direction = "BULLISH" if signal_strength > 0 else signal_direction
        else:
            signal_strength -= 1
            signal_direction = "BEARISH" if signal_strength < 0 else signal_direction
            
        if current_price < lower_bb_value:
            signal_strength += 1
            signal_direction = "BULLISH"
        elif current_price > upper_bb_value:
            signal_strength -= 1
            signal_direction = "BEARISH"
            
        if volume_ratio > 1.5:
            signal_strength = abs(signal_strength) * 1.5
        
        if abs(signal_strength) < 2:
            return None
            
        position = "BUY" if signal_strength > 0 else "SELL"
        
        if len(highs) >= 3 and len(lows) >= 3:
            atr = np.mean([highs[-1] - lows[-1], highs[-2] - lows[-2], highs[-3] - lows[-3]])
        else:
            atr = current_price * 0.02
            
        if position == "BUY":
            stop_loss = current_price - (atr * 1.5)
            take_profit = current_price + (atr * 3)
        else:
            stop_loss = current_price + (atr * 1.5)
            take_profit = current_price - (atr * 3)
            
        risk = abs(current_price - stop_loss)
        reward = abs(take_profit - current_price)
        risk_reward_ratio = reward / risk if risk > 0 else 0
        
        if risk_reward_ratio < 1.5:
            return None
            
        return {
            'symbol': symbol,
            'price': current_price,
            'position': position,
            'signal_strength': abs(signal_strength),
            'rsi': rsi_value,
            'macd': macd_value,
            'volume_ratio': volume_ratio,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_reward_ratio': risk_reward_ratio,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {str(e)}")
        return None

async def analyze_multiple_symbols(selected_symbols):
    """Analyze multiple symbols concurrently"""
    tasks = [analyze_symbol(symbol) for symbol in selected_symbols]
    results = await asyncio.gather(*tasks)
    return [result for result in results if result is not None]

async def generate_signals():
    """Generate trading signals for all symbols"""
    signals = []
    
    for symbol in symbols:
        try:
            signal = await analyze_symbol(symbol)
            if signal:
                signals.append(signal)
                await asyncio.sleep(0.1)
        except Exception as e:
            logger.error(f"Error processing {symbol}: {str(e)}")
            continue
    
    signals.sort(key=lambda x: x['signal_strength'], reverse=True)
    return signals

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a message when the command /start is issued."""
    user = update.effective_user
    
    # Create custom keyboard with analysis options
    keyboard = [
        [KeyboardButton("Analyze Top 5"), KeyboardButton("Get Signals")],
        [KeyboardButton("Analyze Custom"), KeyboardButton("Market Overview")],
        [KeyboardButton("Help"), KeyboardButton("Settings")]
    ]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    
    await update.message.reply_html(
        rf"Hi {user.mention_html()}! 👋",
        reply_markup=reply_markup
    )
    await update.message.reply_text(
        "I'm your AI trading assistant. I analyze crypto markets and provide trading signals.\n\n"
        "Use the buttons below to interact with me:"
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a message when the command /help is issued."""
    help_text = """
🤖 *AI Trading Bot Help*

*Available Commands:*
/start - Start the bot
/analyze5 - Analyze top 5 cryptocurrencies
/analyze_custom - Analyze specific cryptocurrencies
/signals - Get current trading signals
/overview - Market overview
/help - Show this help message

*How it works:*
1. I analyze cryptocurrencies using technical indicators (RSI, MACD, Bollinger Bands)
2. I only show signals with high probability of profit
3. Each signal includes entry price, stop loss, take profit, and risk/reward ratio

*Note:* This is not financial advice. Always do your own research and trade responsibly.
    """
    try:
        await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        logger.error(f"Error sending help message: {str(e)}")
        # Fallback to plain text if Markdown fails
        plain_text = help_text.replace('*', '').replace('_', '').replace('\\', '')
        await update.message.reply_text(plain_text)

async def analyze_top5(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Analyze top 5 cryptocurrencies and provide comparative analysis."""
    message = await update.message.reply_text("🔄 Analyzing top 5 cryptocurrencies...")
    
    try:
        # Select top 5 symbols
        selected_symbols = symbols[:5]
        results = await analyze_multiple_symbols(selected_symbols)
        
        if not results:
            await message.edit_text("No strong signals found among top 5 cryptocurrencies.")
            return
            
        # Prepare comparative analysis
        response = "📊 *Top 5 Cryptocurrency Analysis*\n\n"
        
        # Add summary of signals
        bullish_count = sum(1 for r in results if r['position'] == 'BUY')
        bearish_count = sum(1 for r in results if r['position'] == 'SELL')
        
        response += f"📈 Bullish: {bullish_count} | 📉 Bearish: {bearish_count}\n\n"
        
        # Add details for each cryptocurrency
        for i, result in enumerate(results):
            emoji = "🟢" if result['position'] == 'BUY' else "🔴"
            response += f"{i+1}. {emoji} *{result['symbol']}* - {result['position']}\n"
            response += f"   Price: ${result['price']:.4f} | RSI: {result['rsi']:.1f}\n"
            response += f"   Signal Strength: {result['signal_strength']:.1f}/5\n"
            response += f"   Risk/Reward: 1:{result['risk_reward_ratio']:.2f}\n\n"
        
        # Add overall market sentiment
        if bullish_count > bearish_count:
            sentiment = "BULLISH 📈"
        elif bearish_count > bullish_count:
            sentiment = "BEARISH 📉"
        else:
            sentiment = "NEUTRAL ➡️"
            
        response += f"*Overall Market Sentiment:* {sentiment}\n\n"
        response += "_Analysis completed: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "_"
        
        # Create inline keyboard for detailed views
        keyboard = []
        for result in results[:3]:  # Add buttons for top 3
            keyboard.append([InlineKeyboardButton(
                f"Details {result['symbol']}", 
                callback_data=f"detail_{result['symbol']}"
            )])
        
        reply_markup = InlineKeyboardMarkup(keyboard) if keyboard else None
        
        await message.edit_text(response, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup)
        
    except Exception as e:
        logger.error(f"Error in analyze_top5: {str(e)}")
        await message.edit_text("❌ Error analyzing cryptocurrencies. Please try again later.")

async def analyze_custom(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Let user select which cryptocurrencies to analyze."""
    # Create inline keyboard with cryptocurrency options
    keyboard = []
    row = []
    
    for i, symbol in enumerate(symbols):
        row.append(InlineKeyboardButton(symbol, callback_data=f"analyze_{symbol}"))
        if (i + 1) % 4 == 0:  # 4 buttons per row
            keyboard.append(row)
            row = []
    
    if row:  # Add any remaining buttons
        keyboard.append(row)
    
    # Add select all option
    keyboard.append([InlineKeyboardButton("Analyze Top 5", callback_data="analyze_top5")])
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        "Select cryptocurrencies to analyze:",
        reply_markup=reply_markup
    )

async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle callback queries from inline keyboards."""
    query = update.callback_query
    await query.answer()
    
    data = query.data
    
    if data.startswith("detail_"):
        # Show detailed analysis for a specific symbol
        symbol = data.split("_")[1]
        await show_symbol_detail(query, symbol)
    elif data.startswith("analyze_"):
        if data == "analyze_top5":
            await analyze_top5_callback(query)
        else:
            symbol = data.split("_")[1]
            await analyze_single_symbol(query, symbol)

async def show_symbol_detail(query, symbol):
    """Show detailed analysis for a specific symbol."""
    message = await query.edit_message_text(f"🔍 Analyzing {symbol}...")
    
    try:
        result = await analyze_symbol(symbol)
        
        if not result:
            await message.edit_text(f"No analysis available for {symbol}.")
            return
            
        emoji = "🟢" if result['position'] == 'BUY' else "🔴"
        
        response = f"📊 *Detailed Analysis for {result['symbol']}*\n\n"
        response += f"{emoji} *Position:* {result['position']}\n"
        response += f"💰 *Price:* ${result['price']:.4f}\n"
        response += f"💪 *Signal Strength:* {result['signal_strength']:.1f}/5\n"
        response += f"📉 *RSI:* {result['rsi']:.1f}\n"
        response += f"📊 *MACD:* {result['macd']:.4f}\n"
        response += f"📈 *Volume Ratio:* {result['volume_ratio']:.2f}x\n"
        response += f"🛑 *Stop Loss:* ${result['stop_loss']:.4f}\n"
        response += f"🎯 *Take Profit:* ${result['take_profit']:.4f}\n"
        response += f"⚖️ *Risk/Reward:* 1:{result['risk_reward_ratio']:.2f}\n\n"
        response += "_Analysis completed: " + result['timestamp'] + "_"
        
        await message.edit_text(response, parse_mode=ParseMode.MARKDOWN)
        
    except Exception as e:
        logger.error(f"Error in show_symbol_detail: {str(e)}")
        await message.edit_text("❌ Error retrieving detailed analysis. Please try again later.")

async def analyze_top5_callback(query):
    """Handle analyze top5 callback."""
    await query.edit_message_text("🔄 Analyzing top 5 cryptocurrencies...")
    # We can't directly call analyze_top5, so we need to simulate it
    # For simplicity, we'll just show a message and let the user use the command
    await query.edit_message_text("Please use the /analyze5 command from the main menu to analyze top 5 cryptocurrencies.")

async def analyze_single_symbol(query, symbol):
    """Analyze a single symbol from callback."""
    await show_symbol_detail(query, symbol)

async def send_signals(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send trading signals to user."""
    message = await update.message.reply_text("🔍 Looking for profitable signals...")
    
    try:
        signals = await generate_signals()
        
        if not signals:
            await message.edit_text("No strong trading signals at the moment. Please check back later.")
            return
            
        # Send the strongest signal as the main message
        signal = signals[0]
        emoji = "🟢" if signal['position'] == 'BUY' else "🔴"
        
        response = f"🚀 *Strong Trading Signal* 🚀\n\n"
        response += f"{emoji} *{signal['symbol']}* - {signal['position']}\n"
        response += f"📊 Price: ${signal['price']:.4f}\n"
        response += f"💪 Strength: {signal['signal_strength']:.1f}/5\n"
        response += f"📉 RSI: {signal['rsi']:.1f}\n"
        response += f"🛑 Stop Loss: ${signal['stop_loss']:.4f}\n"
        response += f"🎯 Take Profit: ${signal['take_profit']:.4f}\n"
        response += f"⚖️ Risk/Reward: 1:{signal['risk_reward_ratio']:.2f}\n\n"
        response += "_Signal generated at: " + signal['timestamp'] + "_"
        
        await message.edit_text(response, parse_mode=ParseMode.MARKDOWN)
        
        # Send additional signals if available
        if len(signals) > 1:
            for signal in signals[1:3]:
                await asyncio.sleep(1)
                emoji = "🟢" if signal['position'] == 'BUY' else "🔴"
                additional_response = f"{emoji} *{signal['symbol']}* - {signal['position']}\n"
                additional_response += f"Price: ${signal['price']:.4f} | RSI: {signal['rsi']:.1f}\n"
                additional_response += f"Risk/Reward: 1:{signal['risk_reward_ratio']:.2f}"
                
                await update.message.reply_text(additional_response, parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        logger.error(f"Error in send_signals: {str(e)}")
        await message.edit_text("❌ Error generating signals. Please try again later.")

async def market_overview(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Provide a general market overview."""
    message = await update.message.reply_text("📈 Generating market overview...")
    
    try:
        # Analyze all symbols for overview
        signals = await generate_signals()
        
        if not signals:
            await message.edit_text("No market data available at the moment.")
            return
            
        # Count bullish and bearish signals
        bullish = [s for s in signals if s['position'] == 'BUY']
        bearish = [s for s in signals if s['position'] == 'SELL']
        
        # Calculate average signal strength
        avg_strength = sum(s['signal_strength'] for s in signals) / len(signals) if signals else 0
        
        # Determine market sentiment
        if len(bullish) > len(bearish):
            sentiment = "BULLISH 📈"
            emoji = "🟢"
        elif len(bearish) > len(bullish):
            sentiment = "BEARISH 📉"
            emoji = "🔴"
        else:
            sentiment = "NEUTRAL ➡️"
            emoji = "🟡"
            
        # Get top signals
        top_bullish = sorted(bullish, key=lambda x: x['signal_strength'], reverse=True)[:3]
        top_bearish = sorted(bearish, key=lambda x: x['signal_strength'], reverse=True)[:3]
        
        response = f"🌐 *Market Overview* 🌐\n\n"
        response += f"{emoji} *Market Sentiment:* {sentiment}\n"
        response += f"📊 Total Assets: {len(signals)}\n"
        response += f"🟢 Bullish: {len(bullish)} | 🔴 Bearish: {len(bearish)}\n"
        response += f"💪 Average Signal Strength: {avg_strength:.1f}/5\n\n"
        
        if top_bullish:
            response += "🔥 *Top Bullish Signals:*\n"
            for i, signal in enumerate(top_bullish):
                response += f"{i+1}. {signal['symbol']} (Strength: {signal['signal_strength']:.1f}/5)\n"
            response += "\n"
            
        if top_bearish:
            response += "💥 *Top Bearish Signals:*\n"
            for i, signal in enumerate(top_bearish):
                response += f"{i+1}. {signal['symbol']} (Strength: {signal['signal_strength']:.1f}/5)\n"
            response += "\n"
            
        response += "_Generated at: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "_"
        
        await message.edit_text(response, parse_mode=ParseMode.MARKDOWN)
        
    except Exception as e:
        logger.error(f"Error in market_overview: {str(e)}")
        await message.edit_text("❌ Error generating market overview. Please try again later.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle regular text messages."""
    text = update.message.text.lower()
    
    if 'analyze top' in text or 'top 5' in text:
        await analyze_top5(update, context)
    elif 'analyze custom' in text:
        await analyze_custom(update, context)
    elif 'signal' in text:
        await send_signals(update, context)
    elif 'overview' in text or 'market' in text:
        await market_overview(update, context)
    elif 'help' in text:
        await help_command(update, context)
    else:
        await update.message.reply_text(
            "I'm not sure what you mean. Try one of the buttons or use /help for guidance."
        )

async def send_message_with_fallback(chat_id, text, context, parse_mode=ParseMode.MARKDOWN):
    """Send a message with fallback to plain text if Markdown fails."""
    try:
        await context.bot.send_message(chat_id=chat_id, text=text, parse_mode=parse_mode)
    except Exception as e:
        logger.error(f"Error sending Markdown message: {str(e)}")
        # Fallback to plain text
        plain_text = text.replace('*', '').replace('_', '').replace('\\', '')
        await context.bot.send_message(chat_id=chat_id, text=plain_text)

async def broadcast_signals(context: CallbackContext):
    """Broadcast signals to channel (runs automatically)."""
    try:
        signals = await generate_signals()
        
        if not signals:
            logger.info("No signals to broadcast")
            return
            
        # Filter only strong signals (strength >= 3)
        strong_signals = [s for s in signals if s['signal_strength'] >= 3.0]
        
        if not strong_signals:
            logger.info("No strong signals to broadcast")
            return
            
        # Check if we've already sent these signals recently (within last 6 hours)
        current_time = datetime.now()
        new_signals = []
        
        for signal in strong_signals:
            signal_key = f"{signal['symbol']}_{signal['position']}"
            last_sent = last_signals_sent.get(signal_key)
            
            # If we haven't sent this signal before or it's been more than 6 hours
            if not last_sent or (current_time - last_sent).total_seconds() > 6 * 3600:
                new_signals.append(signal)
                last_signals_sent[signal_key] = current_time
        
        if not new_signals:
            logger.info("No new signals to broadcast")
            return
            
        # Send the strongest signal as the main message
        signal = new_signals[0]
        emoji = "🟢" if signal['position'] == 'BUY' else "🔴"
        
        message = f"🎯 *AUTOMATED TRADING SIGNAL* 🎯\n\n"
        message += f"{emoji} *{signal['symbol']}* - {signal['position']}\n"
        message += f"💰 Price: ${signal['price']:.4f}\n"
        message += f"💪 Strength: {signal['signal_strength']:.1f}/5\n"
        message += f"📊 RSI: {signal['rsi']:.1f}\n"
        message += f"🛑 Stop Loss: ${signal['stop_loss']:.4f}\n"
        message += f"🎯 Take Profit: ${signal['take_profit']:.4f}\n"
        message += f"⚖️ Risk/Reward: 1:{signal['risk_reward_ratio']:.2f}\n\n"
        message += "_Signal generated at: " + signal['timestamp'] + "_"
        message += "\n\n#TradingSignal #Crypto #AlgorithmicTrading"
        
        await send_message_with_fallback(TELEGRAM_CHANNEL_ID, message, context)
        
        # Send additional strong signals if available (with delay between them)
        if len(new_signals) > 1:
            for signal in new_signals[1:3]:  # Limit to 3 signals max
                await asyncio.sleep(2)  # Delay between messages
                emoji = "🟢" if signal['position'] == 'BUY' else "🔴"
                additional_message = f"{emoji} *{signal['symbol']}* - {signal['position']}\n"
                additional_message += f"Price: ${signal['price']:.4f} | RSI: {signal['rsi']:.1f}\n"
                additional_message += f"Risk/Reward: 1:{signal['risk_reward_ratio']:.2f}"
                
                await send_message_with_fallback(TELEGRAM_CHANNEL_ID, additional_message, context)
                
        logger.info(f"Broadcasted {len(new_signals)} signals to channel")
        
    except Exception as e:
        logger.error(f"Error in broadcast_signals: {str(e)}")

def main():
    """Start the bot."""
    # Initialize Binance client
    try:
        client = Client(BINANCE_API_KEY, BINANCE_SECRET_KEY, testnet=False)
        logger.info("Binance client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Binance client: {str(e)}")
        client = None
    
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("analyze5", analyze_top5))
    application.add_handler(CommandHandler("analyze_custom", analyze_custom))
    application.add_handler(CommandHandler("signals", send_signals))
    application.add_handler(CommandHandler("overview", market_overview))
    application.add_handler(CallbackQueryHandler(handle_callback))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Set up job queue for automatic signal broadcasting
    job_queue = application.job_queue
    
    # Run signal broadcasting every 30 minutes
    job_queue.run_repeating(broadcast_signals, interval=1800, first=10)
    
    # Run market analysis every hour
    job_queue.run_repeating(lambda context: asyncio.create_task(generate_signals()), interval=3600, first=30)
    
    logger.info("Bot started successfully")
    application.run_polling()

if __name__ == '__main__':
    main()
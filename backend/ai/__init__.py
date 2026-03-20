"""ai/ — AI agents: Ollama trade confirmation + Ollama in-play detector."""
from backend.ai.ollama_agent import evaluate_setup, evaluate_setups_batch, check_ollama_status, AgentVerdict
from backend.ai.inplay_detector import get_in_play, refresh_in_play, InPlayResult
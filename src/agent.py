# agent.py
import logging
import os
from google.genai import types
from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    ChatContext,
    ChatMessage
)
from livekit.plugins import noise_cancellation, silero, google
from livekit.plugins.elevenlabs import TTS as ElevenLabsTTS  # Import ElevenLabs TTS
from livekit.plugins import assemblyai

# Import the new FAISS-based RAG system
from rag_faiss import rag_lookup

logger = logging.getLogger("agent")
load_dotenv()



# ----------------------
# Assistant class
# ----------------------
class CollegeAssistant(Agent):
    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage):
        # FIXED: Changed from new_message.text_content() to new_message.text_content (removed parentheses)
        query = new_message.text_content
        if not query:
            return

        # Retrieve relevant college info using FAISS RAG
        rag_content = rag_lookup(query, top_k=5)  # Reduced from 200 for better context
        print(f"RAG Content: {rag_content}")
        
        # Add RAG content as context for the model to use
        # Let the model decide whether to use RAG, web search, or both
        turn_ctx.add_message(
            role="system",
            content=f"""Here is relevant information from Shree Devi College's database:
            {rag_content}

            Please use this information when answering questions about Shree Devi College. 
            If the information is insufficient or outdated, you may use web search to supplement your response.
            Always prioritize the college database information over web search results."""
        )

    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a helpful AI assistant for Shree Devi College.
            Only answer questions related to Shree Devi College.
            If asked about other colleges or topics, politely say you only provide info about Shree Devi College.
            
            When answering:
            1. First, use the information provided from Shree Devi College's database
            2. If the database information is insufficient or you need more current information, use web search
            3. Always cite your sources when using web search information
            4. Prioritize accuracy and relevance to Shree Devi College"""
        )

# ----------------------
# Prewarm function
# ----------------------
def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

# ----------------------
# Entry point
# ----------------------
async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    session = AgentSession(
        stt = assemblyai.STT(),
        llm=google.LLM(
            model="gemini-2.5-flash",
            temperature=0.8,
            gemini_tools=[types.GoogleSearch()],
        ),
#         tts = google.beta.GeminiTTS(
#    model="gemini-2.5-flash-tts",
#    voice_name="Zephyr",
#    instructions="Speak in a friendly and engaging tone.",
#   ),
        # Use ElevenLabs TTS
        tts=ElevenLabsTTS(
        model="eleven_multilingual_v2",
        voice_id="ODq5zmih8GrVes37Dizd",
    ),
    )

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=CollegeAssistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC()
        ),
    )

    await session.generate_reply(
        instructions="Greet the user and offer help about Shree Devi College."
    )
    await ctx.connect()

# ----------------------
# Main
# ----------------------
if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))